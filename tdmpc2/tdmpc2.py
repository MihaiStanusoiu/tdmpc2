import os

import torch
import torch.nn.functional as F

from common import math
from common.layers import api_model_conversion
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict

from graphviz import Digraph
from torchviz import make_dot

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		# self.uncompiled_model = self.model
		# if self.cfg.compile:
		# 	self.model = torch.compile(self.model, mode="reduce-overhead")
		lr = torch.tensor(cfg.lr, device=self.device)
		self.optim = torch.optim.Adam([
			# {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._rnn.parameters(), 'lr': self.cfg.lr*torch.tensor(self.cfg.enc_lr_scale, device=self.device)},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
			{'params': [self.model.initial_h] if self.cfg.learned_init_h else []},
		], lr=lr, capturable=True)
		if not self.cfg.freeze_pi:
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=lr*torch.tensor(self.cfg.pi_lr_scale, device=self.device), eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.plan_horizon, self.cfg.action_dim, device=self.device))
		# self._first_update = True
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
			self._burn_in_rollout = torch.compile(self._burn_in_rollout, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp, metrics):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({
			"model": self.model.state_dict(),
			"wm_optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
			"metrics": metrics,
		}, fp)

	def load(self, fp, load_pi_only=False):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.get_default_device(), weights_only=False)

		self.loss = state_dict["metrics"]
		# self.optim.load_state_dict(state_dict['wm_optim'])
		if not self.cfg.freeze_pi:
			self.pi_optim.load_state_dict(state_dict['pi_optim'])
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		if load_pi_only:
			encoder_state_dict = state_dict.copy()
			for k, v in list(state_dict.items()):
				if k.startswith("_pi"):
					state_dict[k.split('_pi.')[1]] = v
				del state_dict[k]
			pi = self.model._pi
			pi.load_state_dict(state_dict)
			for param in pi:
				param.requires_grad = False
			for k, v in list(encoder_state_dict.items()):
				if k.startswith("_encoder"):
					encoder_state_dict[k.split('_encoder.')[1]] = v
				del encoder_state_dict[k]
			encoder = self.model._encoder
			encoder.load_state_dict(encoder_state_dict)
			for param in encoder['state']:
				param.requires_grad = False
			return
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		# assert not set(TensorDict(self.model.state_dict()).keys()).symmetric_difference(set(TensorDict(state_dict).keys()))
		self.model.load_state_dict(state_dict)
		# if self.cfg.compile:
		# 	self.model = torch.compile(self.uncompiled_model, mode="reduce-overhead")
		return

	@property
	def initial_h(self):
		if self.cfg.learned_init_h:
			return torch.tanh(self.model.initial_h)
		return torch.zeros(1, self.cfg.hidden_dim, device=self.device)

	@torch.no_grad()
	def act(self, obs, prev_act, t0=False, h=None, info={}, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		prev_act = prev_act.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			o = self.model.encode(obs, task)
			tensor_dt = None
			if info.get("timestamp") is not None:
				tensor_dt = torch.tensor(info['timestamp'], dtype=torch.float, device=self.device, requires_grad=False).reshape((1, 1))
			z, h = self.model.rnn(o, prev_act, task, h, dt=tensor_dt)
			torch.compiler.cudagraph_mark_step_begin()
			a = self.plan(h.clone(), t0=torch.tensor(t0, device=self.device), dt=tensor_dt, eval_mode=torch.tensor(eval_mode, device=self.device), task=task)
		else:
			z = self.model.encode(obs, task)
			a = self.model.pi(z, h, task)[int(not eval_mode)][0]
		return a.cpu(), h

	@torch.no_grad()
	def _estimate_value(self, h, actions, task, dt=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.plan_horizon):
			reward = math.two_hot_inv(self.model.reward(h, actions[t], task), self.cfg)
			h = self.model.forward(h, actions[t], task, dt=dt)
			G += discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		return G + discount * self.model.Q(h, self.model.pi(h, task)[1], task, return_type='avg')

	@torch.no_grad()
	def _plan(self, h, t0=torch.tensor(False), dt=None, eval_mode=torch.tensor(False), task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.plan_horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_h = h.repeat(self.cfg.num_pi_trajs, 1)
			_dt = dt.repeat(self.cfg.num_pi_trajs, 1) if dt is not None else None
			for t in range(self.cfg.plan_horizon-1):
				pi_actions[t] = self.model.pi(_h, task)[1]
				_h = self.model.forward(_h, pi_actions[t], task, dt=_dt)
			pi_actions[-1] = self.model.pi(_h, task)[1]

		# Initialize state and parameters
		h = h.repeat(self.cfg.num_samples, 1) if h is not None else None
		dt = dt.repeat(self.cfg.num_samples, 1) if dt is not None else None
		mean = torch.zeros(self.cfg.plan_horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.plan_horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.plan_horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.plan_horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(h, actions, task, dt=dt).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, hs, dt, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		# hs = [h]
		# pis = []
		# log_pis = []
		# for _, (z, t) in enumerate(zip(zs.unbind(0), dt.unbind(0))):
		# 	_, pi, log_pi, _ = self.model.pi(z, h, task)
		# 	with torch.no_grad():
		# 		_, h = self.model.rnn(z, pi, task=task, h=h, dt=t)
		# 	pis.append(pi)
		# 	log_pis.append(log_pi)
		# 	hs.append(h)
		# hs = hs[:-1]
		# pis = torch.stack(pis)
		# log_pis = torch.stack(log_pis)
		# hs = torch.stack(hs)

		_, pis, log_pis, _ = self.model.pi(hs, task)

		qs = self.model.Q(hs, pis, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return pi_loss.detach(), pi_grad_norm

	@torch.no_grad()
	def _td_target(self, next_z, next_a, hs, reward, dt, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		# hs = [h]
		# pis = []
		# for _, (z, t) in enumerate(zip(next_z, dt)):
		# 	pi = self.model.pi(z, h, task)[1]
		# 	_, h = self.model.rnn(z, pi, task=task, h=h, dt=t)
		# 	pis.append(pi)
		# 	hs.append(h)
		# hs = hs[:-1]
		# pis = torch.stack(pis)
		# hs = torch.stack(hs)

		# rnn = lambda z, a, h, dt: self.model.rnn(z, a, task, h, dt)
		# f_rnn = functorch.vmap(rnn, in_dims=0, out_dims=0)
		# _, next_hs = f_rnn(next_z, next_a, hs, dt)
		pis = self.model.pi(hs, task)[1]

		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(hs, pis, task, return_type='min', target=True)

	@torch.no_grad()
	def _td_lambda_target(self, next_z, next_a, hs, reward, dt, task):
		pis = self.model.pi(hs, task)[1]
		qs = self.model.Q(hs, pis, task, return_type='min', target=True)
		Rs = [qs[-1]]
		intermediates = (reward + self.discount * (1 - self.cfg.lmbda) * qs)
		for t in reversed(range(self.cfg.horizon - 1)):
			Rs.append(intermediates[t] + self.discount * self.cfg.lmbda * Rs[-1])

		return torch.stack(list(reversed(Rs)), dim=0)

	@staticmethod
	def _mask(value, mask):
		# Apply element-wise multiplication with broadcasting in PyTorch
		return value * mask.to(value.dtype)

	@torch.no_grad()
	def _burn_in_rollout(self, obs_t0, obs, action, hidden, is_first, task=None):
		"""
		Perform a burn-in rollout to initialize hidden states.

		Args:
			obs (torch.Tensor): Observation from the environment.
			action (torch.Tensor): Action taken in the environment.
			hidden (torch.Tensor): Hidden state of the RNN.
			is_first (torch.Tensor): Whether the observation is the first in the episode.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Hidden state after the burn-in rollout.
		"""
		with torch.no_grad():
			z = self.model.encode(obs_t0, task)
			# h = hidden[0]
			h = self.initial_h
			for t, (_obs, _action, _is_first) in enumerate(zip(obs.unbind(0), action.unbind(0), is_first.unbind(0))):
				h = self._mask(h, 1.0 - _is_first.float())
				h = h + self._mask(self.initial_h.detach(), _is_first.float())
				_, h = self.model.next(z, _action, task, h)
				z = self.model.encode(_obs, task)

		return h

	def _update(self, prev_obs, prev_action, prev_dt, hidden, obs, action, reward, dt, is_first, task=None):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""

		h = hidden

		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			next_act = action[1:]
			next_dt = dt[1:]

		if self.cfg.warmup_h:
			with torch.no_grad():
				if prev_obs is not None:
					# ignore prev_obs and prev_act that are zero
					# mask = (prev_obs[0].abs().sum(dim=-1) > 0)  # Shape (seq_len, batch_size)
					# first_nonzero_idx = mask.int().argmax(dim=1)  # Shape (batch_size,)
					# prev_obs = [prev_obs[0][first_nonzero_idx[i]:, i] for i in range(self.cfg.batch_size)]
					# prev_obs = prev_obs[0][:, first_nonzero_idx:-1, :]
					# prev_action = prev_action[0][:, first_nonzero_idx:-1, :]
					prev_z = self.model.encode(prev_obs, task)
					for _, (_a, _z, _dt) in enumerate(
							zip(prev_action.unbind(0), prev_z.unbind(0), prev_dt.unbind(0))):
						_, h = self.model.rnn(_z, _a, task, h, _dt)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		hs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.hidden_dim, device=self.device)

		o = self.model.encode(obs[0], task)
		z, h = self.model.rnn(o, action[0], task, h, dt[0])
		zs[0] = z
		hs[0] = h
		one_step_prediction_error = 0
		consistency_loss = 0
		for t, (_action, _next_z, _dt, _is_first) in enumerate(zip(next_act.unbind(0), next_z.unbind(0), next_dt.unbind(0), is_first.unbind(0))):
			if t < self.cfg.horizon / 2:
				z = h
			z = self.model.next(z, _action, task)
			# z, h = self.model.forward(h, _action, task, dt=_dt)
			_, h = self.model.rnn(_next_z, _action, task, h, _dt)
			consistency_loss = consistency_loss + F.mse_loss(z, h.detach()) * (self.cfg.rho) ** (t)
			if t == 0:
				one_step_prediction_error = consistency_loss

			zs[t+1] = z
			hs[t+1] = h

		# Predictions
		_zs = zs[:-1]
		_hs = hs[:-1]

		# # Q-value discrepancy for timestep t and t+H
		# with torch.no_grad():
		# 	q_hat = self.model.Q(_zs, action, _hs, task, return_type='avg')
		# 	q = self.model.Q(_zs, action, hidden[:-1], task, return_type='avg')
		# 	# get maximum q value to normalize
		# 	q_max = q_hat.max()
		# 	# Q-value discrepancy for t=0 and t=H
		# 	q_discrepancy = (q_hat - q) / q_max
		# 	q_discrepancy_t = q_discrepancy[0]
		# 	q_discrepancy_H = q_discrepancy[-1]

		qs = self.model.Q(_hs, next_act, task, return_type='all')
		reward_preds = self.model.reward(_hs, next_act, task)

		# Compute targets
		with torch.no_grad():
			# td_targets = self._td_target(next_z, next_act, hs[1:].detach(), reward, dt[1:], task)
			td_targets = self._td_lambda_target(next_z, next_act, hs[1:].detach(), reward, dt[1:], task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		# for _, qs_unbind in enumerate(qs.unbind(0)):
		# 	value_loss = value_loss + math.soft_ce(qs_unbind[0], td_targets[0],
		# 										   self.cfg).mean()

		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		# value_loss = value_loss / (self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		# if self._first_update and not self.cfg.compile:
		# 	make_dot(total_loss, dict(list(self.model.named_parameters()))).view()
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		if not self.cfg.freeze_pi:
			# Update policy
			pi_loss, pi_grad_norm = self.update_pi(_hs.detach(),dt, task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()

		return consistency_loss.detach(), reward_loss.detach(), value_loss.detach(), total_loss.detach(), one_step_prediction_error.detach(), grad_norm.detach(), pi_loss.detach(), pi_grad_norm.detach(), hs.detach()
		# self._first_update = False
		# info_dict = {
		# 	"consistency_loss": consistency_loss,
		# 	"reward_loss": reward_loss,
		# 	"value_loss": value_loss,
		# 	"total_loss": total_loss,
		# 	"one_step_prediction_error": one_step_prediction_error,
		# 	"grad_norm": grad_norm,
		# 	"pi_scale": self.scale.value,
		# 	"pi_loss": pi_loss,
		# 	"pi_grad_norm": pi_grad_norm
		# }
		# self.loss = info_dict
		# return info_dict


	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, hist_obs, hist_act, hidden, reward, done, dt, is_first, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		# prev_obs = torch.empty(0, device=self.device)
		# prev_act = torch.empty(0, device=self.device)
		# prev_dt = torch.empty(0, device=self.device)
		# if self.cfg.burn_in > 0:
		# 	prev_obs = obs[:self.cfg.burn_in]
		# 	prev_act = action[:self.cfg.burn_in]
		# 	prev_dt = dt[:self.cfg.burn_in]
		# h = self._burn_in_rollout(obs[0], obs[1:self.cfg.burn_in+1], action[:self.cfg.burn_in], hidden[:self.cfg.burn_in], is_first[:self.cfg.burn_in], **kwargs)
		# return self._update(prev_obs, prev_act, prev_dt, obs[self.cfg.burn_in:], action[self.cfg.burn_in:], reward, dt[self.cfg.burn_in:], is_first, **kwargs)
		if self.cfg.stored_h:
			# numpy to tensor
			h = torch.tensor(hidden, device=self.device).detach()
		else:
			h = self.initial_h.repeat(self.cfg.batch_size, 1)
		consistency_loss, reward_loss, value_loss, total_loss, one_step_prediction_error, grad_norm, pi_loss, pi_grad_norm, hs = self._update(obs[:self.cfg.burn_in], action[:self.cfg.burn_in], dt[:self.cfg.burn_in], h, obs[self.cfg.burn_in:], action[self.cfg.burn_in:], reward, dt[self.cfg.burn_in:], is_first, **kwargs)
		# log h rank
		hs_TxB = hs.reshape(-1, hs.shape[-1])
		hs_rank = torch.linalg.matrix_rank(hs_TxB, tol=1e-5)
		_, S, _ = torch.linalg.svd(hs_TxB)
		effective_rank = torch.exp(-torch.sum(S / (S.sum() + 1e-5) * torch.log(S / (S.sum() + 1e-5) + 1e-5)))
		self.loss = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
			"one_step_prediction_error": one_step_prediction_error,
			"grad_norm": grad_norm,
			"pi_scale": self.scale.value,
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"hs_rank": hs_rank,
			"hs_effective_rank": effective_rank
		})
		return self.loss

