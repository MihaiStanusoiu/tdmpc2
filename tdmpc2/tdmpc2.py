import os

import torch
import torch.nn.functional as F

from common import math
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
		self.uncompiled_model = self.model
		# if self.cfg.compile:
		# 	self.model = torch.compile(self.model, mode="reduce-overhead")
		self.optim = torch.optim.Adam([
			# {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._hmm.parameters()},
			{'params': self.model._rnn.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
			{'params': [self.model.initial_h] if self.cfg.learned_init_h else []},
		], lr=self.cfg.lr, capturable=True)
		if not self.cfg.freeze_pi:
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._first_update = True
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
		self.optim.load_state_dict(state_dict['wm_optim'])
		if not self.cfg.freeze_pi:
			self.pi_optim.load_state_dict(state_dict['pi_optim'])
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		def load_sd_hook(model, local_state_dict, prefix, *args):
			name_map = [
				"weight", "bias", "ln.weight", "ln.bias",
			]
			print("Listing state dict keys (from disk)")
			for k in list(local_state_dict.keys()):
				print("\t", k)

			sd = model.state_dict()
			print("Listing dest state dict keys")
			for k in list(sd.keys()):
				print("\t", k)

			print("Maps:")
			new_sd = dict(sd)
			for cur_prefix in (prefix, "_target"+prefix[:-1]+"_"):
				for key, val in list(local_state_dict.items()):
					if not key.startswith(cur_prefix[:-1]):
						continue
					num = key[len(cur_prefix + "params."):len(cur_prefix + "params.")+1]
					new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
					new_total_key = cur_prefix + 'params.' + new_key
					print("\t", key, '-->', new_total_key)
					del local_state_dict[key]
					new_sd[new_total_key] = val
					if not cur_prefix.startswith("_target"):
						new_total_key = "_detach" + cur_prefix[:-1] + "_" + 'params.' + new_key
						print("\t", 'DETACH', key, '-->', new_total_key)
						new_sd[new_total_key] = val
			local_state_dict.update(new_sd)
			return local_state_dict
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
		# load_sd_hook(self.model, state_dict, "_Qs.")
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
	def observe(self, obs, prev_act, b, dt=None):
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		prev_act = prev_act.to(self.device, non_blocking=True).unsqueeze(0)
		tensor_dt = None
		if dt is not None:
			tensor_dt = torch.tensor(dt, dtype=torch.float, device=self.device,
									 requires_grad=False).reshape((1, 1))
		return self.model.hmm(obs, prev_act, b, tensor_dt)

	@torch.no_grad()
	def act(self, z, t0=False, info={}, eval_mode=False, task=None):
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
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			tensor_dt = None
			if info.get("timestamp") is not None:
				tensor_dt = torch.tensor(info['timestamp'], dtype=torch.float, device=self.device, requires_grad=False).reshape((1, 1))
			torch.compiler.cudagraph_mark_step_begin()
			a = self.plan(z, t0=torch.tensor(t0, device=self.device), dt=tensor_dt, eval_mode=torch.tensor(eval_mode, device=self.device), task=task)
		else:
			a = self.model.pi(z, task)[int(not eval_mode)][0]
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
			z,= self.model.rnn(z, actions[t])
			G += discount * reward
			discount_update = self.discount
			discount = discount * discount_update
		return G + discount * self.model.Q(z, self.model.pi(z)[1], return_type='avg')

	@torch.no_grad()
	def _plan(self, z, t0=torch.tensor(False), dt=None, eval_mode=torch.tensor(False), task=None):
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
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			# _h = h.repeat(self.cfg.num_pi_trajs, 1)
			_dt = dt.repeat(self.cfg.num_pi_trajs, 1) if dt is not None else None
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(z, task)[1]
				_z = self.model.rnn(_z, pi_actions[t])

			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		dt = dt.repeat(self.cfg.num_samples, 1) if dt is not None else None
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions).nan_to_num(0)
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

	def update_pi(self, hs_hat, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		_, pis, log_pis, _ = self.model.pi(hs_hat, task)

		qs = self.model.Q(hs_hat, pis, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		# For some reason, cudagraph prefers to see the zero grad after step
		self.pi_optim.zero_grad(set_to_none=True)

		return pi_loss.detach(), pi_grad_norm

	@torch.no_grad()
	def _td_target(self, next_zs, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
				"""
		pi = self.model.pi(next_zs, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_zs, pi, task, return_type='min', target=True)

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
			z = self.model.hmm(obs_t0, task)
			# h = hidden[0]
			h = self.initial_h
			for t, (_obs, _action, _is_first) in enumerate(zip(obs.unbind(0), action.unbind(0), is_first.unbind(0))):
				h = self._mask(h, 1.0 - _is_first.float())
				h = h + self._mask(self.initial_h.detach(), _is_first.float())
				_, h = self.model.next(z, _action, task, h)
				z = self.model.hmm(_obs, task)

		return h

	def _update(self, prev_obs, prev_action, prev_dt, prev_is_first, obs, action, reward, dt, is_first, task=None):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""


		# Prepare for update
		self.model.train()

		# Encoding memory
		b = self.initial_h.repeat(self.cfg.batch_size, 1)
		# h = self.initial_h.repeat(self.cfg.batch_size, 1)

		z, b = self.model.hmm(prev_obs[0], prev_action[0], b, prev_dt[0])
		next_prev_obs = prev_obs[1:]
		next_prev_act = prev_action[1:]
		next_prev_dt = prev_dt[1:]

		for _, (_a, _obs, _dt) in enumerate(
					zip(next_prev_act.unbind(0), next_prev_obs.unbind(0), next_prev_dt.unbind(0))):
			# _, h = self.model.rnn(z, _a, h)
			z, b = self.model.hmm(_obs, _a, b, _dt)

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+2, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		zs_hat = torch.empty(self.cfg.horizon+2, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		# hs = torch.empty(self.cfg.horizon+2, self.cfg.batch_size, self.cfg.hidden_dim, device=self.device)
		bs = torch.empty(self.cfg.horizon+2, self.cfg.batch_size, self.cfg.hidden_dim, device=self.device)

		bs[0] = b
		# hs[0] = h
		zs[0] = z
		zs_hat[0] = z.detach()
		dynamics_loss = 0
		consistency_loss = 0
		one_step_prediction_error = 0
		for t, (_action, _obs, _dt, _is_first) in enumerate(zip(action.unbind(0), obs.unbind(0), dt.unbind(0), is_first.unbind(0))):
			# (z_{t+1}, h_{t+1}) = f(z_t, a_{t-1}, h_{t-1})
			zs_hat[t+1] = self.model.rnn(zs_hat[t], _action)
			zs[t+1], bs[t+1] = self.model.hmm(_obs, _action, bs[t], _dt)

			# KL Divergence loss
			consistency_loss = consistency_loss + F.mse_loss(zs_hat[t+1].detach(), zs[t+1]) * self.cfg.rho**t
			dynamics_loss = dynamics_loss + F.mse_loss(zs_hat[t+1], zs[t+1].detach()) * self.cfg.rho**t
			# consistency_loss = consistency_loss + torch.nn.KLDivLoss(F.log_softmax(zs_probs_hat[t]), F.softmax(zs_probs[t+1])) * self.cfg.rho**t

			if t == 0:
				one_step_prediction_error = consistency_loss

		# Predictions
		_zs = zs_hat[:-1].detach()

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

		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action)

		# Compute targets
		with torch.no_grad():
			td_targets = self._td_target(zs[1:].detach(), reward, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		dynamics_loss = (dynamics_loss / self.cfg.horizon)
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.dynamics_coef * dynamics_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		if self._first_update and not self.cfg.compile:
			make_dot(total_loss, dict(list(self.model.named_parameters()))).view()
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		if not self.cfg.freeze_pi:
			# Update policy
			pi_loss, pi_grad_norm = self.update_pi(_zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		self._first_update = False
		info_dict = {
			"consistency_loss": consistency_loss,
			"dynamics_loss": dynamics_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,

			# "pi_loss": pi_loss,
			"total_loss": total_loss,
			# "Q_discrepancy_initial_state": q_discrepancy_t,
			# "Q_discrepancy_final_state": q_discrepancy_H,
			"one_step_prediction_error": one_step_prediction_error,
			"grad_norm": grad_norm,
			# "pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		}
		if not self.cfg.freeze_pi:
			info_dict["pi_loss"] = pi_loss
			info_dict["pi_grad_norm"] = pi_grad_norm
		self.loss = info_dict
		return TensorDict(self.loss).detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, done, dt, is_first, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		# h = self._burn_in_rollout(obs[0], obs[1:self.cfg.burn_in+1], action[:self.cfg.burn_in], hidden[:self.cfg.burn_in], is_first[:self.cfg.burn_in], **kwargs)
		return self._update(obs[:self.cfg.burn_in], action[:self.cfg.burn_in], dt[:self.cfg.burn_in], is_first[:self.cfg.burn_in], obs[self.cfg.burn_in:], action[self.cfg.burn_in:], reward, dt[self.cfg.burn_in:], is_first[self.cfg.burn_in:], **kwargs)

