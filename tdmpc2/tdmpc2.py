import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict

from common import tools


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
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			# {'params': self.model._rnn.parameters()},
			{'params': self.model.dynamics.parameters()},
			# {'params': self.model._posterior.parameters()},
			{'params': self.model.heads['reward'].parameters()},
			{'params': self.model.heads['V'].parameters()},
			{'params': self.model.heads['is_first'].parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
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

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@property
	def initial_states(self):
		# return torch.tanh(self.model.initial_h).unsqueeze(0)
		h = torch.zeros(1, self.cfg.hidden_dim, device=self.device)
		z = self.model.dynamics(h)
		return { 'z': z, 'h': h }

	@torch.no_grad()
	def act(self, obs, state, prev_action, t0=False, eval_mode=False, task=None):
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
		# obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			a = self.plan(obs, state, prev_action, t0=t0, eval_mode=eval_mode, task=task)
		else:
			_, _, actions = self.model.imagine(state, self.model.pi, 1, eval_mode)
			a = actions[0]
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, start, actions, task):
		# TODO: lambda-return
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		state = start
		for t in range(self.cfg.horizon):
			# _, h = self.model.next(z, actions[t], task, h)
			state = self.model.dynamics.imagine_with_action(actions[t], state)
			feat = self.model.dynamics.get_feat(state)
			reward = math.two_hot_inv(self.model.heads['reward'](feat), self.cfg)
			G += discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		# _, h = self.model.next(z, self.model.pi(z, task)[1], task, h)
		# z = self.model.dynamics(h)
		return G + discount * self.model.Q(feat, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, state, prev_action, t0=False, eval_mode=False, task=None):
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
		# h = self._mask(h, 1.0 - torch.tensor(float(t0), requires_grad=False))
		# h = h + self._mask(self.initial_states['h'], torch.tensor(float(t0), requires_grad=False))
		# z = self._mask(z, 1.0 - torch.tensor(float(t0), requires_grad=False))
		# z = z + self._mask(self.initial_states['z'], torch.tensor(float(t0), requires_grad=False))
		# # a = self._mask(previous_states['a'], 1.0 - torch.tensor(float(t0), requires_grad=False))
		# # _, h = self.model.next(z, a, task, h)
		# # z = self.model.posterior(h, self.model.encode(obs, task), task)[0]
		# if self.cfg.num_pi_trajs > 0:
		# 	_z = z.repeat(self.cfg.num_pi_trajs, 1)
		# 	_h = h.repeat(self.cfg.num_pi_trajs, 1)
		# 	dreamed_traj = self.dream_trajectory(_z, _h, task)
		# 	pi_actions = dreamed_traj['pis']

		# add time dim
		obs = obs.unsqueeze(1)
		prev_action = prev_action.unsqueeze(1)
		embed = self.model._encoder(obs)
		is_first = torch.tensor([t0], device=self.device).reshape((1, 1, 1))
		post, _ = self.model.dynamics.observe(embed, prev_action, is_first, state)
		post = {k: v.repeat(self.cfg.num_pi_trajs, 1) for k, v in post.items()}
		feats, states, pi_actions = self.model.imagine(post, self.model._pi, self.cfg.horizon)

		# # Initialize state and parameters
		# z_orig = z
		# h_orig = h
		# z = z.repeat(self.cfg.num_samples, 1)
		# h = h.repeat(self.cfg.num_samples, 1) if h is not None else None
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
			# _, h_next = self.model.next(z, actions, task, h)
			# z_next = self.model.dynamics(h_next)
			value = self._estimate_value(state, actions, task).nan_to_num(0)
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

	def update_pi(self, feats, states):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		qs = []
		policy = self.model._pi(feats)
		for feat, state in zip(feats, states):
			succ_feats, succ_states, actions = self.model.imagine(state, policy, 1)
			q = self.model.Q(succ_feats[0], return_type='avg', detach=True)
			qs.append(q)

		qs = torch.stack(qs)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		log_pis = policy.log_prob(actions)
		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return pi_loss.detach(), pi_grad_norm

	# #TODO: make this more efficient
	# def _td_target_t(self, r, t, z_next, task):
	# 	if t == self.cfg.horizon - 1:
	# 		return self.discount * self.model.Q(z_next, task)
	# 	return r + self.discount * (1 - self.cfg.td_lambda) * self.model.Q(z_next, task) + self.cfg.td_lambda * self.model.V(z_next, task)

	@torch.no_grad()
	def _compute_target(self, imag_feat, reward):
		discount = self._config.discount * torch.ones_like(reward)
		value = self.model.Q(imag_feat, target=True).mode()
		target = tools.lambda_return(
			reward[1:],
			value[:-1],
			discount[1:],
			bootstrap=value[-1],
			lambda_=self._config.discount_lambda,
			axis=0,
		)
		weights = torch.cumprod(
			torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
		).detach()
		return target, weights, value[:-1]

	@staticmethod
	def _mask(value, mask):
		# Apply element-wise multiplication with broadcasting in PyTorch
		return value * mask.to(value.dtype)

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

	def dream_trajectory(self, z, h, task):
		"""
		Generate a trajectory of latent states and actions from a given initial state.

		Args:
			z (torch.Tensor): Initial latent state (batched)
			h (torch.Tensor): Initial hidden state (batched)
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Latent states of the trajectory.
			torch.Tensor: Actions of the trajectory.
		"""
		pis_dreamed = []
		pis_mu_dreamed = []
		log_pis_dreamed = []
		pis_log_std_dreamed = []
		zs = [z]
		hs = [h]
		mu, pi, log_pi, log_std = self.model.pi(z.detach(), task)
		pis_dreamed.append(pi)
		pis_mu_dreamed.append(mu)
		log_pis_dreamed.append(log_pi)
		pis_log_std_dreamed.append(log_std)
		for t in range(self.cfg.horizon - 1):
			with torch.no_grad():
				_, h = self.model.next(z, pi, task, h)
				hs.append(h)
				# _, z, _, _ = self.model.dynamics(h)
				z = self.model.dynamics(h)
			zs.append(z)
			mu, pi, log_pi, log_std = self.model.pi(z.detach(), task)
			pis_dreamed.append(pi)
			pis_mu_dreamed.append(mu)
			log_pis_dreamed.append(log_pi)
			pis_log_std_dreamed.append(log_std)

		hs = torch.stack(hs)
		zs = torch.stack(zs)
		pis_dreamed = torch.stack(pis_dreamed)
		pis_mu_dreamed = torch.stack(pis_mu_dreamed)
		log_pis_dreamed = torch.stack(log_pis_dreamed)
		pis_log_std_dreamed = torch.stack(pis_log_std_dreamed)
		r_dreamed = self.model.reward(hs, zs, task)

		#unfold time dimension for all: [T*B, D] -> [T, B, D]
		zs = zs.reshape(self.cfg.horizon, -1, zs.shape[-1])
		hs = hs.reshape(self.cfg.horizon, -1, hs.shape[-1])
		pis_dreamed = pis_dreamed.reshape(self.cfg.horizon, -1, pis_dreamed.shape[-1])
		pis_mu_dreamed = pis_mu_dreamed.reshape(self.cfg.horizon, -1, pis_mu_dreamed.shape[-1])
		log_pis_dreamed = log_pis_dreamed.reshape(self.cfg.horizon, -1, log_pis_dreamed.shape[-1])
		pis_log_std_dreamed = pis_log_std_dreamed.reshape(self.cfg.horizon, -1, pis_log_std_dreamed.shape[-1])
		r_dreamed = r_dreamed.reshape(self.cfg.horizon, -1)

		return { 'zs': zs, 'hs': hs, 'pis': pis_dreamed, 'pis_mu': pis_mu_dreamed, 'log_pis': log_pis_dreamed, 'pis_log_std': pis_log_std_dreamed, 'r': r_dreamed }

	def _update(self, obs, action, reward, is_first, task=None):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""


		# # Compute targets
		# with torch.no_grad():
		# 	next_z = self.model.encode(obs[1:], task)
		# 	td_targets = self._td_target(next_z,  hidden[1:].detach(), reward, task)

		# Prepare for update
		self.model.train()

		# RSSM training
		embed = self.model.encoder(obs)
		post, prior = self.model.dynamics.observe(embed, action, is_first)
		kl_free = self.cfg.kl_free
		dyn_scale = self.cfg.dyn_scale
		rep_scale = self.cfg.rep_scale
		kl_loss, kl_value, dyn_loss, rep_loss = self.model.dynamics.kl_loss(
			post, prior, kl_free, dyn_scale, rep_scale
		)
		assert kl_loss.shape == embed.shape[:2], kl_loss.shape

		feat = self.dynamics.get_feat(post)
		qs = self.model.heads['V'](feat)
		reward_preds = self.model.heads['reward'](feat)
		is_first_preds = self.model.heads['cont'](feat)
		context = dict(
			embed=embed,
			feat=feat,
			kl=kl_value,
			postent=self.dynamics.get_dist(post).entropy(),
		)

		# # fold time dimension for zs and hs: [T, B, D] -> [T*B, D]
		# zs_TxB = zs[:-1].reshape(-1, zs.shape[-1])
		# hs_TxB = hs[:-1].reshape(-1, hs.shape[-1])
		td_targets = self._compute_target(feat, reward)

		# Compute losses
		reward_loss, value_loss, is_first_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, is_first_pred_unbind, is_first_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), is_first_preds.unbind(0), is_first.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			is_first_loss = is_first_loss + math.soft_ce(is_first_preds.unbind, is_first_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		reward_loss = reward_loss / self.cfg.horizon
		is_first_loss = is_first_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			kl_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.is_first_coef * is_first_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		start = {k: v.detach() for k, v in post.items()}
		# imag_feat, imag_state, imag_action = self.model.imagine(start, self.model.pi, self.cfg.horizon)

		# Update policy
		# pi_loss, pi_grad_norm = self.update_pi(dreamed_traj['zs'].detach(), dreamed_traj['hs'].detach(), dreamed_traj['pis'], dreamed_traj['log_pis'], task)
		pi_loss, pi_grad_norm = self.update_pi(feat.detach(), post.detach())

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return TensorDict({
			"dynamics_loss": kl_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"is_first_loss": is_first_loss,
			"pi_loss": pi_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		}).detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, is_first, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		# h = self._burn_in_rollout(obs[0], obs[1:self.cfg.burn_in+1], action[:self.cfg.burn_in], is_first[:self.cfg.burn_in], **kwargs)
		return self._update(obs[self.cfg.burn_in:], action[self.cfg.burn_in:], reward[self.cfg.burn_in:], is_first[self.cfg.burn_in:], **kwargs)

