from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import ncps
from ncps.torch import CfC
from common import layers, math, init


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = CfC(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, cfg.latent_dim, return_sequences=False)
		self.dynamics_initial_h = nn.Parameter(torch.zeros(cfg.hidden_dim))
		self._dynamics.batch_first = False
		# self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.hidden_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		wiring = ncps.wirings.AutoNCP(cfg.hidden_dim, 2 * cfg.action_dim)
		self._pi = CfC(cfg.latent_dim + cfg.task_dim, wiring, return_sequences=False)
		self._pi_act = nn.Mish(inplace=True)
		self.pi_initial_h = nn.Parameter(torch.zeros(cfg.hidden_dim))
		self._Qs = layers.Ensemble([layers.mlp(cfg.hidden_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task, h=None):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if h is None:
			h = torch.tanh(self.dynamics_initial_h)
		z = torch.cat([z, a], dim=-1)
		if z.dim() != 3:
			z = z.unsqueeze(0)
		z_next, h = self._dynamics(z, h)
		return z_next, h
	
	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		with torch.no_grad():
			_, h = self.next(z, a, task)
		return self._reward(h)

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if h is None:
			h = self.pi_initial_h
		is_seq = z.ndim == 3
		if not is_seq:
			# unsqueeze time dim
			z = z.unsqueeze(0)
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		hs = []
		outputs = []
		for i in range(z.shape[0]):
			output, h = self._pi(z[i].unsqueeze(0), h)
			output = self._pi_act(output)
			hs.append(h)
			outputs.append(output)

		# if return_sequences:
		# 	output = torch.stack(outputs, dim=0)
		# 	h = torch.stack(hs, dim=0)
		# else:
		output = outputs[-1]
		h = hs[-1]

		act_next_mu, act_next_log_std = torch.split(output, [self.cfg.action_dim, self.cfg.action_dim], dim=-1)
		act_next_log_std = math.log_std(act_next_log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(act_next_mu)
		log_pi = math.gaussian_logprob(eps, act_next_log_std)
		pi = act_next_mu + eps * act_next_log_std.exp()
		act_next_mu, pi, log_pi = math.squash(act_next_mu, pi, log_pi)
		layer_sizes = [len(self._ncp.wiring.get_neurons_of_layer(i)) for i in range(self._ncp.wiring.num_layers)]

		return [act_next_mu, pi, log_pi, act_next_log_std], h

	def Q(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		with torch.no_grad():
			_, h = self.next(z, a, task)
		out = (self._target_Qs if target else self._Qs)(h)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
