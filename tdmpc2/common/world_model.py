from copy import deepcopy
from enum import Enum

import ncps.wirings
import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init
from ncps.torch import CfC

class Symbols(Enum):
	Q = 'Q'
	r = 'r'
	z_next = 'z_next'

# TODO: get this from config
output_to_intermediate_h_map = {
	Symbols.Q: 1,
	Symbols.r: 0,
	Symbols.z_next: 0
}

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
		wiring = ncps.wirings.AutoNCP(cfg.hidden_dim, 2 * cfg.action_dim)
		# wiring = world_model_wiring(cfg.hidden_units, cfg.latent_dim, cfg.obs_shape[0] + cfg.action_dim)
		self._ncp = CfC(cfg.latent_dim, wiring, return_sequences=False)
		self._ncp.batch_first = False
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(len(wiring.get_neurons_of_layer(output_to_intermediate_h_map[Symbols.z_next])) + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(len(wiring.get_neurons_of_layer(output_to_intermediate_h_map[Symbols.Q])) + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self._reward = layers.mlp(len(wiring.get_neurons_of_layer(output_to_intermediate_h_map[Symbols.r])) + cfg.action_dim + cfg.task_dim, cfg.mlp_dim, max(cfg.num_bins, 1))  # TODO: ID might not be correct here
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

		self._initial_inter_h = [torch.zeros(len(wiring.get_neurons_of_layer(i))) for i in range(0, wiring.num_layers)]

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	@property
	def hidden_state_dim(self, layer_id):
		return len(self._ncp.wiring.get_neurons_of_layer(layer_id))

	@property
	def hidden_layer(self, layer_id):
		if layer_id not in range(self._ncp.wiring.num_layers):
			raise ValueError(f"Invalid layer ID: {layer_id}")
		return self._ncp.rnn_cell._layers[layer_id]

	def _intermediate_h(self, h):
		layer_sizes = [len(self._ncp.wiring.get_neurons_of_layer(i)) for i in range(self._ncp.wiring.num_layers)]
		return torch.split(h, layer_sizes, dim=-1)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		self._initial_inter_h = [h.to(*args, **kwargs) for h in self._initial_inter_h]
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

	@property
	def initial_h(self):
		return torch.cat([torch.tanh(h) for h in self._initial_inter_h], dim=-1)

	def pi(self, z, task, h=None, return_sequences=True):
		"""
		Forward pass through the world model.
		"""
		if h is None:
			h = self.initial_h
		is_seq = z.ndim == 3
		if not is_seq:
			# unsqueeze time dim
			z = z.unsqueeze(0)
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		hs = []
		outputs = []
		for i in range(z.shape[0]):
			output, h = self._ncp(z[i].unsqueeze(0), h)
			hs.append(h)
			outputs.append(output)

		if return_sequences:
			output = torch.stack(outputs, dim=0)
			h = torch.stack(hs, dim=0)
		else:
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

	def next(self, h, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		_h = self._intermediate_h(h)[output_to_intermediate_h_map[Symbols.z_next]]
		if self.cfg.multitask:
			_h = self.task_emb(_h, task)
		_h = torch.cat([_h, a], dim=-1)
		return self._dynamics(_h)
	
	def reward(self, h, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		h = self._intermediate_h(h)[output_to_intermediate_h_map[Symbols.r]]
		if self.cfg.multitask:
			h = self.task_emb(h, task)
		h = torch.cat([h, a], dim=-1)
		return self._reward(h)

	# def pi(self, z, task):
	# 	"""
	# 	Samples an action from the policy prior.
	# 	The policy prior is a Gaussian distribution with
	# 	mean and (log) std predicted by a neural network.
	# 	"""
	# 	if self.cfg.multitask:
	# 		z = self.task_emb(z, task)
	#
	# 	# Gaussian policy prior
	# 	mu, log_std = self._pi(z).chunk(2, dim=-1)
	# 	log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
	# 	eps = torch.randn_like(mu)
	#
	# 	if self.cfg.multitask: # Mask out unused action dimensions
	# 		mu = mu * self._action_masks[task]
	# 		log_std = log_std * self._action_masks[task]
	# 		eps = eps * self._action_masks[task]
	# 		action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
	# 	else: # No masking
	# 		action_dims = None
	#
	# 	log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
	# 	pi = mu + eps * log_std.exp()
	# 	mu, pi, log_pi = math.squash(mu, pi, log_pi)
	#
	# 	return mu, pi, log_pi, log_std

	def Q(self, h, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		h = self._intermediate_h(h)[output_to_intermediate_h_map[Symbols.Q]]
		if self.cfg.multitask:
			h = self.task_emb(h, task)
			
		h = torch.cat([h, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(h)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
