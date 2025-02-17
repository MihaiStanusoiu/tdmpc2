from copy import deepcopy

import torch
import torch.nn as nn

from ncps.torch import CfC, LTC
from common import layers, math, init
from tensordict.nn import TensorDictParams

from common.layers import NormedLinear


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
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		if cfg.rnn_type == 'cfc':
			self._rnn = CfC(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, None,
							backbone_units=cfg.backbone_units, backbone_layers=cfg.backbone_layers,
							backbone_dropout=cfg.backbone_dropout, batch_first=False,
							return_sequences=False)
		elif cfg.rnn_type == 'cfc_pure':
			self._rnn = CfC(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, None,
							backbone_units=cfg.backbone_units, backbone_layers=cfg.backbone_layers,
							backbone_dropout=cfg.backbone_dropout, mode="pure", batch_first=False,
							return_sequences=False)
		elif cfg.rnn_type == 'ltc':
			self._rnn = LTC(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, batch_first=False, return_sequences=False)
		elif cfg.rnn_type == 'lstm':
			self._rnn = nn.LSTM(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, batch_first=False)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.hidden_dim, [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		# self._dynamics = NormedLinear(cfg.hidden_dim, cfg.latent_dim, act=layers.SimNorm(cfg))
		self.initial_h = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
		# self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.hidden_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.hidden_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.hidden_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		self._detach_Qs.params = self._detach_Qs_params
		self._target_Qs.params = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'RNN Readout', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._rnn, self._dynamics, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def non_rnn_params(self):
		# return param generator for all non-RNN parameters
		rnn_params = set(dict(self._rnn.named_parameters()).keys())
		filtered_params = {
			name: param
			for name, param in self.named_parameters()
			if name not in rnn_params
		}
		return set(filtered_params.values())

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

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

	def encode(self, obs, task=None):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def rnn(self, z, a, task=None, h=None, dt=None):
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if z.dim() != 3:
			z = z.unsqueeze(0)
		if a.dim() != 3:
			a = a.unsqueeze(0)
		z = torch.cat([z, a], dim=-1)
		if h is None:
			h = self.initial_h.expand(z.shape[1], -1)
		# if dt is None:
		# 	dt = torch.ones((z.shape[0], z.shape[1], 1), device=h.device, requires_grad=False)
		readout, h = self._rnn(z, h, dt)
		return readout, h

	def next(self, z, a, h, task=None):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a, h], dim=-1)
		z_next = self._dynamics(z)
		return z_next

	def forward(self, z, a, h, task=None, dt=None):
		"""
		Forward pass through the world model.
		"""
		z_next = self.next(z, a, h, task)
		_, h = self.rnn(z_next, a, task, h, dt=dt)
		z_next = self.next(z, a, h, task)
		return z_next, h
	
	def reward(self, z, a, h, task=None):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a, h], dim=-1)
		return self._reward(z)

	def pi(self, z, h, task=None):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, h], dim=-1)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def Q(self, z, a, h, task=None, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a, h], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
