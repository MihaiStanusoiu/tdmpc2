from copy import deepcopy

import torch
import torch.nn as nn

from ncps.torch import CfC
from common import layers, math, init
from tensordict.nn import TensorDictParams

from common import networks, tools


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
		self._encoder = networks.MultiEncoder(
			cfg.obs_shape,
			**cfg.encoder
		)
		self._embed_size = self._encoder.outdim
		self._device = torch.device('cuda:0')
		self.dynamics = networks.RSSM(
			cfg.dyn_stoch,
			cfg.latent_dim,
			cfg.hidden_dim,
			1,
			cfg.dyn_discrete,
			cfg.act,
			True,
			cfg.dyn_mean_act,
			cfg.dyn_std_act,
			cfg.dyn_min_std,
			cfg.unimix_ratio,
			cfg.initial,
			cfg.action_dim,
			self._embed_size,
			self._device
		)
		self.heads = nn.ModuleDict()
		if cfg.dyn_discrete:
			feat_size = cfg.dyn_stoch * cfg.dyn_discrete + cfg.latent_dim
		else:
			feat_size = cfg.dyn_stoch + cfg.dyn_deter

		# self._rnn = CfC(cfg.latent_dim + cfg.action_dim + cfg.task_dim, cfg.hidden_dim, cfg.latent_dim, return_sequences=False)
		# self._dynamics = layers.mlp(cfg.hidden_dim, [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		# self.initial_h = nn.Parameter(torch.zeros(cfg.hidden_dim))
		# self._rnn.batch_first = False
		# self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self.heads['reward'] = networks.MLP(
			feat_size,
			(255,) if cfg.reward_head["dist"] == "symlog_disc" else (),
			cfg.reward_head["layers"],
			cfg.mlp_dim,
			cfg.act,
			cfg.norm,
			dist=cfg.reward_head["dist"],
			outscale=cfg.reward_head["outscale"],
			device=self._device,
			name="Reward",
		)
		self.heads["is_first"] = networks.MLP(
			feat_size,
			(),
			cfg.is_first_head["layers"],
			cfg.mlp_dim,
			cfg.act,
			cfg.norm,
			dist="binary",
			outscale=cfg.is_first_head["outscale"],
			device=self._device,
			name="Cont",
		)
		self.heads['V'] = layers.Ensemble([networks.MLP(
			feat_size,
			(255,) if cfg.V_head["dist"] == "symlog_disc" else (),
			cfg.V_head["layers"],
			cfg.mlp_dim,
			cfg.act,
			cfg.norm,
			dist=cfg.V_head["dist"],
			outscale=cfg.V_head["outscale"],
			device=self._device,
			name="V",
		)])
		self._pi = networks.MLP(
            feat_size,
            (cfg.action_dim,),
            cfg.actor["layers"],
            cfg.mlp_dim,
            cfg.act,
            cfg.norm,
            cfg.actor["dist"],
            cfg.actor["std"],
            cfg.actor["min_std"],
            cfg.actor["max_std"],
            absmax=1.0,
            temp=cfg.actor["temp"],
            unimix_ratio=cfg.actor["unimix_ratio"],
            outscale=cfg.actor["outscale"],
            name="Actor",
        )
		# self._pi = layers.mlp(cfg.latent_dim + cfg.hidden_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		# self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])\
		self._Qs = self.heads['V']
		self.apply(init.weight_init)
		# init.zero_([self.heads['reward'].weight, self._Qs.params["2", "weight"]])


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
		modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'V-functions', 'Is first predictor']
		for i, m in enumerate([self._encoder, self.dynamics, self.heads['reward'], self._pi, self.heads['V'], self.heads['is_first']]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

	def imagine(self, start, policy, horizon, eval_mode=False):
		dynamics = self.dynamics
		flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
		start = {k: flatten(v) for k, v in start.items()}

		def step(prev, _):
			state, _, _ = prev
			feat = dynamics.get_feat(state)
			inp = feat.detach()
			if eval_mode:
				action = policy(inp).mode()
			else:
				action = policy(inp).sample()
			succ = dynamics.img_step(state, action)
			return succ, feat, action

		succ, feats, actions = tools.static_scan(
			step, [torch.arange(horizon)], (start, None, None)
		)
		states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

		return feats, states, actions

	def _policy(self, obs, state, training):
		if state is None:
			latent = action = None
		else:
			latent, action = state
		obs = self._wm.preprocess(obs)
		embed = self._wm.encoder(obs)
		latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
		if self._config.eval_state_mean:
			latent["stoch"] = latent["mean"]
		feat = self._wm.dynamics.get_feat(latent)
		if not training:
			actor = self._task_behavior.actor(feat)
			action = actor.mode()
		elif self._should_expl(self._step):
			actor = self._expl_behavior.actor(feat)
			action = actor.sample()
		else:
			actor = self._task_behavior.actor(feat)
			action = actor.sample()
		logprob = actor.log_prob(action)
		latent = {k: v.detach() for k, v in latent.items()}
		action = action.detach()
		if self._config.actor["dist"] == "onehot_gumble":
			action = torch.one_hot(
				torch.argmax(action, dim=-1), self._config.num_actions
			)
		policy_output = {"action": action, "logprob": logprob}
		state = (latent, action)
		return policy_output, state

	def next(self, z, a, task, h=None):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		if z.dim() != 3:
			z = z.unsqueeze(0)
		z_next, h = self._rnn(z, h)
		#z_next = self._dynamics(h)
		return z_next, h

	def forward(self, obs, a, task=None, h=None):
		"""
		Forward pass through the world model.
		"""
		z = self.encode(obs, task)
		z_next, h = self.next(z, a, task, h)
		return z_next, h
	
	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z, h, task):
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

	def Q(self, z, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

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
