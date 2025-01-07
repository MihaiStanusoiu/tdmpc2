import numpy as np
import gym
from envs.wrappers.time_limit import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False
		self.env.action_repeat = cfg.action_repeat if hasattr(cfg, 'action_repeat') else 2

	def reset(self, **kwargs):
		obs = super().reset(**kwargs)
		obs = obs.astype(np.float32)
		self.env.step(np.zeros(self.env.action_space.shape))
		return obs

	def step(self, action):
		reward = 0
		for _ in range(self.env.action_repeat):
			obs, r, _, info = self.env.step(action.copy())
			reward += r
		obs = obs.astype(np.float32)
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(
			offscreen=True, resolution=(384, 384), camera_name=self.camera_name
		).copy()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-hidden"
	if not cfg.task.startswith('mw-') or not (env_id in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE or env_id in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN):
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id](seed=cfg.seed)
	env = MetaWorldWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
