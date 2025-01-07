import numpy as np
import robosuite as suite
from robosuite import wrappers
import gym

from envs.wrappers.time_limit import TimeLimit

from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class RoboSuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env=env)
		self.cfg = cfg

	def reset(self, **kwargs):
		obs, _ = super().reset(**kwargs)
		obs = obs.astype(np.float32)
		return obs

	def step(self, action):
		obs, reward, d, _, info = super().step(action)
		obs = obs.astype(np.float32)
		return obs, reward, False, info

	def render(self, *args, **kwargs):
		return self.sim.render(
			width=96, height=86, camera_name="frontview",
		).copy()
		# try:
		# 	return self.env.render().copy()
		# except:
		# 	ok = True

def make_env(cfg, eval=False):
	if not cfg.task.startswith("rs"):
		raise ValueError("Invalid task name: {}".format(cfg.task))
	if not cfg.rs_robots in suite.robots.REGISTERED_ROBOTS:
		raise ValueError("Invalid robot name: {}".format(cfg.rs_robots))
	arm_controller_config = suite.load_part_controller_config(default_controller=cfg.rs_controller)
	controller_config = refactor_composite_controller_config(arm_controller_config, cfg.rs_robots, ["right"])

	env = RoboSuiteWrapper(
		wrappers.GymWrapper(
			suite.make(
				env_name=cfg.rs_env_name,
				robots=cfg.rs_robots,
				controller_configs=controller_config,
				has_renderer=True,
				has_offscreen_renderer=True,
				use_camera_obs=False,
				use_object_obs=True,
				reward_shaping=True,
				horizon=cfg.rs_max_episode_steps,
			)
		),
		cfg
	)
	env = TimeLimit(env, max_episode_steps=cfg.rs_max_episode_steps)
	env.max_episode_steps = cfg.rs_max_episode_steps
	env.reset()
	return env
