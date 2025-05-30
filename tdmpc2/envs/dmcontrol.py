from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control.composer import Environment, ObservationPadding, Robot, Entity
from dm_control.composer.variation import noises, distributions
from dm_control.locomotion import tasks
from dm_control.locomotion.walkers.base import Walker
from dm_control.manipulation.place import Place
from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite, composer, manipulation
from torchrl.modules import distributions_maps

from envs.wrappers.observation_delay_wrapper import StochasticDelayWrapper
from envs.wrappers.pomdp_wrapper import POMDPWrapper

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom') + suite._get_tasks('loca')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale
from dm_env import StepType, specs
import gym


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST


	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		wrapped_action_spec = env.action_spec()
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper:
	def __init__(self, env, domain, task, cfg):
		obs_shp = []
		for v in env.observation_spec().values():
			try:
				shp = np.prod(v.shape)
			except:
				shp = 1
			obs_shp.append(shp)
		obs_shp = (int(np.sum(obs_shp)),)
		act_shp = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf,
				dtype=np.float32),
			high=np.full(
				obs_shp,
				np.inf,
				dtype=np.float32),
			dtype=np.float32,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.env = env
		self.dm_env = self.env._env._env._env._env
		self.ts_norm = 1 / self.env.control_timestep()
		self.domain = domain
		self.task = task
		self.max_episode_steps = 500
		self.t = 0
		self.ts = 0.0
		self.delay_enabled = cfg.delay_enabled
		# if self.domain == 'acrobot':
		# 	self.dm_env._physics.model.opt.timestep /= 10
		# 	self.dm_env._n_sub_steps = 10
		self.delay_mu = self.dm_env._n_sub_steps
		self.delay_sigma = cfg.flickering_sigma

	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		return np.concatenate([v.flatten() for v in obs.values()])

	def reset(self):
		self.t = 0
		self.ts = 0.0
		self.env._n_sub_steps = self.delay_mu
		return self._obs_to_array(self.env.reset().observation)

	def get_timestep(self):
		return self.env.control_timestep() * self.ts_norm
	
	def step(self, action):
		if self.delay_enabled:
			self.dm_env._n_sub_steps = max(0, int(np.round(np.random.normal(self.delay_mu, self.delay_mu * self.delay_sigma, 1))[0]))
		self.t += 1
		self.ts += self.env.control_timestep()
		# timestamp = self.substeps * self.dm_env._physics.timestep()
		time_step = self.env.step(action)
		timestamp = self.env.control_timestep() * self.ts_norm
		actual_timestamp = self.dm_env._physics.data.time
		# assert timestamp == actual_timestamp
		info = {
			'timestamp': self.t,
			"prev_act": action,
		}
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.max_episode_steps, info

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


def make_env(cfg):
	"""
	Make DMControl environment.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	# domain, task = cfg.task.split('_', 1)
	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
	if (domain, task) not in suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 visualize_reward=True)
	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, 2)
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	extended_env = ExtendedTimeStepWrapper(env)
	env = TimeStepToGymWrapper(extended_env, domain, task, cfg)
	env = POMDPWrapper(env, domain, cfg)
	if cfg.delay_enabled:
		delay_fn = lambda: np.random.randint(0, 6)
		env = StochasticDelayWrapper(env, env.action_space.shape[0], delay_fn=delay_fn, max_delay=4)
	return env


if __name__ == '__main__':
	# task = suite.load('walker',
	# 				 'walk',
	# 				  # environment_kwargs={},
	# 				 task_kwargs={'random': 0, 'random': np.random.RandomState(0), 'delayed_observation_padding': ObservationPadding.INITIAL_VALUE},
	# 				 visualize_reward=False)
	env = manipulation.load('place_brick_features', seed=0)
	task = env.task
	joint_pos = task.observables['jaco_arm/joints_pos']
	joint_pos.corruptor = noises.Additive(distributions.Normal(scale=0.01))
	joint_pos.delay = noises.Additive(distributions.UniformInteger(low=0, high=3))
	ok = True