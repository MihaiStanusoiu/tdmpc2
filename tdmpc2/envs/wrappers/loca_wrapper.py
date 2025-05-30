import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
from copy import copy

from envs.tasks import reacher
from dm_control.composer import Environment, ObservationPadding, Robot, Entity
from dm_control.composer.variation import noises, distributions
from dm_control.locomotion import tasks
from dm_control.locomotion.walkers.base import Walker
from dm_control.manipulation.place import Place
from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite, composer, manipulation

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom') + suite._get_tasks('loca')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    if self._act_is_dict:
      return self._env.action_space.spaces.copy()
    else:
      return {self._act_key: self._env.action_space}

  def step(self, action):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    return obs


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  def _obs_to_array(self, obs):
    return np.concatenate([v.flatten() for v in obs.values()])

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return self._obs_to_array(obs)

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return self._obs_to_array(obs)

class LOCADMC(DMC):
  # Supports only reacherloca task
  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None,
               loca_phase="phase_1", loca_terminate_reach=0, loca_mode='train', one_way_wall_radius=0.1):
    super().__init__(name, action_repeat, size, camera)
    self._loca_phase = loca_phase
    self._loca_mode = loca_mode

    if self._loca_phase != "phase_1":
      self._env._task.switch_loca_task()

    self._actuators_length = np.array([
      self._env._physics.named.model.body_pos['hand', 'x'],
      self._env._physics.named.model.body_pos['finger', 'x']
    ])
    self._one_way_wall_radius = one_way_wall_radius
    self._loca_terminate_reach = loca_terminate_reach

  def get_target_1_pos(self):
    return self._env._physics.named.data.geom_xpos['target_1', :2]

  def get_finger_pos(self):
    return self._env._physics.named.data.geom_xpos['finger', :2]

  def check_inside_one_way_wall(self):
    target_1_pos = self.get_target_1_pos()
    finger_pos = self.get_finger_pos()
    return np.linalg.norm(finger_pos - target_1_pos) <= self._one_way_wall_radius

  def is_phase_2(self):
    return self._loca_phase == "phase_2"

  def step(self, action):
    prev_inside_wall = self.check_inside_one_way_wall()
    terminate_reach = False
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._action_repeat):
      physics_data_copy = copy(self._env._physics.data)
      time_step = self._env.step(action['action'])
      if prev_inside_wall and (not self.check_inside_one_way_wall()):
        self._env._physics._reload_from_data(physics_data_copy)
        # time_step = self._env.step(0.)
      reward += time_step.reward or 0.0
      if self._loca_terminate_reach and reward:
        terminate_reach = True
        break
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
      'reward': reward,
      'is_first': False,
      'is_last': time_step.last() or terminate_reach,
      'is_terminal': time_step.discount == 0,
      'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
      k: v for k, v in dict(time_step.observation).items()
      if k not in self._ignored_keys})
    return self._obs_to_array(obs)

  def sample_in_one_way_wall(self):
    def sample_xy():
      x = np.random.uniform(low=0, high=self._one_way_wall_radius)
      y = np.sqrt(np.random.uniform(low=0, high=(self._one_way_wall_radius**2 - x**2)))
      return np.array([x, y])

    def inverse_kinematics(pos):
      x, y = pos[0], pos[1]
      a1, a2 = self._actuators_length[0], self._actuators_length[1]
      d = (x ** 2 + y ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
      theta2 = np.arccos(d)
      theta1 = np.arctan(y / x) - np.arctan((a2 * np.sin(theta2)) / (a1 + a2 * np.cos(theta2)))
      # theta2_ = -np.arccos(d)
      # theta1_ = np.arctan(y / x) - np.arctan((a2 * np.sin(theta2_)) / (a1 + a2 * np.cos(theta2_)))
      return theta1, theta2

    def forward_kinematics(theta1, theta2):
      a1, a2 = self._actuators_length[0], self._actuators_length[1]
      x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
      y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
      return np.array([x, y])

    target_1_pos = self.get_target_1_pos()
    while True:
      final_finger_pos = sample_xy() + target_1_pos
      if np.linalg.norm(final_finger_pos) > self._actuators_length.sum():
        continue
      if final_finger_pos[0] == 0.:
        continue
      theta1, theta2 = inverse_kinematics(final_finger_pos)
      if np.isnan(theta1) or np.isnan(theta2):
        continue
      tmp_finger_pos = forward_kinematics(theta1, theta2)
      if np.linalg.norm(final_finger_pos - tmp_finger_pos) > 1e-5:
        theta1 += np.pi
      break
    return theta1, theta2

  def reset(self):
    if self._loca_mode == 'eval':
      obs = super().reset()
      while self.check_inside_one_way_wall():
        obs = super().reset()
      return self._obs_to_array(obs)
    elif not self.is_phase_2():
      return super().reset()
    time_step = self._env.reset()
    theta1, theta2 = self.sample_in_one_way_wall()
    with self._env._physics.reset_context():
      self._env._physics.named.data.qpos['shoulder'] = theta1
      self._env._physics.named.data.qpos['wrist'] = theta2
    obs = {
      'reward': 0.0,
      'is_first': True,
      'is_last': False,
      'is_terminal': False,
      'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
      k: v for k, v in dict(time_step.observation).items()
      if k not in self._ignored_keys})
    return self._obs_to_array(obs)

class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64)):
    self._env = env
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def reset(self):
    obs = self._env.reset()
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.