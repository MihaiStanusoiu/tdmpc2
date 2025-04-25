import collections

import dm_env
import numpy as np


class StochasticDelayDMControlWrapper(dm_env.Environment):
    def __init__(self, env, delay_fn, max_delay=10, keys_to_delay=None):
        self._env = env
        self._delay_fn = delay_fn
        self._max_delay = max_delay
        self._keys_to_delay = keys_to_delay
        self._obs_buffers = {}  # Dict[str, Deque]

    def reset(self):
        ts = self._env.reset()
        self._init_buffers(ts.observation)
        for _ in range(self._max_delay):
            self._push_to_buffers(ts.observation)
        return ts._replace(observation=self._get_delayed_obs())

    def step(self, action):
        ts = self._env.step(action)
        self._push_to_buffers(ts.observation)
        return ts._replace(observation=self._get_delayed_obs())

    def _init_buffers(self, obs):
        self._obs_buffers = {}
        for key, value in obs.items():
            if self._keys_to_delay is None or key in self._keys_to_delay:
                self._obs_buffers[key] = collections.deque(
                    [np.zeros_like(value)] * self._max_delay, maxlen=self._max_delay
                )

    def _push_to_buffers(self, obs):
        for key, buffer in self._obs_buffers.items():
            buffer.append(obs[key])

    def _get_delayed_obs(self):
        delayed_obs = dict(self._env.physics.observation_spec())  # or copy current
        for key, buffer in self._obs_buffers.items():
            delay = min(int(self._delay_fn()), self._max_delay - 1)
            delayed_obs[key] = buffer[-(delay + 1)]
        return delayed_obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def discount_spec(self):
        return self._env.discount_spec()

    def reward_spec(self):
        return self._env.reward_spec()