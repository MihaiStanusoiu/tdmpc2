import gym
import numpy as np
import collections

import torch


class StochasticDelayWrapper(gym.Wrapper):
    def __init__(self, env, action_dim, delay_fn, max_delay=10, keys_to_delay=None):
        super().__init__(env)
        self.delay_fn = delay_fn
        self.max_delay = max_delay
        self.keys_to_delay = keys_to_delay
        self.obs_buffers = {}  # Dict[str, Deque]
        self.act_dim = action_dim
        self.dt_buffer = collections.deque([np.zeros_like(1)] * self.max_delay, maxlen=self.max_delay)
        self.act_buffer = collections.deque([np.zeros_like(action_dim)] * self.max_delay, maxlen=self.max_delay)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._init_buffers(obs)
        for _ in range(self.max_delay):
            self._push_to_buffers(obs, np.zeros(self.act_dim), np.zeros(1))
        obs, _, _ = self._get_delayed_obs(int(self.delay_fn()))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_to_buffers(obs, action, info["timestamp"])
        delay_steps = int(self.delay_fn())
        delayed_obs, delayed_act, delayed_dt = self._get_delayed_obs(delay_steps)
        # info["timestamp"] = float(delay_steps + 1)
        info["timestamp"] = delayed_dt
        info["prev_act"] = delayed_act
        return delayed_obs, reward, done, info

    def _init_buffers(self, obs):
        self.obs_buffers = {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if self.keys_to_delay is None or key in self.keys_to_delay:
                    self.obs_buffers[key] = collections.deque(
                        [np.zeros_like(value)] * self.max_delay, maxlen=self.max_delay
                    )
        else:
            self.obs_buffers["obs"] = collections.deque(
                [np.zeros_like(obs)] * self.max_delay, maxlen=self.max_delay
            )

    def _push_to_buffers(self, obs,act,dt):
        if isinstance(obs, dict):
            for key in self.obs_buffers:
                self.obs_buffers[key].append(obs[key])
        else:
            self.obs_buffers["obs"].append(obs)
        self.act_buffer.append(act)
        self.dt_buffer.append(dt)

    def _get_delayed_obs(self, delay_steps):
        delay = min(delay_steps, self.max_delay - 1)
        if "obs" in self.obs_buffers:
            return self.obs_buffers["obs"][-(delay + 1)], self.act_buffer[-(delay + 1)], self.dt_buffer[-(delay + 1)]
        else:
            return {
                key: buffer[-(delay + 1)]
                for key, buffer in self.obs_buffers.items()
            }, self.act_buffer[-(delay + 1)], self.dt_buffer[-(delay + 1)]
