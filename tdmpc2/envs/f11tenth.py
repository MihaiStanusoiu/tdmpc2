import gym
import numpy as np

from envs.f1tenth_gym.f1tenth_gym.envs import F110Env


class F110EnvWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.max_episode_steps = 1000
        obs_shp = []
        for v in env.observation_space.values():
            try:
                shp = np.prod(v.shape)
            except:
                shp = 1
            obs_shp.append(shp)
        obs_shp = (int(np.sum(obs_shp)),)
        act_shp = env.action_space.shape
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
            low=np.full(act_shp, env.action_space.low),
            high=np.full(act_shp, env.action_space.high),
            dtype=env.action_space.dtype)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(self.cfg.seed)
        return self._obs_to_array(obs)

    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values() if isinstance(v, np.ndarray)] + [np.array([v]) for v in obs.values() if not isinstance(v, np.ndarray)])

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        obs, reward, done, tr, info = self.env.step(action)
        info['truncated'] = tr
        return self._obs_to_array(obs), reward, done or tr, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode=mode).copy()

def make_env(cfg):
    env = None
    if cfg.task == 'f110':
        env = F110EnvWrapper(F110Env(render_mode='rgb_array'), cfg)
    return env
