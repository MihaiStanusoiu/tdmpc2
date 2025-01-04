import gym

from envs.wrappers.pomdp_wrapper import POMDPWrapper
from envs.wrappers.time_limit import TimeLimit

from envs.wrappers.mujoco_wrapper import MujocoWrapper


def make_env(cfg):
    env = POMDPWrapper(cfg.task, cfg.pomdp_type, cfg.flicker_prob, cfg.random_noise_sigma, cfg.random_sensor_missing_prob)
    env = MujocoWrapper(env)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env