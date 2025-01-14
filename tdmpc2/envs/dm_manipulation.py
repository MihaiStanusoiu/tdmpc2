import numpy as np
from dm_control import manipulation
from dm_control.composer.variation import noises, distributions
from dm_control.suite.wrappers import action_scale

from envs.dmcontrol import ActionDTypeWrapper

from envs.dmcontrol import TimeStepToGymWrapper

from envs.dmcontrol import ActionRepeatWrapper, ExtendedTimeStepWrapper


def make_env(cfg):
    env = manipulation.load(cfg.task, seed=cfg.seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, 2)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(env, '', cfg.task)
    if cfg.task == 'place_brick_features':
        observables_to_pertube_keys = ['jaco_arm/joints_pos', 'jaco_arm/joints_vel', 'jaco_arm/joints_torque', 'jaco_arm/jaco_hand/joints_pos',
                                       'jaco_arm/jaco_hand/joints_vel']
    task = env.env.task
    if cfg.random_noise_sigma != 0.0:
        all_observables: dict = task.observables
        observables_to_pertube = [all_observables[key] for key in observables_to_pertube_keys]
        for observable in observables_to_pertube:
            observable.corruptor = noises.Additive(distributions.Normal(scale=cfg.random_noise_sigma))

    return env

if __name__ == "__main__":
    env = manipulation.load('place_brick_features', seed=0)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, 2)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    env = ExtendedTimeStepWrapper(env)
    env = TimeStepToGymWrapper(env, domain, task)
    ok= True