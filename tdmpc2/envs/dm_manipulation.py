import numpy as np
from dm_control import manipulation
from dm_control.composer.variation import noises, distributions

from envs.dmcontrol import ActionDTypeWrapper

from envs.dmcontrol import TimeStepToGymWrapper

def make_env(cfg):
    env = manipulation.load(cfg.task, seed=cfg.seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = TimeStepToGymWrapper(env, '', cfg.task)
    if cfg.task == 'place_brick_features':
        observables_to_pertube_keys = ['jaco_arm/joints_pos', 'jaco_arm/joints_vel', 'jaco_arm/joints_torque', 'jaco_arm/jaco_hand/joints_pos',
                                       'jaco_arm/jaco_hand/joints_vel']
    task = env.env.task
    if cfg.sensor_noise != 0.0:
        all_observables: dict = task.observables
        observables_to_pertube = [all_observables[key] for key in observables_to_pertube_keys]
        for observable in observables_to_pertube:
            observable.corruptor = noises.Additive(distributions.Normal(scale=cfg.sensor_noise))

    return env

if __name__ == "__main__":
    env = manipulation.load('place_brick_features', seed=0)
    env = ActionDTypeWrapper(env, np.float32)
    env = TimeStepToGymWrapper(env, '', 'place_brick_features')
    env.render()
    ok= True