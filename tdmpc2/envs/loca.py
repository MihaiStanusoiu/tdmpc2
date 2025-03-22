
from envs.wrappers.loca_wrapper import LOCADMC
from envs.wrappers.time_limit import TimeLimit
from envs.tasks import reacher

def make_env(cfg, mode):
    env = LOCADMC(cfg.task, cfg.loca_action_repeat, cfg.loca_render_size, cfg.loca_dmc_camera, cfg.loca_phase, cfg.loca_terminate_reach, mode)
    return env