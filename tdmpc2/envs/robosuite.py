import robosuite as suite

def make_env(cfg, eval=False):
	if not cfg.rs_robots in suite.robots.REGISTERED_ROBOTS:
		raise ValueError("Invalid robot name: {}".format(cfg.rs_robots))
	env = suite.make(
		env_name=cfg.rs_env_name,
		robots=cfg.rs_robots,
		has_renderer=eval,
		has_offscreen_renderer=not eval,
		use_camera_obs=False,
		use_object_obs=True,
		reward_shaping=True,
		horizon=200,
	)

	return env
