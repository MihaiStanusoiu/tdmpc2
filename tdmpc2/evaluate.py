import os
import time

from common.logger import Logger
from common.plotting import plot_state_wm_state_correlation, plot_ep_rollout, plot_umap, plot_ep_rollout_video

os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def rollout_trajectory(cfg: dict):
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	cfg.mpc = False
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# initialize logger
	logger = Logger(cfg)

	# Load agent
	agent = TDMPC2(cfg)

	fp = logger.load_agent()
	agent.load(fp)

	real_trajectories = []
	imagined_trajectories = []

	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, info, t, hidden = env.reset(), False, 0, {
			'timestamp': env.get_timestep()}, 0, agent.initial_h.detach()

		while not done:
			dt = None
			if info.get("timestamp") is not None:
				dt = torch.tensor(info['timestamp'], dtype=torch.float, device=agent.device, requires_grad=False).reshape((1, 1))
			pi = agent.act(obs, t0=t == 0, h=hidden, info=info, eval_mode=True, z=agent.model.encode(obs))
			z, h = agent.model.forward(z, pi, h, dt=dt)


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	seed = cfg.seed
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# initialize logger
	logger = Logger(cfg)

	# Load agent
	agent = TDMPC2(cfg)

	fp = logger.load_agent(version=cfg.checkpoint)
	agent.load(fp)

	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	# if cfg.save_video:
	# 	video_dir = os.path.join(cfg.work_dir, 'videos')
	# 	os.makedirs(video_dir, exist_ok=True)

	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes, states, wm_states, one_step_errors, multi_step_errors = [], [], np.array([]), np.array([]), [], []
		state, wm_state = [], []
		for i in range(cfg.eval_episodes):
			set_seed(seed)
			seed += 1
			# Make environment
			env = make_env(cfg)
			one_step_obs_error = []
			multi_step_obs_error = []
			pred_multi_step_obs = []
			pred_obs = []
			actual_obs = []
			rewards = []
			obs, done, ep_reward, t, info = env.reset(task_idx=task_idx), False, 0, 0, {'timestamp': env.get_timestep()}
			hidden_rollout =  None
			pred_obs.append(obs.numpy())
			pred_multi_step_obs.append(obs.numpy())
			actual_obs.append(obs.numpy())
			action = torch.zeros(env.action_space.shape)
			times = []
			if cfg.save_video:
				logger.video.init(env, enabled=True)

			actions = []
			obss = []
			while not done:
				# measure and log inference time
				start_time = time.time_ns()
				action = agent.act(obs, t0=t==0, task=task_idx)
				actions.append(action)
				end_time = time.time_ns()
				times.append((end_time - start_time) // 1_000_000)
				obs, reward, done, info = env.step(action)
				rewards.append(reward)
				obss.append(obs.numpy())
				actual_obs.append(obs.numpy())

				# with torch.no_grad():
				# 	a = action.unsqueeze(0).to('cuda')
				# 	one_step_obs_hat = agent.model.next(hidden_next, a)
				# 	if t % (cfg.plan_horizon) == (cfg.plan_horizon - 1) and t > 0:
				# 		hidden_rollout = hidden_next
				# 	obs_hat = agent.model.next(hidden_rollout, a)
				# 	_, hidden_rollout = agent.model.rnn(obs_hat, a.unsqueeze(0).to('cuda'), h=hidden_rollout, dt=None)
				# 	# obs_hat = agent.model.next(hidden_next, action.unsqueeze(0).to('cuda')).sample()
				# 	obs_hat	= obs_hat.to(obs.device).squeeze(0)
				# 	one_step_obs_hat = one_step_obs_hat.to(obs.device).squeeze(0)
				# 	multi_step_obs_error.append(torch.nn.functional.mse_loss(obs_hat, obs).numpy())
				# 	one_step_obs_error.append(torch.nn.functional.mse_loss(one_step_obs_hat, obs).numpy())
				# 	pred_obs.append(one_step_obs_hat.numpy())
				# 	pred_multi_step_obs.append(obs_hat.numpy())

				ep_reward += reward
				t += 1
				if cfg.save_video:
					logger.video.record(env)

			if cfg.plot_ep_rollout:
				# fig = plot_ep_rollout(np.array(actual_obs), np.array(pred_obs), f"{task} One-step Prediction", cfg.work_dir or None)
				fig_multi_step = plot_ep_rollout_video(np.array(actual_obs), [r'$\cos{\alpha_1}$', r'$\sin{\alpha_1}$', r'$\cos{\alpha_2}$', r'$\sin{\alpha_2}$', r'$\cos{\alpha_3}$', r'$\sin{\alpha_3}$'], f"{task} States", f"states_rollout_video.mp4" or None)
				logger.log_fig(fig_multi_step, f"statistics/states")
				logger.log_video(i+1, f"states_rollout_video.mp4")

				actions = np.expand_dims(actions, 1)
				fig_multi_step = plot_ep_rollout_video(np.array(actions),
													   [r'$\tau_2$'], f"{task} Actions",
													   f"actions_rollout_video.mp4" or None)
				logger.log_fig(fig_multi_step, f"statistics/actions")
				logger.log_video(i + 1, f"actions_rollout_video.mp4")

				rewards = np.expand_dims(rewards, 1)
				fig_multi_step = plot_ep_rollout_video(np.array(rewards),
													   [r'$r$'], f"{task} Rewards",
													   f"rewards_rollout_video.mp4" or None)
				logger.log_fig(fig_multi_step, f"statistics/rewards")
				logger.log_video(i + 1, f"rewards_rollout_video.mp4")

			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			one_step_errors.append(np.array(one_step_obs_error).mean())
			multi_step_errors.append(np.array(multi_step_obs_error).mean())
			time_mean = np.mean(times)
			time_std = np.std(times)
			metrics = dict(
				episode=i,
				episode_reward=ep_reward,
				episode_success=info['success'],
				episode_one_step_error_mean=one_step_errors[i],
				episode_multi_step_error_mean=multi_step_errors[i],
				runtime_mean=time_mean,
				runtime_std=time_std,
			)
			logger.log(metrics, "evaluate_ep")
			if cfg.save_video:
				logger.video.save(i+1, key=f"evaluate/videos/{task}")
		states = np.vstack(state)
		wm_states = np.vstack(wm_state)
		ep_rewards = np.mean(ep_rewards)
		ep_rewards_std = np.std(ep_rewards)
		ep_successes = np.mean(ep_successes)
		one_step_mean = np.mean(one_step_errors)
		one_step_std = np.std(one_step_errors)
		multi_step_mean = np.mean(multi_step_errors)
		multi_step_std = np.std(multi_step_errors)
		metrics = dict(
			task=task_idx,
			episode_rewards=ep_rewards,
			episode_rewards_std=ep_rewards_std,
			episode_success=ep_successes,
			episode_one_step_error_mean=one_step_mean,
			episode_one_step_error_std=one_step_std,
			episode_multi_step_error_mean=multi_step_mean,
			episode_multi_step_error_std=multi_step_std,
		)
		logger.log(metrics, "evaluate_task")
		phase_sep_index = int(cfg.episode_length // 4)
		states_swingup = states[:phase_sep_index]
		states_balance = states[phase_sep_index:]
		wm_states_swingup = wm_states[:phase_sep_index]
		wm_states_balance = wm_states[phase_sep_index:]

		if cfg.plot_state_correlation:
			if 'cartpole' in cfg.task:
				state_labels = ["X", "cos(theta)", "sin(theta)", "X_dot", "theta_dot"]

			fig, r_states, combinations = plot_state_wm_state_correlation(states_swingup, wm_states_swingup, task, cfg.work_dir or None)
			logger.log_state_wm_prediction(fig, state_labels, r_states, combinations, f"statistics/swingup_phase_state_prediction_correlation")

			fig, r_states, combinations = plot_state_wm_state_correlation(states_balance, wm_states_balance, task, cfg.work_dir or None)
			logger.log_state_wm_prediction(fig, state_labels, r_states, combinations,
										   f"statistics/balance_phase_state_prediction_correlation")
		if cfg.plot_umap:
			fig, umap_h = plot_umap(states_swingup, wm_states_swingup, task, cfg.work_dir or None)
			logger.log_fig(fig, f"statistics/swingup_phase_umap_projection")

			fig, umap_h = plot_umap(states_balance, wm_states_balance, task, cfg.work_dir or None)
			logger.log_fig(fig, f"statistics/balance_phase_umap_projection")
		# if cfg.plot_ep_rollout:
		# 	fig, states = plot_ep_rollout(states, wm_states, task, cfg.work_dir or None)
		# 	logger.log_ep_rollout(fig, states, f"statistics/ep_roll
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
