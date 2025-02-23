from time import time, time_ns

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from termcolor import colored

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._tds = []

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_runtime_means, ep_runtime_stds = [], [], [], []
		video_saved = False
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			times = []
			if self.cfg.save_video:
				# self.logger.video.init(self.env, enabled=(i == 0))
				self.logger.video.init(self.env, enabled=True)
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				start_time = time_ns()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				end_time = time_ns()
				times.append((end_time - start_time) // 1_000_000)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_runtime_means.append(np.mean(times))
			ep_runtime_stds.append(np.std(times))
			if self.cfg.save_video and not video_saved and info['success']:
				self.logger.video.save(self._step)
				video_saved = True

		if self.cfg.save_video and not video_saved:
			self.logger.video.save(self._step)

		return dict(
			episode_reward=np.nanmean(ep_rewards),
			cumulative_reward=np.sum(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_runtime_mean=np.nanmean(ep_runtime_means),
			episode_runtime_std=np.nanmean(ep_runtime_stds),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), 0.0)
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		batch_size=(1,))
		return td

	def save(self, metrics, identifier='model'):
		self.logger.save_agent(self.agent, self.buffer, metrics, identifier)

	def load(self, version='latest'):
		"""Load a TD-MPC2 agent."""
		fp = self.logger.load_agent(version)
		self.agent.load(fp, load_pi_only=self.cfg.freeze_pi)
		# self._step = self.agent.loss['step']
		self._step = self.agent.loss['step']
		self._ep_idx = self.agent.loss['episode']
		self._start_time = time() - self.agent.loss['total_time']
		return self.agent.loss

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next, info = {}, True, False, {}

		if self.cfg.checkpoint != '???':
			train_metrics = self.load(str(self.cfg.checkpoint))
			print(colored(f'Loaded agent from {self.cfg.checkpoint}', 'green', attrs=['bold']))

		if self.cfg.load_buffer:
			bfp = self.logger.load_buffer()
			self.buffer.loads(bfp, self._ep_idx)

		success_count = 0
		ep_count = 0
		reset_success_count = False
		log_success_rate = False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				ep_count += 1
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					identifier = f'{self._step}' if not self.cfg.override else 'final'
					self.logger.save_agent(self.agent, self.buffer, metrics=eval_metrics)
					eval_next = False
					reset_success_count = True

				if self._step > 0:
					# if info.has_key('success'):
					log_success_rate = True
					success = info.get('success') or False
					if success:
						success_count += 1
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=success,
					)
					if log_success_rate:
						train_metrics.update(success_rate=success_count / ep_count * 100)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					train_metrics.pop('episode_reward')
					train_metrics.pop('episode_success')
					train_metrics.pop('success_rate')
					if reset_success_count:
						success_count = 0
						reset_success_count = False

					if len(self._tds) > 0:
						self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps and self.buffer.num_eps > 0:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)
				train_metrics.update(
					step=self._step
				)
				self.logger.log(train_metrics, 'train')
			self._step += 1

		self.logger.finish(self.agent)
