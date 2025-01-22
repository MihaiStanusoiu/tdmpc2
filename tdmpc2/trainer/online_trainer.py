from time import time

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
		ep_rewards, ep_successes = [], []
		video_saved = False
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t, hidden, info = self.env.reset(), False, 0, 0, self.agent.initial_h.detach(),  {'timestamp': 0.0}
			if self.cfg.save_video:
				# self.logger.video.init(self.env, enabled=(i == 0))
				self.logger.video.init(self.env, enabled=True)
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action, hidden = self.agent.act(obs, t0=t==0, h=hidden, info=info, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video and not video_saved and info['success']:
				self.logger.video.save(self._step)
				video_saved = True

		if self.cfg.save_video and not video_saved:
			self.logger.video.save(self._step)

		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None, done=False, h=None, is_first=False):
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
			done=torch.tensor(done, dtype=torch.float).unsqueeze(0),
			is_first=torch.ones((1, 1), dtype=torch.float) if is_first else torch.zeros((1, 1), dtype=torch.float),
		batch_size=(1,))
		return td

	def save(self, metrics, identifier='model'):
		self.logger.save_agent(self.agent, self.buffer, metrics, identifier)

	def load(self):
		"""Load a TD-MPC2 agent."""
		fp = self.logger.load_agent()
		self.agent.load(fp, load_pi_only=self.cfg.freeze_pi)
		# buffer_artifact = self.logger._wandb.use_artifact(self.logger._group + '-' + str(self.logger._seed) + '-buffer', type='dataset')
		# buffer_artifact_dir = buffer_artifact.download()
		# TODO: Load buffer

		# self._step = self.agent.loss['step']
		self._step = self.agent.loss['step']
		return self.agent.loss

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next, info = {}, True, False, {'timestamp': 0.0}

		if self.cfg.checkpoint != '???':
			train_metrics = self.load()
			print(colored(f'Loaded agent from {self.cfg.checkpoint}', 'green', attrs=['bold']))

		success_count = 0
		ep_count = 0
		reset_success_count = False
		log_success_rate = False
		h = self.agent.initial_h.detach()
		h_next = h
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
					self.logger.save_agent(self.agent, None, metrics=eval_metrics)
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
				info = {'timestamp': 0.0}
				is_first = True
				h = self.agent.initial_h.detach()
				self._tds = [self.to_td(obs, done=False, is_first=True)]

			# Collect experience
			if self._step > self.cfg.seed_steps and not self.cfg.random_policy:
				if self.cfg.warmup_h:
					# h warmup
					h = self.agent.initial_h.detach()
					if len(self._tds) > 1:
						burn_in_tds = self._tds[-self.cfg.burn_in:-1]
						prev_obs = [td['obs'] for td in burn_in_tds]
						prev_act = [td['action'] for td in burn_in_tds]
						prev_obs = torch.cat(prev_obs).unsqueeze(1).to(self.agent.device)
						prev_act = torch.cat(prev_act).unsqueeze(1).to(self.agent.device)
						with torch.no_grad():
							_, h = self.agent.model.rnn(self.agent.model.encode(prev_obs), prev_act, h=h)
				action, h_next = self.agent.act(obs, t0=len(self._tds)==1, h=h, info=info)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, done, h, is_first=False))
			h = h_next

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
