from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

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
			obs, done, ep_reward, t, state = self.env.reset(), False, 0, 0, self.agent.model.dynamics.initial(1)
			prev_action = torch.zeros((1, self.env.action_space.shape[0]), dtype=torch.float32)
			if self.cfg.save_video:
				# self.logger.video.init(self.env, enabled=(i == 0))
				self.logger.video.init(self.env, enabled=True)
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs.cuda().unsqueeze(0), state, prev_action, t0=t==0, eval_mode=True)
				with torch.no_grad():
					# _, h_next = self.agent.model.next(state['z'], action.cuda().unsqueeze(0), None, state['h'])
					state, _ = self.agent.model.dynamics.observe(self.agent.model._encoder(obs.cuda().unsqueeze(0)), action.cuda.unsqueeze(0), False, state)
					# _, hidden = self.agent.model.forward(obs.cuda().unsqueeze(0), action.cuda().unsqueeze(0), h=hidden)
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

	def to_td(self, obs, action=None, reward=None, h=None, is_first=False):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			is_first=torch.ones((1, 1), dtype=torch.bool) if is_first else torch.zeros((1, 1), dtype=torch.bool),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		success_count = 0
		ep_count = 0
		reset_success_count = False
		log_success_rate = False
		state = self.agent.model.dynamics.initial(1)
		prev_action = torch.zeros((1, self.env.action_space.shape[0]), dtype=torch.float32)
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
					self.logger.save_agent(self.agent, None, identifier=f'{self._step}')
					eval_next = False
					reset_success_count = True

				if self._step > 0:
					# if info.has_key('success'):
					log_success_rate = True
					if info['success']:
						success_count += 1
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
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

					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				is_first = True
				state = self.agent.model.dynamics.initial(1)
				prev_action = torch.zeros((1, self.env.action_space.shape[0]), dtype=torch.float32)
				self._tds = [self.to_td(obs, is_first=True)]

			# Collect experience
			if self._step > self.cfg.seed_steps and not self.cfg.random_policy:
				action = self.agent.act(obs, state, prev_action, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			prev_action = action
			obs, reward, done, info = self.env.step(action)
			with torch.no_grad():
				state, _ = self.agent.model.dynamics.observe(self.agent.model._encoder(obs), action, False, state)
				# _, h_next = self.agent.model.next(state['z'], action.cuda().unsqueeze(0), None, state['h'])
			self._tds.append(self.to_td(obs, action, reward, is_first=False))

			# Update agent
			if self._step >= self.cfg.seed_steps:
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
