

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
	env.render()
	env.step(env.action_space.sample()) # take a random action
env.close()



import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
env.close()





from gym import spaces

class Environment:
	""" A simple environment skeleton """
	def __init__(self):
		self.shares = 0
		self.avg_share = 0
		self.cash = 10
		self.current_price=0

		self.action_spaces = spaces.Discrete(3)

		#Observation Space
		#shares         #Discrete
		#Avgshare Price #Box
		#cash           #Box
		#Current Price  #Box

		self.obesrvation_space = spaces.Tuple(\
									spaces.Discrete(100),\
									spaces.Box(-10,10,1),
									spaces.Box(-100,100,1),\
									spaces.Box(-10,10,1),)

	def step(self, action):
		# Changes the environment based on agents action
		networth = self.cash + self.shares*self.avg_share
		if action==1: #Sell
			if self.shares>0:
				self.avg_share = (self.avg_share*self.shares - self.current_price)/(self.shares-1)
				self.shares-=1
				self.cash += self.current_price
		elif action==2: #Buy
			if self.cash>0:
				self.avg_share = (self.avg_share*self.shares + self.current_price)/(self.shares+1)
				self.shares+=1
				self.cash -= self.current_price
		profit = self.shares * (self.current_price - self.avg_share)
		return next_state, reward, done, info
	def reset(self):
		# Resets the environment to its initial state
		pass
	def render(self):
	   # Show the state of the environment on screen
		pass


class Agent:
	""" A simple agent """
	def __init__(self):
		pass
	def action(self, state):
		if state > 0:
			return 1
		else:
			return 0


import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt
class DQNAgent:
	""" DQN agent """
	def __init__(self, states, actions, max_memory, double_q):
		self.states = states
		self.actions = actions
		self.session = tf.Session()
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=10)
		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.memory = deque(maxlen=max_memory)
		self.eps = 1
		self.eps_decay = 0.99999975
		self.eps_min = 0.1
		self.gamma = 0.90
		self.batch_size = 32
		self.burnin = 100000
		self.copy = 10000
		self.step = 0
		self.learn_each = 3
		self.learn_step = 0
		self.save_each = 500000
		self.double_q = double_q
def build_model(self):
		""" Model builder function """
		self.input = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.states, name='input')
		self.q_true = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
		self.a_true = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
		self.reward = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
		self.input_float = tf.to_float(self.input) / 255.
		# Online network
		with tf.variable_scope('online'):
			self.conv_1 = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
			self.conv_2 = tf.layers.conv2d(inputs=self.conv_1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
			self.conv_3 = tf.layers.conv2d(inputs=self.conv_2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
			self.flatten = tf.layers.flatten(inputs=self.conv_3)
			self.dense = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu)
			self.output = tf.layers.dense(inputs=self.dense, units=self.actions, name='output')
		# Target network
		with tf.variable_scope('target'):
			self.conv_1_target = tf.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
			self.conv_2_target = tf.layers.conv2d(inputs=self.conv_1_target, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
			self.conv_3_target = tf.layers.conv2d(inputs=self.conv_2_target, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
			self.flatten_target = tf.layers.flatten(inputs=self.conv_3_target)
			self.dense_target = tf.layers.dense(inputs=self.flatten_target, units=512, activation=tf.nn.relu)
			self.output_target = tf.stop_gradient(tf.layers.dense(inputs=self.dense_target, units=self.actions, name='output_target'))
		# Optimizer
		self.action = tf.argmax(input=self.output, axis=1)
		self.q_pred = tf.gather_nd(params=self.output, indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))
		self.loss = tf.losses.huber_loss(labels=self.q_true, predictions=self.q_pred)
		self.train = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)
		# Summaries
		self.summaries = tf.summary.merge([
			tf.summary.scalar('reward', self.reward),
			tf.summary.scalar('loss', self.loss),
			tf.summary.scalar('max_q', tf.reduce_max(self.output))
		])
		self.writer = tf.summary.FileWriter(logdir='./logs', graph=self.session.graph)
def copy_model(self):
		""" Copy weights to target network """
		self.session.run([tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('target'), tf.trainable_variables('online'))])
def save_model(self):
		""" Saves current model to disk """
		self.saver.save(sess=self.session, save_path='./models/model', global_step=self.step)
def add(self, experience):
		""" Add observation to experience """
		self.memory.append(experience)
def predict(self, model, state):
		""" Prediction """
		if model == 'online':
			return self.session.run(fetches=self.output, feed_dict={self.input: np.array(state)})
		if model == 'target':
			return self.session.run(fetches=self.output_target, feed_dict={self.input: np.array(state)})
def run(self, state):
		""" Perform action """
		if np.random.rand() < self.eps:
			# Random action
			action = np.random.randint(low=0, high=self.actions)
		else:
			# Policy action
			q = self.predict('online', np.expand_dims(state, 0))
			action = np.argmax(q)
		# Decrease eps
		self.eps *= self.eps_decay
		self.eps = max(self.eps_min, self.eps)
		# Increment step
		self.step += 1
		return action
def learn(self):
		""" Gradient descent """
		# Sync target network
		if self.step % self.copy == 0:
			self.copy_model()
		# Checkpoint model
		if self.step % self.save_each == 0:
			self.save_model()
		# Break if burn-in
		if self.step < self.burnin:
			return
		# Break if no training
		if self.learn_step < self.learn_each:
			self.learn_step += 1
			return
		# Sample batch
		batch = random.sample(self.memory, self.batch_size)
		state, next_state, action, reward, done = map(np.array, zip(*batch))
		# Get next q values from target network
		next_q = self.predict('target', next_state)
		# Calculate discounted future reward
		if self.double_q:
			q = self.predict('online', next_state)
			a = np.argmax(q, axis=1)
			target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
		else:
			target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)
		# Update model
		summary, _ = self.session.run(fetches=[self.summaries, self.train],
									  feed_dict={self.input: state,
												 self.q_true: np.array(target_q),
												 self.a_true: np.array(action),
												 self.reward: np.mean(reward)})
		# Reset learn step
		self.learn_step = 0
		# Write
		self.writer.add_summary(summary, self.step)








