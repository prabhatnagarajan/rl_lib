#!/usr/bin/env python
import numpy as np
from pdb import set_trace
class QLearner:
	def __init__(self, mdp, learning_rate):
		self.mdp = mdp
		self.mdp.check_valid_mdp()
		self.alpha = learning_rate

	'''
	Follows an epsilon greedy policy with epsilon =0.1
	'''
	def learn(self, num_episodes, epsilon=0.15, anneal_rate=0.0001):
		Q = np.zeros((len(self.mdp.states), len(self.mdp.actions)))
		alpha = self.alpha
		for i in range(num_episodes):
			state = self.mdp.start
			if state is None:
				state = np.random.choice(self.mdp.states)
			#Complete an episode
			while not (state == self.mdp.terminal):
				if np.random.uniform() < epsilon:
					action = np.random.choice(self.mdp.actions)
				else:
					action = np.argmax(Q[state])
				reward, next_state = self.mdp.act(state, action)
				Q[state, action] = Q[state, action] + alpha * (reward + self.mdp.gamma * np.amax(Q[next_state,:]) - Q[state, action])
				if np.isnan(Q[state, action]):
					set_trace()
				state = next_state
				alpha = max(alpha - anneal_rate, 0)
		return Q

