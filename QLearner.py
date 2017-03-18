#!/usr/bin/env python
import numpy as np
from pdb import set_trace
class QLearner:
	def __init__(self, mdp, learning_rate, anneal_rate):
		self.mdp = mdp
		self.mdp.check_valid_mdp()
		self.alpha = learning_rate

	'''
	Follows an epsilon greedy policy with epsilon =0.1
	'''
	def learn(self, epsilon=0.1):
		Q = np.zeros((len(self.mdp.states), len(self.mdp.actions)))
		while True:
			state = self.mdp.start
			if state is None:
				state = np.random.choice(self.mdp.states)
			while not (state == self.mdp.terminal):
				if np.random.uniform() < epsilon:
					action = np.random.choice(self.mdp.states)
				else:
					action = np.argmax(Q, axis=1)
				reward, next_state = self.mdp.act(state, action)
				Q[state, action] = Q[state, action] + self.alpha(reward + mdp.gamma * np.amax(Q[next_state,:]) - Q(s, a))
				state = next_state

