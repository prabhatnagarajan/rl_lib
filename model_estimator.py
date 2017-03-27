import numpy as np
from mdp import *
from sklearn.preprocessing import normalize

class ModelEstimator:

	def __init__(self, num_states, num_actions):
		self.rewards = np.zeros(num_states)
		self.reward_counts = np.zeros(num_states)
		self.transitions = np.zeros((num_states, num_actions, num_states))
		self.transition_counts = np.full((num_states, num_actions, num_states), 1.0)

	def add_transition(self, state, action, next_state):
		self.transition_counts[state, action, next_state] = self.transition_counts[state, action, next_state] + 1

	def add_reward(self, state, reward):
		count = self.reward_counts[state]
		self.rewards[state] = self.rewards[state] * (count/(count + 1)) + (1/(count + 1)) * reward
		self.reward_counts[state] = count + 1

	def set_transition_distribution(self):
		num_states = np.shape(self.transitions)[0]
		for state in range(np.shape(self.transition_counts)[0]):
			for action in range(np.shape(self.transition_counts)[1]):
				total = np.sum(self.transition_counts[state, action,:])
				self.transitions[state, action, :] = self.transition_counts[state, action, :]/np.sum(self.transition_counts[state, action,:])
	def get_model(self):
		self.set_transition_distribution()
		return (self.transitions, self.rewards)

	def get_as_mdp(self, gamma, start=None, terminal=None, max_value=1000):
		self.set_transition_distribution()
		#Need to check transition dynamics
		return MDP(self.transitions, self.rewards, gamma, start, terminal, max_value)