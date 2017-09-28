class Demonstration:

	def __init__(self):
		self.total_reward = 0
		self.examples = []

	def add(self, state, action):
		self.examples.append((state, action))

	def set_total_reward(self, total_reward):
		self.total_reward = total_reward

