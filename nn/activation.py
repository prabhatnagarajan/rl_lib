from enum import Enum
import numpy as np

# def sigmoid(value):
# 	return np.exp(value)/(np.exp(value) + 1)

class Activation():
	def sigmoid(self, value):
		return np.exp(value)/(np.exp(value) + 1)

