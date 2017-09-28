from activation import *
import numpy as np
from pdb import set_trace

class Neuron:
	# takes in initialized weights with the final term
	def __init__(self, init_weights, init_bias, activation=Activation.sigmoid):
		self.weights = init_weights
		self.bias = init_bias
		obj = Activation()
		self.activation = obj.sigmoid

	def apply(self, input):
		return self.activation(np.dot(input, self.weights) + self.bias)

