import numpy as np
from neuron import *
import activation
import feedforward

if __name__ == '__main__':
	weights = np.array([1, 2, 3, 4, 4])
	bias = 1.0
	neuron = Neuron(weights, bias)
	output = neuron.apply(np.array([0,0,0,0,0.2]))
	print(output)