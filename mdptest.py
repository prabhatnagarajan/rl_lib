'''
Our MDP examples were taken from Sutton and Barto
'''

from mdp import *
import numpy as np

def run_tests():
	print "stuff"

def _test_value_iteration():


def get_test_mdp():
	transitions = np.zeros((3,2,3))
	transitions[0, 0, 0] = 1.0
	transitions[0, 1, 0] = 0.5
	transitions[0, 1, 1] = 0.5
	transitions[1, 0, 0] = 0.9
	transitions[1, 0, 1] = 0.1
	transitions[1, 1, 1] = 0.5
	transitions[1, 1, 2] = 0.5
	transitions[2, 0, 0] = 0.2
	transitions[2, 0, 2] = 0.8
	transitions[2, 1, 2] = 1.0
	mdp = MDP(transitions, )

if __name__ == '__main__':
	run_tests()