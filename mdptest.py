'''
Our MDP examples were taken from Sutton and Barto
'''

from mdp import *
import numpy as np

def run_tests():
	test_policy_iteration_aima_mdp()

def test_policy_iteration_aima_mdp():
	mdp = get_aima_mdp()
	value = mdp.policy_iteration()[1]
	print value
'''
This MDP is taken from the AIMA book (Russell and Norvig),
pg. 646, Figure 17.1 
'''

def get_aima_mdp():
	transitions = np.zeros((12, 4, 12))
	#Actions: up, down, left, right
	rewards = np.full(12, -0.04)
	rewards[3] = 1.0
	rewards[6] = -1.0
	#terminal state
	rewards[11] = 0.0
	#Terminal States
	transitions[3,:,11] = 1.0
	transitions[6,:,11] = 1.0
	transitions[11,:,11] = 1.0

	#Action - up
	transitions[0, 0, 0] = 0.9
	transitions[0, 0, 1] = 0.1

	transitions[1, 0, 1] = 0.8
	transitions[1, 0, 0] = 0.1
	transitions[1, 0, 2] = 0.1

	transitions[2, 0, 2] = 0.8
	transitions[2, 0, 1] = 0.1
	transitions[2, 0, 3] = 0.1

	transitions[4, 0, 0] = 0.8
	transitions[4, 0, 4] = 0.2

	transitions[5, 0, 2] = 0.8
	transitions[5, 0, 5] = 0.1
	transitions[5, 0, 6] = 0.1

	transitions[7, 0, 4] = 0.8
	transitions[7, 0, 8] = 0.1
	transitions[7, 0, 7] = 0.1

	transitions[8, 0, 8] = 0.8
	transitions[8, 0, 7] = 0.1
	transitions[8, 0, 9] = 0.1

	transitions[9, 0, 5] = 0.8
	transitions[9, 0, 8] = 0.1
	transitions[9, 0, 10] = 0.1

	transitions[10, 0, 6] = 0.8
	transitions[10, 0, 9] = 0.1
	transitions[10, 0, 10] = 0.1

	#Action - down
	transitions[0, 1, 4] = 0.8
	transitions[0, 1, 1] = 0.1
	transitions[0, 1, 0] = 0.1

	transitions[1, 1, 1] = 0.8
	transitions[1, 1, 0] = 0.1
	transitions[1, 1, 2] = 0.1

	transitions[2, 1, 5] = 0.8
	transitions[2, 1, 1] = 0.1
	transitions[2, 1, 3] = 0.1

	transitions[4, 1, 7] = 0.8
	transitions[4, 1, 4] = 0.2

	transitions[5, 1, 9] = 0.8
	transitions[5, 1, 5] = 0.1
	transitions[5, 1, 6] = 0.1

	transitions[7, 1, 7] = 0.9
	transitions[7, 1, 8] = 0.1

	transitions[8, 1, 8] = 0.8
	transitions[8, 1, 7] = 0.1
	transitions[8, 1, 9] = 0.1

	transitions[9, 1, 9] = 0.8
	transitions[9, 1, 8] = 0.1
	transitions[9, 1, 10] = 0.1

	transitions[10, 1, 10] = 0.9
	transitions[10, 1, 9] = 0.1

	#Action - left
	transitions[0, 2, 0] = 0.9
	transitions[0, 2, 4] = 0.1

	transitions[1, 2, 0] = 0.8
	transitions[1, 2, 1] = 0.2

	transitions[2, 2, 1] = 0.8
	transitions[2, 2, 2] = 0.1
	transitions[2, 2, 5] = 0.1

	transitions[4, 2, 4] = 0.8
	transitions[4, 2, 0] = 0.1
	transitions[4, 2, 7] = 0.1

	transitions[5, 2, 5] = 0.8
	transitions[5, 2, 2] = 0.1
	transitions[5, 2, 9] = 0.1

	transitions[7, 2, 7] = 0.9
	transitions[7, 2, 4] = 0.1

	transitions[8, 2, 7] = 0.8
	transitions[8, 2, 8] = 0.2

	transitions[9, 2, 8] = 0.8
	transitions[9, 2, 5] = 0.1
	transitions[9, 2, 9] = 0.1

	transitions[10, 2, 9] = 0.8
	transitions[10, 2, 6] = 0.1
	transitions[10, 2, 10] = 0.1

	#Action - right
	transitions[0, 3, 1] = 0.8
	transitions[0, 3, 4] = 0.1
	transitions[0, 3, 0] = 0.1

	transitions[1, 3, 2] = 0.8
	transitions[1, 3, 1] = 0.2

	transitions[2, 3, 3] = 0.8
	transitions[2, 3, 2] = 0.1
	transitions[2, 3, 5] = 0.1

	transitions[4, 3, 4] = 0.8
	transitions[4, 3, 0] = 0.1
	transitions[4, 3, 7] = 0.1

	transitions[5, 3, 6] = 0.8
	transitions[5, 3, 2] = 0.1
	transitions[5, 3, 9] = 0.1

	transitions[7, 3, 8] = 0.8
	transitions[7, 3, 7] = 0.1
	transitions[7, 3, 4] = 0.1

	transitions[8, 3, 9] = 0.8
	transitions[8, 3, 8] = 0.2

	transitions[9, 3, 10] = 0.8
	transitions[9, 3, 5] = 0.1
	transitions[9, 3, 9] = 0.1

	transitions[10, 3, 10] = 0.9
	transitions[10, 3, 6] = 0.1

	return MDP(transitions, rewards, 1.0)

if __name__ == '__main__':
	run_tests()