from mdp import *
import numpy as np

def run_tests():
	test_policy_iteration_aima_mdp()
	test_value_iteration_aima_mdp()
	test_q_value_iteration_aima_mdp()

def test_policy_iteration_aima_mdp():
	mdp = get_aima_mdp()
	policy, value = mdp.policy_iteration()
	correct_value = np.array([0.812, 0.868, 0.918, 1.0, 0.762, 0.660, -1.0, 0.705, 0.655, 0.611, 0.388, 0, 0, 0])
	np.testing.assert_array_almost_equal(value, correct_value, decimal=2)
	correct_policy = np.array([3, 3, 3, 1, 0, 0, 1, 0, 2, 2, 2, 1, 1, 1], dtype=np.uint32)
	np.testing.assert_array_almost_equal(value, correct_value, decimal=2)
	#ensure the mandatory components are equal, ignore states 3, 6, 11, 12, and 13
	np.testing.assert_array_equal(correct_policy[0:3], policy[0:3])
	np.testing.assert_array_equal(correct_policy[4:6], policy[4:6])
	np.testing.assert_array_equal(correct_policy[7:11], policy[7:11])

def test_value_iteration_aima_mdp():
	mdp = get_aima_mdp()
	value = mdp.value_iteration()
	correct_value = np.array([0.812, 0.868, 0.918, 1.0, 0.762, 0.660, -1.0, 0.705, 0.655, 0.611, 0.388, 0, 0, 0])
	np.testing.assert_array_almost_equal(value, correct_value, decimal=2)


def test_q_value_iteration_aima_mdp():
	mdp = get_aima_mdp()
	Q_opt =  mdp.q_value_iteration()
	for state in mdp.states:
		for action in mdp.actions:
			#check the Bellman optimalityh equations are satisfied (pg. 76, S&B)
			np.testing.assert_almost_equal(np.dot(mdp.transitions[state, action, :], mdp.rewards + mdp.gamma * np.amax(Q_opt, axis=1)), Q_opt[state, action], decimal=2)
'''
This MDP is taken from the AIMA book (Russell and Norvig),
pg. 646, Figure 17.1 
'''

def get_aima_mdp():
	transitions = np.zeros((14, 4, 14))
	#Actions: up, down, left, right
	rewards = np.full(14, -0.04)
	#rewards[3] = 1.0
	#rewards[6] = -1.0
	rewards[11] = 1.0
	rewards[12] = -1.0
	#terminal state
	#rewards[11] = 0.0
	rewards[13] = 0
	#Terminal States
	transitions[3,:,11] = 1.0
	transitions[6,:,12] = 1.0
	transitions[11,:,13] = 1.0
	transitions[12,:,13] = 1.0
	transitions[13,:,13] = 1.0

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