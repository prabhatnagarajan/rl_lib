import sys
sys.path.insert(0, '../..')
import gym
import numpy as np
import random
from pdb import set_trace
import math
from demonstration import *
# from selector import *
from model_estimator import *
# from birl.birl import *
# from birl.prior import *

STATE_SIZE = np.array([3, 3, 8, 5], dtype=np.int32)
SUCCESS_THRESHOLD = 300.0
SOLVE_STREAK = 100
MIN_EPSILON = 0.01
MIN_ALPHA = 0.05
NUM_DEMOS = 5
FAILURE_PERCENTAGE = 0.15

def learn(num_episodes, discount, epsilon=0.15, anneal_rate=0.00, alpha=0.1):
	demos = []
	env = gym.make('BipedalWalker-v2')
	actions = get_actions(env)
	env.render()
	Q = np.zeros((int(np.prod(STATE_SIZE)), len(actions)))
	print(env.observation_space.low)
	print(end.observation_space.high)
	bounds = list(zip(env.observation_space.low, env.observation_space.high))
	#manually bound infinities
	bounds[1] = (-1.5, 1.5)
	bounds[3] = (-math.radians(50), math.radians(50))
	learning_decay = lambda lr, t: max(0.1, min(0.5, 1.0 - math.log10((t + 1) / 25)))
	epsilon_decay = lambda eps, t: max(0.01, min(1.0, 1.0 - math.log10((t + 1) / 25)))
	current_streak = 0
	episode = 0
	model_learner = ModelEstimator(int(np.prod(STATE_SIZE)), len(actions))
	while current_streak < SOLVE_STREAK:
		demo = Demonstration()
		observation = env.reset()
		state = get_discrete_state(observation, bounds)
		done = False
		time = 0
		total_reward = 0
		while True:
			#env.render()
			action = select_action(state, actions, epsilon, Q)
			observation, reward, done, info = env.step(action)
			demo.add(state, action)
			next_state = get_discrete_state(observation, bounds)
			model_learner.add_reward(state, reward)
			model_learner.add_transition(state, action, next_state)
			Q[state, action] = Q[state, action] + alpha * (reward + discount * np.amax(Q[next_state,:]) - Q[state, action])
			#update Epsilon and learning rate
			epsilon = max(MIN_EPSILON, epsilon - 0.001) 
			alpha = max(MIN_ALPHA, alpha - 0.001)
			state = next_state
			#Logistics
			total_reward += reward
			time += 1
			if done:
				print "Episode " + str(episode + 1) + " finished after " + str(time + 1) + " timesteps with total reward " + str(total_reward)
				alpha = get_learning_rate(episode)
				epsilon = get_epsilon(episode)
				break
		if total_reward > SUCCESS_THRESHOLD:
			current_streak += 1
		else:
			current_streak = 0
		episode += 1
		demo.set_total_reward(total_reward)
		demos.append(demo)
	return (Q, demos, model_learner.get_as_mdp(discount))

def get_learning_rate(time):
	return max(MIN_ALPHA, min(0.5, 1.0 - math.log10(float(time + 1) / float(25))))

def get_epsilon(time):
	return max(MIN_EPSILON, min(1, 1.0 - math.log10(float(time + 1)/float(25))))

def select_action(state, actions, epsilon, Q):
	if np.random.uniform() < epsilon:
		return random.choice(actions)
	else:
		return np.argmax(Q[state])

def get_actions(env):
	#On this task, actions are continuous from 
	#[-1, -1, -1, -1] to [1, 1, 1, 1]
	set_trace()
	return range(env.action_space.n)

def get_discrete_state(observation, bounds):
	discrete_state = []
	for i in range(len(observation)):
		discrete_state.append(get_section(observation[i], bounds, i))
	return get_state_num(discrete_state)

def get_state_num(discrete_state):
	state = 0
	for i in range(len(STATE_SIZE)):
		state += discrete_state[i] * np.prod(STATE_SIZE[i+1:])
	return int(state)

def get_section(value, bounds, index):
	if value > bounds[index][1]:
		value = bounds[index][1]
	if value < bounds[index][0]:
		value = bounds[index][0]
	diff = value - bounds[index][0]
	max_diff = bounds[index][1] - bounds[index][0]
	section_size = max_diff/float(STATE_SIZE[index])
	count = 0
	while diff > section_size:
		diff -= section_size
		count += 1
	return int(count)

def play_policy(num_iterations, policy):
	env = gym.make('CartPole-v0')
	actions = get_actions(env)
	bounds = list(zip(env.observation_space.low, env.observation_space.high))
	#manually bound infinities
	bounds[1] = (-1.5, 1.5)
	bounds[3] = (-math.radians(50), math.radians(50))
	current_streak = 0
	episode = 0
	while current_streak < SOLVE_STREAK:
		observation = env.reset()
		state = get_discrete_state(observation, bounds)
		done = False
		time = 0
		total_reward = 0
		while True:
			env.render()
			action = policy[state]
			observation, reward, done, info = env.step(int(action))
			next_state = get_discrete_state(observation, bounds)
			state = next_state
			#Logistics
			total_reward += reward
			time += 1
			if done:
				print "Episode " + str(episode + 1) + " finished after " + str(time + 1) + " timesteps with total reward " + str(total_reward)
				break
		if total_reward > SUCCESS_THRESHOLD:
			current_streak += 1
		else:
			current_streak = 0
		episode += 1

if __name__ == '__main__':
	#main_loop()
	Q, demos, mdp = learn(10000, 0.99, epsilon=1.0, anneal_rate=0.0001, alpha=0.5)
	#Get 5 most spaced out successful demos
	successes = get_successful_demos(demos, NUM_DEMOS, SUCCESS_THRESHOLD)
	failures = get_failure_demos(demos, NUM_DEMOS, FAILURE_PERCENTAGE, SUCCESS_THRESHOLD)

	# birl_format_success_demos = [(demo.total_reward, demo.examples, 400)for demo in demos]
	# policy = birl(mdp, 0.02, 100, 1.0, birl_format_success_demos, 50, 4, PriorDistribution.UNIFORM)
	# play_policy(300, policy)
