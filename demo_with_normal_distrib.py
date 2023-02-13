#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
@File    :   demo_with_normal_distrib.py
@Time    :   2023/02/13 15:27:06
@Author  :   Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   This script generates total reward and correct action ratio figures according to 
chapter 2 of Barto and Sutton's book for the k-armed bandit problem with a gaussian rewards 
ditribution'''

import numpy as np
import matplotlib.pyplot as plt

# General parameters

# Probability of choosing a non-greedy action a time t
C_EPSILON_VEC = [0, 1e-2, 1e-1]
# Number of time steps
C_N_ITER = 1000
# Number of trials for the Monte-Carlo simulation
# <=> number of agents
C_N_AGENTS = 2000
# Number of arms of the bandit problem
C_N_ARMS = 10
# Bandit problem mean for the rewards distirbution
C_MEAN_REWARDS = 0.
# Bandit problem variance for the rewards distirbution
C_VARIANCE_REWARDS = 1.
# Bandit problem variance for the given reward at time t
C_VARIANCE_ACTION_REWARDS = 1.

# Initialize storage for the average reward at time t 
total_reward = np.zeros((C_N_ITER, len(C_EPSILON_VEC)))
# Initialize storage for the ratio of correct action taken at time t
correct_actions_ratio = np.zeros((C_N_ITER, len(C_EPSILON_VEC)))

for n_epsilon, epsilon in enumerate(C_EPSILON_VEC):

    print("*" * 10, f"Epsilon = {epsilon}", "*" * 10)

    # Iter over all agents
    for n_agent in range(C_N_AGENTS):

        if (n_agent + 1) % 100 == 0 :
                print(f"At agent {n_agent + 1} / {C_N_AGENTS}")

        # Get the "base" rewards distribution according to a normal law
        rewards = C_MEAN_REWARDS + np.sqrt(C_VARIANCE_REWARDS) * np.random.randn(C_N_ARMS)
        # Get the best action
        best_action = np.argmax(rewards)

        # Initialize the current agent storage for the action value
        # estimates and action counts
        action_value_estimate = np.zeros(C_N_ARMS)
        action_counts = np.zeros(C_N_ARMS)

        # Let the agent learn
        for n_iter in range(C_N_ITER):

            # Decide wheter a greedy (and so deterministic) move is made
            # or not
            if np.random.ranf() > epsilon :
                action = np.argmax(action_value_estimate)
            else :
                action = np.random.randint(low = 0, high = C_N_ARMS)

            # Update the number of times the current action is chosen
            action_counts[action] += 1

            # Get the corresponding reward
            reward = rewards[action] + np.sqrt(C_VARIANCE_ACTION_REWARDS) * np.random.randn()
            
            # Update the total reward of the current run
            total_reward[n_iter, n_epsilon] += reward

            # If the best action among the k possible is chosen, then its a correct action
            if action == best_action :
                correct_actions_ratio[n_iter, n_epsilon] += 1

            # Compute the action value estimate
            if action_counts[action] >= 2 :
                action_value_estimate[action] +=  1 / action_counts[action] \
                    * (reward - action_value_estimate[action])
            else : 
                action_value_estimate[action] = reward

####################################### Statistics #######################################

# Percentage of optimal action
correct_actions_ratio = correct_actions_ratio / C_N_AGENTS
# Total reward
total_reward = total_reward / C_N_AGENTS

####################################### Plots #######################################

# Total reward figure
fig1, ax1 = plt.subplots()
for n_epsilon in range(len(C_EPSILON_VEC)) :
    ax1.plot(total_reward[:, n_epsilon], label = f"$\epsilon$ = {C_EPSILON_VEC[n_epsilon]}")
ax1.set_xlabel("Time steps")
ax1.set_ylabel("Total reward")
ax1.set_title("Total reward of multiple $\epsilon$-greedy policies on a 10-armed testbed")
ax1.legend()

# Correct action ratio figure
fig2, ax2 = plt.subplots()
for n_epsilon in range(len(C_EPSILON_VEC)) :
    ax2.plot(correct_actions_ratio[:, n_epsilon], label = f"$\epsilon$ = {C_EPSILON_VEC[n_epsilon]}")
ax2.set_xlabel("Time steps")
ax2.set_ylabel("Ratio of optimal actions")
ax2.set_title(f"Ratio of optimal actions of multiple $\epsilon$-greedy policies on a {C_N_ARMS}-armed testbed")
ax2.legend()

plt.show()



