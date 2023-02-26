#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
@File    :   demo_with_gradient_bandit.py
@Time    :   2023/02/26 13:03:53
@Author  :   Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   This script generates total reward and correct action ratio figures 
             according to chapter 2 of Barto and Sutton's book (figure 2.(2) 
             p.38) for the k-armed bandit problem with a gaussian rewards 
             distribution using the Gradient Bandit method.'''

import numpy as np
import matplotlib.pyplot as plt

# General parameters

# Preference learning rate
C_ALPHA_PREF = [0.1, 0.4]
# Average reward learning rate
C_ALPHA_RWD = 0.1
# Number of time steps
C_N_ITER = 1000
# Number of trials for the Monte-Carlo simulation
# <=> number of agents
C_N_AGENTS = 2000
# Number of arms of the bandit problem
C_N_ARMS = 10
# Bandit problem mean for the rewards distirbution
C_MEAN_REWARDS = 4.
# Bandit problem variance for the rewards distirbution
C_VARIANCE_REWARDS = 1.
# Bandit problem variance for the given reward at time t
C_VARIANCE_ACTION_REWARDS = 1.

########################### Gradient Bandit Agents ############################

# 4 experiments: for each value of alpha, agents learn with and without
# baseline

# Initialize storage for the average reward at time t for Gradient Bandit 
# methods
total_reward_gradient = np.zeros((C_N_ITER, len(C_ALPHA_PREF), 2))
# Initialize storage for the ratio of correct action taken at time t for 
# Gradient Bandit methods
correct_actions_ratio_gradient = np.zeros((C_N_ITER, len(C_ALPHA_PREF), 2))

print("\n", "*" * 60)
print("GRADIENT BANDIT METHODS")
print("*" * 60, "\n")

# Iter over all learning rate value
for n_alpha, alpha in enumerate(C_ALPHA_PREF):

    print("*" * 10, f"alpha = {alpha}", "*" * 10)

    # Experiments with and without baseline
    for (n_baseline, baseline) in enumerate([C_MEAN_REWARDS, 0]) :

        # Iter over all agents
        for n_agent in range(C_N_AGENTS):

            if (n_agent + 1) % 100 == 0 :
                    print(f"At agent {n_agent + 1} / {C_N_AGENTS}")

            # Get the "base" rewards distribution according to a normal law
            rewards = C_MEAN_REWARDS + np.sqrt(C_VARIANCE_REWARDS) \
                * np.random.randn(C_N_ARMS)
            # Get the best action
            best_action = np.argmax(rewards)

            # Initialize the current agent storage for the action value
            # estimates and action counts
            preferences = np.zeros(C_N_ARMS)
            probabilities = np.exp(preferences) / np.sum(np.exp(preferences))
            action_counts = np.zeros(C_N_ARMS)

            # Let the agent learn
            for n_iter in range(C_N_ITER):

                # Decide wheter a greedy (and so deterministic) move is made
                # or not
                action = np.random.choice(C_N_ARMS, p = probabilities)

                # Update the number of times the current action is chosen
                action_counts[action] += 1

                # Get the corresponding reward
                reward = rewards[action] + np.sqrt(C_VARIANCE_ACTION_REWARDS) \
                    * np.random.randn()
                
                # Update the total reward of the current run
                total_reward_gradient[n_iter, n_alpha, n_baseline] += reward

                # If the best action among the k possible is chosen, then its 
                # a correct action
                if action == best_action :
                    correct_actions_ratio_gradient[n_iter, n_alpha, n_baseline] += 1

                # Update preferences
                preferences[action] += alpha * (reward - baseline) * (1 - probabilities[action])
                preferences[:action] -= alpha * (reward - baseline) * probabilities[:action]
                preferences[action + 1:] -= alpha * (reward - baseline) * probabilities[action + 1:]

                # Update probabilities
                probabilities = np.exp(preferences) / np.sum(np.exp(preferences))

                if baseline != 0 :
                    baseline = reward * C_ALPHA_RWD + (1 - C_ALPHA_RWD) * baseline



############################### Statistics ####################################

# Percentage of optimal action (Gradient Bandit methods)
correct_actions_ratio_gradient = correct_actions_ratio_gradient / C_N_AGENTS

# Total reward ((Gradient Bandit methods)
total_reward_gradient = total_reward_gradient / C_N_AGENTS

################################# Plots #######################################

# Total reward figure
fig1, ax1 = plt.subplots()
for n_alpha in range(len(C_ALPHA_PREF)) :
    ax1.plot(total_reward_gradient[:, n_alpha, 0], label = f"$\\alpha$ = {C_ALPHA_PREF[n_alpha]}")
    ax1.plot(total_reward_gradient[:, n_alpha, 1], label = f"$\\alpha$ = {C_ALPHA_PREF[n_alpha]}, no baseline")
ax1.set_xlabel("Time steps")
ax1.grid()
ax1.set_ylabel("Total reward")
ax1.set_title(f"Total reward of multiple\nGradient Bandit policies on a {C_N_ARMS}-armed testbed")
ax1.legend()
plt.savefig("barro_sutton_examples/figures/gradient_bandit_rwd.png", dpi = 300)

# Correct action ratio figure
fig2, ax2 = plt.subplots()
for n_alpha in range(len(C_ALPHA_PREF)) :
    ax2.plot(correct_actions_ratio_gradient[:, n_alpha, 0], label = f"$\\alpha $ = {C_ALPHA_PREF[n_alpha]}")
    ax2.plot(correct_actions_ratio_gradient[:, n_alpha, 1], label = f"$\\alpha $ = {C_ALPHA_PREF[n_alpha]}, , no baseline")
ax2.set_xlabel("Time steps")
ax2.grid()
ax2.set_ylabel("Ratio of optimal actions")
ax2.set_title(f"Ratio of optimal actions of multiple\nGradient Bandit policies on a {C_N_ARMS}-armed testbed")
ax2.legend()
plt.savefig("barro_sutton_examples/figures/gradient_bandit_action.png", dpi = 300)



