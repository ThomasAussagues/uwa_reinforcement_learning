#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   greedy_experiment.py
@Time    :   2023/02/10 18:51:22
@Author  :   Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues, lucas.fourest, selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Desc    :   None
'''

"""This script reproduces the experiments on epsilon-greedy policies in chpater 2 of Baro and 
Sutton's book."""

import matplotlib.pyplot as plt
from agent import Testbed
from policies import EpsilonGreedy
from update_rules import IncrementalUpdateRule
from rewards import RewardsNormalDistribution


# Parameters
C_N_ARMS = 10
C_N_AGENTS = 1000
C_N_MAX_ITER = 2000
C_MEAN_REWARD = 0.
C_STD_REWARD = 1.
C_STD_OBSERVATIONS = 1.
C_EPSILON_VALUES = [0, 0.1, 0.01]

# Create figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# Run a testbed for each epsilon value
for n_epsilon, epsilon in enumerate(C_EPSILON_VALUES):
    
    # Get the udpate rule
    update_rule = IncrementalUpdateRule()
    # Get the espilon-greedy policy
    policy = EpsilonGreedy(epsilon=epsilon)
    # Get the rewards distribution
    rewards_distribution = RewardsNormalDistribution(mean_reward=C_MEAN_REWARD,
     std_reward=C_STD_REWARD, std_observations=C_STD_OBSERVATIONS, n_arms=C_N_ARMS)
    # Rn the testbed
    average_total_reward, correct_action_counts_ratio = Testbed(n_arms=C_N_ARMS, 
    n_agents=C_N_AGENTS, n_max_iter=C_N_MAX_ITER, policy=policy, update_rule=update_rule, 
    rewards_distribution=rewards_distribution).run()
    # Total reward figure
    ax1.plot(average_total_reward, label = f"$\epsilon$ = {C_EPSILON_VALUES[n_epsilon]}")
    ax1.set_xlabel("Time steps")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Total reward of multiple $\epsilon$-greedy policies on a 10-armed testbed")
    ax1.legend()
    # Correct action ratio figure
    ax2.plot(correct_action_counts_ratio, label = f"$\epsilon$ = {C_EPSILON_VALUES[n_epsilon]}")
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Ratio of optimal actions")
    ax2.set_title(f"Ratio of optimal actions of multiple $\epsilon$-greedy policies on a {C_N_ARMS}-armed testbed")
    ax2.legend()

plt.show()
