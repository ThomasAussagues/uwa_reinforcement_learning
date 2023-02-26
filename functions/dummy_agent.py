#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dummy_agent.py
@Time    :   2023/02/26 16:21:37
@Author  :   Thomas Aussaguès, Lucas Fourest and Selman Sezgin 
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   None 
'''

import numpy as np
from functions.rewards import get_reward

def train_dummy_agent(per_values, bitrates, n_agents=1000, n_iter=1000):
    """Train dummy agents by taking the modulation that maximizes
    the bitrate at each iteration.

    Parameters
    ----------
   
    per_values: array-like
        TEP values of each modulation method
    birates: array-like
        The birates associated to each action
    n_agent: int, default 1000
        Number of agents
    n_iter: int, default 1000
        Number of time steps
        
    Returns
    -------
    rewards: array-like
        rewards for each agent at each iteration
    action_counts: array-like
        Number of action selections for each agent
        at each iteration

    """

    ## Initialization
    # Number of actions ie number of modulation methods
    k = per_values.size 
    # Number of action selections for each agent
    # at each iteration
    action_counts = np.zeros((n_iter, n_agents, k)) 
    # Rewards for each agent at each iteration
    rewards = np.zeros((n_iter, n_agents))
   
    ## Learning loop
    for n_iter_current in range(n_iter) :

        # Iter through every agents
        for n_agent_current in range(n_agents):
            
            # Get next action
            action = np.argmax(bitrates)
            # Get the corresponding reward
            reward = get_reward(per_values, bitrates, action)
            # Update the action count
            action_counts[n_iter_current + 1 :, n_agent_current, action] = action_counts[n_iter_current, n_agent_current, action] + 1
            # Store the reward
            rewards[n_iter_current, n_agent_current] = reward

    return rewards, action_counts