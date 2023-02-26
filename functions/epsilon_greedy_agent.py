#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   epsilon_greedy_agent.py
@Time    :   2023/02/23 13:22:33
@Author  :   Thomas Aussaguès, Lucas Fourest and Selman Sezgin 
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   None
'''

import numpy as np
from functions.rewards import get_reward

def get_action_epsilon_greedy(Q, eps):
    """Perform an epsilon-greedy action selection.

    Parameters
    ----------
    Q: array-like
        Estimated action values
    eps: float
        Probability of selecting a non-greedy action instead of the greedy one

    Returns
    -------
    action: int
        An action
    """
    if np.random.random() <= eps:
        action = np.random.randint(Q.size)
    else:
        action = np.argmax(Q)
    return action

def train_agent_epsilon_greedy(per_values, bitrates, n_agents=1000, n_iter=1000, eps=.05, alpha = None):
    """Train agents using incrementally computed sample averages and
    epsilon-greedy action selection.

    Parameters
    ----------
    per_values: array-like
        TEP values of each modulation method
    birates: array-like
        The birates associated to each action
    n_iter: int, default 1000
        Number of time steps
    n_agent: int, default 1000
        Number of agents
    eps: float, default 0.05
        Probability of selecting a non-greedy action instead of the greedy one
    alpha: float, 
        Learning rate, if set to None, the sample-average method is used: 
        the learning rate is set to 1 / number of times action a has been chosen 
        between first and current iteration
        
    Returns
    -------
    rewards: array-like
        rewards for each agent at each iteration
    action_values: array-like
        Estimated action values for each agent
        at each iteration
    action_counts: array-like
        Number of action selections for each agent
        at each iteration
    """

    ## Initialization
    # Number of actions ie number of modulation methods
    k = per_values.size 
    # Estimated action values for each agent
    # at each iteration
    action_values = np.zeros((n_iter, n_agents, k))
    # Number of action selections for each agent
    # at each iteration
    action_counts = np.zeros((n_iter, n_agents, k)) 
    # Rewards for each agent at each iteration
    rewards = np.zeros((n_iter, n_agents))
   
    
    for n_iter_current in range(n_iter) :

        # Iter through every agents
        for n_agent_current in range(n_agents):
            
            # Get next action
            action = get_action_epsilon_greedy(action_values[n_iter_current, n_agent_current, :], eps)
            # Get the corresponding reward
            reward = get_reward(per_values, bitrates, action)
            # Update the action count
            action_counts[n_iter_current + 1 :, n_agent_current, action] = action_counts[n_iter_current, n_agent_current, action] + 1
            
            # If alpha is set to None, the learning-rate is set to
            # 1 / number of times action a has been chosen between first 
            # and current iteration
            if alpha is None :
                alpha = 1 / action_counts[n_iter_current, n_agent_current, action] 

            # Update of the action-value estimate for the selected action
            action_values[n_iter_current + 1 : , n_agent_current, action] =\
                 action_values[n_iter_current, n_agent_current, action] +\
                    alpha * (reward - action_values[n_iter_current, n_agent_current, action])

            # Update the mean reward
            rewards[n_iter_current, n_agent_current] = reward

    return rewards, action_values, action_counts

    