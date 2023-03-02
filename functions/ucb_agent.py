#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ucb_agent.py
@Time    :   2023/02/26 16:22:30
@Author  :   Thomas Aussaguès, Lucas Fourest and Selman Sezgin 
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   None 
'''


import numpy as np
from functions.rewards import get_reward

def get_action_ucb(action_values, action_counts, c = 2):
    """Perform action selection according to the Upper 
    Confidence Band policy.

    Parameters
    ----------
    action_values: array-like
        Estimated action values
    action_counts: array-like
        Action counts
    c: float, default 2
        parameter of the UCB 

    Returns
    -------
    action: int
        An action
    """
    
   
    # Upper Confidence Band
    action = np.argmax(action_values + c * np.sqrt(np.log(np.sum(action_counts) + 1) / (action_counts + 1e-12)))
    return action

def train_ucb_agent(per_values, bitrates, n_agents=1000, n_iter=1000, c=2., alpha = None):
    """Train agents using incrementally computed sample averages and
    Upper Confidence Band action selection.

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
    eps: float, default 0.05
        Probability of selecting a non-greedy action instead of the greedy one
    c: float
        parameter of the UCB 
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
   
    ## Learning loop
    for n_iter_current in range(n_iter) :

        # Iter through every agents
        for n_agent_current in range(n_agents):
            
            # Get next action
            action = get_action_ucb(action_values[n_iter_current, n_agent_current, :], action_counts[n_iter_current, n_agent_current, :], c)
            # Get the corresponding reward
            R = get_reward(per_values, bitrates, action)
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
                    alpha * (R - action_values[n_iter_current, n_agent_current, action])

            # Store the reward
            rewards[n_iter_current, n_agent_current] = R

    return rewards, action_values, action_counts




def train_ucb_agent_non_stat(per_values, bitrates, n_agents=1000, n_iter=1000, c=2., alpha = None):
    """Train agents using incrementally computed sample averages and
    Upper Confidence Band action selection in the non-stationary case.

    Parameters
    ----------
   
    per_values: array-like
        PER values of each modulation method at each
        iteration
    birates: array-like
        The birates associated to each action
    n_agent: int, default 1000
        Number of agents
    n_iter: int, default 1000
        Number of time steps
    eps: float, default 0.05
        Probability of selecting a non-greedy action instead of the greedy one
    c: float
        parameter of the UCB 
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
    k = per_values.shape[1]
    # Estimated action values for each agent
    # at each iteration
    action_values = np.zeros((n_iter, n_agents, k))
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
            action = get_action_ucb(action_values[n_iter_current, n_agent_current, :], action_counts[n_iter_current, n_agent_current, :], c)
            # Get the corresponding reward
            R = get_reward(per_values[n_iter_current, :], bitrates, action)
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
                    alpha * (R - action_values[n_iter_current, n_agent_current, action])

            # Store the reward
            rewards[n_iter_current, n_agent_current] = R

    return rewards, action_values, action_counts