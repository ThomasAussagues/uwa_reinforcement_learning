#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   gradient_agent.py
@Time    :   2023/02/26 16:14:32
@Author  :   Thomas Aussaguès, Lucas Fourest and Selman Sezgin 
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   None '''


import numpy as np
from functions.rewards import get_reward


def get_action_gradient(probabilities) :

    """Returns the next action to take 
    accoding to the input probability distribution

    Parameters
    ----------
    probabilities: array-like,
        the actions probabilities

    Returns
    -------
    action: int
        An action
    """

    # Next action
    action = np.random.choice(probabilities.shape[0], p = probabilities)

    return action


def train_agent_gradient(per_values, bitrates, n_agents=1000, n_iter=1000, alpha_pref = .1, alpha_rwd = .1, init_mean_reward = 0.):
    
    
    """Train agents using the Gradient Bandit algorithm.

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
    alpha_pref: float, default .1
        Learning rate for the preferences
    alpha_rwd: float, default .1
        Learning rate for the mean reward estimate, if set to None,
        the learning rate is set to 1 / number of times action 
        a has been chosen between first and current iteration
    init_mean_reward: float, default 0.
        initial value for the mean reward estimate
        
    Returns
    -------
    rewards: array-like
        rewards for each agent at each iteration
    probabilities: array-like
        Estimated action probabilities for each agent
        at each iteration
    action_counts: array-like
        Number of action selections for each agent
        at each iteration
    """

    ## Initialization
    # Number of actions ie number of modulation methods
    k = per_values.size    
    # Preferences for each agent at each iteration
    preferences = np.zeros((n_iter, n_agents, k))
    # Probabilities for each agent at each iteration
    probabilities = np.zeros((n_iter, n_agents, k))
    # Initial probability distbrutions : unfiform
    probabilities[0, :, :] = (
        np.exp(preferences[0, :, :]) 
        / np.sum(np.exp(preferences[0, :, :]), axis = 1)[:, np.newaxis]
        )
    # Number of action selections for each agent
    # at each iteration
    action_counts = np.zeros((n_iter, n_agents, k)) 
    # Rewards for each agent at each iteration
    rewards = np.zeros((n_iter, n_agents))
    # Mean reward for each agent at each iteration 
    mean_rewards = init_mean_reward + np.zeros((n_iter, n_agents))
   
    
    for n_iter_current in range(n_iter) :
        
        # Iter through every agents
        for n_agent_current in range(n_agents):
            
            # Get next action
            a = get_action_gradient(probabilities[n_iter_current, n_agent_current, :])
            # Get the corresponding reward
            R = get_reward(per_values, bitrates, a)
            # Update the action count
            action_counts[n_iter_current :, n_agent_current, a] = action_counts[n_iter_current - 1, n_agent_current, a] + 1
            
            # If alpha is set to None, the learning-rate is set to
            # 1 / number of times action a has been chosen between first 
            # and current iteration
            if alpha_rwd is None :
                alpha_rwd = 1 / action_counts[n_iter_current, n_agent_current, a] 

            # Store the reward
            rewards[n_iter_current, n_agent_current] = R

            # Update the mean reward
            mean_rewards[n_iter_current, n_agent_current] = (
                alpha_rwd * R 
                + (1 - alpha_rwd) * mean_rewards[n_iter_current - 1, n_agent_current]
                )

            if n_iter_current < n_iter - 1 :
                # Update preferences
                # Chosen action 
                preferences[n_iter_current + 1, n_agent_current, a] = (
                    preferences[n_iter_current, n_agent_current, a] 
                    + alpha_pref * (R - mean_rewards[n_iter_current, n_agent_current]) 
                    * (1 - probabilities[n_iter_current, n_agent_current, a])
                    )
                # Other actions
                preferences[n_iter_current + 1, n_agent_current, : a] = (
                    preferences[n_iter_current, n_agent_current, : a] 
                    - alpha_pref * (R - mean_rewards[n_iter_current, n_agent_current]) 
                    * probabilities[n_iter_current, n_agent_current, a]
                    )
                preferences[n_iter_current + 1, n_agent_current, a + 1 :] = (
                    preferences[n_iter_current, n_agent_current, a + 1: ] 
                    + alpha_pref * (R - mean_rewards[n_iter_current, n_agent_current])
                    * probabilities[n_iter_current, n_agent_current, a]
                    )

                # Update probabilities 
                probabilities[n_iter_current + 1, :, :] = (
                    np.exp(preferences[n_iter_current + 1, :, :] 
                    - np.max(preferences[n_iter_current + 1, :, :], axis = 1)[:, np.newaxis]) 
                    / np.sum(np.exp(preferences[n_iter_current + 1, :, :] 
                    - np.max(preferences[n_iter_current + 1, :, :], axis = 1)[:, np.newaxis]), axis = 1)[:, np.newaxis]
                )
            

    return rewards, probabilities, action_counts

    