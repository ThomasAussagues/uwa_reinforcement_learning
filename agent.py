#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   agent.py
@Time    :   2023/02/10 18:17:57
@Author  :   Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues, lucas.fourest, selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Desc    :   None
'''

import numpy as np

class Agent():

    """Implements a reinforcement leanring agent for the k-armed bandits problem with the 
    provided policy, update rule and rewards.
    
    Attributes
    ----------
    
    n_arms                     : int, the number of arms of the bandit problem
    n_max_iter                 : int, the number learning steps of the agent
    policy                     : the policy for taking the next action
    update_rule                : the update rule for the action value estimates
    rewards_distribution       : the rewards distribution at any time
    init_action_value_estimates: array-like, the initial rewrads distribution
    total_reward               : array-like, the received reward at each iteration
    correct_action             : array-like, the number of times a correct action has been selected
    action_counts              : array-like, the number of times each selected has been selected
    action_values_estimates    : array-like, theb current action-value estimates
    """

    def __init__(self, n_arms :int, n_max_iter : int, policy, update_rule, rewards_distribution,
     init_action_value_estimates = None) -> None:

        "Inits the Agent class."
    
        self.n_arms = n_arms
        self.n_max_iter = n_max_iter
        self.policy = policy
        self.update_rule = update_rule
        self.rewards = rewards_distribution
        self.n_iter = 0
        # Storage
        # Storing the total reward at each time step
        self.total_reward = np.zeros(n_max_iter)
        # Storing the number of times each action is chosen
        self.action_counts = np.zeros(self.n_arms)
        # Storing the number of times a correct action is chosen
        self.correct_action_counts = np.zeros(self.n_max_iter)
        # Storing the estimated reward for each action
        if init_action_value_estimates is None :
            self.action_value_estimates = np.zeros(self.n_arms)
        else :
            self.action_value_estimates = init_action_value_estimates

    def action(self) -> None:

        "Makes the agent do an action."

        # Get the next action according to the provided 
        # policy
        next_action = self.policy.get_next_action(self.action_value_estimates)
        # If the next action is indeed the correct one, update the correct actions count
        if next_action == self.rewards.get_correct_next_action():
            self.correct_action_counts[self.n_iter] += 1
        # Get the associated reward
        next_reward = self.rewards.get_next_reward(next_action)
        # Update the action count
        self.action_counts[next_action] += 1
        # Update the total reward
        self.total_reward[self.n_iter] += next_reward
        # Update the action reward estimate
        self.action_value_estimates = self.update_rule.update_estimates(
            self.action_counts, self.action_value_estimates, next_action, next_reward)
        # Update the iterations number
        self.n_iter += 1


class Testbed():

    """Implemented a testbedt for the k-armed bandits problem 
    "according to the chapter 2 of Barto and Sutton's book
    
    Attributes
    ----------

    n_arms                     : int, the number of arms of the bandit problem
    n_agents                   : int, the number of agents
    n_max_iter                 : int, the number learning steps of the agent
    policy                     : the policy for taking the next action
    update_rule                : the update rule for the action value estimates
    rewards_distribution       : the rewards distribution at any time
    init_action_value_estimates: array-like, the initial rewrads distribution
    total_reward               : array-like, the received reward at each iteration
    correct_action_counts      : array-like, the number of times a correct action has been selected
    
    """

    def __init__(self, n_arms, n_agents, n_max_iter, policy, update_rule, rewards_distribution,
    init_action_value_estimates = None) -> None:

        "Inits the testbed class"
        
        
        self.n_arms = n_arms
        self.n_agents = n_agents
        self.n_max_iter = n_max_iter
        self.policy = policy
        self.update_rule = update_rule
        self.rewards_distribution = rewards_distribution
        self.init_action_value_estimates = init_action_value_estimates
        self.total_reward = np.zeros(self.n_max_iter)
        self.correct_actions_counts= np.zeros(n_max_iter)

        print("New testbed")
        print("*" * 80)

    def run(self):

        for agent_index in range(self.n_agents) :
            
            if (agent_index + 1) % 100 == 0 :
                print(f"At agent {agent_index + 1} / {self.n_agents}")

            # Get the rewards ditrsibtuion
            self.rewards_distribution.reset()
            # Create the agent
            agent = Agent(n_arms=self.n_arms, n_max_iter=self.n_max_iter, policy=self.policy
            , update_rule=self.update_rule, rewards_distribution=self.rewards_distribution)
            # Let the agent learn
            for _ in range(self.n_max_iter):
                agent.action()
            # Update the total reward
            self.total_reward += agent.total_reward
            # Update the correct action counts
            self.correct_actions_counts += agent.correct_action_counts

        # Compute the average total reward
        self.average_total_reward = self.total_reward / self.n_agents
        # Compute the correct actions ratio
        self.correct_actions_counts_ratio = self.correct_actions_counts / self.n_agents 

        return self.average_total_reward, self.correct_actions_counts_ratio






