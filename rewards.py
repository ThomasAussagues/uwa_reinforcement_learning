#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rewards.py
@Time    :   2023/02/10 18:03:31
@Author  :   Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues, lucas.fourest, selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Desc    :   None
'''

import numpy as np

class RewardsNormalDistribution():

    """
    Implements the normal distributed rewards according to chapter 2 of Barto
    and Sutton's book
    
    Attributes 
    ----------

    mean_reward     : float, initial rewards distribution mean
    std_reward      : float, initial rewards distribution standard deviation
    std_observations: float, observed rewards standard deivation
    n_arms          : int, number of arms
    rewards         : array-like, rewards
    
    """

    def __init__(self, mean_reward : float, std_reward : float, std_observations : float,
    n_arms : int) -> None:

        """Inits the RewardsNormalDistribution sample class and 
        creates the rewards distribution"""
        
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.std_observations = std_observations
        self.n_arms = n_arms
        # Initial rewards according to previous attributes
        self.rewards = self.mean_reward + self.std_reward * np.random.randn(self.n_arms)
        
    def get_next_reward(self, action : int) -> float:

        """Given an action returns a noisy observation
        of the corresponding reward.

        Parameters
        ----------

        action: int, the chosen action

        Returns
        --------

        float, the corresponding reward
        
        """

        return self.rewards[action] + self.std_observations * np.random.randn()

    def get_correct_next_action(self) -> int:
        
        """Returns the correct action.

        Returns
        --------

        float, the corresponding reward
        
        """

        return np.argmax(self.rewards)


    def reset(self) -> None:

        "Draws an new initial rewards distribution"

        self.rewards = self.mean_reward + self.std_reward * np.random.randn(self.n_arms)

