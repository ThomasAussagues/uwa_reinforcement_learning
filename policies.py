#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   policies.py
@Time    :   2023/02/10 18:09:22
@Author  :   Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues, lucas.fourest, selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Desc    :   None
'''


import numpy as np

class EpsilonGreedy():

    """Implements the epsilon-greedy policy according to chapter 2 of Barto
    and Sutton's book. The agent chooses a greedy action with probability 
    1 - epsilon and a random one with probability epsilon.
    
    Attributes
    ----------
    
    epsilon: float, the probability of chosing a non-greedy action
    """

    def __init__(self, epsilon : float) -> None:
        
        "Inits the EpsilonGreedy class"

        self.epsilon = epsilon

    def get_next_action(self, action_values_estimates : np.ndarray) -> int:

        """Returns the next action according to the current epsilon-greedy
        policy.
        
        Parameters
        ----------
        
        action_values_estimates: array-like, theb current action-value estimates
        
        Returns
        -------
        
        next_action: int, the next action"""

        # Decide wheter a greedy (and so deterministic move) is made
        # or not
        if np.random.ranf() > self.epsilon :
            next_action = np.argmax(action_values_estimates)
        else :
            next_action = np.random.randint(low = 0, high = action_values_estimates.shape[0])

        return next_action