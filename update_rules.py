#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   update_rules.py
@Time    :   2023/02/10 18:14:44
@Author  :   Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Version :   1.0
@Contact :   {thomas.aussagues, lucas.fourest, selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest & Selman Sezgin
@Desc    :   None
'''

import numpy as np

class IncrementalUpdateRule():

    """Implementation of the incremental update rule for computing the next action value
    estimates according to the chapter 2 of Barto and Sutton's book.
    """

    def __init__(self) -> None:
        "Inits the IncrementalUpdateRule class."
        pass

    def update_estimates(self, action_counts : np.ndarray, action_value_estimates : np.ndarray
    , next_action : int, next_reward : float) -> np.ndarray:

        """Updates the action values estimate using the incremental update rule.
        
        Parameters
        ----------
        
        action_counts         : array-like, the number of times each selected has been selected
        action_value_estimates: array-like, the current action value estimates
        next_action           : int, the next action
        nexr_reward           : float, the next reward
        
        Returns
        --------
        
        action_value_estimates: array-like, the updated action value estimates
        """
      
        action_value_estimates[next_action] += 1 / action_counts[next_action]\
            * (next_reward - action_value_estimates[next_action])

        return action_value_estimates
