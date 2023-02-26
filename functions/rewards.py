#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   rewards.py
@Time    :   2023/02/26 16:21:55
@Author  :   Thomas Aussaguès, Lucas Fourest and Selman Sezgin 
@Version :   1.0
@Contact :   {thomas.aussagues,lucas.fourest,selman.sezgin}@imt-atlantique.net
@License :   (C)Copyright 2023, Thomas Aussaguès, Lucas Fourest, Selman Sezgin
@Desc    :   None 
'''


import numpy as np


def get_reward(tep_values, bitrates,  action):
    """Return a reward according to the selected action and the data.

    Parameters
    ----------
    tep_values: array-like
        The error rates
    birates: array-like
        The birates associated to each action
    action: int
        The selected action

    Returns
    -------
    reward: float
        A reward 
    """
    if np.random.random() < tep_values[action]:
        reward = 0  # Non-transmitted packet
    else:
        reward = bitrates[action]
    return reward 