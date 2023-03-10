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
    """Return a reward according to the selected action and the corresponding
    bitrate. If the packet is transmitted, then the bitrate is returned 
    as reward, else the reward is 0.

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
        The reward 
    """
    if np.random.random() < tep_values[action] :
        # Non-transmitted packet
        reward = 0  
    else :
        # Transmitted packet
        reward = bitrates[action]
    return reward 


def get_reward_shannon(powers, tep_values, bitrates, snrdB, bandwidthHz,
                       action):
    """Return a reward according to the selected action, the corresponding
    power and bitrate. If the packet is transmitted, then the squared bitrate 
    divided by the power is returned as reward, else the reward is minus 
    Shannon's capacity

    Parameters
    ----------
    powers: array-like
        The powers array
    tep_values: array-like
        The error rates
    birates: array-like
        The birates associated to each action
    snrdB: float
        The signal to noise ratio in dB
    bandwidthHz: float
        the channel bandwidth in Hz
    action: int
        The selected action

    Returns
    -------
    reward: float
        The reward 
    """
    if np.random.random() < tep_values[action] :
        # Non-transmitted packet
        # Shannon's capacity: 
        # (Bandwidth in Hz) * log2( 1 + linear SNR) / Relative power
        # SNR in linear units
        snr_lin= 10 ** (snrdB / 10)
        # Reward
        reward = - bandwidthHz * np.log2(1 + snr_lin) / powers[action]

    else :
        # Transmitted packet
        reward = bitrates[action] ** 2 / powers[action]
    return reward

def get_expected_reward_shannon(powers, per_values, bitrates, v_snr_db,
                                bandwidthHz, action):
    """Return the expected reward according to the selected action, 
    the corresponding power and bitrate. If the packet is transmitted, 
    then the squared bitrate divided by the power is returned as reward, 
    else the reward is minus Shannon's capacity

    Parameters
    ----------
    powers: array-like
        The powers array
    per_values: array-like
        The error rates
    birates: array-like
        The birates associated to each action
    v_snr_db: array-like
        The signal to noise ratio array in dB
    bandwidthHz: float
        the channel bandwidth in Hz
    action: int
        The selected action

    Returns
    -------
    reward: float
        The reward 
    """

    # Compute the linear SNR
    snr_lin = 10 ** (v_snr_db / 10)
    # Compute the expected reward
    expected_reward = (1 - per_values[:, action]) \
        * bitrates[action] ** 2 / powers[action] \
        - per_values[:, action] * bandwidthHz \
        * np.log2(1 + snr_lin) / powers[action]
    
    return expected_reward