# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:16:04 2019

@author: WilliamShih

Another try of Q-learning
"""
import numpy as np

# initialize parameters
gamma = 0.8   # discount factor 
alpha = 0.9   # learning rate

# define state space
states = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8
}

# define action space
actions = {'left': 0, 'right': 1, 'up': 2, 'down': 3}

# create Q table
Q = np.array(np.zeros([9, 4]))

