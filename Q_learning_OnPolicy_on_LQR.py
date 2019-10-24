# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:54:10 2019
@author: WilliamShih
"""
"""
This program simulates the algorithm proposed in the paper 
"Reinforcement Learning Applied to Linear Quadratic Regulation",
following code flow shown in Figure(1) in the paper

Use 1-D system, state/action/reward/value are all scalars
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# initializing parameters
E, F = 1, 1  # LQR quadratic reward function parameters
A, B = 1, 1  # LQR state transition parameters
U = 0.5      # LQR initial stabilizing policy
K = 1        # quadratic state-value function parameter
gamma = 0.9  # discount factor
alpha = 0.9  # learning rate

t, k = 0, 0
s0 = 0       # initial state

# build Q-function parameter matrix, eq.(2)
h11 = E + gamma*A*K*A
h12 = gamma*A*K*B
h21 = gamma*B*K*A
h22 = F + gamma*B*K*B
h = np.array([[h11, h12], [h21, h22]], dtype = 'float')

# initialize RLS estimator for every episode
q0 = 0  # value of current state-action pair (s0, a0)
q1 = 0  # value of next state-action pair (s1, a1)

for episode in range(1):
        
    # in an episode, converge Q-func parameters based on a fixed policy
    for time in range(100):
        # get action for current state using current policy
        a0 = U*s0
        # action output + exploration factor
        a0 += np.random.normal(0, 1)
        # get immediate reward for current state s0 & action a0
        reward = s0*E*s0 + a0*F*a0
        
        # get next state s1 for LQR model
        s1 = A*s0 + B*a0
        # define action a1 for next state based on current policy
        a1 = U*s1
        
        # build [s0, a0] & [s1, a1] matrices for Q(s, a) function (eq.(1))
        s0_a0 = np.array([[s0], [a0]], dtype = 'float')
        s1_a1 = np.array([[s1], [a1]], dtype = 'float')
        
        # get Q(s0, a0) at current time step, eq.(1)
        q0 = np.dot(np.transpose(s0_a0), h)
        q0 = np.dot(q0, s0_a0)
        # get Q(s1, a1) at current time step, eq.(1)
        q1 = np.dot(np.transpose(s1_a1), h)
        q1 = np.dot(q1, s1_a1)
        # update Q(s0, a0) using current Q(s0, a0) & Q(s1, a1) and immediate reward, eq.(4)
        q0 = (1 - alpha) * q0 + alpha * (reward + gamma * q1)
        
        # state moves forward
        s0 = s1
        t += 1
    
    # estimated Q-func parameters maatrix = q0 * [s0, a0] * [s0, a0]^T
    h_est = q0 * np.linalg.inv(np.dot(s0_a0, np.transpose(s0_a0)))
    # improve policy using estimated Q-func parameters: U = -(h22^-1)*h21
    U = -1 * (1 / h_est[1, 1]) * h_est[1, 0]
    # set estimated Q-func parameters for next episode
    h = h_est
    # episode index
    k += 1
