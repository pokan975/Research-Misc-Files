# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:54:10 2019
@author: WilliamShih
"""
"""
This program simulates the algorithm proposed in the paper 
"Reinforcement Learning Applied to Linear Quadratic Regulation",
following code flow shown in Figure(1) in the paper
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# initializing parameters
E, F = 1, 1  # quadratic reward function parameters
A, B = 1, 1  # state transition parameters
U = -0.5      # initial stabilizing policy
K = 1        # quadratic state-value function parameter
gamma = 0.9  # discount factor
alpha = 0.9  # learning rate

t = 0
s0 = 1       # initial state

# build Q-function parameter matrix, eq.(2)
h11 = E + gamma*A*K*A
h12 = gamma*A*K*B
h21 = gamma*B*K*A
h22 = F + gamma*B*K*B
h = np.array([[h11, h12], [h21, h22]], dtype = 'float')

# initialize RLS estimator for every episode
q0 = 0  # value of current state-action pair (s0, a0)
q1 = 0  # value of next state-action pair (s1, a1)

steps = 1
state = np.zeros((steps,2))
action = np.zeros((steps,3))

for time in range(steps):
    # get action for current state using current policy
    a0 = U*s0
    # action output + exploration factor
    action[time, 2] = np.random.normal(0, 1)
    a0 = a0 + action[time, 2]
    
    # get immediate reward for current state s0 & action a0
    reward = s0*E*s0 + a0*F*a0
    
    # get next state s1 for LQR model
    s1 = A*s0 + B*a0
    # the action that minimizes the Q-func for next state
    a1 = -1 * (1 / h[1, 1]) * h[1, 0] * s1
    
    # build [s0, a0] & [s1, a1] matrices for Q(s, a) function (eq.(1))
    s0_a0 = np.array([[s0], [a0]], dtype = 'float')
    s1_a1 = np.array([[s1], [a1]], dtype = 'float')
    
    # get Q(s0, a0) at current time step, eq.(1)
    temp = np.dot(np.transpose(s0_a0), h)
    q0 = np.dot(temp, s0_a0)
    # get Q(s1, a1) at current time step, eq.(1)
    temp = np.dot(np.transpose(s1_a1), h)
    q1 = np.dot(temp, s1_a1)
    # update Q(s0, a0) using current Q(s0, a0) & Q(s1, a1) and immediate reward, eq.(3)
    q0 = (1 - alpha) * q0 + alpha * (reward + gamma * q1)
    
    h_est = q0 * np.linalg.inv(np.dot(s0_a0, np.transpose(s0_a0)))
    
    
    # move forward state
    state[time, 0] = s0
    action[time, 0] = a0
    state[time, 1] = s1
    action[time, 1] = a1
    s0 = s1
    t += 1



plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
x = range(steps)
plt.plot(x, state[:, 0], 'b', label = '$s_0$')
plt.plot(x, state[:, 1], 'r', label = '$s_1$')
plt.plot(x, action[:, 0], 'g', label = '$a_0$')
plt.plot(x, action[:, 1], 'k', label = '$a_1$')
plt.plot(x, action[:, 2], 'p', label = 'noise')
plt.xlabel("step")
plt.grid()
plt.legend()
plt.show()
