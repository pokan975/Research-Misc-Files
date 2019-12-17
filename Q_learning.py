# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:16:04 2019

@author: WilliamShih

Another try of Q-learning
"""
import numpy as np

np.random.seed(0)

# initialize parameters
gamma = 0.8    # discount factor 
alpha = 0.9    # learning rate
epsilon = 0.5  # initial epsilon in epsilon-greedy policy
episodes = 10  # num of total episodes

# size of the 2-D playground
fieldX = 10
fieldY = 10

states = [(x, y) for x in range(fieldX) for y in range(fieldY)]

# action index: 0: up, 1: left, 2: down, 3: right
actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

# state-action value table
# action index: 0: up, 1: left, 2: down, 3: right
Qtable = np.zeros((fieldX*fieldY, 4), "float")
Qtable = dict(zip(states, Qtable))

# create playground, every cell(position) represens a state
field = np.random.randint(0, 20, (fieldX, fieldY))

# create end position at right bottom corner
endX = fieldX - 1
endY = fieldY - 1
# set reward for end state
field[endX, endY] = 999


def action(posX, posY, eps):
    # extract available actions for current state
    avail_actions = Qtable[(posX, posY)]
    if np.random.uniform(0, 1) > eps:
        act = np.argmax(avail_actions)
    else:
        act = np.random.randint(0, 4)
    
    # take action, move to next position
    newX = posX + actions[act][0]
    newY = posY + actions[act][1]
    

    # if position on the border, cannot go beyond border
    if newX < 0 or newX > (fieldX - 1):
        newX = posX
    if newY < 0 or newY > (fieldY - 1):
        newY = posY
        
    return newX, newY, act


def reward(curX, curY, newX, newY):
    # if agent does not move, assign penalty, other positive reward
    if curX == newX and curY == newY:
        r = -10
    else:
        r = field[newX, newY]
        
    return r  # assign -1 penalty for every time step consumption


def test(X, Y):
    path = []
    while (X != endX) or (Y != endY):
        
        act = np.argmax(Qtable[(X, Y)])
        newX = X + actions[act][0]
        newY = Y + actions[act][1]
        
        path.append((X, Y, a))
        X, Y = newX, newY
        
    print(path)



# train Q table
for episode in range(episodes):
    
    # for every episode, start at random positions
    # if start & end pos too close, re-sample positions
    while True:
        init_pos = np.random.randint(0, fieldX*fieldY - 1)
        startX = states[init_pos][0]
        startY = states[init_pos][1]
        
        if (abs(startX - endX) + abs(startY - endY)) > 1:
            break
        
    step = 0

    # do learning process
    while ((startX != endX) or (startY != endY)) and (step < 50000):
        # take action in every state
        nextX, nextY, a = action(startX, startY, epsilon)
        
        # get immediate reward
        rwd = reward(startX, startY, nextX, nextY)
        
        
        # max_a Q(s, a) in next state
        if startX == nextX and startY == nextY:
            maxQ = -10
        else:
            maxQ = np.max(Qtable[(nextX, nextY)])
        # calc temporal difference
        TD = rwd + gamma * maxQ - Qtable[(startX, startY)][a]
        # update Q(s, a)
        Qtable[(startX, startY)][a] = Qtable[(startX, startY)][a] + alpha * TD
        
        # move to next position
        startX, startY = nextX, nextY
        step += 1
        
        # after every episode ends, slightly shrink epsilon so that agent takes
        # more exploitation than exploration increasingly
        epsilon = epsilon * 0.95


