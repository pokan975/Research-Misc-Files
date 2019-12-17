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
fieldX = 8
fieldY = 8

states = [(x, y) for x in range(fieldX) for y in range(fieldY)]

# action index: 0: up, 1: left, 2: down, 3: right
actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

# =============================================================================
# state-action value table
# action index: 0: up, 1: left, 2: down, 3: right
# =============================================================================
Qtable = np.zeros((fieldX*fieldY, 4), "float")
Qtable = dict(zip(states, Qtable))

# create playground, every cell(position) represens a state
field = np.random.randint(0, 20, (fieldX, fieldY))

# term rightmost bottom cell as end state
endX = fieldX - 1
endY = fieldY - 1
# set reward for end state
field[endX, endY] = 999


# =============================================================================
# Define action function:
# input: current state (X, Y coord)
#        epsilon (exploration parameter)
# output: next state (X, Y coord)
#         action taken (index)
# given state, select an action from available actions in action-state table
# using epsilon-greedy algo. (uniformly pick a number between 0 and 1, if greater
# than epsilon, take action with max value, if not, randomly pick one action)
# If action taken would make agent pass border, then agent will stay put
# =============================================================================
def action(posX: int, posY: int, eps: float):
    # extract available actions for current state
    avail_actions = Qtable[(posX, posY)]
    # uniformly pick a number between 0 and 1
    # if > epsilon, take action with max value
    if np.random.uniform(0, 1) > eps:
        act = np.argmax(avail_actions)
    # if =< epsilon, randomly select an action
    else:
        act = np.random.randint(0, 4)
    
    # take action, move to next state
    newX = posX + actions[act][0]
    newY = posY + actions[act][1]
    

    # check if next state exceeds the border, agent have to stay at orig state
    if newX < 0 or newX > (fieldX - 1):
        newX = posX
    if newY < 0 or newY > (fieldY - 1):
        newY = posY
    
    # return next state and action taken
    return newX, newY, act


# =============================================================================
# Define reward function:
# input: current state (X, Y coord)
#        next state (X, Y coord)
# output: reward (int value)
# if agent takes acion and move to next state, give reward, which value is 
# shown on the cell of next state
# if agent takes action but stays at same state, give fixed neg reward
# =============================================================================
def reward(curX: int, curY: int, newX: int, newY: int):
    # if agent has taken action but does not move to new state, assign penalty
    if curX == newX and curY == newY:
        r = -10
    # otherwise, assign pos reward
    else:
        r = field[newX, newY]
    
    # return reward for the action
    return r


# =============================================================================
# Define test function:
# input: initial state (X, Y coord)
# output: path (cell list)
# after Q table is trained, given an initial state, show the optimal path from
# initial state to terminal state using Q table
# =============================================================================
def test(X: int, Y: int):
    path = []
    while (X != endX) or (Y != endY):
        # pick action with max value
        act = np.argmax(Qtable[(X, Y)])
        # take action and move to next state
        newX = X + actions[act][0]
        newY = Y + actions[act][1]
        # record cell and action taken on the route
        path.append((X, Y, a))
        # update state
        X, Y = newX, newY
        
    print(path)



# =============================================================================
# Training Q table 
# initial state to terminal state using Q table
# =============================================================================
for episode in range(episodes):
    
    # for every episode, start at random state
    # if start & end pos too close, re-sample state
    while True:
        init_pos = np.random.randint(0, fieldX*fieldY - 1)
        startX = states[init_pos][0]
        startY = states[init_pos][1]
        if (abs(startX - endX) + abs(startY - endY)) > 1:
            break
        
    # record how many steps it takes for each episode to get the terminal state
    step = 0

    # do learning process, to avoid infinite loop, limit episode to max 50000 steps
    while ((startX != endX) or (startY != endY)) and (step < 50000):
        # take action in every state
        nextX, nextY, a = action(startX, startY, epsilon)
        
        # get immediate reward
        rwd = reward(startX, startY, nextX, nextY)
        
        
        # extract max_a Q(s, a) in next state for Q table update
        # I made a little tweak here:
        # if stay at same state, to prevent agent from taking max_a Q(s, a) from
        # current Q table of current state, always assign a fixed penalty as
        # max_a Q(s, a)
        if startX == nextX and startY == nextY:
            maxQ = -10
        # otherwise, extract max Q value for next state
        else:
            maxQ = np.max(Qtable[(nextX, nextY)])
        # calc temporal difference
        TD = rwd + gamma * maxQ - Qtable[(startX, startY)][a]
        # update Q(s, a) for current state
        Qtable[(startX, startY)][a] = Qtable[(startX, startY)][a] + alpha * TD
        
        # update current state
        startX, startY = nextX, nextY
        step += 1
        
    # after every episode ends, slightly shrink epsilon so that agent takes
    # more exploitation than exploration increasingly
    epsilon = epsilon * 0.95