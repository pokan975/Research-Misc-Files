# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:26:49 2019
@author: WilliamShih
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from functools import reduce

np.random.seed(0)
# concentration parameter
alpha = 2
# =============================================================================
# This part of code simulates the realization of Dirichlet process using
# stick-breaking approach, taking stanard Gaussian as base distribution
# =============================================================================
def Stick_Breaking(num_weights, alpha):
    betas = np.random.beta(1, alpha, size = num_weights) 
    betas[1:] *= np.cumprod(1 - betas[:-1])
    return betas

# number of samples
n = 10
sample = np.zeros(n)

# generate realization from G_0 (base distribution)
x = np.random.normal(0., 1., n)
# generate theta from beta distribution
theta = np.random.beta(1, alpha, n)
# compute Dirichlet samples
sample[0] = theta[0]

for i in range(1, n):
    sample[i] = theta[i]
    sample[i] *= reduce(lambda x, y: x*y, 1 - theta[:i])

# another func to generate samples from Dirichlet process
# referred from stackoverflow
weights = Stick_Breaking(n, alpha)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
# plot base distribution
xx = np.linspace(-3, 3, 100)
yy = np.fromiter(map(lambda x: st.norm(0., 1.).pdf(x), xx), dtype = np.float)

plt.stem(x, sample)
plt.plot(xx, yy, 'g-', label = "base dist.")
plt.xlabel(r"$\theta_i$ (from standard normal)")
plt.ylabel(r"Prob. Weight ($\pi_i$)")
plt.legend()
plt.grid()
plt.show()
# check if samples sum to 1
print("Sum of total prob. weights:")
print(sum(sample), sum(weights))


# =============================================================================
# This part of code simulates the realization of Dirichlet process using
# Chinese restaurant process (CRP)
# =============================================================================
# initialize table array
tables = [1, alpha]
# record the growth of table number
tablenum = [1]
# size of dataset
customer = 5000

for c in range(1, customer):
    # set prob. of new customer selecting each table
    table_prob = np.array(tables) / sum(tables)
    # table selection as a realization of categorical distribution
    assignment = np.random.multinomial(1, table_prob)
    # show which table customer selects
    t = np.argwhere(assignment)
    
    # customer selects new table, add a table with 1 customer
    if t[0][0] == len(tables) - 1:
        tables.insert(-1, 1)
        tablenum.append(tablenum[-1] + 1)
    # customer selects one of current tables, add 1 to that table
    else:
        tables[t[0][0]] += 1
        tablenum.append(tablenum[-1])
        
print("\nChinese restaurant process \nTable population")
for i, t in enumerate(tables[:-1]):
    print("{0:5} {1:10}".format(i+1, tables[i]))
plt.plot(tablenum)
plt.xlabel("customer index")
plt.ylabel("table number")
plt.grid()
plt.show()