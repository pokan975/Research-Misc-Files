# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:26:49 2019
@author: WilliamShih
This program simulates generating samples from Dirichlet process using
stick-breaking approach, takes standard Gaussian as base distribution
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from functools import reduce

np.random.seed(1)

def Stick_Breaking(num_weights, alpha):
    betas = np.random.beta(1, alpha, size = num_weights) 
    betas[1:] *= np.cumprod(1 - betas[:-1])
    return betas

# number of samples
n = 10
# alpha for Dirichlet distribution
alpha = 1

sample = np.zeros(n)

# generate realization from G_0 (base distribution)
x = np.random.normal(0., 1., n)
# generate theta from beta distribution
theta = np.random.beta(alpha, 1, n)
# compute Dirichlet samples
sample[0] = theta[0]

for i in range(1, n):
    sample[i] = theta[i]
    sample[i] *= reduce(lambda x, y: x*y, 1 - theta[:i])

# another func to generate samples from Dirichlet process
# referred from stackoverflow
weights = Stick_Breaking(n, alpha)

plt.rcParams['font.size'] = 12
# plot base distribution
xx = np.linspace(-3, 3, 100)
yy = np.fromiter(map(lambda x: st.norm(0., 1.).pdf(x), xx), dtype = np.float)

plt.stem(x, sample, label = "a realization")
plt.plot(xx, yy, 'g-', label = "base dist.")
plt.xlabel("$G_0$ (Gaussian)")
plt.ylabel("Weight")
plt.legend()
plt.grid()
plt.show()
# check if samples sum to 1
print("Sum of all weights:")
print(sum(sample), sum(weights))
