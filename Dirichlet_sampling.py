# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:26:49 2019

@author: WilliamShih
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# number of realizations
n = 10
alpha = 1

x = np.random.normal(0., 1., (n, 1))
v = np.random.beta(alpha, 1, (n, 1))

sample = np.zeros((n, 1))
sample[0, 0] = v[0, 0]

for i in range(1, n):
    sample[i, 0] = v[i, 0]
    for j in range(i):
        sample[i, 0] *= (1 - v[j, 0])
        
plt.bar(x, range(n))
plt.show()