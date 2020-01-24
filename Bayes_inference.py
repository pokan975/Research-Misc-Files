# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:37:30 2020

@author: William
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# generate samples
mean = np.array([1, 2])
covr = np.array([[1, 0.5], [0.5, 2]])

sample = st.multivariate_normal(mean, covr).rvs(200)

# define intital hyperparameters of prior normal-inverse Wishart
mu = 1
lmbda = 1
psi = 1
nu = 1

plt.plot(sample[:, 0], sample[:, 1], ".")
plt.show()