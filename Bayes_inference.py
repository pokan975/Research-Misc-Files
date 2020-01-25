# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:37:30 2020
@author: William
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import itertools


class NormalInvWishart(object):
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / self.lmbda), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    foo[i,j] = np.sqrt(st.chi2.rvs(self.nu - (i + 1) + 1))
                else:
                    foo[i,j]  = np.random.normal(0, 1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


# generate samples
mean = 5
var = 3

sample = st.norm(mean, var).rvs(200)

# hyperparameters of prior normal
mu = 0
sigma = np.var(sample)

# hyperparameter of prior inverse gamma
alpha = 1

m = np.linspace(-5, 5, 100)
t = np.linspace(0, 10, 100)

prior_m = st.norm(mu, np.sqrt(sigma)).pdf(m)
prior_t = st.invgamma(alpha).pdf(t)
prior = [(x, y) for x in prior_m for y in prior_t]

plt.hist2d(m, t, normed = False, cmap = 'plasma')
plt.show()
