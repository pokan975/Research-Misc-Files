# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12


# =============================================================================
# Likelihood Function:
# compute the joint likelihood of the samples given every mean and variance sampled
# from prior distribution. Since samples are assumed drawn i.i.d,
# joint likelihood can be factored into product of likelihood of each sample
# take normal distribution as likelihood distribution
# =============================================================================
def Likelihood(data, mu, var):
    # initialize the joint likelihood value
    like = 1
    
    for datum in data:
        # calc likelihood of each sample, then multiply them
        pdf_val = st.norm(mu, np.sqrt(var)).pdf(datum)
        like *= pdf_val
    
    # return likelihood value of the sample set
    return like


# =============================================================================
# generate samples (dataset) from a normal distribution
# =============================================================================
# parameters of true distribution that samples are from
mean = 5  # true mean
var = 3   # true variance
# sample from true distribution
sample = st.norm(mean, var).rvs(20)


# =============================================================================
# mean & variance joint prior distribution
# divide the joint prior into 2 indep. priors for mean & variance, respectively
# for variance, use its reciprocal, precision, for prior dist. instead
# so Normal inverse Gamma (joint prior) => Normal (mean) * inverse Gamma (precision)
# =============================================================================
# define value ranges of mean & precision priors
mean_prior = np.linspace(0, 10, 50)
prec_prior = np.linspace(0.1, 1, 50)


# hyperparameters of mean prior (normal distribution)
hyper_mu = 3
hyper_sigma = np.var(sample)

# hyperparameter of precision prior (inverse gamma dist.) (beta = 1)
alpha = 0.1


# calc mean & precision priors probability values, respectively
mean_prior_pdf = st.norm(hyper_mu, np.sqrt(hyper_sigma)).pdf(mean_prior)
prec_prior_pdf = st.invgamma(alpha).pdf(prec_prior)

prior = np.zeros((len(prec_prior), len(mean_prior)))

# combine the joint prior as product of mean prior & precision prior
for i in range(prec_prior.shape[0]):
    prior[i, :] = prec_prior_pdf[i] * mean_prior_pdf[:]


# plot the 2D colormap of the joint prior
prior = np.flipud(prior)
# imshow: extent = [x_min , x_max, y_min , y_max] sets x, y axis values
plt.imshow(prior, cmap = "plasma", extent = [0, 10, 0.1, 1], aspect = "auto")
plt.xlabel("mean value")
plt.ylabel("precision value")
plt.title("prior (Normal Inverse Gamma) distribution\ncolor represents prior probability value")
plt.show()


# =============================================================================
# mean & variance joint posterior distribution
# for every (mean, precision) pair in joint prior, calc its corresponding
# proportional posterior = likelihood * prior
# doesn't take prob. of sample set into account, it's constant for mean & precision
# =============================================================================
posterior = np.zeros((len(prec_prior), len(mean_prior)))

# for every (mean, precision) point in joint prior, calc posterior
for i in range(prec_prior.shape[0]):
    for j in range(mean_prior.shape[0]):
        # variance = 1/(precision)
        vari = prec_prior[i]**(-1)
        # calc likelihood given (mean, variance) parameter pair
        L = Likelihood(sample, mean_prior[j], vari)
        # calc proportional posterior
        posterior[i, j] = prec_prior_pdf[i] * mean_prior_pdf[j] * L
        

# plot the 2D colormap of the joint posterior
posterior = np.flipud(posterior)
# imshow: extent = [x_min , x_max, y_min , y_max] sets x, y axis values
plt.imshow(posterior, cmap = "plasma", extent = [0, 10, 0.1, 1], aspect = "auto")
plt.xlabel("mean value")
plt.ylabel("precision value")
plt.title("posterior distribution\ncolor represents prior probability value")
plt.show()
