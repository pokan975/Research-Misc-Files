# -*- coding: utf-8 -*-
"""
@author: William
Gibbs sampling for DPMM
this algorithms corresponds to the algo. 2 of Neal's paper 
"Markov Chain Sampling Methods for Dirichlet Process Mixture Models"
we use Gaussian for both base dist. (G0) of DP prior and mixture components
for simplification, the variance of components are constant & the same
base dist. is only over mean of all components
"""

import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

sys.getdefaultencoding()
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
np.random.seed(1)

# =============================================================================
# Parameters
# =============================================================================
# global parameter, fixed
pi = np.pi
comp_var = 1  # vaiances for all mixture components are constant
g0_mean = 10  # mean of base distribution
g0_var = 2  # variance of base distribution
alpha = 2  # concentration parameter of DP prior

# tunable parameters
N = 200  # number of samples
components = 3  # number of mixture components
iterations = 10**4  # times of Gibbs sampling iteration


# =============================================================================
# Draw samples from finite GMM
# =============================================================================
def sampleMM(n, K, print_out = False):
    '''
    Parameters
    ----------
    n : int
        number of total samples.
    K : int
        number of mixture components.

    Returns
    -------
    val : list[float]
        generate samples from a GMM.
    '''
    # use symmetric Dirichlet to generate mixing proportions of all components,
    # large parameters b/c we want the proportions to be even
    comp_prob = np.random.dirichlet([5.] * K, 1)
    # transform to list b/c np.multinomial needs input type as list 
    comp_prob = list(comp_prob[0])
    # use Gaussian dist. to generate the mean for each component,
    # take round just for simplification
    comp_mean = np.round(np.random.normal(g0_mean, g0_var, K), 2)
    if print_out == True:
        print("component mean: ", comp_mean)

    # pick component for each sample
    pick_component = np.random.multinomial(1, comp_prob, size = n)
    # get the indices of picked components
    mix_component = np.nonzero(pick_component)[1]

    # generate random samples
    val = np.zeros(n)
    std = np.sqrt(comp_var)
    for i, m in enumerate(mix_component):
        # given index, draw sample from corresponding component
        val[i] = np.random.normal(comp_mean[m], std, 1)
        
    return val
    


# =============================================================================
# Gibbs sampling function
# =============================================================================
class Gibbs_sampler(object):
    def __init__(self, size, iters, data):
        '''
        Parameters
        ----------
        size : int
            number of samples.
        iters : int
            number of Gibbs sampling iterations.
        data : array
            observed samples.

        Returns
        -------
        None.
        '''
        self.size = size
        self.iters = iters
        self.data = data
        # record the number of clusters
        self.cluster_num = [1]
        # record the members of each cluster, each cluster represented as a list
        self.clusters = [list(self.data)]
        # record the history of partitions over samples
        self.partition_history = []
        # record the mean of each cluster
        self.cluster_mu = [np.random.normal(g0_mean, np.sqrt(g0_var))]
        # initialize all cluster assignment indicators as zero 
        # (all observations assigned to 1st cluster = index 0 initially)
        self.c = np.zeros(self.size, "int")
        
    def likelihood_new(self, mu, y):
        '''
        Parameters
        ----------
        mu : float
            variable of likelihood of mean given a sample.
        y : float
            value of the sample.

        Returns
        -------
        float
            likelihood value of support of G0 given a sample.
        '''
        a = 2 * pi * np.sqrt(g0_var)
        b1 = (1 + g0_var) * (mu**2)
        b2 = 2 * (g0_var * y + g0_mean) * mu
        b3 = g0_var * (y**2) + (g0_mean**2)
        b = -(b1 - b2 + b3) / (2 * g0_var)
        return np.exp(b) / a


    def G0_posterior(self, y):
        '''
        Parameters
        ----------
        y : list[float]
            all samples assigned to certain cluster.
        
        Returns
        -------
        mu : float
            the new mean of certain cluster drawn from posterior given G0 and 
            samples assigned to this cluster.
        '''
        var = comp_var * g0_var / (comp_var + len(y) * g0_var)
        mean = (g0_mean/g0_var + sum(y)/comp_var) * var
        mu = np.random.normal(mean, np.sqrt(var))
        return mu


    def Gibbs_iteration(self):
        # initialize the size of each cluster (last one always represents newly created one)
        cluster_size = [self.size, alpha]
        comp_std = np.sqrt(comp_var)
        # Gibbs sampling iteration
        for t in range(self.iters):
        # =============================================================================
        #   # sample c[i] sequentially
        # =============================================================================
            # the cluster num for this sampling starts from the num of last time
            self.cluster_num.append(self.cluster_num[-1])    
            for i in range(self.size):
                   
                # remove sample[i] from current assignment
                # current assignment for sample[i] is c[i]
                self.clusters[self.c[i]].remove(self.data[i])
                cluster_size[self.c[i]] -= 1
                # assert cluster_size[self.c[i]] >= 0
                
                # sample[i] was singleton of its orig. cluster and now it left, cluster num - 1
                if cluster_size[self.c[i]] == 0:
                    self.cluster_num[-1] -= 1
        
                # build probability vector for cluster assignment
                prior_prob = np.array(cluster_size) / (self.size - 1 + alpha)
                
                # build the posterior prob. of being assigned to each existing cluster
                likelihood = st.norm(self.cluster_mu, comp_std).pdf(self.data[i])
                posterior_old = np.multiply(prior_prob[0:-1], likelihood)
                
                # build the posterior prob. of being assigned to new cluster
                # obtain likelihood by integrating over sample space of mu 
                likelihood, err = quad(self.likelihood_new, -np.inf, np.inf, args = (self.data[i],))
                posterior_new = np.array([prior_prob[-1] * likelihood])
                
                # combine above 2 cases to build the multinomial choices of assignment
                posterior = np.concatenate((posterior_old, posterior_new), axis = 0)
                # normalize these parameters to make them valid probabilities
                posterior = posterior / sum(posterior)
                # select cluster based on the posterior prob. and extract its index
                c_i_new = np.random.multinomial(1, posterior).nonzero()[0][0]
                # update indicator of sample[i]
                self.c[i] = c_i_new
        
                # sample[i] assigned to new cluster
                if c_i_new == len(cluster_size) - 1:
                    # add a cluster with 1 member and assign sample[i] to it
                    cluster_size.insert(-1, 1)
                    self.clusters.append([self.data[i]])
                    # draw parameter for newly created cluster
                    self.cluster_mu.append(self.G0_posterior([self.data[i]]))
                    # cluster num + 1
                    self.cluster_num[-1] += 1
        
                # sample[i] assigned to one of current clusters
                else:
                    cluster_size[c_i_new] += 1
                    self.clusters[c_i_new].append(self.data[i])
            
            # reassign indices in indicator set c by removing empty clusters 
            temp = self.c
            for i, val in enumerate(cluster_size):
                if val == 0:
                    self.c = np.where(temp > i, self.c - 1, self.c)
                
            # remove empty clusters from sets clusters, cluster_size
            while 0 in cluster_size:
                cluster_size.remove(0)
                self.clusters.remove([])
                # self.cluster_mu.pop(i)  # comment out
                
            # =============================================================================
            #   # update mu for each existing cluster sequentially
            # =============================================================================
            # for non-empty cluster, update its parameter using its members
            self.cluster_mu[:] = list(map(self.G0_posterior, self.clusters))
            
            # record the partition of samples at each iteration
            # (pad each partition with empty clusters to a large enough number of clusters)
            pad_length = 30 - len(cluster_size)
            self.partition_history.append(np.pad(cluster_size, (0, pad_length), "constant", constant_values = 0))



# =============================================================================
# Main function starts here
# =============================================================================
# generate data for fixed sample size
samples = sampleMM(N, components, True)
# do Gibbs sampling for this data set
G = Gibbs_sampler(N, iterations, samples)
G.Gibbs_iteration()

# print experimental parameters
print("components of GMM: ", components)
print("data points: ", N)
print("\u03B1: ", alpha)

# plot histogram of data
plt.hist(samples, bins = 50, density = True)
plt.xlabel("value")
plt.ylabel("normalized histogram")
plt.title("sample distribution")
plt.show()

# plot num of clusters vs. iterations
ii = np.arange(0, iterations + 1)
plt.plot(ii, G.cluster_num, "-o")
plt.xlabel("iteration")
plt.ylabel("cluster number")
plt.grid()
plt.show()

# plot the clusters vs. avg number of samples at each cluster
p = np.array(G.partition_history)
# take avg over all partition histories for each cluster
p_avg = np.mean(p, axis = 0)
ii = np.arange(1, 30 + 1, 1)
plt.bar(ii, p_avg)
plt.xlabel("cluster index")
plt.ylabel("avg. number of samples")
plt.grid()
plt.show()


# =============================================================================
# # test different sample sizes
# sample_size = np.arange(150, 260, 10)
# iterations = 8000
# # expected number of clusters of Gibbs sampling result for each sample size
# avg_cluster = []
# 
# for nn in sample_size:
#     print("size: ", nn)
#     samples = sampleMM(nn, components)
#     G = Gibbs_sampler(nn, iterations, samples)
#     G.Gibbs_iteration()
#     # compute expected number of clusters for each sampling result
#     avg_cluster.append(np.mean(G.cluster_num))
#     
# # plot number of clusters vs. number of samples
# plt.plot(sample_size, avg_cluster)
# plt.xlabel("sample size")
# plt.ylabel("cluster number")
# plt.title("number of clusters vs. sample size")
# plt.grid()
# plt.show()
# =============================================================================
