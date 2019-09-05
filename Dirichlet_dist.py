import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# size of sample set
N = 500
# list of Dirichlet parameter sets (K = 3)
alphas = [(1., 1., 1.), (0.5, 0.5, 0.5), (2., 4., 10.), (0.1, 0.5, 0.7)]

# generate Dirichlet samples & plot them
for alpha in alphas:
    samples = st.dirichlet(alpha).rvs(N)
    
    ax = plt.gca(projection = '3d')
    plt.title(r'$\alpha$ = {}'.format(alpha))
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    ax.view_init(azim = 40)
    ax.set_xlabel(r'$p_1$')
    ax.set_ylabel(r'$p_2$')
    ax.set_zlabel(r'$p_3$')
    plt.show()


# use standardized Gamma distribution to generate Dirichlet
a = 3  # choose the parameter set of alpha
gamma1 = st.gamma(alphas[a][0]).rvs(size = (N, 1))
gamma2 = st.gamma(alphas[a][1]).rvs(size = (N, 1))
gamma3 = st.gamma(alphas[a][2]).rvs(size = (N, 1))
Diri = np.concatenate((gamma1, gamma2, gamma3), axis = 1)

for i in range(N):
    # each component as normalized Gamma realization
    norm = sum(Diri[i, :])
    Diri[i, :] /= norm

ax = plt.gca(projection = '3d')
plt.title(r'$\alpha$ = {}, gen from indep Gamma'.format(alphas[a]))
ax.scatter(Diri[:, 0], Diri[:, 1], Diri[:, 2])
ax.view_init(azim = 40)
ax.set_xlabel(r'$p_1$')
ax.set_ylabel(r'$p_2$')
ax.set_zlabel(r'$p_3$')
plt.show()