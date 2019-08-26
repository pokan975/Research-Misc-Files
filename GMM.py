import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


N = 8000
comp_prob = [0.5, 0.3, 0.2]
mean = [0.0, 4.0, 6.0]
var = [1.5, 1.0, 1.2]

sample = np.zeros((N,1))
for i in range(N):
    component = np.random.multinomial(1, comp_prob, size = 1)
    point = component[0,0]*np.random.normal(mean[0], var[0], 1) + \
            component[0,1]*np.random.normal(mean[1], var[1], 1) + \
            component[0,2]*np.random.normal(mean[2], var[2], 1)
    sample[i] = point
    
plt.hist(sample, bins = 50)
plt.show()


x = np.arange(3)
y = np.arange(3)
z = np.column_stack([x.ravel(), y.ravel()])
a = st.multivariate_normal([0,0], 1*np.eye(2)).pdf([0,0])
