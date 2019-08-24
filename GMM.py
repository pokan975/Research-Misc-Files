import numpy as np
import matplotlib.pyplot as plt


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