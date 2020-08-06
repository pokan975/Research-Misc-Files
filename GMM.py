import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

plt.rcParams['figure.dpi'] = 80
plt.rcParams['font.size'] = 10
np.random.seed(0)

N = 500  # sample size
comp_prob = [0.3, 0.4, 0.3]  # probability of each component
mean = [0.0, 3.0, 6.0]  # mean of each Gaussian component
var = [1.5, 1.0, 1.2]   # variance of each Gaussian component

# generate random sample
sample = np.zeros((N,1))
for i in range(N):
    component = np.random.multinomial(1, comp_prob, size = 1)
    point = component[0,0] * np.random.normal(mean[0], np.sqrt(var[0]), 1) + \
            component[0,1] * np.random.normal(mean[1], np.sqrt(var[1]), 1) + \
            component[0,2] * np.random.normal(mean[2], np.sqrt(var[2]), 1)
    sample[i] = point
    
# plot histogram of samples
plt.hist(sample, bins = 50)
plt.show()


# compute probabilities of samples
sample_prob = np.zeros((N,1))
for i in range(N):
    prob = comp_prob[0] * st.norm(mean[0], np.sqrt(var[0])).pdf(sample[i]) + \
           comp_prob[1] * st.norm(mean[1], np.sqrt(var[1])).pdf(sample[i]) + \
           comp_prob[2] * st.norm(mean[2], np.sqrt(var[2])).pdf(sample[i])
    sample_prob[i] = prob
    

# plot the PDF of the GMM
x = np.linspace(-3, 9, N)
xpdf = comp_prob[0] * st.norm(mean[0], np.sqrt(var[0])).pdf(x)
ypdf = comp_prob[1] * st.norm(mean[1], np.sqrt(var[1])).pdf(x)
zpdf = comp_prob[2] * st.norm(mean[2], np.sqrt(var[2])).pdf(x)
    
sup = xpdf + ypdf + zpdf
plt.plot(x, xpdf, 'b', label = '$\mu$ = 0, $\sigma^2$ = 1.5, $\pi$ = 0.3')
plt.plot(x, ypdf, 'r', label = '$\mu$ = 3, $\sigma^2$ = 1, $\pi$ = 0.4')
plt.plot(x, zpdf, 'g', label = '$\mu$ = 6, $\sigma^2$ = 1.2, $\pi$ = 0.3')
plt.plot(x, sup, 'k')
plt.legend()
plt.show()