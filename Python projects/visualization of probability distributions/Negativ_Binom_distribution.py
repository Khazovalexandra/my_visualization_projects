import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom

r = np.array([2, 5, 10])
p = 0.5
cols = ['r', 'b', 'g']

fig, ax = plt.subplots(1,len(r),figsize=(4*len(r),4), sharey=True)
x = np.arange(nbinom.ppf(0.01, r, p).min(), nbinom.ppf(0.99, r, p).max())

for i in range(len(r)):
    rv = nbinom(r[i], p)
    ax[i].bar(x, rv.pmf(x), alpha=0.5, color=cols[i])
    ax[i].set_title('Neg. binomial PMF for $r={}, p={}$'.format(r[i], p))

    fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$p_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Negativ_Binom_distribution.jpg")
plt.show()