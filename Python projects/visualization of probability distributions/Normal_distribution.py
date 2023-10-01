import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fig, ax = plt.subplots(1,2,figsize=(4.8*2,4), sharey=True)

a = 0
b = [0.7, 1, 2]
x = np.arange(a-3*max(b), a+3*max(b), 0.01)
cols = ['r', 'b', 'g']

for i in range(len(b)):
    rv = norm(a, b[i])
    ax[0].plot(x, rv.pdf(x), alpha=0.5, color=cols[i], label='$\sigma={}$'.format(b[i]))
    ax[1].plot(x, rv.cdf(x), alpha=0.5, color=cols[i])
    ax[0].set_title('PDF of Normal')
    ax[1].set_title('CDF of Normal')
    ax[1].grid()
    ax[0].legend(bbox_to_anchor=(2.19, 1.0), loc='upper left')

    fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
    fig.text(0.08, 0.5, r'$f_X\rightarrow$', ha='center', va='center', rotation='vertical')
    fig.text(0.52, 0.5, r'$F_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Normal_distribution.jpg")
plt.show()