import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

fig, ax = plt.subplots(1,2,figsize=(4.8*2,4), sharey=True)

a = 2
b = 4
cols = ['r', 'b', 'g']

rv = uniform(a, b-a)
x = np.arange(a, b, 0.01)

ax[0].plot(x, rv.pdf(x), alpha=0.5, color=cols[0])
ax[0].set_title('PDF of Unif over $[{},{}]$'.format(a,b))
x = np.arange(a-0.2, a, 0.01)
ax[0].plot(x, rv.pdf(x), alpha=0.5, color=cols[0])
x = np.arange(b+0.01, b+0.2, 0.01)
ax[0].plot(x, rv.pdf(x), alpha=0.5, color=cols[0])
ax[0].scatter([a,b], rv.pdf([a,b]), s=50, alpha=0.5, color=cols[0])
ax[0].scatter([a,b], rv.pdf([a-0.01,b+0.01]), s=50, facecolors='none', edgecolors='r')
ax[0].plot([a,a], rv.pdf([a-0.01,a]), '--', alpha=0.5, color=cols[0])
ax[0].plot([b,b], rv.pdf([b,b+0.01]), '--', alpha=0.5, color=cols[0])
x = np.arange(a-0.2, b+0.2, 0.01)

ax[1].plot(x, rv.cdf(x), alpha=0.5, color=cols[1])
ax[1].set_title('CDF of Unif over $[{},{}]$'.format(a,b))
ax[1].grid()

fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
fig.text(0.08, 0.5, r'$f_X\rightarrow$', ha='center', va='center', rotation='vertical')
fig.text(0.52, 0.5, r'$F_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Unif_distribution.jpg")
plt.show()