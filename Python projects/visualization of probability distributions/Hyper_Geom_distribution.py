import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom

N, m = 52, 4
n = [5, 15, 25]
x = np.arange(0, m+1)

cols = ['r', 'b', 'g']
fig, ax = plt.subplots(1,len(n),figsize=(4.8*len(n),4), sharey=True)

for i in range(len(n)):
    rv = hypergeom(N, m, n[i])
    ax[i].bar(x, rv.pmf(x), alpha=0.5, color=cols[i])
    ax[i].set_title('PMF of number of aces for $n={}$'.format(int(n[i])))
    ax[i].grid()

    fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
    fig.text(0.09, 0.5, r'$p_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Hyper_Geom_distribution.jpg")
plt.show()