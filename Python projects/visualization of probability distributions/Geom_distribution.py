import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import geom

n=10

x = np.arange(1,n+1,1)
xs = [str(x[i]-1) for i in range(len(x))]
p = [0.3,0.5,0.7]
width = 0.35 # the width of the bars
cols = ['r', 'b', 'g']

fig, axs = plt.subplots(1,len(p), figsize=(4*len(p),4), sharey=True)

for i in range(len(p)):
    axs[i].bar(xs, geom.pmf(x, p[i]), alpha=0.5, color=cols[i])
    axs[i].set_title('Geometric PMF for $p={}$'.format(p[i]))
    axs[i].grid()
    fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$p_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Geom_distribution.jpg")
plt.show()