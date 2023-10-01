import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

n=10
x = np.arange(0,11,1)
p1=0.3
p2=0.7
y1 = binom.pmf(x, n, p1)
y2 = binom.pmf(x, n, p2)

width = 0.35 # the width of the bars

fig, ax = plt.subplots()
ax.grid()
rects1 = ax.bar(x - width/2, y1, width, alpha = 0.6, label='$p={}$'.format(p1))
rects2 = ax.bar(x + width/2, y2, width, alpha = 0.6, label='$p={}$'.format(p2))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'$p_X\rightarrow$')
ax.set_xlabel(r'$x\rightarrow$')
ax.set_title('Binomial PMFs when $n=10$ and $p={},{}$'.format(p1,p2))
ax.legend()

fig.tight_layout()

plt.savefig("Python projects/visualization of probability distributions/Binom_distribution_1.jpg")
plt.show()

n=10

x = np.arange(0,n+1,1)
xs = [str(x[i]) for i in range(len(x))]
p = [0.3,0.5,0.7]
width = 0.35 # the width of the bars
cols = ['r', 'b', 'g']

fig, axs = plt.subplots(1, len(p), figsize=(4*len(p), 4), sharey=True)

for i in range(len(p)):
    axs[i].bar(xs, binom.pmf(x, n, p[i]), alpha=0.5, color=cols[i])
    axs[i].set_title('Binomial PMF for $n={}$, $p={}$'.format(n,p[i]))

plt.savefig("Python projects/visualization of probability distributions/Binom_distribution_2.jpg")
plt.show()