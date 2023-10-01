import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint
low = [1,-3,0]
high = [6,5,21]

cols = ['r', 'b', 'g']
fig, ax = plt.subplots(1,len(low),figsize=(4.8*len(low),4), sharey=True)

for i in range(len(low)):
    rv = randint(low[i], high[i])
    x = np.arange(low[i], high[i])
    ax[i].bar(x, rv.pmf(x), alpha=0.5, color=cols[i])
    ax[i].set_title('PMF of DUnif over ${},{},\ldots, {}$'.format(x.min(),x.min()+1, x.max()))
    ax[i].grid()

    fig.text(0.5, 0.04, r'$x\rightarrow$', ha='center', va='center')
    fig.text(0.08, 0.5, r'$p_X\rightarrow$', ha='center', va='center', rotation='vertical')

plt.savefig("Python projects/visualization of probability distributions/Discret_Unif_distribution.jpg")
plt.show()