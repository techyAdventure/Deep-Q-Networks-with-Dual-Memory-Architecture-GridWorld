import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_d3qn = np.loadtxt("scores_seed7_d3qn.txt")
scores_ma_d3qn = moving_average(scores_d3qn, n=100)
scores_ma_d3qn_1000 = moving_average(scores_d3qn, n=1000)

plt.plot(np.arange(len(scores_ma_d3qn)), scores_ma_d3qn,
         alpha=0.2, color='r')
plt.plot(np.arange(len(scores_ma_d3qn_1000)),
         scores_ma_d3qn_1000, label="D3QN", color='r')

plt.ylabel('Score')
plt.xlabel('Episode ')
plt.legend()
plt.savefig('graph.png')
plt.show()
