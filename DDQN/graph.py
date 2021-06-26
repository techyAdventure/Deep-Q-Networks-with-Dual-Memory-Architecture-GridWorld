import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_ddqn = np.loadtxt("scores_100k_1_r3_temp.txt")
scores_ma_ddqn = moving_average(scores_ddqn, n=100)
scores_ma_ddqn_1000 = moving_average(scores_ddqn, n=1000)

plt.plot(np.arange(len(scores_ma_ddqn)), scores_ma_ddqn,
         alpha=0.2, color='b')
plt.plot(np.arange(len(scores_ma_ddqn_1000)),
         scores_ma_ddqn_1000, label="DDQN", color='b')

plt.ylabel('Score')
plt.xlabel('Episode ')
plt.legend()
plt.savefig('graph.png')
plt.show()
