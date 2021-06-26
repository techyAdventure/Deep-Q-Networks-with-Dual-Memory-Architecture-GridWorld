import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_dqn = np.loadtxt("scores_2x.txt")
scores_ma_dqn = moving_average(scores_dqn, n=100)
scores_ma_dqn_1000 = moving_average(scores_dqn, n=1000)

plt.plot(np.arange(len(scores_ma_dqn)), scores_ma_dqn,
         alpha=0.2, color='b')
plt.plot(np.arange(len(scores_ma_dqn_1000)),
         scores_ma_dqn_1000, label="DQN", color='b')

plt.ylabel('Score')
plt.xlabel('Episode ')
plt.legend()
plt.savefig('graph.png')
plt.show()
