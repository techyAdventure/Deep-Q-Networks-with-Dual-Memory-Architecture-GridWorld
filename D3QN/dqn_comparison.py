import numpy as np
import matplotlib.pyplot as plt



def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#plt.figure(figsize=(15, 10))

#DQN_target
# scores_dqn_target = np.loadtxt("E:/Dual Memory SAR/DQN_DEMO_EFAZ/scores_.txt")

# #C:\Users\Towsif\Desktop\New folder\DQNS\dqn_target
# scores_ma_dqn_target = moving_average(scores_dqn_target, n = 100)

# scores_ma_dqn_target_1000 = moving_average(scores_dqn_target, n = 1000)

# plt.plot(np.arange(len(scores_ma_dqn_target)), scores_ma_dqn_target, alpha = 0.1, color = 'b')
# plt.plot(np.arange(len(scores_ma_dqn_target_1000)), scores_ma_dqn_target_1000, alpha = 1, color = 'b', label = "DQN_target_mem")

# #DQN_target_mem
scores_dqn_target_mem = np.loadtxt("E:/Dual Memory SAR/DDQN_DUAL_MEM/Scores/scores_d2qn.txt")
scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, alpha = 0.1, color = 'r')
plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem)), scores_ma_dqn_target_1000_mem, alpha = 1, color = 'r', label = "DDQN")


# scores_dqn_target_mem = np.loadtxt("E:/Dual Memory SAR/DDQN_DUAL_MEM/Scores/scores_random.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, alpha = 0.1, color = 'g')
# plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem)), scores_ma_dqn_target_1000_mem, alpha = 1, color = 'g', label = "DDQN with random pop")

# #DQN_target_mem_new
scores_dqn_target_mem_new = np.loadtxt("E:/Dual Memory SAR/DDQN_DUAL_MEM/Scores/scores_final.txt")
scores_ma_dqn_target_mem_new = moving_average(scores_dqn_target_mem_new, n = 100)
scores_ma_dqn_target_1000_mem_new = moving_average(scores_dqn_target_mem_new, n = 1000)
plt.plot(np.arange(len(scores_ma_dqn_target_mem_new)), scores_ma_dqn_target_mem_new, alpha = 0.1, color = 'b')
plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem_new)), scores_ma_dqn_target_1000_mem_new, alpha = 1, color = 'b', label = "DDQN with Dual Memory")

#DQN_target_mem_randomized
# scores_dqn_target_mem_randomized = np.loadtxt("C:/Users/Towsif/Desktop/499 code/Dual Memory DQNS/DDQN dual mem/DQN_DUAL_MEM_Randomized/scores_dqn.txt")
# maxs = np.max(scores_dqn_target_mem_randomized)
# print('rand',maxs)
# scores_ma_dqn_target_mem_randomized = moving_average(scores_dqn_target_mem_randomized, n = 100)
# maxs = np.max(scores_ma_dqn_target_mem_randomized)
# print('rand moving avg 100',maxs)
# scores_ma_dqn_target_1000_mem_randomized = moving_average(scores_dqn_target_mem_randomized, n = 1000)
# maxs = np.max(scores_ma_dqn_target_1000_mem_randomized)
# print('rand moving avg 1000',maxs)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem_randomized)), scores_ma_dqn_target_mem_randomized, alpha = 0.1, color = 'r')
# plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem_randomized)), scores_ma_dqn_target_1000_mem_randomized, alpha = 1, color = 'r', label = "DQN_target_mem_randomized")

# #Doubel_DQN
# scores_ddqn= np.loadtxt("C:/Users/Towsif/Desktop/499 code/DQNS/DDQN/scores_ddqn.txt")
# scores_ma_ddqn = moving_average(scores_ddqn, n = 100)
# scores_ma_ddqn_1000 = moving_average(scores_ddqn, n = 1000)
# plt.plot(np.arange(len(scores_ma_ddqn)), scores_ma_ddqn, alpha = 0.1, color = 'b')
# plt.plot(np.arange(len(scores_ma_ddqn_1000)), scores_ma_ddqn_1000, alpha = 0.8, color = 'b', label = "Double_DQN")

# # #Doubel_DQN_mem
# # scores_ddqn_mem= np.loadtxt("C:/Users/Towsif/Desktop/499 code/Dual Memory DQNS/DDQN dual mem/scores_ddqn_dual_mem.txt")
# # scores_ma_ddqn_mem = moving_average(scores_ddqn_mem, n = 100)
# # scores_ma_ddqn_1000_mem = moving_average(scores_ddqn_mem, n = 1000)
# # plt.plot(np.arange(len(scores_ma_ddqn_mem)), scores_ma_ddqn_mem, alpha = 0.1, color = 'r')
# # plt.plot(np.arange(len(scores_ma_ddqn_1000_mem)), scores_ma_ddqn_1000_mem, alpha = 0.8, color = 'r', label = "Double_DQN_mem")

# #Doubel_DQN_mem_new
# scores_ddqn_mem_new = np.loadtxt("C:/Users/Towsif/Desktop/ume new code/DDQN_DUAL_MEM/scores_100k.txt")
# scores_ma_ddqn_mem_new = moving_average(scores_ddqn_mem_new, n = 100)
# scores_ma_ddqn_1000_mem_new = moving_average(scores_ddqn_mem_new, n = 1000)
# plt.plot(np.arange(len(scores_ma_ddqn_mem_new)), scores_ma_ddqn_mem_new, alpha = 0.1, color = 'g')
# plt.plot(np.arange(len(scores_ma_ddqn_1000_mem_new)), scores_ma_ddqn_1000_mem_new, alpha = 0.8, color = 'g', label = "Double_DQN_mem_new")



# #Double_dueling_DQN
# scores_d3qn_target = np.loadtxt("C:/Users/Towsif/Desktop/New folder/DQNS/double_dueling_dqn/scores_100k_d3qn.txt")
# scores_ma_d3qn_target = moving_average(scores_d3qn_target, n = 100)
# scores_ma_d3qn_target_1000 = moving_average(scores_d3qn_target, n = 1000)
# plt.plot(np.arange(len(scores_ma_d3qn_target)), scores_ma_d3qn_target, alpha = 0.1, color = 'r')
# plt.plot(np.arange(len(scores_ma_d3qn_target_1000)), scores_ma_d3qn_target_1000, alpha = 0.6, color = 'r', label = "Double_dueling_DQN")

# #Prioritized Experience Replay
# scores_per_target = np.loadtxt("C:/Users/Towsif/Desktop/New folder/DQNS/per/scores_per.txt")
# scores_ma_per_target = moving_average(scores_per_target, n = 100)
# scores_ma_per_target_1000 = moving_average(scores_per_target, n = 1000)
# plt.plot(np.arange(len(scores_ma_per_target)), scores_ma_per_target, alpha = 0.1, color = 'y')
# plt.plot(np.arange(len(scores_ma_per_target_1000)), scores_ma_per_target_1000, alpha = 0.6, color = 'y', label = "PER")


plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend()
plt.savefig('DQN_graph.png')
plt.show()