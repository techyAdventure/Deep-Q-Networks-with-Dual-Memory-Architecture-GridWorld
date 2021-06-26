import torch
from agent import Agent
import json
from environment import Environment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

env = Environment()
agent = Agent(state_size=2, action_size=4, seed=3)

n_episodes = 100000
max_t = 300
eps_start = 1.0
eps_end = 0.01  # 0.1
eps_decay = 0.999936 # 0.9955
GAMMA = 0.99


def train_agent(agent, env, target_folder, eps_start=eps_start, eps_decay=eps_decay, eps_end=eps_end, gamma=GAMMA, n_episodes=n_episodes, max_t=max_t):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score = 300.0
    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        score = 0
        for t in range(max_t):
            #env.render()
            # time.sleep(0.2)
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            # time.sleep(0.2)
            # if (env.drone_y == env.man_y):
            #     print('\nReward: {:.4f}'.format(reward))
            #print(f'\nStates : {state}')
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        # if i_episode % 10 == 0:
        #     torch.save(agent.qnetwork_local.state_dict(),
        #                "dqn_agent{}.pkl".format(i_episode))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= max_score: 
            max_score = np.mean(scores_window)
            torch.save(agent.qnetwork_local.state_dict(),
                       'checkpoint_100k_1_r3_temp.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
    return scores

start_time = time.time()
scores_ddqn = train_agent(agent, env, target_folder="double_dqn/")
end_time = time.time()

scores_ddqn_np = np.array(scores_ddqn)
np.savetxt("scores_100k_1_r3_temp.txt", scores_ddqn_np)

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "Execution time: %d hours : %02d minutes : %02d seconds" % (hour, minutes, seconds)

n = end_time-start_time
train_time = convert(n)
print(train_time)

mem1 = agent.mem1_learn_counter
mem3 = agent.mem3_learn_counter

train_info_dictionary = {'algorithm': 'DQN_dual_mem', 'eps_start': eps_start, 'eps_end': eps_end,
                         'eps_decay': eps_decay, 'episodes': n_episodes, 'train_time': train_time, 'memory 1 use': mem1, 'memory 3 use': mem3}

train_info_file = open('train_info.json', 'w')
json.dump(train_info_dictionary, train_info_file)
train_info_file.close()


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

scores_ma_ddqn = moving_average(scores_ddqn, n=100)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_ma_ddqn)), scores_ma_ddqn)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.savefig('graph_100k_1_r3_temp.png')
plt.show()
