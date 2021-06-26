from agent import Agent
import torch
from environment import Environment
import time

env = Environment()

agent = Agent(state_size=2, action_size=4, seed=2)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_100k_1_r3_temp.pth'))

for i in range(10):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        state, reward, done = env.step(action)
        env.render()
        time.sleep(0.4)
        if done:
            time.sleep(1)
            break
