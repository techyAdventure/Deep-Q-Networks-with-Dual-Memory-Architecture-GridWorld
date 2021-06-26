import torch
import torch.nn as nn
import torch.nn.functional as F


class DuellingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=64, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuellingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.value = nn.Linear(fc1_units, 1)
        self.advantage = nn.Linear(fc1_units , action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        value = self.value(x)
        advantage = self.advantage(x)
        
        Q = value+advantage-torch.mean(advantage, dim=1, keepdim=True)

        return value, advantage, Q
