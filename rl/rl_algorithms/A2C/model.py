import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ACPolicy(nn.Module):
    """
    implements both actor and critic in one model
    Input size: 4 
    Actor Output size: 2
    Critic output size: 1   
    """
    def __init__(self, input_size, disc_action_size, lstm_hidden_size):
        super(ACPolicy, self).__init__()
        self.affine1 = nn.Linear(input_size, 64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size)
        self.affine2 = nn.Linear(64, 128)

        # actor's layer
        self.action_head = nn.Linear(128, disc_action_size)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, h_lstm):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        x, hx = self.lstm(x, h_lstm) 

        x = F.relu(self.affine2(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, hx