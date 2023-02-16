import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition',
                        ('old_state', 'action', 'new_state', 'reward',
                         'mask', 'policy', 'log_policy',
                         'state_value_est', 'entropy'))

"""TODO: Implement maximum capicity"""

class Memory(object):
    def __init__(self):
        
        self.memory = []
        self.position = 0

    def push(self, old_state, action, new_state, reward, mask, policy, log_policy ,state_value_est, entropy):
        """Saves a transition."""
        self.memory.append(Transition(old_state, action, new_state, reward, mask, policy, log_policy ,state_value_est, entropy))

    def sample(self):
        transitions = Transition(*zip(*self.memory))
        return transitions

    def __len__(self):
        return len(self.memory)