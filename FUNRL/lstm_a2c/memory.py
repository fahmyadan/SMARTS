import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition',
                        ('w_state', 'man_state', 'new_w_state', 'new_m_state', 'actions', 'reward',
                         'mask', 'goal', 'policy', 'm_lstm', 'w_lstm',
                         'm_value', 'w_value_ext', 'w_value_int', 'm_state', 'entropy'))


class Memory(object):
    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, w_state, man_state, new_w_state, new_m_state, actions, reward,
             mask, goal, policy, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state, entropy=None):
        """Saves a transition."""
        self.memory.append(Transition(w_state, man_state, new_w_state, new_m_state, actions, reward, mask,
                           goal, policy, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state, entropy))

    def sample(self):
        transitions = Transition(*zip(*self.memory))
        return transitions

    def __len__(self):
        return len(self.memory)