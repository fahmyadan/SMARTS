import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def select_action(state, model, SavedAction, device):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()
    entropy = m.entropy()
    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value, entropy))

    # the action to take (left or right)
    return action.item()