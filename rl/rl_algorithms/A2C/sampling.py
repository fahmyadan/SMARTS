import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def select_action(state, model ,SavedAction, device, h_lstm):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value, h_lstm  = model(state, h_lstm= h_lstm) #Forward pass through model 

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item() , h_lstm 