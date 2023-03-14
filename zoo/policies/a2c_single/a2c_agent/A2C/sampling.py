import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np 
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, model, SavedAction, device):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take 
    return action.item()



def action_inference(observations, model, agent_obs_size):

    agent_obs_array = np.array([observations[1]['ego_lane_dist'], observations[1]['ego_ttc'],
                                    observations[0].ego_vehicle_state.position, observations[0].ego_vehicle_state.linear_velocity,
                                    observations[0].ego_vehicle_state.angular_velocity]).reshape(1,agent_obs_size)


    state = torch.from_numpy(agent_obs_array).float().to(device)

    probs, state_value = model(state)

    m = Categorical(probs)
    action = m.sample()

    return action.item()



