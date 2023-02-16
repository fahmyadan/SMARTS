import torch
import numpy as np
# from skimage.color import rgb2gray
# from skimage.transform import resize
from torch.distributions import Categorical
from typing import Dict, Tuple, List, Any, Sequence



def get_action(policies, num_actions):
    m={}
    actions = {}
    entropy = {}
    for key in policies.keys():

        m[key] = Categorical(policies[key])
        actions[key] = m[key].sample()
        #print(f'policies size is {policies.get(key).size()} and actions size is {actions.get(key).size()}')
        #actions[key] = actions[key].data.cpu().numpy()
        #print(f'actions length {len(actions[key])}')
        actions[key] = int(actions[key])
        entropy[key] = m[key].entropy()
    # m = Categorical(policies)
    # actions = m.sample()
    # actions = actions.data.cpu().numpy()
    # actions = int(actions)
    return actions, policies, entropy

def select_worker_action(worker_policy: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor] :
    
    m = Categorical(worker_policy)

    action = m.sample()
    worker_action = int(action)
    worker_entropy = m.entropy()
    
    worker_log_prob = m.log_prob(action)

    return worker_action, worker_entropy, worker_log_prob

def select_manager_action(manager_policy:torch.Tensor):

    m =  Categorical(manager_policy)

    manager_action = m.sample()
    
    manager_entropy = m.entropy()

    manager_log_prob = m.log_prob(manager_action)

    return manager_action, manager_entropy, manager_log_prob


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)
    return grad_norm