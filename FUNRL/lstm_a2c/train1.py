import numpy as np
import torch
import torch.nn.functional as F
from FUNRL.lstm_a2c.utils import get_grad_norm
from torchviz import make_dot, make_dot_from_trace

# Machine epsilon for rounding precision of floating point numbers
eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_return(rewards, masks, gamma, state_values):

    # R= V(s;theta). Calculate true value using rewards from the environment 
    returns = []
    R = 0 

    batch_size = len(rewards)

    for r in rewards[::-1]: 

        R = r + gamma *R

        returns.insert(0,R)
    
    returns = torch.Tensor(returns).to(device)
    normalised_returns = (returns - returns.mean()) / (returns.std() + eps)

    return normalised_returns

def train_worker(worker_net, worker_optimiser, worker_transition, args): 

    pass

def train_manager(manager_net, manager_critic_optimiser, manager_transitions, args): 

    manager_return = calc_return(manager_transitions.reward, manager_transitions.mask, args.m_gamma, manager_transitions.state_value_est)
    m_adv = []
    manager_actor_loss =[]
    manager_critic_loss = []
    num_ops = 0
    sum_sq_error = torch.nn.MSELoss(reduction='sum')
    for entropy, state_value_est, log_policy, R  in zip(manager_transitions.entropy, manager_transitions.state_value_est , manager_transitions.log_policy ,manager_return):

        manager_advantage = R.item() - state_value_est.item()
        m_adv.append(manager_advantage)

        policy_component = -log_policy * manager_advantage

        manager_actor_loss.append(policy_component) 


        value_component = sum_sq_error(state_value_est, R)
        manager_critic_loss.append(value_component)
        num_ops += 1
        

        # TODO: Incorporate entropy regularisation?
    

    manager_critic_optimiser.zero_grad()

    # Sum up actor and critic losses 
    if not num_ops:
        raise RuntimeError("No ops done ")


    manager_total_critic_loss = torch.stack(manager_critic_loss).sum()
    manager_total_actor_loss = torch.stack(manager_actor_loss).sum()
    manager_net_named_parameters = dict(manager_net.named_parameters()) 
    manager_net_critic_parameters = manager_net.critic.parameters()

    manager_critic_graph = make_dot(manager_total_critic_loss, manager_net_named_parameters, show_saved=True, show_attrs=True)

    manager_total_actor_loss.backward()

    # manager_total_critic_loss.backward(retain_graph=True)


    manager_critic_optimiser.step()

    

    return manager_total_critic_loss