import numpy as np
import torch
import torch.nn.functional as F
from FUNRL.lstm_a2c.utils import get_grad_norm
#from SMARTS.FUNRL.lstm_a2c.utils import get_grad_norm
import time
from data_utils import *
#from SMARTS.data_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Test Commit"""
def get_returns(rewards, masks, gamma, values):
    returns = torch.zeros_like(rewards)
    #returns = torch.zeros(len(rewards), 3)
    running_returns = values[-1].squeeze()

    for t in reversed(range(0, len(rewards)-1)):
        #print('check training')
        #print(f'reward size is {len(rewards)} mask size is {len(masks)} ')
        if len(rewards) > len(masks):
            print('rewards and mask do not match')
        running_returns = rewards[t] + gamma * running_returns * masks[t]

        returns[t] = running_returns
    if returns.std() != 0:
        returns = (returns - returns.mean()) / returns.std()

    return returns
"""
To Do: Refactor train_model():
    - Add repeat operations to data_utils()
"""
def train_model(net, optimizer, transition, args, Worker_IDs):
    start = time.time()
    actions ={w_id:[] for w_id in Worker_IDs}
    entropy = {w_id: [] for w_id in Worker_IDs}
    goal = []
    reward = {w_id: [] for w_id in Worker_IDs}
    policies = {w_id: [] for w_id in Worker_IDs}
    log_policies = {w_id: [] for w_id in Worker_IDs}
    new_w_state = {w_id: [] for w_id in Worker_IDs}
    w_values = {w_id: [] for w_id in Worker_IDs}
    w_lstm = {w_id: [] for w_id in Worker_IDs}

    man_state = []
    m_value = []
    for stuff in transition.actions:
        for keys,values in stuff.items():
            actions[keys].append(values)
    for keys in actions.keys():
        actions[keys]= tuple(actions[keys])
        actions[keys] = torch.Tensor(actions[keys]).long().to(device)
    for stuff in transition.entropy:
        for keys,values in stuff.items():
            entropy[keys].append(values)
    for keys in actions.keys():
        entropy[keys]= tuple(entropy[keys])
        entropy[keys] = torch.Tensor(entropy[keys]).to(device)
    for stuff in transition.reward:
        for keys,values in stuff.items():
            reward[keys].append(values)
    for keys in actions.keys():
        reward[keys]= tuple(reward[keys])
        reward[keys] = torch.Tensor(reward[keys]).to(device)
    for stuff in transition.goal:
        goal.append(stuff)
    goal = tuple(goal)
    goal = torch.stack(goal).to(device)
    for stuff in transition.policy:
        for keys,values in stuff.items():
            policies[keys].append(values)
    for keys in policies.keys():
        policies[keys]= tuple(policies[keys])
    for keys, values in policies.items():
        policies[keys] = torch.stack(values).to(device)
    for stuff in transition.new_w_state:
        for keys,values in stuff.items():
            new_w_state[keys].append(values)
    for keys in new_w_state.keys():
        new_w_state[keys]= tuple(new_w_state[keys])
    for keys, values in new_w_state.items():
        new_w_state[keys] = torch.stack(values).to(device)
    for stuff in transition.w_values:
        for keys,values in stuff.items():
            w_values[keys].append(values)
    for keys in w_values.keys():
        w_values[keys]= tuple(w_values[keys])
        w_values[keys] = torch.Tensor(w_values[keys])
    for stuff in transition.m_state:
        man_state.append(stuff)
    man_state = tuple(man_state)
    man_state = torch.stack(man_state).to(device)
    for stuff in transition.m_value:
        m_value.append(stuff)
    m_value=tuple(m_value)
    m_value= torch.stack(m_value).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    w_returns = {}
    for keys, values in w_values.items():
        w_returns[keys] = get_returns(reward[keys], masks, args.w_gamma, w_values[keys])

    for stuff in transition.w_lstm:
        for keys,values in stuff.items():
            w_lstm[keys].append(values[0])

    """
    Compute junction manager return 
    """
    manager_reward = torch.Tensor(transition.manager_reward).to(device)
    manager_returns = get_returns(manager_reward, masks, args.m_gamma, m_value)

    """
    Calculate intrinsic return as cosine sim between worker hidden states and manager goal
    """
    intrinsic_reward = []
    for key, val in w_lstm.items():
        cos_sum = 0
        for i in range(0, len(w_lstm[key]) - 1 ):
            beta = goal[i+1]
            alpha = w_lstm[key][i]
            cosine_sim = F.cosine_similarity(alpha, beta)
            cos_sum = cos_sum + cosine_sim
        intrinsic_reward.append(cos_sum)
    intrinsic_return = abs(sum(intrinsic_reward))



    """
    Compute manager loss (actor and critic) 
    """
    m_critic_loss = torch.zeros_like(manager_returns).to(device)

    for i in range(len(manager_returns)):
        m_critic_loss[i] = (manager_returns[i] - m_value[i])**2

    total_manager_critic_loss = sum(m_critic_loss)
    log_goals = torch.mean(torch.log(F.softmax(goal)), dim=2)
    m_actor_loss = torch.zeros_like(log_goals).to(device)
    m_adv = torch.zeros_like(manager_returns).to(device)

    for i in range(len(manager_returns)):
        m_adv[i] = manager_returns[i] - m_value[i]
        m_actor_loss[i] = -1*m_adv[i] * log_goals[i]


    total_manager_actor_loss = sum(m_critic_loss)

    """
    To Do: Calculate workers' critic and actor loss (actor loss calculated using int return) 
    """


    for key,values in policies.items():
        for i in range(0, policies[key].size()[0]):
            log_policies[key].append(torch.log(policies[key][i] + 1e-5))
        log_policies[key] = tuple(log_policies[key])
    for key,values in log_policies.items():
        log_policies[key] = torch.stack(values).to(device)


    """
    To Do: Compute worker critic loss. Worker ret and values have different dimensions. 
        Pad returns with 0s when workers have been removed from simulation 
        Done
    """
    steps = len(transition.actions)
    # for key,vals in w_returns.items():
    #     if w_returns[key].size()[0] <  steps:
    #         size = steps - w_returns[key].size()[0]
    #         w_returns[key] = zero_pad_tensor(w_returns[key], size)
    """
    Pad w_values and w_returns to be equal in size. Sometimes an agent dies so some will have different sizes. 
    """
    # for key in w_values.keys():
    #     if w_values[key].size()[0] < steps: 
    #         size = steps - w_values[key].size()[0]
    #         w_values[key] = zero_pad_tensor(w_values[key], size)

    w_adv ={}
    w_actor_loss = {ids:[] for ids in Worker_IDs}
    w_total_actor_loss = {}
    for key, val in w_returns.items():
        for i in range(len(val)):
            w_adv[key] = w_returns[key][i]+ intrinsic_return - w_values[key][i]

            entropy_component = policies[key][i]*log_policies[key][i]
            entropy_component = sum(sum(entropy_component)) * args.entropy_coef
            w_actor_loss[key].append(log_policies[key][i][0][actions[key][i]] * w_adv[key] - entropy_component)
        w_total_actor_loss[key] = -1* sum(w_actor_loss[key])


    w_total_critic_loss = {ids : [] for ids in Worker_IDs}
    for keys in w_returns.keys():
        w_total_critic_loss[keys] = F.mse_loss(w_returns[keys].to(device), w_values[keys].to(device), reduction='sum')
        w_total_critic_loss[keys].requires_grad =True




    """
    Note: Compute loss for manager and worker seperately not together 
        Implement a weighted loss function, where highest entropy gained by workers during the episode are weighted more 
    """
    avg_entropy ={}
    for key in entropy.keys():
        avg_entropy[key] = entropy[key].mean()

    worker_weights = {}
    for key in avg_entropy.keys():
        worker_weights[key] = avg_entropy[key]/ max(avg_entropy.values())

    total_manager_loss = total_manager_actor_loss + total_manager_critic_loss
    weighted_worker_actor_loss = {}
    weighted_worker_critic_loss ={}
    for key, values in w_total_actor_loss.items():

        weighted_worker_actor_loss[key] = worker_weights[key] * w_total_actor_loss[key]
        weighted_worker_critic_loss[key] = worker_weights[key] * w_total_critic_loss[key]


    # TODO: Add entropy to loss for exploration
    
    optimizer.zero_grad()
    total_weighted_worker_actor_loss = sum(weighted_worker_actor_loss.values())
    total_weighted_worker_critic_loss = sum(weighted_worker_critic_loss.values())
    #For some reason, the workers' actors loss is sometimes getting NaN values... Ignore and pretend everything is ok!
    combined_loss = total_manager_loss + total_weighted_worker_critic_loss #+ total_weighted_worker_actor_loss
    #combined_loss.backward(retain_graph=True)
    # print(f'manager actor loss version {total_manager_actor_loss._version} manager critic loss version {total_manager_critic_loss._version}')
    # print(f'worker critic version {total_weighted_worker_critic_loss._version} worker actor version {total_weighted_worker_actor_loss._version}')

    total_manager_actor_loss.backward(retain_graph=True)
    total_manager_critic_loss.backward(retain_graph=True)
    total_weighted_worker_critic_loss.backward(retain_graph=True)

    total_weighted_worker_actor_loss.backward(retain_graph=True)

    grad_norm = get_grad_norm(net)
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    # return total_weighted_worker_critic_loss, total_manager_critic_loss, total_manager_actor_loss, grad_norm
    return combined_loss, grad_norm