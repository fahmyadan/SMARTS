import numpy as np
import torch
import torch.nn.functional as F
from FUNRL.lstm_a2c.utils import get_grad_norm
import time
from data_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Test Commit"""
def get_returns(rewards, masks, gamma, values):
    returns = torch.zeros_like(rewards)
    #returns = torch.zeros(len(rewards), 3)
    running_returns = values[-1].squeeze()

    for t in reversed(range(0, len(rewards)-1)):
        #print('check training')
        print(f'reward size is {len(rewards)} mask size is {len(masks)} ')
        if len(rewards) > len(masks):
            print('rewards and mask do not match')
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        #print(f'running returns = {running_returns}')
        returns[t] = running_returns
    print('end of running return statement')
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
    w_values_int = {w_id: [] for w_id in Worker_IDs}
    w_values_ext = {w_id: [] for w_id in Worker_IDs}
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
    for stuff in transition.w_value_int:
        for keys,values in stuff.items():
            w_values_int[keys].append(values)
    for keys in w_values_int.keys():
        w_values_int[keys]= tuple(w_values_int[keys])
        w_values_int[keys] = torch.Tensor(w_values_int[keys])
    for stuff in transition.w_value_ext:
        for keys,values in stuff.items():
            w_values_ext[keys].append(values)
    for keys in w_values_ext.keys():
        w_values_ext[keys]= tuple(w_values_ext[keys])
        w_values_ext[keys] = torch.Tensor(w_values_ext[keys])
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
    for keys, values in w_values_ext.items():
        w_returns[keys] = get_returns(reward[keys], masks, args.w_gamma, w_values_ext[keys])
    """
    To Do: Implement a different reward and calculate new return for manager 
    At present, manager reward is calculated as an avg of worker rewards in each step
    """
    m_returns = {}
    for keys, values in w_values_ext.items():
        m_returns[keys]= get_returns(reward[keys], masks, args.m_gamma, m_value)
    m_vals = []
    for values in m_returns.values():
        m_vals.append(values)
    m_ret = torch.zeros_like(m_value)
    steps = m_value.size()[0]
    for i in range(len(m_vals)):
        if m_vals[i].size()[0]< steps:
            pad = steps - m_vals[i].size()[0]
            padding = (0,0,0,pad)
            pads = torch.nn.ZeroPad2d(padding)
            m_vals[i] = pads(m_vals[i]).reshape(steps,1)

    m_ret = torch.mean(torch.stack(m_vals), dim=0)
    m_returns = m_ret
    """
    To Do: Determine why adjacent states (step i to i+1) are the same for m_state and w_state???
    """
    # todo: how to get intrinsic reward before 10 steps
    intrinsic_return= {}
    intrinsic_reward = []
    for keys, values in w_values_int.items():

        for i in range(args.horizon, len(reward[keys])):
            cos_sum = 0
            for j in range(1, args.horizon + 1):
                alpha = man_state[i] - man_state[i-j]
                beta = goal[i-j]
                cosine_sim = F.cosine_similarity(alpha, beta)
                cos_sum = cos_sum + cosine_sim
            intrinsic_reward.append(cos_sum/args.horizon)
    intrinsic_reward = torch.cat(intrinsic_reward)
    int_returns ={}
    intrinsic_masks = torch.ones_like(intrinsic_reward).to(device)
    for key in w_values_int.keys():
        int_returns[key] = get_returns(intrinsic_reward, intrinsic_masks, args.w_gamma, w_values_int[key])


    m_loss = torch.zeros_like(m_returns).to(device)
    w_loss={}
    for key in w_returns.keys():
        w_loss[key] = torch.zeros_like(w_returns[key]).to(device)
    w_adv ={}
    action_inst ={}
    for i in range(0, steps - args.horizon):
        m_adv = m_returns[i] - m_value[i]
        alpha = man_state[i + args.horizon] - man_state[i]
        beta = goal[i]
        cosine_sim = F.cosine_similarity(alpha, beta)
        m_loss[i] = -m_adv * cosine_sim
    """
    To Do: Investigate why worker advantage functions and policies are not computed correctly 
    """
    for key,values in policies.items():
        for i in range(0, policies[key].size()[0]):
            log_policies[key].append(torch.log(policies[key][i] + 1e-5))
        log_policies[key] = tuple(log_policies[key])
    for key,values in log_policies.items():
        log_policies[key] = torch.stack(values).to(device)
        # log_policy[key] = log_policy[key].gather(-1, actions[key][i])

    for key in w_returns.keys():
        for i in range(0, len(reward[key])-args.horizon):

            w_adv[key] = w_returns[key][i] + int_returns[key][i] - w_values_ext[key][i] - w_values_int[key][i]
            action_inst[key] = actions[key][i]


            #log_policy[key] = log_policy[0][action_inst[key]]
            # w_advantage = w_returns[i].to(device) + returns_int[i].to(device) - w_values_ext[i].squeeze(-1) - w_values_int[i].squeeze(-1)
            # action_inst = actions[i].item()
            # #log_policy = log_policy.gather(-1, actions[i].unsqueeze(-1))
            # log_policy = log_policy[0][action_inst]
            # w_loss[i] = - w_advantage * log_policy.squeeze(-1)

            #
            # log_policy[key] = torch.log(policies[key][i] +1e-5)
            # w_adv[key] = w_returns[key][i] + int_returns[key][i] - w_values_ext[key][i] - w_values_int[key][i]
            #
            # action_inst[key] = actions[key][i]
    manager_mean_actor_loss = m_loss.mean()
    worker_mean_loss = {}
    for key, val in w_loss.items():
        worker_mean_loss[key] = val.mean()

    manager_value_loss = F.mse_loss(m_value.squeeze(), m_returns.detach())
    """
    To Do: Compute worker critic loss. Worker ret and values have different dimensions. 
        Pad returns with 0s when workers have been removed from simulation 
        Done
    """

    for key,vals in w_returns.items():
        if w_returns[key].size()[0]<  steps:
            size = steps - w_returns[key].size()[0]
            w_returns[key] = zero_pad_tensor(w_returns[key], size)
            #w_returns[key] = zero_padding()
    #Compute Intrinsic/Extrinsic Loss for each worker
    w_loss_value_ext= {}
    w_loss_value_int= {}

    for key, val in w_returns.items():
        w_loss_value_ext[key] = F.mse_loss(w_values_ext[key].to(device).unsqueeze(1), w_returns[key].detach().to(device))
        w_loss_value_int[key] = F.mse_loss(w_values_int[key].to(device).unsqueeze(1), int_returns[key].detach().to(device))
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

    total_manager_loss = manager_mean_actor_loss + manager_value_loss
    weighted_worker_loss = {}
    for key, values in worker_mean_loss.items():

        weighted_worker_loss[key] = worker_weights[key] *(w_loss_value_int[key] + w_loss_value_ext[key] + worker_mean_loss[key] )

    combined_loss = total_manager_loss + sum(weighted_worker_loss.values()) - ((sum(avg_entropy.values())/len(avg_entropy))*args.entropy_coef)
    # loss = w_loss + w_loss_value_ext + w_loss_value_int + m_loss + m_loss_value - entropy.mean()*args.entropy_coef
    # TODO: Add entropy to loss for exploration

    optimizer.zero_grad()

    combined_loss.backward(retain_graph=True)
    grad_norm = get_grad_norm(net)
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    return combined_loss, grad_norm