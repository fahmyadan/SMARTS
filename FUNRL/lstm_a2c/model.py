import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any

"""Test Commit"""
class Manager(nn.Module):
    def __init__(self, num_actions, N_Workers):
        super(Manager, self).__init__()
        self.N_workers = N_Workers
        self.num_actions = num_actions
        self.fc = nn.Linear(N_Workers*num_actions * 16, N_Workers*num_actions * 16)
        self.fc2 = nn.Linear(N_Workers*num_actions * 16, num_actions * 16)
        # todo: change lstm to dilated lstm
        self.lstm = nn.LSTMCell(num_actions * 16, hidden_size=num_actions * 16)
        # todo: add lstm initialization
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2 = nn.Linear(50, 1)

        self.fc_actor = nn.Linear(num_actions*16, num_actions *16 )

    def forward(self, inputs,N_workers, num_actions, device):
        """
        To do:
        Concat all worker observations to be processed by the manager
        Return one goal given all observations and one value

        DONE
        """

        self.hello = 'hello'
        x, (hx, cx) = inputs
        n_workers = len(x)
        full_obs = []
        for values in x.values():
            full_obs.append(values)
        full_obs = torch.stack(tuple(full_obs)).to(device)
        print(full_obs.size())
        """
        To Do: When agent's are removed from simulation N_workers <4. self.fc() expects input of 256
                If N_workers <4, pad the full obs with N (1,64) zeros along last dimension 
        """

        if full_obs.size()[0] < 4:

            pad = 4 - full_obs.size()[0]
            padding = (0,0,0,0,0,pad)
            pads = torch.nn.ZeroPad2d(padding)
            full_obs = pads(full_obs)

        full_obs = self.fc(full_obs.reshape(1,self.num_actions*self.N_workers *16).to(device))
        full_obs = F.relu(self.fc(full_obs))
        full_obs = self.fc2(full_obs)
        # for key,values in x.items():
        #     x[key] = self.fc(values)
        #     x[key] = F.relu(self.fc(x[key]))
        state = full_obs
        hx, cx = self.lstm(state, (hx,cx))
        goal = self.fc_actor(cx)
        goal = F.softmax(goal)
        m_value = F.relu(self.fc_critic1(goal))
        m_value = self.fc_critic2(m_value)
        """
        Use Frobenius norm to deal with noisy goals 
        """
        goal_norm = torch.norm(goal, p='fro' , dim=1).unsqueeze(1)
        goal = goal/goal_norm

        return goal, (hx, cx), m_value, state


class Worker(nn.Module):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        super(Worker, self).__init__()

        self.lstm = nn.LSTMCell(num_actions * 16, hidden_size=num_actions * 16)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # linear projection of goal has no bias
        self.fc = nn.Linear(num_actions * 16, 16, bias=False)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic1_out = nn.Linear(50, 1)

        self.fc_critic2 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2_out = nn.Linear(50, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        x, w_lstm, goals = inputs
        # x = torch.squeeze(x)
        #x = torch.squeeze(x, 1)
        #print('worker input size ', x.size())
        """
        To Do: Refactor to allow w_lstm to be flexible when 'Worker_N' does not exist
        """
        n_workers = len(x)



        for keys, values in x.items():
            w_lstm[keys]= self.lstm(x[keys], (w_lstm[keys][0], w_lstm[keys][1]) )

        if n_workers < 4:
            print('worker object lengths',len(x))
            print('worker perceptz z keys',x.keys())

        value_ext = {}
        value_int = {}
        for keys, values in w_lstm.items():
            value_ext[keys]= F.relu(self.fc_critic1(w_lstm[keys][0]))
            value_int[keys]= F.relu(self.fc_critic2(w_lstm[keys][0]))
            value_ext[keys]= self.fc_critic1_out(value_ext[keys])
            value_int[keys]= self.fc_critic2_out(value_int[keys])

        # value_ext = F.relu(self.fc_critic1(hx))
        # value_ext = self.fc_critic1_out(value_ext)
        # value_int = F.relu(self.fc_critic2(hx))
        # value_int = self.fc_critic2_out(value_int)
        worker_embed = {}
        for keys in w_lstm.keys():
            worker_embed[keys]= w_lstm[keys][0].view(w_lstm[keys][0].size(0),
                                                     self.num_actions,
                                              16)
        goal_embed = goals.sum(dim=1)
        goal_embed = self.fc(goal_embed).unsqueeze(-1)

        policy ={}
        for key in worker_embed.keys():
            policy[key]= torch.bmm(worker_embed[key], goal_embed)
            policy[key]= policy[key].squeeze(-1)
            policy[key]= F.softmax(policy[key], dim=-1)

        return policy, w_lstm, value_ext, value_int


class Percept(nn.Module):
    def __init__(self,observation_size, num_actions):
        super(Percept, self).__init__()
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=16,
        #     kernel_size=8,
        #     stride=4)
        # self.conv2 = nn.Conv2d(
        #     in_channels=16,
        #     out_channels=32,
        #     kernel_size=4,
        #     stride=2)
        self.fc = nn.Linear(observation_size, num_actions * 16)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = x.view(x.size(0), -1)
        out = {}
        for key, value in x.items():
            out[key] = F.relu(self.fc(value))
        return out


class FuN(nn.Module):
    def __init__(self, observation_size, num_actions, horizon, N_Workers):
        super(FuN, self).__init__()
        self.percept = Percept(observation_size, num_actions)
        self.manager = Manager(num_actions, N_Workers)
        self.worker = Worker(num_actions)
        self.horizon = horizon

    def forward(self, w_states: Dict[str, List], m_states , m_lstm, w_lstm, goals_horizon,N_workers, num_actions, device):
        percept_z = self.percept(w_states)
        m_inputs = (percept_z, m_lstm)
        goal, m_lstm, m_value, m_state = self.manager(m_inputs, N_workers, num_actions, device)

        # todo: at the start, there is no previous goals. Need to be checked

        goals_horizon = torch.cat([goals_horizon[:, 1:], goal.unsqueeze(1)], dim=1)

        w_inputs = (percept_z, w_lstm, goals_horizon)
        policy, w_lstm, w_value_ext, w_value_int = self.worker(w_inputs)
        return policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state


