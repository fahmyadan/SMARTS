import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any


class Manager(nn.Module):
    def __init__(self, num_actions):
        super(Manager, self).__init__()
        self.fc = nn.Linear(num_actions * 16, num_actions * 16)
        # todo: change lstm to dilated lstm
        self.lstm = nn.LSTMCell(num_actions * 16, hidden_size=num_actions * 16)
        # todo: add lstm initialization
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2 = nn.Linear(50, 1)

        self.fc_actor = nn.Linear(50, num_actions)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        N_workers = len(x)
        for key,values in x.items():
            x[key] = self.fc(values)
            x[key] = F.relu(self.fc(x[key]))
        state = x
        lstm_vals = {}
        goal ={}
        goal_norm ={}

        value_fun = {}
        for key, values in state.items():
            lstm_vals[key] = self.lstm(state[key], (hx,cx))
            goal[key] = lstm_vals[key][1]
            value_fun[key] =self.fc_critic2(F.relu(self.fc_critic1(goal[key])))
            goal_norm[key] = torch.norm(goal[key], p=2, dim=1).unsqueeze(1)
            goal[key] = goal[key]/goal_norm[key].detach()


        return goal, (hx, cx), value_fun, state


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
        for keys, values in w_lstm.items():
            w_lstm[keys]= self.lstm(x[keys], (w_lstm[keys][0], w_lstm[keys][1]) )

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
            goal_embed = {}
        for keys in goals.keys():
            goals[keys]= goals[keys].sum(dim=1)
            goal_embed[keys]= self.fc(goals[keys].detach())
            goal_embed[keys]=goal_embed[keys].unsqueeze(-1)


        # goals = goals.values().sum(dim=1)
        # goal_embed = self.fc(goals.detach())
        # goal_embed = goal_embed.unsqueeze(-1)
        policy ={}
        for key in worker_embed.keys():
            policy[key]= torch.bmm(worker_embed[key], goal_embed[key])
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
    def __init__(self, observation_size, num_actions, horizon):
        super(FuN, self).__init__()
        self.percept = Percept(observation_size, num_actions)
        self.manager = Manager(num_actions)
        self.worker = Worker(num_actions)
        self.horizon = horizon

    def forward(self, w_states: Dict[str, List], m_states , m_lstm, w_lstm, goals_horizon):
        percept_z = self.percept(w_states)
        m_inputs = (percept_z, m_lstm)
        goal, m_lstm, m_value, m_state = self.manager(m_inputs)

        # todo: at the start, there is no previous goals. Need to be checked
        w_goals_horizon ={}
        for worker, goals in goal.items():
            w_goals_horizon[worker] = torch.cat([goals_horizon[:,1:], goals.unsqueeze(1)], dim=1)

        # goals_horizon = torch.cat([goals_horizon[:, 1:], goal.unsqueeze(1)], dim=1)

        w_inputs = (percept_z, w_lstm, w_goals_horizon)
        policy, w_lstm, w_value_ext, w_value_int = self.worker(w_inputs)
        return policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state


