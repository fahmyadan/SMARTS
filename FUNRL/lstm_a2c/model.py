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
        #print('passing through worker')
        x, (hx, cx), goals = inputs
        x = torch.squeeze(x)
        #x = torch.squeeze(x, 1)
        #print('worker input size ', x.size())
        hx, cx = self.lstm(x.view(1,64), (hx, cx))

        value_ext = F.relu(self.fc_critic1(hx))
        value_ext = self.fc_critic1_out(value_ext)

        value_int = F.relu(self.fc_critic2(hx))
        value_int = self.fc_critic2_out(value_int)

        worker_embed = hx.view(hx.size(0),
                               self.num_actions,
                               16)

        goals = goals.sum(dim=1)
        goal_embed = self.fc(goals.detach())
        goal_embed = goal_embed.unsqueeze(-1)

        policy = torch.bmm(worker_embed, goal_embed)
        policy = policy.squeeze(-1)
        policy = F.softmax(policy, dim=-1)
        #print('completed worker')
        return policy, (hx, cx), value_ext, value_int


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
        self.worker1 = Worker(num_actions)
        self.worker2 = Worker(num_actions)
        self.worker3 = Worker(num_actions)
        self.worker4 = Worker(num_actions)
        self.horizon = horizon

    def forward(self, w_states: Dict[str, List], m_states , m_lstm, w_lstm, goals_horizon):
        percept_z = self.percept(w_states)
        m_inputs = (percept_z, m_lstm)
        goal, m_lstm, m_value, m_state = self.manager(m_inputs)

        # todo: at the start, there is no previous goals. Need to be checked
        goals_horizon = torch.cat([goals_horizon[:, 1:], goal.unsqueeze(1)], dim=1)

        w_inputs = (percept_z, w_lstm, goals_horizon)
        policy, w_lstm, w_value_ext, w_value_int = self.worker(w_inputs)
        return policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state


