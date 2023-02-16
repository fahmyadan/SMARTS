import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any, Sequence


class ManagerAC(nn.Module):

    def __init__(self, manager_in_size:int, actor_out_size:int, hidden_size:int) -> None:
        super(ManagerAC, self).__init__() 

        self.input_size = manager_in_size
        self.actor_out_size = actor_out_size
        self.hidden_size = hidden_size

        self.critic = nn.Sequential(
            nn.Linear(self.input_size, out_features=hidden_size, bias=True), #FC1
            nn.ReLU(), 
            # TODO: Add Recurrent Layer
            nn.Linear(hidden_size, 1), #FC2
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(self.input_size, out_features=hidden_size*4, bias= True), #FC1
            nn.ReLU(),
            # TODO: Add Recurrent Layer
            nn.Linear(hidden_size*4, out_features=actor_out_size, bias=True), #FC2
            nn.Softmax()   
        )
        

    def forward(self, manager_observations: torch.Tensor) -> Sequence[Tuple[torch.Tensor,torch.Tensor]]: 

        manager_state_value = self.critic(manager_observations)

        manager_latent_state = self.actor(manager_observations)

        return manager_latent_state, manager_state_value


class WorkerAC(nn.Module):

    def __init__(self, worker_action_size:int, worker_raw_obs_size:int, manager_action_size:int) -> None:
        super(WorkerAC, self).__init__()

        self.worker_combined_obs_size = worker_raw_obs_size+manager_action_size
        self.worker_action_size = worker_action_size

        self.critic = nn.Sequential(
            nn.Linear(self.worker_combined_obs_size, self.worker_combined_obs_size*4, bias=True), #FC1
            nn.ReLU(),
            #TODO: ADD recurrent layer
            nn.Linear(self.worker_combined_obs_size*4, 1, bias=True), #FC2
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(self.worker_combined_obs_size, self.worker_combined_obs_size*4, bias=True), #FC1
            nn.ReLU(), 
            nn.Linear(self.worker_combined_obs_size*4, self.worker_combined_obs_size*2, bias =True), #FC2
            nn.ReLU(), 
            #TODO: ADD Recurrent Layer
            nn.Linear(self.worker_combined_obs_size*2, worker_action_size),
            nn.Softmax()
        )
    
    def forward(self, combined_obs:torch.Tensor) -> Sequence[Tuple[torch.Tensor,torch.Tensor]]:

        worker_state_value = self.critic(combined_obs)

        worker_action_prob = self.actor(combined_obs)

        return worker_action_prob, worker_state_value


        







