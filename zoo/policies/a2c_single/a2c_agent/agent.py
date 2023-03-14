import torch 

from smarts.core.agent import Agent
from smarts.core.sensors import Observation

from .A2C.model import ACPolicy
from .A2C.sampling import action_inference

agent_obs_size = 15
model_path = '/home/fahmy/PhD/Experiments/SMARTS/model_paras/checkpoint/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def load_parameters(): 

    a2c_model = ACPolicy(input_size=agent_obs_size, disc_action_size=4)
    a2c_model.load_state_dict(torch.load(model_path))
    a2c_model.eval()
    a2c_model.to(device=device)
    return a2c_model


class LaneAgent(Agent): 

    def __init__(self):

        self.trained_model = load_parameters()
        
    
        return None 



    def act(self, obs: Observation ,**configs):

        sampled_action = action_inference(obs, self.trained_model, agent_obs_size)

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[sampled_action]

        
        return lane_actions 



