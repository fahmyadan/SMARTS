import argparse
import gym
from matplotlib.pyplot import step
import numpy as np
from itertools import count
from collections import namedtuple
import pathlib 
import os
import datetime
import shutil
import atexit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter



# Import A2C model, sampling and train logic

from rl_algorithms.A2C.model import ACPolicy
from rl_algorithms.A2C.sampling import select_action
from rl_algorithms.A2C.train import finish_episode

# Import SMARTS core functions and env wrappers 
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec
from smarts.env.custom_observations import lane_ttc_observation_adapter

# Risk indices functions 
from risk_indices.risk_obs import risk_obs



parser = argparse.ArgumentParser(description='SMARTS Actor Critic Implementation')
parser.add_argument( "--num_episodes", help="The number of episodes to run the simulation for.",type=int,default=50,)
parser.add_argument("scenarios",help="A list of scenarios. ",type=str,nargs="*",)
parser.add_argument("--headless", help="Run the simulation in headless mode.", action="store_true")
parser.add_argument('--env_name', type=str, default="smarts.env:hiway-v0", help='The name of the gym environment')
parser.add_argument('--a2c_gamma', default=0.99, help='DIscount factor')
parser.add_argument('--log_interval', default=5, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_step', default=100, help='Number of steps to take in SMARTS env')
parser.add_argument('--lr', default=7e-3, help='')
parser.add_argument('--eps', default=1e-4, help='')
args = parser.parse_args()


###################################################################################################################################
#################################################### SMARTS Agent Spec/Interface ##################################################
###################################################################################################################################

class LaneAgent(Agent): 

    def act(self, obs: Observation, sampled_action:int ,**configs):

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[sampled_action]
        
        return lane_actions 


# Adapted observation space to return ttc and dtc for ego 

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)
ttc_threshold = 1000
ttc_weight = 0.9
ttc_dist_weight = 0.9

def observation_adapter(env_obs):
    ttc_obs = lane_ttc_observation_adapter.transform(env_obs)

    risk_dict = risk_obs(env_obs)
    return env_obs, ttc_obs, risk_dict

def reward_adapter(env_obs, env_reward):
    risk_dict = risk_obs(env_obs)

    total_ego_risk = sum(risk_dict.values())

    mag_jerk = np.linalg.norm(env_obs.ego_vehicle_state.linear_jerk)

    if len(env_obs.events.collisions )!= 0:
        print('collision reward activated')
        env_reward = -1 
        
    
    else: 

        env_reward = (0.5 * env_obs.ego_vehicle_state.speed) + (0.5* env_obs.distance_travelled) - (0.1* total_ego_risk) - (0.1 * mag_jerk)

    return env_reward

"""
Instantiate A2C model and optimiser 

Agent obs space = [TTC(3,), DTC(3,), Position(3,), Linear_velocity(3,), Angular_vel(3,)]
"""

agent_obs_size = 15 

a2c = ACPolicy(input_size=agent_obs_size, disc_action_size=4)
a2c_optimizer = optim.Adam(a2c.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_agents = 1 
agent_ids = [f'Worker_{i}' for i in range(1,n_agents+1)] 

# Instantiate the memory object / replay buffer
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

#Specify log directory 
log_dir = f'./runs/latest/'

###################################################################################################################################
################################################################ Main Logic ##################################################### 
###################################################################################################################################

def main():
    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scenarios/intersections/4lane')
    args.scenarios = [scenario_dir]
    args.headless = True 
    args.num_step = 1000
    args.num_episodes = 500
    args.log_interval = 5

    # a2c_agent_interface = AgentInterface(action=ActionSpaceType.Lane, max_episode_steps=args.num_step, neighborhood_vehicles=NeighborhoodVehicles(radius=25))
    a2c_agent_interface = AgentInterface.from_type(requested_type=AgentType.Laner, neighborhood_vehicles=NeighborhoodVehicles(radius=100), 
                                                   accelerometer=True, max_episode_steps = args.num_step)
    a2c_agent_spec = AgentSpec(interface=a2c_agent_interface, agent_builder=LaneAgent, reward_adapter=reward_adapter, observation_adapter=observation_adapter)

    agent_specs = {agent_id: a2c_agent_spec for agent_id in agent_ids}

    env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=args.scenarios,
    agent_specs=agent_specs,
    headless=args.headless,
    sumo_headless=False,
    sumo_port= 45761)

    # Wrapper from MultiAgent to Single Agent env 

    env = SingleAgent(env=env)
    env.seed(500)
    torch.manual_seed(500)

    a2c.train()
    a2c.to(device=device)

    # TODO: Running reward initial value is arbitrary; experiment with better initial value 

    running_reward = 10 
    running_loss = 0 
 
    
    writer = SummaryWriter(log_dir=log_dir)

    for episode in episodes(n=args.num_episodes):

        # Build the agent 

        agents = { agent_id: agent_spec.build_agent() for agent_id, agent_spec in agent_specs.items() }
        # Reset Env and episode reward 

        observations = env.reset()
        episode_reward = 0 

        agent_obs_array = np.array([observations[1]['ego_lane_dist'], observations[1]['ego_ttc'],
                                    observations[0].ego_vehicle_state.position, observations[0].ego_vehicle_state.linear_velocity,
                                    observations[0].ego_vehicle_state.angular_velocity]).reshape(1,agent_obs_size)

        
        # TODO: Set up Traci connection for manager obs


        episode.record_scenario(env.scenario_log)
        done = False
        steps = 0

        while not done: 

            agent_action = select_action(state=agent_obs_array, model=a2c, SavedAction=SavedAction, device=device)

            agent_lane_actions = agents['Worker_1'].act(obs=observations, sampled_action= agent_action)

            observations, reward, done, info = env.step(agent_lane_actions)
            # print(f'action : {agent_lane_actions}')
           
            episode.record_step(observations, episode_reward, done, info)
           
            a2c.rewards.append(reward)
            episode_reward += reward
            
            # Reassign observations for forward pass: [TTC(3,), DTC(3,), Position(3,), Linear_velocity(3,), Angular_vel(3,)]
            agent_obs_array = np.array([observations[1]['ego_lane_dist'], observations[1]['ego_ttc'],
                                        observations[0].ego_vehicle_state.position, observations[0].ego_vehicle_state.linear_velocity,
                                        observations[0].ego_vehicle_state.angular_velocity]).reshape(1,agent_obs_size)
            steps+=1

            if done: 
                print(f'episode done after {steps} steps')

        # After episode done==True, training logic.

        running_reward = (0.1 * episode_reward) + (0.9 * running_reward)

        total_ac_loss = finish_episode(model=a2c, args=args, optimizer=a2c_optimizer, device=device)
        
        running_loss+= total_ac_loss.item()
        
        # Log the loss every 5 episodes

        if episode.index   %  args.log_interval == 0: 
            print(f'Last reward {episode_reward} \n average reward {running_reward} \n running avg AC_Loss {running_loss/5}')
            writer.add_scalar('training_loss', running_loss/5, episode.index+5)
            running_loss = 0
            writer.flush()
           


 



       


    return None

#Function called when program exists
@atexit.register
def cleanup():
    shutil.rmtree(log_dir)
    




if __name__ == "__main__": 

    args = parser.parse_args()

    if not args.scenarios:
     args.scenarios = [
        str(pathlib.Path(__file__).absolute().parents[1] / "scenarios" / "loop")
    ]
    build_scenario(args.scenarios)
    main()







