from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.env.custom_observations import lane_ttc_observation_adapter
import gym
import numpy as np 
import sys 

from .risk_indices.risk_obs import risk_obs

a2c_agent_interface = AgentInterface.from_type(requested_type=AgentType.Laner, neighborhood_vehicles=NeighborhoodVehicles(radius=100), 
                                                   accelerometer=True, max_episode_steps = 100)

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
    max_speed: float = 20.0 
    max_distance: float = 5
    max_acc: float = 5
    max_jerk: float = max_acc / 0.1
    risk_dict = risk_obs(env_obs)

    total_ego_risk = sum(risk_dict.values())

    mag_jerk = np.linalg.norm(env_obs.ego_vehicle_state.linear_jerk)

    if len(env_obs.events.collisions )!= 0:
        print('collision reward activated')
        env_reward = -1 
    
    elif env_obs.ego_vehicle_state.speed < 2:
        # To discourage the vehicle from stopping 
 
        env_reward = -0.1
        
    else: 

        norm_speed = env_obs.ego_vehicle_state.speed / max_speed
        norm_distance = env_obs.distance_travelled / max_distance
        norm_jerk = mag_jerk / max_jerk


        env_reward = (0.5 * norm_speed) + (0.5* norm_distance) - (0.2* total_ego_risk) - (0.1 * norm_jerk)

    return env_reward


