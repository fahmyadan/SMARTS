from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, Waypoints, DoneCriteria, AgentType
from smarts.env.hiway_env import HiWayEnv
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.sstudio import build_scenario
from smarts.core.agent import Agent
from smarts.core.sensors import Observation
import numpy as np 

from smarts.env.custom_observations import lane_ttc_observation_adapter

from rl.risk_indices.risk_obs import risk_obs


class LaneAgent(Agent): 

    def act(self, obs: Observation, sampled_action:int ,**configs):

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[sampled_action]
        
        return lane_actions 



def intersection_v0_env(
    headless: bool = True,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None):

    
    n_agents = 4
    AGENT_IDS = ["Agent %i" % i for i in range(n_agents)]
    scenarios = [str(Path(__file__).absolute().parents[2]           
                / "scenarios"
                 / "intersections"
                 / "4lane")]

    build_scenario(scenario=scenarios)

    done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=True,
        on_shoulder=True,
        wrong_way=True,
        not_moving=False,
        agents_alive=None,)
    agent_interfaces = AgentInterface.from_type(requested_type=AgentType.Laner, neighborhood_vehicles=NeighborhoodVehicles(radius=100), 
                                                   accelerometer=True, max_episode_steps = 1000, done_criteria=done_criteria)
    agent_spec = AgentSpec(interface=agent_interfaces, agent_builder=LaneAgent, observation_adapter=observation_adapter, 
                            reward_adapter=reward_adapter, action_adapter=action_adapter, info_adapter=info_adapter)
    agent_specs = {agent_id: agent_spec for agent_id in AGENT_IDS}

    env = HiWayEnv(scenarios=scenarios, agent_specs=agent_specs, headless=headless, visdom=visdom, 
                   sumo_headless=sumo_headless, envision_record_data_replay_path=envision_record_data_replay_path)
                   
    agents = { agent_id: agent_spec.build_agent() for agent_id, agent_spec in agent_specs.items() }

    env.episode_limit = 1000


    return env


    
agent_obs_size = 16
def observation_adapter(env_obs):

    ttc_obs = lane_ttc_observation_adapter.transform(env_obs)

    risk_dict = risk_obs(env_obs)

    observations = env_obs, ttc_obs, risk_dict
    total_risk = np.array(sum(observations[-1].values()))
    agent_obs_array = np.array([observations[1]['ego_lane_dist'], observations[1]['ego_ttc'],
                                    observations[0].ego_vehicle_state.position, observations[0].ego_vehicle_state.linear_velocity,
                                    observations[0].ego_vehicle_state.angular_velocity])

    agent_obs_array = np.append(agent_obs_array, total_risk).reshape(1,agent_obs_size)

    return agent_obs_array

def reward_adapter(env_obs, env_reward):

    max_speed: float = 20.0 
    max_distance: float = 6
    total_distance: float = 180
    max_acc: float = 5
    max_jerk: float = max_acc / 0.1
    risk_dict = risk_obs(env_obs)

    total_ego_risk = sum(risk_dict.values())

    mag_jerk = np.linalg.norm(env_obs.ego_vehicle_state.linear_jerk)

    if len(env_obs.events.collisions )!= 0:
        print('collision reward activated')
        env_reward = -5
    
    elif env_obs.ego_vehicle_state.speed < 2:
        # To discourage the vehicle from stopping 
 
        env_reward = -1
        
    else: 

        norm_speed = env_obs.ego_vehicle_state.speed / max_speed
        norm_distance = env_obs.distance_travelled / max_distance
        norm_jerk = mag_jerk / max_jerk


        env_reward = (0.5 * norm_speed) + (0.7 * norm_distance) - (0.1* total_ego_risk) - (0.1 * norm_jerk)

    return env_reward 

def action_adapter(act:int) -> str:

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[int(act)]
        
        return lane_actions 

def info_adapter(obs, reward, info): 
    print(f'info adapter check')

    info = {}


    info['collision'] = len(obs.events.collisions) > 0
    info['reached_goal'] = obs.events.reached_goal
    
    return info 
    









