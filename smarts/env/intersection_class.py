import logging
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import time
import gym
import numpy as np 
from gym.spaces import flatdim
from pathlib import Path 

from envision.client import Client as Envision
from smarts.core import agent, seed as smarts_seed
from smarts.core import smarts
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.utils.logging import timeit
from smarts.core.utils.visdom_client import VisdomClient
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles, DoneCriteria
from smarts.core.agent import Agent
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


from smarts.env.custom_observations import lane_ttc_observation_adapter

from rl.risk_indices.risk_obs import risk_obs


class LaneAgent(Agent): 

    def act(self, obs: Observation, sampled_action:int ,**configs):

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[sampled_action]
        
        return lane_actions 


class IntersectionEnv(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS."""

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_specs: Dict[str, AgentSpec],
        sim_name: Optional[str] = None,
        shuffle_scenarios: bool = True,
        headless: bool = True,
        visdom: bool = False,
        fixed_timestep_sec: Optional[float] = None,
        seed: int = 42,
        num_external_sumo_clients: int = 0,
        sumo_headless: bool = True,
        sumo_port: Optional[str] = None,
        sumo_auto_start: bool = True,
        endless_traffic: bool = True,
        envision_endpoint: Optional[str] = None,
        envision_record_data_replay_path: Optional[str] = None,
        zoo_addrs: Optional[str] = None,
        timestep_sec: Optional[
            float
        ] = None,  # for backwards compatibility (deprecated)
        **kwargs
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self.seed(seed)
        self.env_info = kwargs
        self.n_agents = self.env_info['n_agents']
        self.sumo_headless = sumo_headless
        self.sumo_port = sumo_port
        self.num_external_sumo_clients = num_external_sumo_clients
        self.headless = headless 
        self.envision_record_data_replay_path = envision_record_data_replay_path
        self.envision_endpoint = envision_endpoint
        self.sim_name = sim_name
        self.fixed_timestep_sec = fixed_timestep_sec


        if timestep_sec and not fixed_timestep_sec:
            warnings.warn(
                "timestep_sec has been deprecated in favor of fixed_timestep_sec.  Please update your code.",
                category=DeprecationWarning,
            )
        if not fixed_timestep_sec:
            fixed_timestep_sec = timestep_sec or 0.1
        self.action_space = gym.spaces.Discrete(4)
    
        scenarios = [str(Path(__file__).absolute().parents[2]/"scenarios"/scenarios)]        
                

        self.scenarios = scenarios


        self._agent_specs = agent_specs 

        
        self._dones_registered = 0

        self.agent_interfaces = {
            agent_id: agent.interface for agent_id, agent in agent_specs.items()
        }

        
        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(self._agent_specs.keys()),
            shuffle_scenarios,
        )

        self._smarts = None #To be created by env.setup




    @property
    def agent_specs(self) -> Dict[str, AgentSpec]:
        """Agents' specifications used in this simulation.

        Returns:
            (Dict[str, AgentSpec]): Agents' specifications.
        """
        return self._agent_specs

    @property
    def scenario_log(self) -> Dict[str, Union[float, str]]:
        """Simulation steps log.

        Returns:
            Dict[str, Union[float,str]]: A dictionary with the following keys.
                fixed_timestep_sec - Simulation timestep.
                scenario_map - Name of the current scenario.
                scenario_routes - Routes in the map.
                mission_hash - Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_routes": scenario.route or "",
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    def seed(self, seed: int) -> int:
        """Sets random number generator seed number.

        Args:
            seed (int): Seed number.

        Returns:
            int: Seed number.
        """
        smarts_seed(seed)
        return seed

    def step(
        self, agent_actions
    ) -> Tuple[
        Dict[str, Observation], Tuple[float], Dict[str, bool], Dict[str, Any]
    ]:
        """Steps the environment.

        Args:
            agent_actions (Dict[str, Any]): Action taken for each agent.

        Returns:
            Tuple[ Dict[str, Observation], Dict[str, float], Dict[str, bool], Dict[str, Any] ]:
                Observations, rewards, dones, and infos for active agents.
        """

        epymarl_actions = self.epymarl_actions(agent_actions)
        # agent_actions = {
        #     agent_id: self._agent_specs[agent_id].action_adapter(action)
        #     for agent_id, action in agent_actions.items()
        # }        # done_criteria = DoneCriteria(
        # collision=True,
        # off_road=True,
        # off_route=True,
        # on_shoulder=True,
        # wrong_way=True,
        # not_moving=False,
        # agents_alive=None,)
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in epymarl_actions.items()
        }

        assert isinstance(agent_actions, dict) and all(
            isinstance(key, str) for key in agent_actions.keys()
        ), "Expected Dict[str, any]"

        observations, rewards, dones, extras = None, None, None, None
        with timeit("SMARTS Simulation/Scenario Step", self._log):
            observations, rewards, dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {"score": value, "env_obs": observations[agent_id]}
            for agent_id, value in extras["scores"].items()
        }

        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_specs)

        rewards = self.epymarl_rewards(rewards)
        infos = self._info(infos)

        return observations, rewards, dones["__all__"], infos

    def reset(self) -> Dict[str, Observation]:
        """Reset the environment and initialize to the next scenario.

        Returns:
            Dict[str, Observation]: Agents' observation.
        """
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0

        if self._smarts is None: 
            self._smarts = self.build_smarts()
            self._smarts.setup(scenario)

        env_observations = self._smarts.reset(scenario)

        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }

        return observations

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        """Closes the environment and releases all resources."""
        if self._smarts is not None:
            self._smarts.destroy()
            self._smarts = None

    def get_env_info(self):

        env_info ={}
        env_info['n_agents'] = self.n_agents
        env_info['n_actions'] = self.env_info['n_actions']
        env_info['obs_shape'] = self.env_info['obs_shape']
        env_info['episode_limit'] = self.env_info['episode_limit']
        env_info['state_shape'] = 64  #env_info['obs_shape'] * env_info['n_agents'] 
        env_info['name'] = 'smarts'

        return env_info

    def get_state(self):
        """
        Gets global state 
        # """
        # obsers = {
        #     agent_id: self._agent_specs[agent_id].observation_adapter(obs)
        #     for agent_id, obs in env_observations.items()
        # }

        a_ids = [ids for ids in self._agent_specs.keys()]

        smarts_observations, _, _, _  =  self._smarts.agent_manager.observe(self._smarts)
        
        # Mask observation if agent is not present in full list (i.e. has crashed)
        """
        Be aware that masking/padding the obs space will add noise to the model. 
        By adding zeros, values like risk, ttc etc will get distorted. Lets try it for now
        """
        observations = {agent_id: self._agent_specs[agent_id].observation_adapter(obs)
                         for agent_id, obs  in smarts_observations.items()}

        for ids in smarts_observations.keys():
            
            if len(smarts_observations[ids].via_data.hit_via_points) >0:

                print(f'success. {ids} hit a via point in last step')

        if len(observations.keys()) != len(a_ids):
                observations = {agent_id: self._agent_specs[agent_id].observation_adapter(smarts_observations[agent_id]) 
                                if agent_id in observations.keys() 
                                else np.zeros([1,16]) 
                                for agent_id in a_ids}
                print('check mask')
        self.agent_obs = smarts_observations
        self.state_list = []

        for vals in observations.values():
            self.state_list.append(vals)
        
        self.state = np.hstack(self.state_list)

        # # obvs = []

        # # for ids, obs in observations.items():
        #     # self.state[ids] = np.array([obs.ego_vehicle_state.position, obs.ego_vehicle_state.linear_velocity, obs.ego_vehicle_state.angular_velocity]).reshape(1,9)
        #     # obvs.append(self.state[ids])

        # dummy = np.random.randn(1,36)

        return self.state 

    def get_avail_actions(self):
        agent_ids = [agentid for agentid in self._agent_specs.keys()]
        avail_actions = []
  
        for agent_id in agent_ids:

            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """


        avail_actions = self.agent_interfaces[agent_id].action

        if avail_actions is ActionSpaceType.Lane: 

            actions = np.arange(0,4)
            return actions
        else:
            return avail_actions

    def get_obs(self):
        """
        Get partial observation 
        """

        # agent_ids = [agentid for agentid in self._agent_specs.keys()]
        # observation = []

        # for ids in agent_ids:

        #     observation.append(self.get_obs_agent(ids))

        # print(f'check obs {observation}')
        return np.vstack(self.state_list)



    def get_obs_agent(self, agent_id):
        
        state = self.state

        print('check id obs ', agent_id, 'state observed ', state.get(agent_id))

        if state.get(agent_id) is None: 

            return None 
        else: 

            return state[agent_id]
    
    def build_smarts(self):
        envision_client = None
        if not self.headless or self.envision_record_data_replay_path:
            envision_client = Envision(
                endpoint=self.envision_endpoint,
                sim_name=self.sim_name,
                output_dir=self.envision_record_data_replay_path,
                headless=self.headless,
            )
            

        traffic_sim = SumoTrafficSimulation(
        headless=self.sumo_headless,
        num_external_sumo_clients=self.num_external_sumo_clients,
        sumo_port=self.sumo_port )

        self._smarts = SMARTS(
            agent_interfaces=self.agent_interfaces,
            traffic_sim=traffic_sim,
            envision=envision_client,
            fixed_timestep_sec=self.fixed_timestep_sec,

        )

        agents = { agent_id: agent_spec.build_agent() for agent_id, agent_spec in self._agent_specs.items() }

        return self._smarts
    
    def epymarl_actions(self, actions):
        agent_ids = [agentid for agentid in self._agent_specs.keys()]

        action_dict = {}

        for i in range(len(actions)):

            action_dict[agent_ids[i]] = actions[i].cpu().numpy()

        return action_dict

    def epymarl_rewards(self, rewards:Dict[str,float]) -> float:

        """
        Reward from the env should be returned as single value 
        """

        return sum(tuple(rewards.values())) 

    def _info(self,info):
        _info = {'collision': False , 'reached_goal': True}

        for agents in info.values():
            _info['collision'] |= agents['collision']
            _info['reached_goal'] &= agents['reached_goal']

        return _info

agent_obs_size = 16
def observation_adapter(env_obs):

    ttc_obs = lane_ttc_observation_adapter.transform(env_obs)

    risk_dict = risk_obs(env_obs)

    observations = env_obs, ttc_obs, risk_dict
    total_risk = np.array(sum(observations[-1].values()))
    agent_obs_array = np.array([observations[1]['ego_lane_dist'], observations[1]['ego_ttc'],
                                observations[0].ego_vehicle_state.position, observations[0].ego_vehicle_state.linear_velocity,
                                observations[0].ego_vehicle_state.angular_velocity],
                                    )
    # neighbor_states = observations[0].neighborhood_vehicle_states
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
        env_reward = -10
    
    elif env_obs.ego_vehicle_state.speed < 2:
        # To discourage the vehicle from stopping 
 
        env_reward = -1
        
    else: 

        norm_speed = env_obs.ego_vehicle_state.speed / max_speed
        norm_distance = env_obs.distance_travelled / max_distance
        norm_jerk = mag_jerk / max_jerk


        env_reward = (0.2 * norm_speed) + (0.5 * norm_distance) - (0.5* total_ego_risk) - (0.2 * norm_jerk)

    return env_reward 

def action_adapter(act:int) -> str:

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[int(act)]
        
        return lane_actions 

def info_adapter(obs, reward, info): 

    info = {}


    info['collision'] = len(obs.events.collisions) > 0
    info['reached_goal'] = obs.events.reached_goal
    
    return info 

