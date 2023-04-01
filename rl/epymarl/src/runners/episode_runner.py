from typing import Dict, Any
from unittest.mock import ANY
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

from smarts.env.intersection_env import intersection_v0_env
from smarts.env.intersection_class import observation_adapter, reward_adapter, action_adapter, info_adapter, LaneAgent

from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, AgentType
from smarts.zoo.agent_spec import AgentSpec



class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        # self.env = intersection_v0_env()

        self.agent_specs = self.agent_env_interface(self.args.agent_interface, self.args.env_info)

        self.env = env_REGISTRY[self.args.env](agent_specs = self.agent_specs, **self.args.env_args)
        self.episode_limit = self.args.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def agent_env_interface(self, env_args: Dict[str,Any], env_info) -> Dict[str,AgentSpec]:

        args_interface = env_args
        agent_ids = [f'Agent {i}' for i in range(1, env_info['n_agents']+1)]


        if args_interface['agent_type'] == 'Laner':

            req_type = AgentType.Laner
        else: 
            raise ValueError('Unknown Agent type requested')

        _agent_interface = AgentInterface.from_type(requested_type=req_type,
                                                    neighborhood_vehicles=NeighborhoodVehicles(radius=args_interface['neighbourhood_vehicle_radius'])), 

        _agent_specs = AgentSpec(interface= _agent_interface, agent_builder=LaneAgent, reward_adapter=reward_adapter,
                                observation_adapter=observation_adapter, action_adapter=action_adapter, info_adapter=info_adapter)

        
        agent_specs = {agent_id: _agent_specs for agent_id in agent_ids}
        return agent_specs
    
    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            print(f'top of while loop')

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            
            self.batch.update(pre_transition_data, ts=self.t)
   

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            print(f'check actions {actions}')
            _, reward, terminated, env_info = self.env.step(actions[0])
            print(f'check reward : {reward} \n INFO {env_info} \n terminated {terminated}')
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            print(f'out of while loop')

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
