import os
import gym
import argparse
import numpy as np

import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from model import FuN
from utils import *
from train import train_model
from env import EnvWorker
from memory import Memory
import pathlib

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.core.utils.episodes import episodes

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="smarts.env:hiway-v0", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--m_gamma', default=0.999, help='')
parser.add_argument('--w_gamma', default=0.99, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', default=12, help='')
parser.add_argument('--num_step', default=400, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.01, help='')
parser.add_argument('--lr', default=7e-4, help='')
parser.add_argument('--eps', default=1e-5, help='')
parser.add_argument('--horizon', default=9, help='')
parser.add_argument('--clip_grad_norm', default=5, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
parser.add_argument('--scenarios', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )

def get_action(policies, num_actions):
    m = Categorical(policies)
    actions = m.sample()
    actions = actions.data.cpu().numpy()
    return actions

def main():
    parser = default_argument_parser("feudal-learning")
    args = parser.parse_args()
    args.scenarios = ['/home/tslab/Desktop/SMARTS/scenarios/loop']  #Relative file path To Do: Change to absolute
    args.horizon = 9
    args.save_path = './save_model/'
    args.num_envs = 1
    args.env_name = "smarts.env:hiway-v0"
    args.render = False
    args.num_step = 40
    args.headless = True

    max_episodes =10

    agent_interface = AgentInterface(debug=True, waypoints=True, action=ActionSpaceType.LaneWithContinuousSpeed,
                                     max_episode_steps=max_episodes)
    agent_spec = AgentSpec(
        interface=agent_interface,
        agent_builder=ChaseViaPointsAgent,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=args.headless,
        sumo_headless=True,
    )

    env = SingleAgent(env=env)
    env.seed(500)
    torch.manual_seed(500)

    observation_size = agent_spec.interface.waypoints.lookahead
    num_actions = agent_spec.interface.action.value
    print('observation size:', observation_size)
    print('action size:', num_actions)
    print("cuda is ", torch.cuda.is_available())
    print(device)

    net = FuN(observation_size,num_actions, args.horizon)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

    net.to(device)
    net.train()

    global_steps = 0
    score = np.zeros(args.num_envs)
    count = 0
    grad_norm = 0

    histories = torch.zeros([args.num_envs, observation_size]).to(device)

    m_hx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    w_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    w_lstm = (w_hx, w_cx)

    goals_horizon = torch.zeros(args.num_envs, args.horizon + 1, num_actions * 16).to(device)

    for episode in episodes(n=max_episodes):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)
        count += 1

        for i in range(args.num_step):
            net_output = net.forward(histories.to(device), m_lstm, w_lstm, goals_horizon)
            policies, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state = net_output
            actions = get_action(policies, num_actions)





if __name__ == "__main__":
    args.scenarios = ['/home/tslab/Desktop/SMARTS/scenarios/loop']
    build_scenario(args.scenarios)

    main()

