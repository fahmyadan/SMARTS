import os
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from FUNRL.lstm_a2c.model import FuN
from FUNRL.lstm_a2c.utils import *
from FUNRL.lstm_a2c.train import train_model
from FUNRL.lstm_a2c.env import EnvWorker
from FUNRL.lstm_a2c.memory import Memory
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

import os


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

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    writer = SummaryWriter('logs')
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'loop')
    parser = default_argument_parser("feudal-learning")
    args = parser.parse_args()
    args.scenarios = [scenario_dir]  #Relative file path To Do: Change to absolute
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

    observation_size = 3 #agent_spec.interface.waypoints.lookahead
    num_actions = agent_spec.interface.action.value
    print('observation size:', observation_size)
    print('action size:', num_actions)
    print("cuda is ", torch.cuda.is_available())
    print(device)

    net = FuN(observation_size, num_actions, args.horizon)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

    net.to(device)
    net.train()

    global_steps = 0
    score = np.zeros(args.num_envs)
    count = 0
    grad_norm = 0

    state = torch.zeros([observation_size]).to(device)

    m_hx = torch.zeros(num_actions * 16).to(device)
    m_cx = torch.zeros(num_actions * 16).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(num_actions * 16).to(device)
    w_cx = torch.zeros(num_actions * 16).to(device)
    w_lstm = (w_hx, w_cx)

    goals_horizon = torch.zeros(args.horizon + 1, num_actions * 16).to(device)

    score_history = []
    for episode in episodes(n=max_episodes):
        score = 0
        memory = Memory()
        agent = agent_spec.build_agent()
        observation = env.reset()
        state = torch.Tensor([observation.ego_vehicle_state.linear_acceleration]).to(device)
        episode.record_scenario(env.scenario_log)
        count += 1

        for i in range(args.num_step):
            net_output = net.forward(state.to(device), m_lstm, w_lstm, goals_horizon)
            policies, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state = net_output
            actions, policies, entropy = get_action(policies, num_actions)

            episode = 0
            steps = 0
            score = 0
            collisions = 1 # -> Threshold for collisions; if veh has crashed once crashed ==True
            crashed = False

            if args.render:
                env.render()

            next_state, reward, done, info = env.step({'SingleAgent': actions + 1})

            if collisions <=  len(info['SingleAgent']['env_obs'].events.collisions): #if length of list of collisions >1 crashed is true
                crashed = True
                collisions = info['SingleAgent']['env_obs'].events.collisions

            steps += 1
            score += reward

            if done and crashed:
                # print('{} episode | score: {:2f} | steps: {}'.format(
                #     episode, score, steps
                # ))
                episode += 1
                steps = 0
                score = 0
                crashed = False
                collisions = 5
                break

            if crashed:
                crashed = False
                break

            next_state = torch.Tensor(next_state).to(device)
            reward = np.asarray(reward)
            mask = np.asarray([1])

            memory.push(state, next_state,
                        actions, reward, mask, goal,
                        policies, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state, entropy)
            state = next_state

        if done:
            entropy = - policies * torch.log(policies + 1e-5)
            entropy = entropy.mean().data.cpu()
            plcy = policies.tolist()[0]
            print('global steps {} | score: {:.3f} | entropy: {:.4f} | grad norm: {:.3f} ! eps: {:.5f} | policy {}'.format(global_steps,
                                                                                              score[i], entropy,
                                                                                              grad_norm, eps, plcy))
            if i == 0:
                writer.add_scalar('log/score', score[i], global_steps)

        score_history.append(score)

        transitions = memory.sample()
        loss, grad_norm = train_model(net, optimizer, transitions, args)

        m_hx, m_cx = m_lstm
        m_lstm = (m_hx.detach(), m_cx.detach())
        w_hx, w_cx = w_lstm
        w_lstm = (w_hx.detach(), w_cx.detach())
        goals_horizon = goals_horizon.detach()
        # avg_loss.append(loss.cpu().data)

        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)

        plt.plot(range(args.episodes), score_history)
        plt.plot(range(9, args.episodes), moving_average(score_history), color='green')
        plt.title('Average agent reward per episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()

if __name__ == "__main__":
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'loop')

    args.scenarios = [scenario_dir]
    build_scenario(args.scenarios)

    main()

