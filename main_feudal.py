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

from FUNRL.lstm_a2c.model import FuN
from FUNRL.lstm_a2c.utils import *
from FUNRL.lstm_a2c.train import train_model
from FUNRL.lstm_a2c.memory import Memory

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
parser.add_argument('--m_gamma', default=0.1, help='')
parser.add_argument('--w_gamma', default=0.1, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', default=12, help='')
parser.add_argument('--num_episodes', default=20, help='')
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

"""

Moving average function that computes the cummulative sum of a given array, a, and returns the average for every 10 data
points/ arrays  

"""
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    writer = SummaryWriter('logs')
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'loop')
    # parser = default_argument_parser("feudal-learning")
    # args = parser.parse_args()
    args.scenarios = [scenario_dir]  #Relative file path To Do: Change to absolute
    args.horizon = 9
    args.save_path = './save_model/'
    args.num_envs = 1
    args.env_name = "smarts.env:hiway-v0"
    args.render = False
    args.num_step = 100
    args.headless = True
    """
    Build an agent by specifying the interface. Interface captures the observations received by an agent in the env
    and specifies the actions the agent can take to impact the env 
    """
    agent_interface = AgentInterface(debug=True, waypoints=True, action=ActionSpaceType.Lane,
                                     max_episode_steps=args.num_step)
    agent_spec = AgentSpec(
        interface=agent_interface,
        agent_builder=ChaseViaPointsAgent,
    )
    """
    Make the HiwayEnv environment. 
    Args: 
    scenario: Sequence[str]
    agent_specs: Dict[str, AgentSpec 
    headless: bool 
    """
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=args.headless,
        sumo_headless=True,
    )

    env = SingleAgent(env=env) #Wrapper for gym.env change output of step and reset
    env.seed(500)
    torch.manual_seed(500)

    observation_size = 3 #agent_spec.interface.waypoints.lookahead
    num_actions = 4
    print('observation size:', observation_size)
    print('action size:', num_actions)
    print("cuda is ", torch.cuda.is_available())
    print(device)

    #Instantiate the FuN model object (Creating the percept, manager and worker in the __init__)
    net = FuN(observation_size, num_actions, args.horizon)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

    net.to(device)
    net.train()

    global_steps = 0
    score = np.zeros(args.num_envs)
    count = 0
    grad_norm = 0
    #Initialize state -> zeros torch.Size([3])
    state = torch.zeros([observation_size]).to(device)

    #Initialize hidden and cell state of Manager and Worker
    #torch.size([1,3*16])
    #changed from : m_hx = torch.zeros(1, num_actions * 16).to(device)
    # same m_cx and worker hidden and cell states
    m_hx = torch.zeros(observation_size, num_actions * 16).to(device)
    m_cx = torch.zeros(observation_size, num_actions * 16).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(observation_size ,num_actions * 16).to(device)
    w_cx = torch.zeros(observation_size, num_actions * 16).to(device)
    w_lstm = (w_hx, w_cx)
    #Used to be goals_horizon = torch.zeros(1, args.horizon + 1, num_actions * 16).to(device)
    goals_horizon = torch.zeros(observation_size, args.horizon + 1, num_actions * 16).to(device)

    score_history = []
    loss_history = []
    for episode in episodes(n=args.num_episodes):
        score = 0
        memory = Memory()
        #Build the agent @ the start of each episode
        agent = agent_spec.build_agent()
        #Reset the env @ start of each episode and log the observations. observation contains all observations in SMARTS
        observation = env.reset()
        """
        specify the state as a subset of observations to be passed through the model; 
        in this case we ar only using linear acceleration
        
        """
        state_rep = [observation.ego_vehicle_state.linear_velocity,observation.ego_vehicle_state.position, observation.ego_vehicle_state.linear_acceleration]
        state = torch.Tensor(state_rep).to(device)
        episode.record_scenario(env.scenario_log)
        count += 1
        #if count == 2:
         #   print(f'count = {count}, break ')
        steps = 0
        """
        In each episode, we step (take actions) the environment. The agent model (FuN) receives the most recent 
        state+histories and uses the NNs to estimate the best action to take given the new state. 
         
        """
        for i in range(args.num_step):
            net_output = net.forward(state.to(device), m_lstm, w_lstm, goals_horizon)
            policies, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state = net_output

            pol = policies[0]
            actions, policies, entropy = get_action(pol, num_actions)
            """
            Actions available to ActionSpaceType.Lane are [keep_lane, slow_down, change_lane_left, change_lane_right]
            see smarts.core.controllers.__init__  
            """
            lane_actions = {0:'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3:'change_lane_right'}

            #print(f'action taken is {lane_actions[actions]}')
            episode = 0
            #steps = 0
            collisions = 1 # -> Threshold for collisions; if veh has crashed once crashed ==True
            crashed = False

            if args.render:
                env.render()
            #Step the environment by taking the actions predicted by FuN model.
            #observation, reward, done, info = env.step({'SingleAgent': actions})
            observation, reward, done, info = env.step(lane_actions[actions])
            #Record the new state after taking an action
            #next_state = observation.ego_vehicle_state.linear_acceleration
            next_state = [observation.ego_vehicle_state.linear_velocity,observation.ego_vehicle_state.position, observation.ego_vehicle_state.linear_acceleration]

            #Increment steps and sum the reward
            steps += 1
            #print(f'steps = {steps}')
            score += reward



            #Pass the new state as a tensor to GPU
            next_state = torch.Tensor([next_state]).to(device)
            reward = np.asarray([reward])
            mask = np.asarray([1])
            """
            Memory function that saves the transition from one state to the next, logging the states, actions, rewards 
            and network weights each time.  
            """
            memory.push(state, next_state,
                        actions, reward, mask, goal,
                        policies, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state, entropy)
            if done:
                break
            #End of step loop, assign the state to be passed to FuN(manager and worker) as the most recent state and repeat loop.
            state = next_state
            #print('new step')
        #If done criteria == True, calculate entropy -> H(x) = -P * log(P)
        if done:
            entropy = - policies * torch.log(policies + 1e-5)
            entropy = entropy.mean().data.cpu()
            plcy = policies.tolist()[0]
            print('global steps {} | score: {:.3f} | entropy: {:.4f} | grad norm: {:.3f} | policy {}'.format(steps,
                                                                                             score, entropy,
                                                                                              grad_norm, pol))
            if i == 0:
                writer.add_scalar('log/score', score[i], steps)

        score_history.append(score)
        """
        For each episode, compute the loss as a cosine similarity between 2 vectors (m_state & goal)
        """

        transitions = memory.sample()
        #print('training model called')
        loss, grad_norm = train_model(net, optimizer, transitions, args)
        loss_history.append(loss.item())
        m_hx, m_cx = m_lstm
        m_lstm = (m_hx.detach(), m_cx.detach())
        w_hx, w_cx = w_lstm
        w_lstm = (w_hx.detach(), w_cx.detach())
        goals_horizon = goals_horizon.detach()
        # avg_loss.append(loss.cpu().data)

        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)

    plt.plot(range(args.num_episodes), score_history)
    plt.plot(range(9, args.num_episodes), moving_average(score_history), color='green')
    plt.title('Average agent reward per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()
    plt.plot(range(args.num_episodes), loss_history)
    plt.title('Losses per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'loop')

    args.scenarios = [scenario_dir]
    build_scenario(args.scenarios)

    main()

