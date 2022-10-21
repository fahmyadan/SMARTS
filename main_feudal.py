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
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.core.utils.episodes import episodes
from smarts.env.custom_observations import lane_ttc_observation_adapter

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
parser.add_argument('--num_episodes', default=500, help='')
parser.add_argument('--num_step', default=500, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.1, help='')
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

class WorkerAgent(Agent):
    def act(self, actions):
        lane_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}
        w_actions ={}
        for key, value in actions.items():
            w_actions[key] = lane_actions[value]
        # print(w_actions)
        return w_actions





N_Workers = 4
Worker_IDS = [f'Worker_{i}' for i in range(1,N_Workers+1)]


"""

Moving average function that computes the cummulative sum of a given array, a, and returns the average for every 10 data
points/ arrays  

"""
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
"""

Adapters for transforming raw sensor observations to a more useful form. 

"""

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

    return env_obs, ttc_obs


def reward_adapter(env_obs, env_reward):
    adapt_obs = observation_adapter(env_obs)
    obs_ttc = adapt_obs['ego_ttc']
    obs_ttc_dist = adapt_obs['ego_lane_dist']
    for ttc in obs_ttc:
        if ttc > ttc_threshold:
            env_reward = -1
            return env_reward

    ttc_norm = obs_ttc.mean()/max(obs_ttc)
    ttc_dist_norm = obs_ttc_dist.mean()/max(obs_ttc_dist)
    env_reward = (ttc_weight *ttc_norm) + (ttc_dist_weight*ttc_dist_norm) + env_obs.distance_travelled

    return env_reward

def main():
    writer = SummaryWriter('logs')
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'intersections/4lane')
    # parser = default_argument_parser("feudal-learning")
    # args = parser.parse_args()
    args.scenarios = [scenario_dir]  #Relative file path To Do: Change to absolute
    args.horizon = 9
    args.save_path = './save_model/'
    args.num_envs = 1
    args.env_name = "smarts.env:hiway-v0"
    args.render = False
    args.num_step = 500
    args.headless = True
    """
    Build an agent by specifying the interface. Interface captures the observations received by an agent in the env
    and specifies the actions the agent can take to impact the env 
    """
    worker_interface = AgentInterface(debug=True, waypoints=True, action=ActionSpaceType.Lane,
                                     max_episode_steps=args.num_step, neighborhood_vehicles=NeighborhoodVehicles(radius=25))
    worker_spec = AgentSpec(
        interface=worker_interface,
        agent_builder=WorkerAgent,
        #reward_adapter=reward_adapter,
        observation_adapter=observation_adapter
    )
    agent_specs = { worker_id: worker_spec for worker_id in Worker_IDS}

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
        agent_specs=agent_specs,
        headless=args.headless,
        sumo_headless=True,
    )
    env.seed(500)
    torch.manual_seed(500)

    observation_size = 13 + (5*3)  #13 + xyz position of N=5 neighbor vehicles
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
    state = torch.zeros([observation_size,observation_size]).to(device)

    #Initialize hidden and cell state of Manager and Worker
    #torch.size([1,3*16])
    #changed from : m_hx = torch.zeros(1, num_actions * 16).to(device)
    # same m_cx and worker hidden and cell states
    m_hx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(args.num_envs ,num_actions * 16).to(device)
    w_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    w_lstm = {key : (w_hx, w_cx) for key in Worker_IDS}
    goals_horizon = torch.zeros(args.num_envs, args.horizon + 1, num_actions * 16).to(device)

    score_history = {w_id: [] for w_id in Worker_IDS}

    loss_history = []
    for episode in episodes(n=args.num_episodes):

        memory = Memory()
        #Build the agent @ the start of each episode
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }

        #Reset the env @ start of each episode and log the observations. observation contains all observations in SMARTS
        observations = env.reset()
        """
        specify the state as a subset of observations to be passed through the model; 
        in this case we ar only using linear acceleration
        
        """
        # w_state_rep = [torch.Tensor(ttc_obs['distance_from_center']),
        #              torch.Tensor(ttc_obs['angle_error']),
        #              torch.Tensor(ttc_obs['speed']),
        #              torch.Tensor(ttc_obs['steering']),
        #              torch.Tensor(ttc_obs['ego_ttc']),
        #              torch.Tensor(ttc_obs['ego_lane_dist']),
        #              ]
        #Initialise 0 score/reward for all workers
        score = 0
        scores = {keys: score for keys in observations.keys()}

        worker_states ={}
        for key, value in observations.items():
            worker_states[key] = [value[1], value[0].ego_vehicle_state.position,
                                  [value[0].neighborhood_vehicle_states[i].position
                                   for i in range(len(value[0].neighborhood_vehicle_states))]]

        worker_tensors = {}
        for key, value in worker_states.items():
            worker_tensors[key] = (torch.Tensor(value[0]['distance_from_center']),
                                   torch.Tensor(value[0]['angle_error']),
                                   torch.Tensor(value[0]['speed']),
                                   torch.Tensor(value[0]['steering']),
                                   torch.Tensor(value[0]['ego_ttc']),
                                   torch.Tensor(value[0]['ego_lane_dist']),
                                   torch.Tensor(value[1]),
                                   torch.Tensor(value[2]))

        manager_state = {}
        for key, value in observations.items():
            manager_state[key] = value[0].ego_vehicle_state.linear_velocity

        m_avg_velocity = sum(manager_state.values())/N_Workers
        w_states = {}
        for key, values in worker_tensors.items():
            w_states[key] = [values[i].to(device) for i in range(len(values))]

        #Pad Neighbor pos with 0z if <5
        for key in w_states.keys():
            if w_states[key][7].size()[0] <5:
                pad = 5 - w_states[key][7].size()[0]
                padding = (0,0,0,pad)
                pads = torch.nn.ZeroPad2d(padding)
                w_states[key][7] = pads(w_states[key][7]).reshape(15,1)
        #Concatenate worker states into single tensor
        for key in w_states.keys():
            w_states[key][0]= torch.cat(w_states[key][0:7]).reshape(13,1)
            w_states[key][1]= w_states[key][7]
            w_states[key] = torch.cat(w_states[key][0:2]).reshape(1,observation_size)

        man_states = torch.Tensor(m_avg_velocity).to(device)
        episode.record_scenario(env.scenario_log)
        count += 1
        #if count == 4:
         #   print(f'count = {count}, break ')
        steps = 0
        """
        In each episode, we step (take actions) the environment. The agent model (FuN) receives the most recent 
        state+histories and uses the NNs to estimate the best action to take given the new state. 
         
        """
        for i in range(args.num_step):
            net_output = net.forward(w_states, man_states, m_lstm, w_lstm, goals_horizon)
            policies, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state = net_output
            actions, policies, entropy = get_action(policies, num_actions)
            """
            Actions available to ActionSpaceType.Lane are [keep_lane, slow_down, change_lane_left, change_lane_right]
            see smarts.core.controllers.__init__  
            """
            if args.render:
                env.render()
            #Step the environment by taking the actions predicted by FuN model.
            #observation, reward, done, info = env.step({'SingleAgent': actions})
            # for key in agents.keys():
            #     agents[key] = agents[key].act()

            w_act = [agent.act(actions) for agent in agents.values()]

            observation, reward, done, info = env.step(w_act[0])
            #Record the new state after taking an action
            new_w_states = {}
            new_w_tensors = {}
            for key, value in observations.items():
                new_w_states[key] = [value[1], value[0].ego_vehicle_state.position,
                                      [value[0].neighborhood_vehicle_states[i].position
                                       for i in range(len(value[0].neighborhood_vehicle_states))]]
            for key, value in new_w_states.items():
                new_w_tensors[key] = (torch.Tensor(value[0]['distance_from_center']),
                                       torch.Tensor(value[0]['angle_error']),
                                       torch.Tensor(value[0]['speed']),
                                       torch.Tensor(value[0]['steering']),
                                       torch.Tensor(value[0]['ego_ttc']),
                                       torch.Tensor(value[0]['ego_lane_dist']),
                                       torch.Tensor(value[1]),
                                       torch.Tensor(value[2]))

            new_m_state = {}
            for key, value in observations.items():
                new_m_state[key] = value[0].ego_vehicle_state.linear_velocity

            m_avg_velocity = sum(new_m_state.values()) / N_Workers
            new_w_states = {}
            for key, values in new_w_tensors.items():
                new_w_states[key] = [values[i].to(device) for i in range(len(values))]

            # Pad Neighbor pos with 0z if <5
            for key in new_w_states.keys():
                if new_w_states[key][7].size()[0] < 5:
                    pad = 5 - new_w_states[key][7].size()[0]
                    padding = (0, 0, 0, pad)
                    pads = torch.nn.ZeroPad2d(padding)
                    new_w_states[key][7] = pads(new_w_states[key][7]).reshape(15, 1)
            # Concatenate worker states into single tensor
            for key in new_w_states.keys():
                new_w_states[key][0] = torch.cat(new_w_states[key][0:7]).reshape(13, 1)
                new_w_states[key][1] = new_w_states[key][7]
                new_w_states[key] = torch.cat(new_w_states[key][0:2]).to(device).reshape(1, observation_size)

            new_m_state = torch.Tensor(m_avg_velocity).to(device)
            # next_state = [torch.Tensor(observation['distance_from_center']),
            #          torch.Tensor(observation['angle_error']),
            #          torch.Tensor(observation['speed']),
            #          torch.Tensor(observation['steering']),
            #          torch.Tensor(observation['ego_ttc']),
            #          torch.Tensor(observation['ego_lane_dist'])
            #     ]
            for value in new_w_states.keys():
                new_w_states[key]= new_w_states[key].to(device)


            #Increment steps and sum the reward
            steps += 1

            for key, value in reward.items():

                scores[key] = scores[key] + value

            episode.record_step(observations, reward, done, info)

            """
            To Do: Pass manager and worker states to GPU
            """

            reward = {key: np.asarray([value]) for key, value in reward.items()}

            mask = np.asarray([1])

            memory.push(w_states, man_states, new_w_states, new_m_state,
                        actions, reward, mask, goal,
                        policies, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state, entropy)
            if done['__all__']:
                break
            #End of step loop, assign the state to be passed to FuN(manager and worker) as the most recent state and repeat loop.
            w_states = new_w_states
            man_states = new_m_state

            """
            End of Step loop 
            """

        #If done criteria == True, calculate entropy -> H(x) = -P * log(P)
        if done['__all__']:
            for key, value in entropy.items():
                entropy[key] = -policies[key] * torch.log(policies[key]+ 1e-5)
                entropy[key] = entropy[key].mean().data.cpu()

            # entropy = - policies * torch.log(policies + 1e-5)
            # entropy = entropy.mean().data.cpu()
            plcy ={}
            for key, value in policies.items():
                plcy[key] = value.tolist()[0]
            print('action are {} | global steps {} | score: {} | entropy: {} | grad norm: {} | policy {}'.format(w_act[0],steps,
                                                                                             scores, entropy,
                                                                                              grad_norm, policies))
            if i == 0:
                writer.add_scalar('log/score', score[i], steps)
        for key, val in scores.items():
            score_history[key].append(val)

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

    #plt.plot(range(args.num_episodes), moving_average(loss_history))
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
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'intersections/4lane')

    args.scenarios = [scenario_dir]
    build_scenario(args.scenarios)

    main()

