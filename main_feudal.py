import math
import os
import time

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

from data_utils import *

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

from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from sumo_interfacing import TraciMethods
from risk_indices import *
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
parser.add_argument('--num_episodes', default=200, help='')
parser.add_argument('--num_step', default=100, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.5, help='')
parser.add_argument('--lr', default=7e-4, help='')
parser.add_argument('--eps', default=1e-4, help='')
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
"""
To Do: Implement one reward for manager and workers
"""
def reward_adapter(env_obs, env_reward):
    # adapt_obs = observation_adapter(env_obs)
    # obs_ttc = adapt_obs['ego_ttc']
    # obs_ttc_dist = adapt_obs['ego_lane_dist']
    # for ttc in obs_ttc:
    #     if ttc > ttc_threshold:
    #         env_reward = -1
    #         return env_reward
    #
    # ttc_norm = obs_ttc.mean()/max(obs_ttc)
    # ttc_dist_norm = obs_ttc_dist.mean()/max(obs_ttc_dist)
    # env_reward = (ttc_weight *ttc_norm) + (ttc_dist_weight*ttc_dist_norm) + env_obs.distance_travelled
    if len(env_obs.events.collisions )!= 0:
        env_reward = -100

    return env_reward

def manager_reward(queues, wait_time, time_loss):

    sum_queue = sum(queues)
    sum_wait = sum(wait_time)
    scaled_time = 10 * time_loss

    negative_reward = -1 * (sum_queue + sum_wait + scaled_time)

    return negative_reward

def compute_risk_indices(traci_conn, veh_list):
    veh_positions = TraciMethods(traci_conn).get_vehicle_positions(veh_list)
    veh_speeds = TraciMethods(traci_conn).get_vehicle_speeds(veh_list)
    veh_angles = TraciMethods(traci_conn).get_vehicle_angles(veh_list)
    scale = 1.5
    veh_width = 1.8
    veh_length = 5

    detection_threshold = 100 # radius of detection circle, only consider vehicle within the circle
    for veh1 in veh_list:
        veh1_pos = np.array(veh_positions[veh1])
        veh1_vel = np.array(veh_speeds[veh1])
        veh1_ang = veh_angles[veh1]
        rotation_matrix = np.matrix([[math.cos(veh1_ang), -math.sin(veh1_ang)],
                                     [math.sin(veh1_ang), math.cos(veh1_ang)]])

        front = rear = left = right = None
        veh1_dist_front = veh1_dist_rear = veh1_dist_left = veh1_dist_right = detection_threshold
        r_lon_front = r_lon_rear = r_lat_left = r_lat_right = 0
        ttc_front = ttc_rear = ttc_left = ttc_right = float('inf')
        drac_index = 0
        for veh2 in veh_list:
            veh2_pos = np.array(veh_positions[veh2])
            vec = veh2_pos - veh1_pos
            if veh1 == veh2 or np.linalg.norm(vec) > detection_threshold:
                continue
            # transfer to the frame of veh1
            dist_vec = rotation_matrix * vec.reshape(2, 1)
            # veh2 in front of veh 1
            if dist_vec[1] >= 0 and abs(dist_vec[0]) <= scale * veh_width:
                if front is None or veh1_dist_front > dist_vec[1]:
                    front = veh2
                    veh1_dist_front = dist_vec[1]
            elif dist_vec[1] < 0 and abs(dist_vec[0]) <= scale * veh_width:
                if rear is None or veh1_dist_rear > dist_vec[1]:
                    rear = veh2
                    veh1_dist_rear = dist_vec[1]
            # veh2 at left of veh 1
            if dist_vec[0] <= 0 and abs(dist_vec[1]) <= scale * veh_length:
                if left is None or veh1_dist_left > dist_vec[0]:
                    left = veh2
                    veh1_dist_left = dist_vec[0]
            elif dist_vec[0] > 0 and abs(dist_vec[1]) <= scale * veh_length:
                if right is None or veh1_dist_right > dist_vec[0]:
                    right = veh2
                    veh1_dist_right = dist_vec[0]

        if front is not None:
            front_pos = np.array(veh_positions[front])
            vec = front_pos - veh1_pos
            front_vel = np.array(veh_speeds[front])
            dist_vec = rotation_matrix * vec.reshape(2, 1)
            vel_vec = rotation_matrix * front_vel.reshape(2, 1)
            safeLonDis, safeLonDisBrake = safe_lon_distances(vel_vec[0], veh1_vel[0])
            r_lon_front = risk_index(safeLonDis, safeLonDisBrake, abs(dist_vec[1]))
            drac_index = drac(dist_vec[1], veh1_vel[0], veh1_vel[1], vel_vec[0], vel_vec[1])
            ttc_front = ttc_compute(dist_vec[1], veh1_vel[0] - vel_vec[0])
        if rear is not None:
            rear_pos = np.array(veh_positions[rear])
            vec = rear_pos - veh1_pos
            rear_vel = np.array(veh_speeds[rear])
            dist_vec = rotation_matrix * vec.reshape(2, 1)
            vel_vec = rotation_matrix * rear_vel.reshape(2, 1)
            safeLonDis, safeLonDisBrake = safe_lon_distances(veh1_vel[0], vel_vec[0])
            r_lon_rear = risk_index(safeLonDis, safeLonDisBrake, abs(dist_vec[1]))
            ttc_rear = ttc_compute(dist_vec[1], veh1_vel[0] - vel_vec[0])
        if left is not None:
            left_pos = np.array(veh_positions[left])
            vec = left_pos - veh1_pos
            left_vel = np.array(veh_speeds[left])
            dist_vec = rotation_matrix * vec.reshape(2, 1)
            vel_vec = rotation_matrix * left_vel.reshape(2, 1)
            safeLatDis, safeLatDisBrake = safe_lat_distances(vel_vec[1], veh1_vel[1])
            r_lat_left = risk_index(safeLatDis, safeLatDisBrake, abs(dist_vec[0]))
            ttc_left = ttc_compute(dist_vec[0], veh1_vel[1] - vel_vec[1])
        if right is not None:
            right_pos = np.array(veh_positions[right])
            vec = right_pos - veh1_pos
            right_vel = np.array(veh_speeds[right])
            dist_vec = rotation_matrix * vec.reshape(2, 1)
            vel_vec = rotation_matrix * right_vel.reshape(2, 1)
            safeLatDis, safeLatDisBrake = safe_lat_distances(veh1_vel[0], vel_vec[0])
            r_lat_right = risk_index(safeLatDis, safeLatDisBrake, abs(dist_vec[0]))
            ttc_right = ttc_compute(dist_vec[0], veh1_vel[1] - vel_vec[1])
        risk_index_lon = max(r_lon_front, r_lon_rear)
        risk_index_lat = max(r_lat_left, r_lat_right)
        uni_risk_index = risk_index_unified(risk_index_lon, risk_index_lat)
        ttc = min(ttc_front, ttc_rear, ttc_left, ttc_right)
        # print(veh1, uni_risk_index, ttc, drac_index)

"""Test Commit"""
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
    args.render = True
    args.num_step = 50
    args.headless = True
    # args.headless = False

    worker_interface = AgentInterface(debug=True, waypoints=True, action=ActionSpaceType.Lane,
                                     max_episode_steps=args.num_step, neighborhood_vehicles=NeighborhoodVehicles(radius=25))
    worker_spec = AgentSpec(
        interface=worker_interface,
        agent_builder=WorkerAgent,
        reward_adapter=reward_adapter,
        observation_adapter=observation_adapter
    )
    agent_specs = { worker_id: worker_spec for worker_id in Worker_IDS}

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs=agent_specs,
        headless=args.headless,
        sumo_headless=True,
        sumo_port= 45761
    )
    env.seed(500)
    torch.manual_seed(500)

    observation_size = 13 + (5*3)  #13 + xyz position of N=5 neighbor vehicles
    manager_obs_size = 17 #(8queue + 8 wait + 1 cumm loss)
    num_actions = 4
    print('observation size:', observation_size)
    print('action size:', num_actions)
    print("cuda is ", torch.cuda.is_available())
    print(device)

    #Instantiate the FuN model object (Creating the percept, manager and worker in the __init__)
    net = FuN(observation_size, num_actions, args.horizon, N_Workers, manager_obs_size=manager_obs_size)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

    net.to(device)
    net.train()
    count = 1
    grad_norm = 0
    state = torch.zeros([observation_size,observation_size]).to(device)

    #Initialize hidden and cell state of Manager and Worker
    m_lstm, w_hx, w_cx = init_lstm_weights(args.num_envs, num_actions, k=16, device=device)
    w_lstm = {key : (w_hx, w_cx) for key in Worker_IDS}
    goals_horizon = torch.zeros(args.num_envs, args.horizon + 1, num_actions * 16).to(device)

    score_history = {w_id: [] for w_id in Worker_IDS}


    loss_history = []

    avg_worker_reward_history =[]
    #episodic_queues =
    for episode in episodes(n=args.num_episodes):
        print(f'episode count {count}')
        avg_episode_reward = []
        memory = Memory()
        #Build the agent @ the start of each episode
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }

        #Reset the env @ start of each episode and log the observations. observation contains all observations in SMARTS
        observations = env.reset()
        traci_conn = env.get_info()
        # edit get_info in hiway_env to retrieve more traci info
        edges_list = TraciMethods(traci_conn=traci_conn).get_edges_list()[16:24]
        veh_list = TraciMethods(traci_conn).get_vehicle_list()
        print("check1", len(veh_list))

        edge_queue = TraciMethods(traci_conn=traci_conn).get_edge_vehicle_number(edges_list)
        all_queues = [i for i in edge_queue.values()]
        all_queues = np.asarray(all_queues)

        edge_wait = TraciMethods(traci_conn=traci_conn).get_edge_waiting_time(edges_list)
        all_wait = [i for i in edge_wait.values()]
        all_wait = np.asarray(all_wait).reshape(8,1)
        cumm_timeloss = np.asarray(TraciMethods(traci_conn).get_cumm_timeloss(veh_list))


        #Initialise 0 score/reward for all workers
        score = 0
        scores = {keys: score for keys in observations.keys()}

        worker_tensors = worker_observations(observations, device)

        manager_state = np.vstack((all_queues,all_wait,cumm_timeloss))
        # for key, value in observations.items():
        #     manager_state[key] = value[0].ego_vehicle_state.linear_velocity
        #
        # m_avg_velocity = sum(manager_state.values())/N_Workers
        #Process the data from tuple to list for mutability
        worker_states = process_w_states(worker_tensors, device)
        #Pad Neighbor pos with 0z if <5
        worker_states = zero_padding(worker_states,neighbour_idx=7,n_neighbours=5)

        #Concatenate worker states into (1,observation_size)) tensor
        worker_states = concat_states(worker_states, observation_size)

        man_states = torch.Tensor(manager_state).to(device)
        episode.record_scenario(env.scenario_log)
        steps = 0
        
        for i in range(args.num_step):
            print(f'forward pass {steps}')
            net_output = net.forward(manager_state=man_states, w_states=worker_states, m_lstm=m_lstm, w_lstm=w_lstm,
                                     goals_horizon=goals_horizon, N_workers=N_Workers, num_actions=num_actions, device=device)
            policies, goal, goals_horizon, m_lstm, new_w_lstm, m_value, w_values , m_state = net_output
            actions, policies, entropy = get_action(policies, num_actions)
            """
            To Do: Fix the discrepancy between man_state and m_state in the model 
            """
            if args.render:
                env.render()
            #Step the environment by taking the actions predicted by FuN model.
            #observation, reward, done, info = env.step({'SingleAgent': actions})
            # for key in agents.keys():
            #     agents[key] = agents[key].act()

            w_act = [agent.act(actions) for agent in agents.values()]

            observations, reward, done, info = env.step(w_act[0])
            #traci observations
            traci_conn = env.get_info()
            veh_list = TraciMethods(traci_conn).get_vehicle_list()
            print("check2", len(veh_list))
            compute_risk_indices(traci_conn, veh_list)

            new_edge_queue = TraciMethods(traci_conn=traci_conn).get_edge_vehicle_number(edges_list)
            new_all_queues = [i for i in new_edge_queue.values()]
            new_all_queues = np.asarray(new_all_queues)
            new_edge_wait = TraciMethods(traci_conn=traci_conn).get_edge_waiting_time(edges_list)
            new_all_wait = [i for i in new_edge_wait.values()]
            new_all_wait = np.asarray(new_all_wait).reshape(8, 1)
            new_cumm_timeloss = np.asarray(TraciMethods(traci_conn).get_cumm_timeloss(veh_list))
            """
            Compute manager reward after some time step, dt
            """

            junction_manager_reward = manager_reward(queues=new_all_queues, wait_time=new_all_wait,
                                                     time_loss=new_cumm_timeloss)

            #Record the new state after taking an action
            new_w_tensor= worker_observations(observations, device)
            new_w_tensor= process_w_states(new_w_tensor, device)
            new_w_states= zero_padding(new_w_tensor, neighbour_idx=7, n_neighbours=5)
            new_w_states= concat_states(new_w_states, observation_size)

            new_manager_state = np.vstack((new_all_queues, new_all_wait, new_cumm_timeloss))
            new_man_state = torch.Tensor(new_manager_state).to(device)

            """    
            To Do: Review w-state & m-state to find out why model states do not change at each step 
            """
            #Increment steps and sum the reward
            steps += 1

            for key in reward.keys():
                score_history[key].append(reward.get(key))

            episode.record_step(observations, reward, done, info)

            """
            To Do: Pass manager and worker states to GPU
            """

            reward = {key: np.asarray([value]) for key, value in reward.items()}

            mask = np.asarray([1])

            memory.push(worker_states, man_states, new_w_states, new_man_state,
                        actions, reward, junction_manager_reward ,mask, goal,
                        policies, m_lstm, new_w_lstm, m_value, w_values, m_state, entropy)
            if done['__all__']:
                break
            #End of step loop, assign the state to be passed to FuN(manager and worker) as the most recent state and repeat loop.
            worker_states= new_w_states
            man_states = new_man_state
            w_lstm = new_w_lstm
        
     

            """
            To Do: Check if entropy should be > 1 ???
            """
            # traci.close()
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

        transitions = memory.sample()
        #print('training model called')
        loss, grad_norm = train_model(net, optimizer, transitions, args, Worker_IDS)
        print(f'training check {count}')
        loss_history.append(loss.item())
        m_hx, m_cx = m_lstm
        m_lstm = (m_hx.detach(), m_cx.detach())
        #w_hx, w_cx = w_lstm
        #w_lstm = (w_hx.detach(), w_cx.detach())
        goals_horizon = goals_horizon.detach()

        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)
        """    
        Calculate avg worker reward per episode
        """
        for vals in score_history.values():
            worker_total_reward = sum(vals)
            avg_episode_reward.append(worker_total_reward)

        avg_episode_reward = sum(avg_episode_reward)/N_Workers
        avg_worker_reward_history.append(avg_episode_reward)
        count+=1 
        if count == 2: 
            print('next episode')
        print('episode reset')


    print(f'size of avg_worker_reward is {len(avg_worker_reward_history)} and num_episode is {args.num_episodes}')
    plt.plot(range(args.num_episodes), avg_worker_reward_history)
    plt.plot(range(9, args.num_episodes), moving_average(avg_worker_reward_history), color='green')
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

