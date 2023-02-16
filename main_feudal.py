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

from FUNRL.lstm_a2c.model1 import ManagerAC, WorkerAC
from FUNRL.lstm_a2c.utils import *
from FUNRL.lstm_a2c.train1 import calc_return, train_manager, train_worker
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

torch.autograd.set_detect_anomaly(True)


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

        worker_lane_actions = lane_actions[actions]

        return worker_lane_actions

N_Workers = 1
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

    if len(env_obs.events.collisions )!= 0:
        print('Negative reward activated')
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
        # if drac_index or uni_risk_index > 0 : 

            # print(f'Risk Indices for {veh1} {uni_risk_index, ttc, drac_index}')

"""Test Commit"""
"""
TODO: The Manager's output/action should provide some semantic meaning to the worker and the environment. Maybe stop/slow down/speed up??  
"""
def main():
    writer = SummaryWriter('logs')
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'intersections/4lane')
    # parser = default_argument_parser("feudal-learning")
    # args = parser.parse_args()
    args.scenarios = [scenario_dir]  
    args.horizon = 9
    args.save_path = './save_model/'
    args.num_envs = 1
    args.env_name = "smarts.env:hiway-v0"
    args.render = True
    args.num_step = 3
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
    env = SingleAgent(env=env)

    env.seed(500)
    torch.manual_seed(500)

    observation_size = 15  
    manager_obs_size = 17 #(8queue + 8 wait + 1 cumm loss)
    num_actions = 4
    manager_policy_size = 5 
    manager_sampled_action = 1 
    print('observation size:', observation_size)
    print('action size:', num_actions)
    print("cuda is ", torch.cuda.is_available())
    print(device)

    # Instantiate Manager + Worker model 

    manager_net = ManagerAC(manager_in_size=manager_obs_size,actor_out_size=manager_policy_size, hidden_size= 10)
    manager_optim = optim.Adam(manager_net.parameters(), lr= 5e-2)
    manager_critic_optim = optim.Adam(manager_net.critic.parameters(), lr=5e-2)
    manager_actor_optim = optim.Adam(manager_net.actor.parameters(), lr=5e-2)

    worker_net = WorkerAC(worker_action_size=num_actions, worker_raw_obs_size=observation_size, manager_action_size=manager_sampled_action)
    worker_optim = optim.Adam(worker_net.parameters(), lr= 5e-2)

    manager_net.to(device=device)
    manager_net.train()
    worker_net.to(device=device)
    worker_net.train()
    
    count = 1
    grad_norm = 0

    manager_loss_history = []
    worker1_loss_history = []
    worker2_loss_history = []

    manager_memory = Memory()
    worker_memory = Memory()
    avg_worker_reward_history =[]
    #episodic_queues =
    for episode in episodes(n=args.num_episodes):
        print(f'episode count {count}')
        avg_episode_reward = []

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


        # worker_tensors = multi_worker_observations(observations, device)

        worker_state = worker_obs_to_tensor(observations).to(device, torch.float32)
        worker_state_tensors = worker_state.reshape(1, observation_size)




        manager_state = torch.from_numpy(np.vstack((all_queues,all_wait,cumm_timeloss))).to(device, torch.float32)
        manager_state_tensors = manager_state.reshape(1,manager_obs_size)

        

        episode.record_scenario(env.scenario_log)
        steps = 0

        done = False

        while not done:
            print(f'forward pass {steps}')
            manager_net_parameters = dict(manager_net.named_parameters())

            manager_out = manager_net(manager_state_tensors)
            managers_policy_latent_state, manager_state_value = manager_out

            manager_action, manager_entropy, manager_log_prob = select_manager_action(managers_policy_latent_state)

            manager_action_out = manager_action.reshape(1,1)
            
            # TODO: Input to workerAC is manager's policy as opposed to sampled action. Investigate if replacing with action helps.  

            combined_obs = torch.cat([worker_state_tensors, manager_action_out], dim=1) 

            worker_out = worker_net(combined_obs)

            worker_policy_probs, worker_state_value = worker_out

            worker_actions, worker_entropy, worker_log_prob = select_worker_action(worker_policy=worker_policy_probs) 

            
            # actions, policies, entropy = get_action(policies, num_actions)

            if args.render:
                env.render()


            worker_lane_actions = agents['Worker_1'].act(worker_actions)
            

            observations, reward, done, info = env.step(worker_lane_actions)
            

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


            junction_manager_reward = manager_reward(queues=new_all_queues, wait_time=new_all_wait,
                                                     time_loss=new_cumm_timeloss)

            junction_manager_reward_tensor = torch.from_numpy(junction_manager_reward).to(device, torch.float32)
            #Record the new worker and manager state after taking an action

            new_worker_state = worker_obs_to_tensor(observations) 
            new_worker_state_tensor = new_worker_state.to(device, torch.float32).reshape(1,observation_size)


            new_manager_state = np.vstack((new_all_queues, new_all_wait, new_cumm_timeloss))
            new_manager_state_tensor = torch.from_numpy(new_manager_state).to(device, torch.float32).reshape(1,manager_obs_size)


            #Increment steps 
            steps += 1


            episode.record_step(observations, reward, done, info)

            reward_tensors = torch.Tensor([reward]).to(device, torch.float32)



            mask = torch.ones(1).to(device)


            manager_memory.push(manager_state_tensors, manager_action_out, new_manager_state_tensor, junction_manager_reward_tensor, mask, managers_policy_latent_state,
                                manager_log_prob, manager_state_value.detach(), manager_entropy)

            
            worker_memory.push(worker_state_tensors, worker_actions, new_worker_state_tensor, reward_tensors, mask, worker_policy_probs, worker_log_prob ,worker_state_value, worker_entropy)

            worker_state_tensors = new_worker_state_tensor
            manager_state_tensors = new_manager_state_tensor

            episode.record_step(observations, reward, done, info)

        print('Exited step loop ')
        if done:

            print(f'agent_action are {worker_lane_actions} | global steps {steps} | last_score: {reward} | entropy: {worker_entropy} | grad norm: {grad_norm} | policy {worker_policy_probs}')
       
            writer.add_scalar('log/score', reward, steps)
            
        
        manager_transitions = manager_memory.sample()
        worker_transitions = worker_memory.sample()

        manager_AC_loss = train_manager(manager_net, manager_critic_optim, manager_transitions, args)

        count+=1


        print(f'reward for in last episode {reward} ')
       
        manager_loss_history.append(manager_AC_loss)


        # if count % args.save_interval == 0:
        #     ckpt_path = args.save_path + 'model.pt'
        #     torch.save(net.state_dict(), ckpt_path)
        """    
        Calculate avg worker reward per episode
        """

        # avg_episode_reward = sum(avg_episode_reward)/N_Workers
        # avg_worker_reward_history.append(avg_episode_reward)
        # count+=1 
        # if count == 2: 
        #     print('next episode')
        # print('episode reset')


    print(f'size of avg_worker_reward is {len(avg_worker_reward_history)} and num_episode is {args.num_episodes}')
    plt.plot(range(args.num_episodes), avg_worker_reward_history)
    plt.plot(range(9, args.num_episodes), moving_average(avg_worker_reward_history), color='green')
    plt.title('Average agent reward per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()
    plt.plot(range(args.num_episodes), manager_loss_history)
    plt.title('Losses per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    scenario_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenarios'), 'intersections/4lane')

    args.scenarios = [scenario_dir]
    build_scenario(args.scenarios)

    main()

