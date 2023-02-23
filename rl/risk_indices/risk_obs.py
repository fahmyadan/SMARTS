from dataclasses import dataclass
from typing import Callable, Dict

import gym
import numpy as np
import time

from smarts.core.coordinates import Heading
from smarts.core.sensors import Observation
from smarts.core.utils.math import squared_dist, vec_2d, vec_to_radians, position_to_ego_frame

from risk_indices.risk_indices import safe_lon_distances

@dataclass
class Adapter:
    """An adapter for pairing an action/observation transformation method with its gym
    space representation.
    """

    space: gym.Space
    transform: Callable


_RISK_INDICES_OBS = gym.spaces.Dict(
    {
        "rel_distance_min": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
        "rel_vel": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
    }
)


def risk_obs(obs: Observation):

    ego_pos = obs.ego_vehicle_state.position
    ego_lin_vel = obs.ego_vehicle_state.linear_velocity # dx/dt, dy/dt 
    ego_heading = obs.ego_vehicle_state.heading.__float__() #heading angle in radians

    cos, sin = np.cos(ego_heading) , np.sin(ego_heading)

    rotation_matrix_3d = np.array(((cos,-sin, 0), (sin, cos, 0), (0,0,1)))

    neighbors = obs.neighborhood_vehicle_states

    neigh_pos = []
    neigh_speed = []
    local_frame_dist_dict = {}
    local_frame_vel_dict = {}
    if ego_lin_vel[1] > 5: 
        print(f'ego lin vel {ego_lin_vel}')
        time.sleep(5)

    for neighbor in neighbors: 

        neigh_pos.append(neighbor.position) #neighbor position 
        neigh_speed.append(neighbor.speed)

        neigh_id = neighbor.id

        ego_distance = neighbor.position - ego_pos

        local_frame_distance = np.dot(rotation_matrix_3d,ego_distance.reshape(3,1))
        smarts_ego_frame_pos = position_to_ego_frame(position=neighbor.position, ego_position=ego_pos, ego_heading=ego_heading)
        # Note: smarts_ego_frame_pos[1]> 0 is in front, x> 0 is to the right 

        local_frame_dist_dict[neigh_id] = smarts_ego_frame_pos

        # Convert scalar speed to linear velocity
        neigh_heading = neighbor.heading.__float__() 



        #TODO: Convert neighbor heading to ego frame 
        neigh_rel_heading = neigh_heading - ego_heading  
        neigh_x_dot = neighbor.speed * np.sin(neigh_rel_heading)
        neigh_y_dot = neighbor.speed * np.cos(neigh_rel_heading)
        neigh_linear_vel = np.array([neigh_x_dot, neigh_y_dot, 0]) #assume 0 for velocity in z direction 
        local_frame_vel = np.dot(rotation_matrix_3d, neigh_linear_vel)

        local_frame_vel_dict[neigh_id] = local_frame_vel
    
    # Combine neighbor velocity and distance into one dictionary {veh_id: (dist, vel)...} -> relative to ego  
    local_frame_paras = {} 

    for keys in local_frame_dist_dict.keys(): 
            local_frame_paras[keys] = (np.array(local_frame_dist_dict[keys]), local_frame_vel_dict[keys], ego_lin_vel)

    #Check if neighbor vehicle is in front of ego and assign vf, vr accordingly

    _risk_long_inputs = front_check(local_frame_paras) 

    _risk_lat_inputs = left_check(local_frame_paras)

    #TODO:Compute d_long min and d_lat min 

    if len(neighbors) > 3: 
        _risk_long_inputs = front_check(local_frame_paras) 
        _risk_lat_inputs = left_check(local_frame_paras)

        _ = safe_lon_distances(_risk_long_inputs)

        # print('check') 
        # print(f'ego frame dist {local_frame_dist_dict}')
        # time.sleep(5)
    


    risk_obs = {'rel_distance_min': 50, 'rel_vel': 20}
    

    return risk_obs


risk_indices_obs_adapter = Adapter(space=_RISK_INDICES_OBS, transform=risk_obs)

def front_check(local_frame): 

    risk_long_inp = {}

    for keys, vals in local_frame.items():

        local_dist, local_vels, ego_vel  = vals
        d_long_curr = local_dist[1]
        if d_long_curr >= 0:  #Neighbor in front .
            # if meet lateral threshold
            #TODO: Add threshold 
            #TODO: Nested if for neighbours that are in front 
            v_f = np.linalg.norm(local_vels)
            v_r = np.linalg.norm(ego_vel)
            risk_long_inp[keys] = (v_f, v_r, d_long_curr)
            # else:
            #    long risk =0 

        else: 
            #TODO: Elseif threshold and less than 0 
            v_f = np.linalg.norm(ego_vel)
            v_r = np.linalg.norm(local_vels)
            risk_long_inp[keys] = (v_f, v_r, d_long_curr)

    return risk_long_inp


def left_check(local_frame):

    risk_lat_inputs = {} 

    for keys, vals in local_frame.items():
        local_dist, local_vels, ego_vel  = vals

        d_lat_curr =  local_dist[0] 

        if d_lat_curr >= 0: #Neighbor on RHS
            v_lhs = ego_vel[1]
            v_rhs = local_vels[0]
            risk_lat_inputs[keys] = (v_lhs, v_rhs, d_lat_curr)
        else: 
            v_lhs = local_vels[0]
            v_rhs = ego_vel[1]
            risk_lat_inputs[keys] = (v_lhs, v_rhs, d_lat_curr)

    return risk_lat_inputs




        


            
            
        
    
    
  