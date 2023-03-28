from dataclasses import dataclass
from typing import Callable, Dict

import gym
import numpy as np
import time

from smarts.core.coordinates import Heading
from smarts.core.sensors import Observation
from smarts.core.utils.math import squared_dist, vec_2d, vec_to_radians, position_to_ego_frame, velocity_to_ego_frame

# from risk_indices.risk_indices import safe_lon_distances, safe_lat_distances,risk_index, risk_index_unified, alt_risk_index_unified
from .risk_indices import safe_lon_distances, safe_lat_distances,risk_index, risk_index_unified, alt_risk_index_unified

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
    ego_heading = obs.ego_vehicle_state.heading.__float__()
    #heading angle in radians: Starts at north and moves anti clockwise

    cos, sin = np.cos(ego_heading) , np.sin(ego_heading)

    rotation_matrix_3d = np.array(((cos,-sin, 0), (sin, cos, 0), (0,0,1)))

    neighbors = obs.neighborhood_vehicle_states

    neigh_pos = []
    neigh_speed = []
    local_frame_dist_dict = {}
    local_frame_vel_dict = {}
    if ego_lin_vel[1] > 5: 
        print(f'ego lin vel {ego_lin_vel}')
  
    if neighbors:

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




            neigh_global_long_vel = neighbor.speed * np.cos(neigh_heading)
            neigh_global_lat_vel = neighbor.speed * np.sin(neigh_heading)
            neigh_vel_vect = np.array([[neigh_global_long_vel],[neigh_global_lat_vel]])

            new_vel_vect = velocity_to_ego_frame(neighbor_velocity_vector=neigh_vel_vect, 
                                    neighbor_heading=neigh_heading, ego_heading=ego_heading)
            
            #TODO: Check sign of longitudinal velocity; vehicles moving in opposite directions should have opposite sign.

            local_frame_vel_dict[neigh_id] = new_vel_vect
    
    # Combine neighbor velocity and distance into one dictionary {veh_id: (dist, vel)...} -> relative to ego  
    local_frame_paras = {} 

    for keys in local_frame_dist_dict.keys(): 
            local_frame_paras[keys] = (np.array(local_frame_dist_dict[keys]), local_frame_vel_dict[keys], ego_lin_vel)

    #Check if neighbor vehicle is in front of ego and assign vf, vr accordingly

    _risk_long_inputs = front_check(local_frame_paras) 

    _risk_lat_inputs = left_check(local_frame_paras)

    safe_long_distances_all = safe_lon_distances(_risk_long_inputs)

    safe_lat_distances_all = safe_lat_distances(_risk_lat_inputs)
    
    safe_distances_all = (safe_long_distances_all, safe_lat_distances_all)


    long_lat_risk = risk_index(safe_distances_all) # (Long risk, Lat risk) 

    unif_risks = risk_index_unified(long_lat_risk)
    alt_unif_risk = alt_risk_index_unified(long_lat_risk)
    

    return alt_unif_risk


risk_indices_obs_adapter = Adapter(space=_RISK_INDICES_OBS, transform=risk_obs)

veh_width = 1.8 
veh_length = 5
scale = 1.5 #safety factor 

def front_check(local_frame): 
    """
    Check if vehicle is in front and within the threshold to be considered for long_risk calc: 
    Args:
    local_frame: Dict[str:(dist, neigh_vel, ego_vel)] with distance and vel relative to ego frame
    Returns: 
    risk_long_inp: Dict[Tuple[bool ,np.array,np.array,float]] bool=False if threshold not met 
                    and risk_long_inp[id][1:]=None 
                    else bool=True and risk_long_inp[id][1:] = v_f, vr, d_long_current 
    """

    #TODO: Check v_f/v_r if it is correct longitudinal velocity or the magnitude of vector
    risk_long_inp = {}

    for keys, vals in local_frame.items():

        local_dist, local_vels, ego_vel  = vals
        d_long_curr = local_dist[1]
        horiz = local_dist[0]
        if d_long_curr >= 0:  #Neighbor in front .

            if abs(horiz) <= scale * veh_width:  # if meet lateral threshold
                
                v_f = np.linalg.norm(local_vels)
                v_r = np.linalg.norm(ego_vel)
                front = True 
                risk_long_inp[keys] = ({'front':front}, v_f, v_r, d_long_curr)

            else:
                front= False
                risk_long_inp[keys] = ({'front':front}, None, None, None)

        elif d_long_curr < 0 and abs(horiz) <= scale * veh_width: 
            v_f = np.linalg.norm(ego_vel)
            v_r = np.linalg.norm(local_vels)
            rear = True 
            risk_long_inp[keys] = ({'rear':rear},v_f, v_r, d_long_curr)
        
        else: 
            rear= False
            risk_long_inp[keys] = ({'rear':rear}, None, None, None)

    return risk_long_inp


def left_check(local_frame):
    #TODO: Check v_lhs/v_rhs if it is correct lateral velocity or the magnitude of vector
    risk_lat_inputs = {} 

    for keys, vals in local_frame.items():
        local_dist, local_vels, ego_vel  = vals

        d_lat_curr =  local_dist[0] 
        vertical = local_dist[1]

        if d_lat_curr >= 0: #Neighbor on RHS

            if abs(vertical) <= scale * veh_length: # Meets vertical threshold zone
                v_lhs = ego_vel[1]
                v_rhs = local_vels[1]
                right= True #Neighbor is on RHS 
                risk_lat_inputs[keys] = ({'right':right},v_lhs, v_rhs, d_lat_curr)
            else: 
                right = False 
                risk_lat_inputs[keys] = ({'right':right},None, None, None)

        elif d_lat_curr < 0 and abs(vertical) <= scale * veh_length: 
            left = True #Neighbor is on LHS 
            v_lhs = local_vels[1]
            v_rhs = ego_vel[1]
            risk_lat_inputs[keys] = ({'left':left},v_lhs, v_rhs, d_lat_curr)
        else: 
            left = False
            risk_lat_inputs[keys] = ({'left':left},None, None, None)


    return risk_lat_inputs



            
        
    
    
  