"""
Useful functions
"""
import numpy as np
import torch
from smarts.core.sensors import Observation
from typing import Dict, List, Tuple, Any

def init_lstm_weights(num_envs, num_actions, k, device):

    m_hx = torch.zeros(num_envs, num_actions*k).to(device)
    m_cx = torch.zeros(num_envs, num_actions * k).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(num_envs, num_actions * k).to(device)
    w_cx = torch.zeros(num_envs, num_actions * k).to(device)

    return m_lstm, w_hx, w_cx


def worker_observations(obs: Dict[str,Tuple[Observation,Dict[str, Any]]], device):
    worker_states = {}
    for key, value in obs.items():
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

    for values in worker_tensors.values():
        values = [values[i].to(device) for i in range(len(values))]

    return worker_tensors

def process_w_states(worker_tensors, device):

    worker_states = {}

    for key, value in worker_tensors.items():
        worker_states[key] = [value[i].to(device) for i in range(len(value))]

    return worker_states


def zero_padding(w_tensor, neighbour_idx, n_neighbours):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    for key in w_tensor.keys():
        if w_tensor[key][neighbour_idx].size()[0] < n_neighbours:
            pad = n_neighbours - w_tensor[key][neighbour_idx].size()[0]
            padding = (0, 0, 0, pad)
            pads = torch.nn.ZeroPad2d(padding)
            if w_tensor[key][neighbour_idx].dim() < 2:
                # print(f'we are in w_tensor length{len(w_tensor)}  for key {key} with padding length {len(padding)} '
                #       f'and input size {w_tensor[key][neighbour_idx].dim()}')
                w_tensor[key][neighbour_idx] = torch.zeros(3 * n_neighbours, 1).to(device)
                # print(f'no neighbour position so created new tensor')
            else:
                w_tensor[key][neighbour_idx] = pads(w_tensor[key][neighbour_idx]).reshape(3 * n_neighbours, 1)
                # print(f'padding operation completed for {key} and padding len {len(padding)}')



            # if len(w_tensor) < 4:
            #
            #     if w_tensor[key][neighbour_idx].dim() < 2:
            #         w_tensor[key][neighbour_idx] = torch.zeros(3*n_neighbours,1)
            #         print('check')
            #     w_tensor[key][neighbour_idx] = pads(w_tensor[key][neighbour_idx]).reshape(3 * n_neighbours, 1)
            # w_tensor[key][neighbour_idx] = pads(w_tensor[key][neighbour_idx]).reshape(3 * n_neighbours, 1)

    return w_tensor
def zero_pad_tensor(tensor, size):

    padding  = (0,0,0,size)
    pads = torch.nn.ZeroPad2d(padding)
    tensor = pads(tensor)
    return tensor




def concat_states(worker_states, observation_size):
    for key in worker_states.keys():
        worker_states[key][0] = torch.cat(worker_states[key][0:7]).reshape(13, 1)
        worker_states[key][1] = worker_states[key][7]
        #if worker_states[key][1] != 1 or worker_states[key]
        #print(f'shapes of worker {key} states are {worker_states[key][0].shape} and {worker_states[key][1].shape}')
        if worker_states[key][1].shape[1] != 1:
            if worker_states[key][1].shape[0] > 5:
               # print('our extra dimension is ',worker_states[key][1].shape[0])
                worker_states[key][1] = worker_states[key][1][0:5]
                #print('our new dimension is ', worker_states[key][1].shape)
                worker_states[key][1] = worker_states[key][1].reshape(15, 1)
                worker_states[key] = torch.cat(worker_states[key][0:2]).reshape(1, observation_size)
            else:
                worker_states[key][1]= worker_states[key][1].reshape(15,1)
                worker_states[key] = torch.cat(worker_states[key][0:2]).reshape(1, observation_size)
        else:
            worker_states[key] = torch.cat(worker_states[key][0:2]).reshape(1, observation_size)
    return worker_states

#Plot moving average of reward every n steps
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

