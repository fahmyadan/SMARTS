import traci
import numpy as np
import sys, os, math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare env var SUMO_HOME")

class TraciMethods():

    def __init__(self, traci_conn):
        self.vehicle = traci_conn.vehicle
        self.edge = traci_conn.edge
        self.lane = traci_conn.lane

    def get_edges_list(self):
        edges_list = self.edge.getIDList()
        return edges_list

    def get_vehicle_list(self):
        veh_list = self.vehicle.getIDList()
        return veh_list

    def get_lane_list(self):
        lane_list = self.lane.getIDList()
        return lane_list
    """
    Edge Methods
    """
    def get_edge_travel_time(self):
        all_edges = self.edge.getIDList()
        travel_times = {}
        for edge_id in all_edges:
            travel_times[edge_id] = self.edge.getTraveltime(edge_id)

        return travel_times

    def get_edge_waiting_time(self, edges_list):
        waiting_times = {ids: [] for ids in edges_list}
        for edge_id in edges_list:
            waiting_times[edge_id] = self.edge.getWaitingTime(edgeID=edge_id)
        return waiting_times

    def get_cumm_timeloss(self, vehicle_ids):
        veh_timeloss = {ids: [] for ids in vehicle_ids}
        for veh_id in vehicle_ids:
            veh_timeloss[veh_id] = self.vehicle.getTimeLoss(veh_id)

        cumm_timeloss = sum(veh_timeloss.values())
        return cumm_timeloss

    def get_edge_vehicle_number(self, edges_list):

        edge_queues ={ids: [] for ids in edges_list}
        for edge_id in edges_list:
            edge_queues[edge_id].append(self.edge.getLastStepVehicleNumber(edgeID=edge_id))

        return edge_queues


    """
    Vehicle Methods
    """
    def get_vehicle_positions(self, vehicle_ids):
        veh_positions = {ids: [] for ids in vehicle_ids}
        for veh_id in vehicle_ids:
            veh_positions[veh_id] = self.vehicle.getPosition(vehID=veh_id)
        return veh_positions

    def get_vehicle_speeds(self, vehicle_ids):
        veh_speeds = {ids: [] for ids in vehicle_ids}
        for veh_id in vehicle_ids:
            longi = self.vehicle.getSpeed(vehID=veh_id)
            lat = self.vehicle.getLateralSpeed(vehID=veh_id)
            veh_speeds[veh_id] = [longi, lat]
        return veh_speeds

    def get_vehicle_angles(self, vehicle_ids):
        veh_angles = {ids: [] for ids in vehicle_ids}
        for veh_id in vehicle_ids:
            veh_angles[veh_id] = self.vehicle.getAngle(vehID=veh_id) * math.pi / 180
        return veh_angles
