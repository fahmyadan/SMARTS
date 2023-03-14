import numpy as np
from typing import Dict, Sequence, Tuple 
import time
### CONSTANTS

# time it takes rear car to react and begin braking (both longitudinal and lateral)
# https://trl.co.uk/sites/default/files/PPR313_new.pdf
# seconds
responseTime = 1.5

# max braking of front car (longitudinal) Bokare, P. S., & Maurya, A. K. (2017). Acceleration-deceleration behaviour
# of various vehicle types. Transportation research procedia, 25, 4733-4749. m/s^2
accFrontMaxBrake = 2.5

# max acceleration of rear car during response time (longitudinal) Bokare, P. S., & Maurya, A. K. (2017).
# Acceleration-deceleration behaviour of various vehicle types. Transportation research procedia, 25, 4733-4749. m/s^2
accRearMaxResp = 2.5

# min braking of rear car (longitudinal)
# m/s^2
accRearMinBrake = 1

# max braking capability of rear car (longitudinal) replace for ACC_REAR_MIN_BRAKE when calculating
# 'safeDistanceBrake' Bokare, P. S., & Maurya, A. K. (2017). Acceleration-deceleration behaviour of various vehicle
# types. Transportation research procedia, 25, 4733-4749. m/s^2
accRearMaxBrake = 4

# max acceleration of both cars towards each other during response time (lateral)
# m/s^2
accMaxResp = 1

# min braking of both cars (lateral)
# https://doi.org/10.1007/s12544-013-0120-2
# m/s^2
accMinBrake = 1.5

# max braking capability of both cars (lateral)
# replace for ACC_MIN_BRAKE when calculating 'safeDistanceBrake'
# https://doi.org/10.1007/s12544-013-0120-2
# m/s^2
accMaxBrake = 4

# longitudinal risk propensity exponent > 0
riskPropLon = 1

# lateral risk propensity exponent > 0
riskPropLat = 1

# TTC Threshold - in seconds
TTC_THRESHOLD = 5


# Function to calculate safe longitudinal distance
def safe_lon_distances(risk_long_inputs):
    """
    All speed and acceleration inputs must be for the longitudinal axis
    RISK_LONG_INPUTS : {'neighbor_id': (speed_front, speed_rear, d_long_curr)...} 
    responseTime: time it takes rear car to react and begin braking
    speedFront: current velocity of front car
    accFront: current accelaration of front car
    accFrontMaxBrake: max braking of front car
    speedRear: current velocity of rear car
    accRear: current accelaration of rear car
    accRearMaxResp: max acceleration of rear car during response time
    accRearMinBrake: min braking of rear car
    """
    #TODO: Check 'sign' for vehicle moving in opposite direction 
    
    safe_dlon_min = {}

    for keys, values in risk_long_inputs.items(): 

        # Front check threshold not met -> None 
        if None in values: 

            d_lon_min = None 
            d_lon_min_brake = None 
            d_long_curr = None 
            safe_dlon_min[keys]= (d_lon_min, d_lon_min_brake, d_long_curr)
        
        else: # Threshold met 

            front_or_rear, speed_front, speed_rear, d_long_curr = values
            sq_term = (speed_rear + responseTime*accRearMaxResp)**2
            d_lon_min = (speed_rear * responseTime) + (0.5 * np.power(responseTime,2) * accRearMaxResp) + (0.5 * sq_term / accRearMinBrake ) \
                                    - (0.5 *  np.power(speed_front,2) / accFrontMaxBrake)

            d_lon_min_brake = (speed_rear * responseTime) + (0.5 * np.power(responseTime,2) * accRearMaxResp) + (0.5 * sq_term / accRearMaxBrake ) \
                            - (0.5 *  np.power(speed_front,2) / accFrontMaxBrake)
             
            safe_dlon_min[keys] = (d_lon_min, d_lon_min_brake ,d_long_curr)
         
    return safe_dlon_min


# Function to calculate safe lateral distance
def safe_lat_distances(risk_lat_inputs):
    """
    All speed and acceleration inputs must be for the lateral axis
    responseTime: time it takes rear car to react and begin braking
    speedLeft: current velocity of left car
    speedRight: current velocity of right car
    accMaxResp: max acceleration of both cars towards each other during response time
    accMinBrake: min braking of both cars
    """
    safe_dlat ={}
    for keys, vals in risk_lat_inputs.items():
        if None not in vals:
            left_or_right, speed_left, speed_right, d_lat_curr = vals

            # TODO: Explain -> assuming vehicles travel towards each other during response time for boundary condition
            vLeft = speed_left - responseTime * accMaxResp
            vRight = speed_right + responseTime * accMaxResp
            # left vehicle move to left
            if vLeft <= 0:

                if vRight <= 0: 
                    """
                    If left + right vehs are moving to the left, both need to brake as much as possible to avoid collision
                    """
                    acc_left_min_gap = acc_right_max_gap = accMaxBrake
                    acc_left_max_gap = acc_right_min_gap = accMinBrake
                
                else: #right vehicle moving to the right 
                    """
                    Left veh brakes at maximum brake limit and right veh brakes at minimum limit
                    """
                    acc_left_min_gap = acc_right_min_gap = accMaxBrake
                    acc_left_max_gap = acc_right_max_gap = accMinBrake
            
            else: #Left veh moving to the right  
                
                if vRight <= 0: #Right veh moving left
                    acc_left_min_gap = acc_right_min_gap = accMinBrake
                    acc_left_max_gap = acc_right_max_gap = accMaxBrake
                
                else: #Right veh moving right 
                    acc_left_min_gap = acc_right_max_gap = accMinBrake
                    acc_left_max_gap = acc_right_min_gap = accMaxBrake

            
            sign_left = [1,-1] [vLeft.item()<=0] #left vehicle moving left sign=-1
            sign_right = [1,-1][vRight.item()<=0]

            d_lat_min = 0.5 * (speed_left + vLeft) * responseTime + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_min_gap - (
            0.5 * (speed_right + vRight) * responseTime + 0.5 * sign_right * np.power(vRight, 2) / acc_right_min_gap)

            d_lat_min_brake = 0.5 * (speed_left + vLeft) * responseTime + 0.5 * sign_left * np.power(vLeft, 2) / \
                      acc_left_max_gap - (0.5 * (speed_right + vRight) * responseTime + 0.5 * sign_right *
                                          np.power(vRight, 2) / acc_right_max_gap)

            safe_dlat[keys] = (d_lat_min, d_lat_min_brake, d_lat_curr)

        else: 
            d_lat_min = None
            d_lat_min_brake = None 
            d_lat_curr = None 

            safe_dlat[keys] = (d_lat_min, d_lat_min_brake, d_lat_curr)
    
    return safe_dlat


# Function to calculate the longitudinal or lateral risk index [0,1]
def risk_index(safe_distances_all):
    """
    All inputs must be  longitudinal safeDistance: safe longitudinaldistance (use
    function SafeLonDistance) safeDistanceBrake: safe longitudinal distance under max
    braking capacity (use function SafeLonDistance with max braking acceleration) distance: current
    longitudinal distance between cars
    """
    r_long = {}
    r_lat = {}
    
    long_distance, lat_distance = safe_distances_all
    long_lat_distances = {k: [long_distance[k], lat_distance[k]] for k in long_distance.keys()}
    
    # long_lat = {key: [(safe_long_distance, safe_lat_distance)]}. safe_long/lat: (dmin,dminbrake, dcurrent)
   
    for keys, vals in long_lat_distances.items(): 
        if None in vals[0]: #long risk = 0
            r_long[keys] = 0
        elif None not in vals[0]: 
            safe__long_distance, safe_long_distance_brake , long_distance = vals[0]

            if abs(long_distance) >= safe__long_distance and abs(long_distance) > 0:
                r_long[keys] = 0

            elif safe__long_distance >= abs(long_distance) and abs(long_distance) >= safe_long_distance_brake:

                r_long[keys] = 1- ((abs(long_distance)- safe_long_distance_brake)/ (safe__long_distance - safe_long_distance_brake))
            else: 
                r_long[keys] = 1
        if None in vals[1]:
            r_lat[keys] = 0
        elif None not in vals[1]: 
            safe_lat_distance, safe_lat_distance_brake, lat_distance = vals[1]
            if abs(lat_distance) >= safe_lat_distance and abs(lat_distance) > 0:
                r_lat[keys] = 0

            elif safe_lat_distance >= abs(lat_distance) and abs(lat_distance) >= safe_lat_distance_brake:

                r_lat[keys] = 1- ((abs(lat_distance)- safe_lat_distance_brake)/ (safe_lat_distance - safe_lat_distance_brake))
            else: 
                r_lat[keys] = 1
        

    r = {car: (r_long[car], r_lat[car]) for car in r_long.keys()}

    return r


# Function to calculate the unified risk index [0,1]
def risk_index_unified(risk_all):
    """
    riskLon: longitudinal risk index (use function RiskIndex with longitudinal inputs)
    riskPropLon: longitudinal risk propensity exponent > 0
    riskLat: lateral risk index (use function RiskIndex with lateral inputs)
    riskPropLat: lateral risk propensity exponent > 0
    TODO: Invesitgate why both long + lat risk > 0 to have a unified risk >0
    """
    risk_u = {}
    for keys in risk_all.keys():
        risk_lon, risk_lat = risk_all[keys]

        risk_u[keys] =  np.power(risk_lon, riskPropLon) * np.power(risk_lat, riskPropLat)

    return risk_u

def alt_risk_index_unified(risk_all):
    """
    riskLon: longitudinal risk index (use function RiskIndex with longitudinal inputs)
    riskPropLon: longitudinal risk propensity exponent > 0
    riskLat: lateral risk index (use function RiskIndex with lateral inputs)
    riskPropLat: lateral risk propensity exponent > 0
    TODO: Invesitgate why both long + lat risk > 0 to have a unified risk >0
    """
    risk_u = {}
    for keys in risk_all.keys():
        risk_lon, risk_lat = risk_all[keys]

        if risk_lon >0 and risk_lat >0:

            risk_u[keys] =  np.power(risk_lon, riskPropLon) * np.power(risk_lat, riskPropLat)
        
        elif risk_lon > 0 and risk_lat == 0: 

            risk_u[keys] =  np.power(risk_lon, riskPropLon)

        elif risk_lat > 0 and risk_lon ==0: 

             risk_u[keys] = np.power(risk_lat, riskPropLat)
           
        else: #both long + lat == 0 
            risk_u[keys] =  np.power(risk_lon, riskPropLon) * np.power(risk_lat, riskPropLat)
          

    return risk_u


def ttc_compute(distance, sd):
    """
    distance: gap between ego and target vehicle
    ego_speed: speed of ego vehicle in default direction
    veh_speed: speed of target vehicle in default direction
    """
    if sd < 1e-6:
        return 6
    return abs(distance) / sd


def drac(distance, ego_speed_longi, ego_speed_lateral, veh_speed_longi, veh_speed_lateral):
    """
    ego_pos: pos of ego vehicle
    ego_speed_longi: longitudinal speed of ego vehicle
    ego_speed_lateral: lateral speed of of ego vehicle
    veh_pos: pos of leading vehicle
    veh_speed_longi: longitudinal speed of leading vehicle
    veh_speed_lateral: lateral speed of of leading vehicle
    """
    ego_velocity = np.array([ego_speed_longi, ego_speed_lateral])
    veh_velocity = np.array([veh_speed_longi, veh_speed_lateral])
    # TODO: make sure response time and velocity coincide with units
    if ego_speed_longi < veh_speed_longi:
        return 0

    return np.power(np.linalg.norm(ego_velocity) - np.linalg.norm(veh_velocity), 2) / \
           (distance - ego_speed_longi * responseTime)


def tet(delta_time, prev_value, ttc_val):
    """
    Calculate the time-exposed time-to-collision - cumulative value throughout the simulation
    delta_time: time step size in the simulation
    prev_value: previous value of tet
    ttc_val: ttc at current time step
    """
    if ttc_val < TTC_THRESHOLD:
        return prev_value + delta_time
    return prev_value


def tit(prev_value, ttc_val):
    """
    Calculate the time-integrated time-to-collision (TIT) - cumulative value
    prev_value: previous value of tit
    ttc_val: ttc at current time step
    """
    if ttc_val < TTC_THRESHOLD:
        return prev_value + (TTC_THRESHOLD - ttc_val)
    return prev_value
