import numpy as np

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
def safe_lon_distances(speed_front, speed_rear):
    """
    All speed and acceleration inputs must be for the longitudinal axis
    responseTime: time it takes rear car to react and begin braking
    speedFront: current velocity of front car
    accFront: current accelaration of front car
    accFrontMaxBrake: max braking of front car
    speedRear: current velocity of rear car
    accRear: current accelaration of rear car
    accRearMaxResp: max acceleration of rear car during response time
    accRearMinBrake: min braking of rear car
    """
    if speed_rear + responseTime * accRearMaxResp < 0:
         sign = 1
    else:
        sign = -1

    safeLonDis = 0.5 * np.power(speed_front, 2) / accFrontMaxBrake - speed_rear * responseTime - 0.5 * \
                 np.power(responseTime, 2) * accRearMaxResp - 0.5 * sign * np.power(speed_rear + responseTime *
                                                                                    accRearMaxResp, 2) / accRearMinBrake
    safeLonDisBrake = 0.5 * np.power(speed_front, 2) / accFrontMaxBrake - speed_rear * responseTime - 0.5 * \
                      np.power(responseTime, 2) * accRearMaxResp - 0.5 * sign * np.power(speed_rear + responseTime *
                                                                                  accRearMaxResp, 2) / accRearMaxBrake
    return safeLonDis, safeLonDisBrake


# Function to calculate safe lateral distance
def safe_lat_distances(speed_left, speed_right):
    """
    All speed and acceleration inputs must be for the lateral axis
    responseTime: time it takes rear car to react and begin braking
    speedLeft: current velocity of left car
    speedRight: current velocity of right car
    accMaxResp: max acceleration of both cars towards each other during response time
    accMinBrake: min braking of both cars
    """
    # assuming vehicles travel towards each other during response time for boundary condition
    vLeft = speed_left - responseTime * accMaxResp
    vRight = speed_right + responseTime * accMaxResp
    # left vehicle move to left
    if vLeft >= 0:
        # right vehicle move to left
        if vRight >= 0:
            acc_left_min_gap = acc_right_max_gap = accMaxBrake
            acc_left_max_gap = acc_right_min_gap = accMinBrake

        # right vehicle move to right
        else:
            acc_left_min_gap = acc_right_min_gap = accMaxBrake
            acc_left_max_gap = acc_right_max_gap = accMinBrake

    # left vehicle move to right
    else:
        # right vehicle move to left
        if vRight >= 0:
            acc_left_min_gap = acc_right_min_gap = accMinBrake
            acc_left_max_gap = acc_right_max_gap = accMaxBrake

        # right vehicle move to right
        else:
            acc_left_min_gap = acc_right_max_gap = accMinBrake
            acc_left_max_gap = acc_right_min_gap = accMaxBrake
    if vLeft < 0:
        sign_left = -1
    else:
        sign_left = 1

    if vRight < 0:
        sign_right = -1
    else:
        sign_right = 1

    safeLatDis = 0.5 * (speed_left + vLeft) * responseTime + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_min_gap - (
            0.5 * (speed_right + vRight) * responseTime + 0.5 * sign_right * np.power(vRight, 2) / acc_right_min_gap)
    safeLatDisBrake = 0.5 * (speed_left + vLeft) * responseTime + 0.5 * sign_left * np.power(vLeft, 2) / \
                      acc_left_max_gap - (0.5 * (speed_right + vRight) * responseTime + 0.5 * sign_right *
                                          np.power(vRight, 2) / acc_right_max_gap)
    return safeLatDis, safeLatDisBrake


# Function to calculate the longitudinal or lateral risk index [0,1]
def risk_index(safe_distance, safe_distance_brake, distance):
    """
    All inputs must me either longitudinal or lateral safeDistance: safe longitudinal/lateral distance (use
    function SafeLonDistance/SafeLatDistance) safeDistanceBrake: safe longitudinal/lateral distance under max
    braking capacity (use function SafeLonDistance/SafeLatDistance with max braking acceleration) distance: current
    longitudinal/lateral distance between cars
    """
    if safe_distance + distance > 0:
        r = 0
    elif safe_distance_brake + distance <= 0:
        r = 1
    else:
        r = 1 - (safe_distance_brake + distance) / (safe_distance_brake - safe_distance)
    return r


# Function to calculate the unified risk index [0,1]
def risk_index_unified(risk_lon, risk_lat):
    """
    riskLon: longitudinal risk index (use function RiskIndex with longitudinal inputs)
    riskPropLon: longitudinal risk propensity exponent > 0
    riskLat: lateral risk index (use function RiskIndex with lateral inputs)
    riskPropLat: lateral risk propensity exponent > 0
    """
    r = np.power(risk_lon, riskPropLon) * np.power(risk_lat, riskPropLat)
    return r


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
