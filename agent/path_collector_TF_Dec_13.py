import carla
import math
import torch
import time
import os

import numpy as np
import matplotlib.pyplot as plt

 


import cv2 
import time, sys
print('\n\n\n\nRemember to update your path of carla folder....\n\n\n')
time.sleep(1)
sys.path.append('.')
sys.path.append('./carla/PythonAPI')
sys.path.append('./carla/PythonAPI/carla')


import math

import agent.hybrid_A_star_TF_Dec_13 as hybrid_a_star
import matplotlib.pyplot as plt

import open3d as o3d #### for lidar point clouds
from matplotlib import cm

#from agent.dubins_path import calc_dubins_path


VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


W = 2.16  # width of car
LF = 3.8  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
MOVE_STEP = 0.2
WD = 0.7 * W
WB = 2.9
TR = 0.4
TW = 0.8



from data_generation import parking_position
from collections import deque
from agents.tools.misc import get_speed ### carla folder

from functools import reduce
def convex_hull_graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm. 
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l


from scipy.spatial.transform import Rotation as Rot
def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                    fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

def plot_car(x, y, yaw):
    car_color = '-k'
    c, s = math.cos(yaw), math.sin(yaw)
    rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2] #rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)
    plt.plot(x,y,'b*') ##rear center
    plt.plot(car_outline_x, car_outline_y, car_color)

########### controller ##################
## from controller.controller import VehiclePIDController 
class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """
    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, offset, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, offset, **args_lateral)

    def run_step(self, target_speed, waypoint, direction = 1): ### add a direction to enable reverse tracking
        
        SPEED_LIMIT = 5.4

        current_speed = get_speed(self._vehicle)

        current_steering = self._lat_controller.run_step(waypoint, direction)
        
        steer_angle = abs(current_steering)
        steering_multiplier = 1.0 - 0.3 * steer_angle  # Reduce speed up to 70% when turning
        
        # Apply direction-based speed reduction
        # if direction != 1:  # If reversing
        #     steering_multiplier *= 0.4  # Additional 60% reduction for reverse
        
        # Calculate modified target speed based on steering and direction
        modified_target_speed = min(target_speed * steering_multiplier, SPEED_LIMIT)
        
        # If we're close to speed limit, reduce target speed further
        if current_speed > SPEED_LIMIT * 0.9:  # Within 90% of speed limit
            modified_target_speed = min(modified_target_speed, current_speed)
        
        # Longitudinal control with modified target speed
        acceleration = self._lon_controller.run_step(modified_target_speed, direction, waypoint)

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            acc = min(acceleration, self.max_throt) 
            acc = (np.tanh(acc)+1)/2
            control.throttle = acc #min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        ##### Steering regulation: changes cannot happen abruptly, can't steer too much.
        steering_offset = 0.08
        if current_steering > self.past_steering + steering_offset:   # 0.1:
            current_steering = self.past_steering + steering_offset   # 0.1
        elif current_steering < self.past_steering - steering_offset: # 0.1:
            current_steering = self.past_steering - steering_offset   # 0.1    


        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)  

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """
    def __init__(self, vehicle, offset = 0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)
        self._offset = offset

    def run_step(self, target_speed, direction, waypoint, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))
        if direction == 1:
            target_speed = 8
        if direction == -1:
            vehicle_transform = self._vehicle.get_transform() 
            # Get the ego's location and forward vector
            ego_loc = vehicle_transform.location
            v_vec = vehicle_transform.get_forward_vector()
            v_vec = np.array([np.round(v_vec.x,1), np.round(v_vec.y,1), 0.0])

            # Get the vector vehicle-target_wp
            if self._offset != 0:
                # Displace the wp to the side
                w_tran = waypoint.transform
                r_vec = w_tran.get_right_vector()
                w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                            y=self._offset*r_vec.y)
            else:
                w_loc = waypoint.location ##transform.

            w_vec = np.array([np.round(w_loc.x,1) - np.round(ego_loc.x,1), np.round(w_loc.y,1) - np.round(ego_loc.y,1), 0.0])

            # Calculate the angle difference between w_vec and v_loc 
            self.diff = np.degrees(np.arccos((-1) * np.dot(v_vec, w_vec) / (np.linalg.norm(v_vec) * np.linalg.norm(w_vec)))) # -1 * v_vec = ego's backward vector
            print(f"Angle difference (radians): {self.diff}")

            # Limit target speed when diff > threshold, threshold is according to observation  in experience
            if self.diff > 6:
                target_speed = 2.5
            if self.diff > 8:
                target_speed = 1

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """


        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """
    def __init__(self, vehicle, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, direction=1):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform(), direction)
   
    
    def _pid_control(self, waypoint, vehicle_transform, direction = 1):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        if direction == 1: ### forward
            # Get the ego's location and forward vector
            ego_loc = vehicle_transform.location
            v_vec = vehicle_transform.get_forward_vector()
            v_vec = np.array([np.round(v_vec.x,1), np.round(v_vec.y,1), 0.0])

            # Get the vector vehicle-target_wp
            if self._offset != 0:
                # Displace the wp to the side
                w_tran = waypoint.transform
                r_vec = w_tran.get_right_vector()
                w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                            y=self._offset*r_vec.y)
            else:
                w_loc = waypoint.location ##transform.

            w_vec = np.array([np.round(w_loc.x,1) - np.round(ego_loc.x,1), np.round(w_loc.y,1) - np.round(ego_loc.y,1), 0.0])
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec)/(np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)) 

            _cross = np.cross(v_vec, w_vec)  ### check if the angle difference is left or right? u_1*v_2 -u_2*v1
            if _cross[2] < 0: #### means the ego needs to rotate right to catch the target 
                _dot *= -1.0

            self._e_buffer.append(_dot)

            if len(self._e_buffer) >= 2:
                _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
                _ie = sum(self._e_buffer) * self._dt
            else:
                _de = 0.0
                _ie = 0.0

            result = np.tanh((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie)) 
            return result

        else: 
            # Get the ego's location and forward vector
            ego_loc = vehicle_transform.location
            v_vec = vehicle_transform.get_forward_vector()
            v_vec = np.array([np.round(v_vec.x,1), np.round(v_vec.y,1), 0.0])

            # Get the vector vehicle-target_wp
            if self._offset != 0:
                # Displace the wp to the side
                w_tran = waypoint.transform
                r_vec = w_tran.get_right_vector()
                w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                            y=self._offset*r_vec.y)
            else:
                w_loc = waypoint.location ##transform.

            w_vec = np.array([np.round(w_loc.x,1) - np.round(ego_loc.x,1), np.round(w_loc.y,1) - np.round(ego_loc.y,1), 0.0])
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec)/(np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)) 

            _cross = np.cross(v_vec, w_vec)  ### check if the angle difference is left or right? u_1*v_2 -u_2*v1
            if _cross[2] < 0: #### means the ego needs to rotate right to catch the target 
                _dot *= -1.0
            
            self._e_buffer.append(_dot)  ##let's try negative to see if it works

            if len(self._e_buffer) >= 2:
                _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
                _ie = sum(self._e_buffer) * self._dt
            else:
                _de = 0.0
                _ie = 0.0

            #result = np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)  
            result = np.tanh((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie)) 
            return result   


#########################################
def calculate_curvature(p1, p2, p3):
    """
    Calculate the curvature given three points.
 
    Parameters:
        p1, p2, p3: Tuples representing the points (x, y).
 
    Returns:
        curvature: The curvature value (float). If the points are collinear, returns 0.
    """
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
 
    # Calculate the determinant (area of the triangle formed by the points)
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
 
    # Calculate the side lengths of the triangle
    a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
 
    # Avoid division by zero for degenerate triangles
    if a == 0 or b == 0 or c == 0:
        return 0.0
 
    # Semi-perimeter
    s = (a + b + c) / 2
 
    s_a = abs(s - a)
    s_b = abs(s - b)
    s_c = abs(s - c) # Avoid floating calculation error
    area = math.sqrt(s * s_a * s_b * s_c)

    # If the area is zero, the points are collinear, and curvature is zero
    if area == 0:
        return 0.0
 
    # Radius of the circumcircle
    radius = (a * b * c) / (4 * area)
 
    # Curvature is the reciprocal of the radius
    curvature = 1 / radius
 
    return curvature
 

lat_p = 1.5   # 1.0
lat_i = 0.1   # 0.08
lat_d = 0.2   # 0.001

lateral_par = {'K_P': lat_p, 'K_D': lat_d, 'K_I': lat_i, 'dt': 0.03}
longitudinal_par = {'K_P': 1.0, 'K_D': 0.1, 'K_I': 0.1, 'dt':0.03}
max_steering =  0.8

class Real_control:
    def __init__(self, cx, cy, cyaw):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.len = len(self.cx)
        self.s0 = 0

    def nearest_index(self, ego_x, ego_y):
        """
        find the index of the nearest point to current position.
        :param node: current information
        :return: nearest index
        """
        dx = [ego_x - x for x in self.cx]
        dy = [ego_y - y for y in self.cy]
        dist = np.hypot(dx, dy)
        self.s0 += np.argmin(dist[self.s0:self.len]) 
        return min(self.s0+2, self.len-1) ##self.s0 #### do we need to consider the case when s0 >= len  

class Path_collector:
    def __init__(self, data_generator, round):
        self.rgb_rear = None
        self.rgb_right = None
        self.rgb_left = None
        self.rgb_front = None

        self.net_eva = data_generator
        self.world = data_generator.world
        self.player = data_generator.world._player

        ################# bounding box info #####################
        self.bbox = self.player.bounding_box.get_local_vertices() 
        ##########################################################
        self.bbox_min = [self.bbox[0].x, self.bbox[0].y, -1.6+self.bbox[0].z] ### with respect to lidar z
        self.bbox_max = [self.bbox[7].x, self.bbox[7].y, -1.6+self.bbox[7].z]
        
        self.solution_required = True
        if self.net_eva._parking_goal_index <= 31:
            self.target_yaw = -180
        else:
            self.target_yaw = 0    
        self.current_target = [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y, self.target_yaw]

        ##### get target with respect to the Carla world coordinate (vehicle center)
        self.target_x, self.target_y = self.net_eva._parking_goal.x, self.net_eva._parking_goal.y 
        ### shift the center to rear center in world coordinate system
        self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
        self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))

    
        self.path_stage = None ### Breaking the path into multiple segments and using this index to access the segment
        self.current_stage = None
        self.r_trajectory = None
        
        self.prev_dist = None

        self.round = round
        self.exp_name = 'seed_'+str(self.round)+'_'+'target_index_'+str(self.net_eva._parking_goal_index)
        self.soft_start_thre = 0.55
        
         
        self.process_frequency = 1 #3   # process sensor data for every 3 steps 0.1s
        self.step = -1

        self.trans_control = carla.VehicleControl()
      
        self.init_agent()

        plt.ion()


    
    def init_agent(self):
        self.pid_controller = VehiclePIDController(self.player, lateral_par, longitudinal_par, max_throttle=1.0, max_steering=max_steering)

        self.step = -1
   
        self.my_control = carla.VehicleControl()
        self.ego_speed = []
        self.finetuning = False
        self.segment_len = 0
        self.path_stage = None 
        self.current_stage = None
        self.task_idx = -1

    

    def tick(self, carla_world):
        if self.current_target[:2] != [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y] or self.world._need_init_ego_state: ### we also need to consider the initial position change for ego
            print('Parking task changes, hence we need to find a new path')
            if self.net_eva._parking_goal_index <= 31:
                self.target_yaw = -180
            else:
                self.target_yaw = 0    
            self.current_target = [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y, self.target_yaw]
            self.solution_required = True
            ##### get target with respect to the Carla world coordinate (vehicle center)
            self.target_x, self.target_y = self.net_eva._parking_goal.x, self.net_eva._parking_goal.y 
            ### shift the center to rear center in world coordinate system
            self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
            self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))
            self.r_trajectory = None
            self.current_stage = None
            self.exp_name = 'seed_'+str(self.round)+'_'+'target_index_'+str(self.net_eva._parking_goal_index)

            self.world._need_init_ego_state = False
            self.finetuning = False
            self.my_control = carla.VehicleControl()


        self.step += 1

        # stop 1s for new eva
        if self.step < 30: ### 30 frames per second
            self.player.apply_control(carla.VehicleControl())
            #self.net_eva._ego_transform_generator.get_data_gen_ego_transform()
            #self.player.set_transform(self.net_eva.ego_transform)
            return

        if self.step % self.process_frequency == 0:
            data_frame = self.world.sensor_data_frame

            if not data_frame:
                return

            data = self.get_model_data(data_frame)
            try:
                voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(data['point_list'], 0.1) 
            except:
                voxelGrid = None  

            if self.solution_required:   
                #print('ego angle: ', self.player.get_transform().location.x, self.player.get_transform().location.y, self.player.get_transform().rotation.yaw)                                            ## semantic  ## lidar pc
                result, ox_real, oy_real, gx, gy, gyaw = self.simple_solution(voxelGrid)
                # result, ox_real, oy_real, gx, gy, gyaw = self.hybrid_A_star_solution(data['feng'], voxelGrid)
                
                #sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ## ego rear center pos in BEV coordinate system (right hand)

                if result is not None:
                    self.solution_required = False
                    self.path_stage = 0 ### go through the path segments one by one, starts with the first segment
                    self.ego_traject = [] ### to log ego trajectory under the controller
                    self.hist_use = 10000 ### a large number
                   
                    self.start_time = time.time()
                    
                    #### in current ego frame coordinate system
                    x, y, yaw, direction = result

                    x = x[::-1]
                    y = y[::-1]
                    yaw = yaw[::-1]
                    direction = [-i for i in direction[::-1]]

                    ###### convert the path to world coordinate
                    print('Convert the path to world coordinate')
                    self.positions = []
                    for k in range(0, len(x), 1):
                        y_graph = x[k]-16.0  #Rear center in ego coordinate x heading (since this path uses rear center)
                        x_graph = y[k]-16.0  
                        yaw_graph = 90-np.rad2deg(yaw[k])  #np.deg2rad(90)-(yaw[k])  ## right hand to left hand
                        ###first convert this point from ego coordinate to world coordinate, both are left hand
                        y_world = math.cos(self.ab_yaw_init)*y_graph - math.sin(self.ab_yaw_init)*x_graph + self.a_y_init
                        x_world = math.sin(self.ab_yaw_init)*y_graph + math.cos(self.ab_yaw_init)*x_graph + self.b_x_init
                        yaw_world = np.deg2rad(yaw_graph)-self.ab_yaw_init ### notice here self.ab_yaw_init is negative
                        #### Now we only care about rear center
                        ###Get the center in world coordinate
                        # x_world += 1.37*math.cos(yaw_world)
                        # y_world += 1.37*math.sin(yaw_world)
                        #location = carla.Location(x=x_world, y= y_world, z = 0.2)
                        #carla_world.debug.draw_point(location, size=0.02,  color=carla.Color(255,0,0),life_time=30)
                        self.positions.append([x_world, y_world, yaw_world, direction[k]])

                    final_yaw_world = np.deg2rad(self.target_yaw) 
                    self.positions.append([self.net_eva._parking_goal.x-1.37*math.cos(final_yaw_world), self.net_eva._parking_goal.y-1.37*math.sin(final_yaw_world), final_yaw_world, direction[k]])  
                    
                    np_vec = np.array(self.positions)
                    #np.savetxt('rear_path_{}.txt'.format(self.exp_name), np_vec)  ##np.loadtxt('rear_path.txt') ==> a np array

                    oy_real_w, ox_real_w = [], []
                    for ele1, ele2 in zip(ox_real, oy_real):
                        oy_real_w.append(math.cos(self.ab_yaw_init)*(ele1-16) - math.sin(self.ab_yaw_init)*(ele2-16)+ self.a_y_init)
                        ox_real_w.append(math.sin(self.ab_yaw_init)*(ele1-16) + math.cos(self.ab_yaw_init)*(ele2-16)+ self.b_x_init)

                    
                    # plt.cla()
                    # #plt.scatter(self.player.get_transform().location.y, self.player.get_transform().location.x, label='ego center')
                    # plt.plot(oy_real_w, ox_real_w, "sk", markersize=1)
                    # plt.scatter(self.player.get_transform().location.y-1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)), self.player.get_transform().location.x-1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)), label='ego rear start')
                    # plt.scatter(self.target_y, self.target_x, label='target rear')
                    # plt.plot(np_vec[:, 1],np_vec[:,0], label='planned path rear') ### using UE coordinate system
                    # plt.legend()
                    # plt.title('Visulize the planned path in world coordinate for 5 seconds')
                    # plt.show()
                    # plt.pause(5)
                    # plt.close()  

                    # ### Assign position
                    # transform = self.player.get_transform()
                    # transform.location.x = self.net_eva._parking_goal.x
                    # transform.location.y = self.net_eva._parking_goal.y 
                    # #angle = np.random.choice([90, -90])
                    # transform.rotation.yaw = 180 #float(angle)
                    # self.player.set_transform(transform) 
                    # #carla_world.tick()

                    self.mul_pos = []
                    start_dir = self.positions[0][-1]
                    ##### do we also need to add ego?
                    #tmp = [[self.player.get_transform().location.x, self.player.get_transform().location.y, np.rad2deg(self.player.get_transform().rotation.yaw), 0]]
                    tmp = []
                    for i in range(len(self.positions)):
                        if self.positions[i][-1] != start_dir: ### direction changes
                            self.mul_pos.append(tmp)
                            tmp = [self.positions[i]]
                            start_dir = self.positions[i][-1]
                        else:
                            tmp.append(self.positions[i])  
                    self.mul_pos.append(tmp) 
                    print('\nIt should only contain two segments', len(self.mul_pos))
                    
                    self.task_idx += 1


                else:
                    print('\nHybrid A* fails to find a path, soft movement to trigger a new search position...')
                    if self.player.get_transform().rotation.yaw > 0: ## heading right
                        diff_yaw = (90 - self.player.get_transform().rotation.yaw+180)%360 - 180
                    else:
                        diff_yaw = (-90 - self.player.get_transform().rotation.yaw+180)%360 - 180 
                    value_filter = np.clip(diff_yaw*0.1, -1.0, 1.0)       
                    self.player.apply_control(carla.VehicleControl(throttle=self.soft_start_thre, steer= value_filter))
                    self.soft_start_thre *= 0.95
                return    
            else:
                ego_center_current_x = self.player.get_transform().location.x ##np.round(,1)
                ego_center_current_y = self.player.get_transform().location.y 

                ego_rear_x =  ego_center_current_x - 1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)) 
                ego_rear_y =  ego_center_current_y - 1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)) 
                dist =  math.hypot(ego_rear_x - self.mul_pos[self.path_stage][-1][0],   ego_rear_y - self.mul_pos[self.path_stage][-1][1])

                ############################ the following two values are hyperparameters #############################
                if (dist >= self.hist_use and dist <= 0.5) or dist < 0.2 or self.finetuning: #or dist < 0.1
                    v = self.player.get_velocity()
                    feng_speed = math.sqrt(v.x ** 2 + v.y ** 2)
                    # print('feng_speed >= 1e-3 {} and not self.finetuning {}'.format(feng_speed >= 1e-3, not self.finetuning))
                    if feng_speed >= 1e-3 and not self.finetuning: ### first make sure the vehicle stops when finishing to track one segment
                        self.my_control.brake = 1.0 #### stop vehicle moving after reaching the target position
                        self.my_control.throttle = 0.0
                        self.my_control.steer = 0.0
                        self.player.apply_control(self.my_control)
                        #print('one segment is done now brake...')
                        return
                    print('path segment completed and switch to the next path...', ' This is the final segment: ', self.path_stage == len(self.mul_pos)-1)
                    
        
                    if self.path_stage == len(self.mul_pos)-1:
                        print('We completed all paths')
                        
                        #plot trajectory for record
                        traj_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'e2e_parking', 'trajectories')
                        file_path = os.path.join(traj_path, f'task_{self.task_idx}_goal {self.net_eva._parking_goal_index}.png')
                        if not os.path.exists(traj_path):
                            os.makedirs(traj_path) #make directory
                        if not os.path.exists(file_path):
                            np_vec = np.array(self.positions) ##[::2]
                            np_ego = np.array(self.ego_traject)
                            plt.cla()
                            plt.scatter(np_ego[:, -1], np_ego[:, 0], label='ego rear')
                            plt.scatter(np_vec[:, 1], np_vec[:, 0], label='planned path')
                            plt.scatter(self.target_y, self.target_x, label='target rear')
                            try:
                                plt.scatter(self.r_trajectory.cy[waypoint_index], self.r_trajectory.cx[waypoint_index], label='current target')
                            except:
                                pass
                            plt.plot(np_ego[:, 1], np_ego[:, 0], label='ego path')
                            plt.axes().set_xticks(np.arange(int(min(np.round(np_vec[:, 1]))), int(max(np.round(np_vec[:, 1]))), 0.1), minor=True)
                            plt.axes().set_yticks(np.arange(int(min(np.round(np_vec[:, 0]))), int(max(np.round(np_vec[:, 0]))), 0.1), minor=True)
                            plt.grid()
                            plt.grid(which='minor', alpha=0.3)
                            plt.title('p-{} i-{} d-{}-max-steer-{}'.format(lat_p, lat_i, lat_d, max_steering))
                            plt.legend()
                            plt.pause(0.001)
                            
                            plt.savefig(file_path)
                        
                        ##reach_goal = abs(self.player.get_transform().location.x-self.net_eva._parking_goal.x) < 0.4 and abs(self.player.get_transform().location.y-self.net_eva._parking_goal.y) < 0.4
                        reach_goal = np.hypot(self.player.get_transform().location.x-self.net_eva._parking_goal.x, self.player.get_transform().location.y-self.net_eva._parking_goal.y) < 0.5
                        reach_goal = reach_goal and min(abs(self.player.get_transform().rotation.yaw), 180-abs(self.player.get_transform().rotation.yaw)) < 1.0
                        #print(min(abs(self.player.get_transform().rotation.yaw), 180-abs(self.player.get_transform().rotation.yaw)) < 1.0)
                        print('Task done? ', reach_goal)
                        if reach_goal: ###  
                            print('\n\nIt takes {} seconds to finish one experiment'.format(time.time()-self.start_time), ' simulation time: ', self.step/30)
                            # #### trigger a new task
                            # self.net_eva.start_next_parking()
                            self.my_control.brake = 1.0
                            self.my_control.throttle = 0.0
                            self.my_control.steer = 0.0
                            self.my_control.reverse = True #### the evaluator uses reverse as one of its conditions to check if goal is reached or not
                            self.player.apply_control(self.my_control)
                            return
                        else:
                            self.finetuning = True ### requires to finetune the current position though already finishing tracking
                            ### save the ego trajectory
                            rear_x_ego = self.player.get_transform().location.x - 1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)
                            rear_y_ego = self.player.get_transform().location.y - 1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)
                            self.ego_traject.append([rear_x_ego, rear_y_ego])
                            #### assign new segement to track
                            diff_yaw = (self.target_yaw - self.player.get_transform().rotation.yaw+180)%360 - 180
                            if abs(diff_yaw) > 5: #0.5: ### 5 degrees
                                print('finetuning by moving forward to adjust the heading angle')
                                self.net_eva.skip_saving = True
                                value_filter = np.clip(diff_yaw*0.05, -1.0, 1.0) 
                                self.player.apply_control(carla.VehicleControl(throttle=0.2, steer=value_filter))
                                return
                            else: ### backwards for 
                                print('backwards...')
                                self.net_eva.skip_saving = False  # Allow saving
                                self.player.apply_control(carla.VehicleControl(throttle=0.1, steer=-np.clip(diff_yaw*0.2, -1.0, 1.0) , reverse=True))
                                return 
 
                    else: 
                        self.path_stage = min(self.path_stage+1, len(self.mul_pos)-1) ###may not need
                        self.hist_use = 10000
                        ### reinitilize the controll buffer ### which is used in I and D sections
                        #self.pid_controller = VehiclePIDController(self.player, lateral_par, longitudinal_par, max_throttle=1.0, max_steering=max_steering)  
                                 
                else:
                    self.hist_use = dist    

                # import pdb; pdb.set_trace()   
                ###### only for new tracking segments from the derived path.    
                if self.current_stage != self.path_stage:
                    self.current_stage = self.path_stage
                    path_lists = np.array(self.mul_pos[self.path_stage][::2]) #[::2]
                    self.r_trajectory = Real_control(path_lists[:,0], path_lists[:,1], path_lists[:,2])
                    self.segment_len = len(path_lists[:,0])

                #### current path to track
                dx_feng = [ego_rear_x - x for x in np.array(self.mul_pos[self.path_stage])[:,0]]
                dy_feng = [ego_rear_y - y for y in np.array(self.mul_pos[self.path_stage])[:,1]]
                #index_f = np.argmin(np.hypot(dx_feng, dy_feng))
                #print('index_f: ', index_f, np.min(np.hypot(dx_feng, dy_feng)))   
                if np.min(np.hypot(dx_feng, dy_feng)) >= 0.5:
                    print('\n\nEgo deviates from the planned path, replan triggered...') 
                    self.solution_required = True
                    self.my_control.brake = 1.0 #### stop vehicle moving after reaching the target position
                    self.my_control.throttle = 0.0
                    self.my_control.steer = 0.0
                    self.player.apply_control(self.my_control)
                    self.current_stage = None
                    self.r_trajectory = None
                    return    
                     
                waypoint_index = self.r_trajectory.nearest_index(ego_rear_x, ego_rear_y)   #We add extra steps in the nearest_index function

                #### stright uses a smalled gain for lateral_par
                #### use the curvature to determine the PID parameters
                kappa = calculate_curvature([self.r_trajectory.cx[waypoint_index-2], self.r_trajectory.cy[waypoint_index-2]], [self.r_trajectory.cx[waypoint_index-1], self.r_trajectory.cy[waypoint_index-1]], [self.r_trajectory.cx[waypoint_index], self.r_trajectory.cy[waypoint_index]])
                #print('angle diff: ', abs(self.r_trajectory.cyaw[waypoint_index]-self.player.get_transform().rotation.yaw), ' curvature: ', kappa)
                #if abs(self.r_trajectory.cyaw[waypoint_index]-self.player.get_transform().rotation.yaw)<2: #kappa < 0.05: #self.path_stage == 0 or self.path_stage == 2: 
                #print('this is kappa',kappa)
                if abs(kappa) <= 0.01:
                    lateral_par_new = {'K_P': 0.01, 'K_D': 0.01, 'K_I': 0.00, 'dt': 0.03}
                    self.pid_controller = VehiclePIDController(self.player, lateral_par_new, longitudinal_par, max_throttle=1.0, max_steering=max_steering)
                    #print('straight lane control')
                else: ### works for curvatures
                    #print('curvature control')
                    self.pid_controller = VehiclePIDController(self.player, lateral_par, longitudinal_par, max_throttle = 0.3, max_steering=max_steering)    

                
                #print('current pos', str(self.player.get_transform().location), ' dist to end target: ', dist) ## self.r_trajectory.cx[waypoint_index], self.r_trajectory.cy[waypoint_index],
            
                ### convert path point into carla.transform
                location = carla.Location(x=self.r_trajectory.cx[waypoint_index], y= self.r_trajectory.cy[waypoint_index], z=0)
                rotation = carla.Rotation(pitch=0.0, yaw=np.rad2deg(self.r_trajectory.cyaw[waypoint_index]), roll=0.0) 
                waypoint = carla.Transform(location, rotation)

                ###### PID controller expects the unit to be km/h: 2.24m/s*3.6 = 8.064km/h 
                target_speed = 5 #5.4 #8.064 #4 #4.47 #2.24 ### similar to 5 miles/hour #4.2 #0.21 #3
                if waypoint_index > len(self.mul_pos[self.path_stage])-2: ## 6
                    target_speed *= 0.1  #0.2

                 
                if self.mul_pos[self.path_stage][-1][-1] == 1: ### forward  
                    self.my_control = self.pid_controller.run_step(target_speed, waypoint)  
                    self.my_control.gear = 1
                else:
                    self.my_control = self.pid_controller.run_step(target_speed, waypoint, -1)
                    self.my_control.reverse = True  
                           
                if dist <= 0.3:  ### To avoid speeding
                    self.my_control.brake = 0.8 

                #print(dist, waypoint_index, self.segment_len, self.my_control.throttle, self.my_control.steer, self.my_control.brake)
        

        self.player.apply_control(self.my_control)  

        rear_x_ego = self.player.get_transform().location.x - 1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)
        rear_y_ego = self.player.get_transform().location.y - 1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)
        self.ego_traject.append([rear_x_ego, rear_y_ego])

        np_vec = np.array(self.positions) ##[::2]
        np_ego = np.array(self.ego_traject)

        do_plot = False # Set to True if you want to enable plotting
        if do_plot:
            plt.cla()
            plt.scatter(np_ego[:, -1], np_ego[:, 0], label='ego rear')
            plt.scatter(np_vec[:, 1], np_vec[:, 0], label='planned path')
            plt.scatter(self.target_y, self.target_x, label='target rear')
            try:
                plt.scatter(self.r_trajectory.cy[waypoint_index], self.r_trajectory.cx[waypoint_index], label='current target')
            except:
                pass
            plt.plot(np_ego[:, 1], np_ego[:, 0], label='ego path')
            plt.axes().set_xticks(np.arange(int(min(np.round(np_vec[:, 1]))), int(max(np.round(np_vec[:, 1]))), 0.1), minor=True)
            plt.axes().set_yticks(np.arange(int(min(np.round(np_vec[:, 0]))), int(max(np.round(np_vec[:, 0]))), 0.1), minor=True)
            plt.grid()
            plt.grid(which='minor', alpha=0.3)
            plt.title('p-{} i-{} d-{}-max-steer-{}'.format(lat_p, lat_i, lat_d, max_steering))
            plt.legend()
            plt.pause(0.001)



    #################################################################################################
    def simple_solution(self, voxels=None):
        ox = []
        oy = []
        indexes = [v.grid_index for v in voxels.get_voxels()]
        occupied_list = np.array([np.round(voxels.get_voxel_center_coordinate(index), 1) for index in indexes])
        
        ox = (occupied_list[:,0]+16.0).tolist()
        oy = (occupied_list[:,1]+16.0).tolist()

        ox_real = ox.copy()  ## ox, oy can add some extra fake points to limit the solution space
        oy_real = oy.copy()

        # ##### limit the searching space
        # for i in range(6,26): ### add search boundary
        #     ox.append(6)
        #     oy.append(i)
        #     ox.append(26)
        #     oy.append(i)
        # for i in range(6, 26):
        #     ox.append(i)
        #     oy.append(26) 
        #     ox.append(i)
        #     oy.append(6)  
        ##### in graph coordinate right hand,        
        sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ### assume the RB = 1m ==> 2.37-1= 1.37 (16.0-1.37=14.63)
        ##### transform target rear center to current frame coordinate
        self.ab_yaw_init = np.deg2rad(-self.player.get_transform().rotation.yaw) ## clockwise is negative in ab coordinate
        self.b_x_init = self.player.get_transform().location.x
        self.a_y_init = self.player.get_transform().location.y
        
        #### +16 and swap x, y axis
        gyaw = np.deg2rad(90-self.target_yaw)-self.ab_yaw_init  ### in the right hand
        gx = 16 + math.cos(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.sin(self.ab_yaw_init)*(self.target_x- self.b_x_init)  
        gy = 16 - math.sin(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.cos(self.ab_yaw_init)*(self.target_x - self.b_x_init)  

        # max_c = 0.19
        # path_i = calc_dubins_path(sx, sy, syaw, gx, gy, gyaw, max_c)   
        # result = [path_i.x, path_i.y, path_i.yaw, path_i.mode]  ##x, y, yaw, direction    

        # plt.cla()
        # plt.plot(ox, oy, "sk", markersize=1)
        # plot_car(sx, sy, syaw)
        # plot_car(gx, gy, gyaw) 
        # plt.plot(path_i.x, path_i.y, "sb", markersize=1)
        # plt.legend()
        # plt.title('Visulize the input')
        # plt.show()
        # plt.pause(10)

        # path_i = calc_dubins_path(gx, gy, gyaw, sx, sy, syaw, max_c)   
        # plt.cla()
        # plt.plot(ox, oy, "sk", markersize=1)
        # plot_car(sx, sy, syaw)
        # plot_car(gx, gy, gyaw) 
        # plt.plot(path_i.x, path_i.y, "sb", markersize=1)
        # plt.legend()
        # plt.title('Visulize the input')
        # plt.show()
        # plt.pause(10)


        start_time = time.time()  
        print('Starting to find a path...')  

        result = hybrid_a_star.hybrid_astar_planning(gx, gy, gyaw, sx, sy, syaw, ox, oy)
        # plt.cla()
        # plt.plot(ox, oy, "sk", markersize=1)
        # plot_car(sx, sy, syaw)
        # plot_car(gx, gy, gyaw) 
        # plt.plot(result[0], result[1], "sb", markersize=1)
        # plt.legend()
        # plt.title('Visulize the input')
        # plt.show()
        # plt.pause(5)
        # plt.close()


        if result is not None and len(result[0]) <= 200:
            #########################################
            print('It takes {} seconds to find a path'.format((time.time()-start_time)), 'total waypoints: ', len(result[0]))
            return result, ox_real, oy_real, gx, gy, gyaw 
        else:
            print('It takes {} seconds but still cannot find a path'.format((time.time()-start_time)))  
            return None, ox_real, oy_real, gx, gy, gyaw  
    #################################################################################################        





    def hybrid_A_star_solution(self, semantic_img, voxels=None):
        if voxels is None:
            ### crop
            center = semantic_img.shape### 640*640*3
            w, h = 320, 320
            x = center[1]/2 - w/2
            y = center[0]/2 - h/2
            crop_img = semantic_img[int(y):int(y+h), int(x):int(x+w), 0] ###only use Red channel
            ox = []
            oy = []
            ox_real = []
            oy_real = []
            for i in range(6,26): ### add search boundary
                ox.append(6)
                oy.append(i)
                ox.append(26)
                oy.append(i)
            for i in range(6, 26):
                ox.append(i)
                oy.append(26) 
                ox.append(i)
                oy.append(6)    

            cell_size = 0.1
            N, M =  crop_img.shape
            for i in range(N): ## row 
                for j in range(M): ## column
                    if crop_img[i][j] == 81 or crop_img[i][j] == 153 or crop_img[i][j] == 128 or crop_img[i][j] == 50: ### Ground (81, 0, 81), Pole (153, 153, 153), Road (128, 64, 128), Lane (50, 234, 157)
                        continue
                    elif 134<=i<=185 and 148<=j<=171: ###tesla model 3 size range (we canuse https://pixspy.com/ to get) 
                        ## cv2.imwrite('test.png', semantic_img[int(y):int(y+h), int(x):int(x+w), :]) ## H:4.823m, W: 2.226m 
                        continue
                    else:
                        for x_a, y_a in [[-cell_size/2,-cell_size/2], [cell_size/2, cell_size/2], [-cell_size/2, cell_size/2], [cell_size/2, -cell_size/2]]: 
                            x_ = j*cell_size+x_a
                            y_ = (N-i)*cell_size-y_a
                           
                            if 6 < y_ < 26: 
                                ox.append(x_)
                                oy.append(y_)
                                ox_real.append(x_)
                                oy_real.append(y_)

                        #### we might only use the center for simiplification purpose and speed up the path searching        
                        # x_ = j*cell_size
                        # y_ = (N-i)*cell_size ### shift to right-hand x,y coordinate
                        # if 6 < y_ < 26: ### we only care about the local region centered on ego. 6 < x_< 26 and 
                        #     ox.append(x_)
                        #     oy.append(y_)
                        #     ox_real.append(x_)
                        #     oy_real.append(y_)
            
        else:
            print('Using voxel grid')
            ox = []
            oy = []
            indexes = [v.grid_index for v in voxels.get_voxels()]
            occupied_list = np.array([np.round(voxels.get_voxel_center_coordinate(index), 1) for index in indexes])
            
            ox = (occupied_list[:,0]+16.0).tolist()
            oy = (occupied_list[:,1]+16.0).tolist()

            ###### more accurate occ
            # for direction in [[-0.05, -0.05], [-0.05, 0.05], [0.05, -0.05], [0.05, 0.05]]:
            #     ox += (occupied_list[:,0]+16.0+direction[0]).tolist()
            #     oy += (occupied_list[:,1]+16.0+direction[1]).tolist()
 
            ox_real = ox.copy()  ## ox, oy can add some extra fake points to limit the solution space
            oy_real = oy.copy()

            # ##### limit the searching space
            # for i in range(6,26): ### add search boundary
            #     ox.append(6)
            #     oy.append(i)
            #     ox.append(26)
            #     oy.append(i)
            # for i in range(6, 26):
            #     ox.append(i)
            #     oy.append(26) 
            #     ox.append(i)
            #     oy.append(6)  

        print('hybrid_A_star_solution input obstacle length: ', len(ox)) 
        ##### in graph coordinate right hand,        
        sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ### assume the RB = 1m ==> 2.37-1= 1.37 (16.0-1.37=14.63)
        ##### transform target rear center to current frame coordinate
        self.ab_yaw_init = np.deg2rad(-self.player.get_transform().rotation.yaw) ## clockwise is negative in ab coordinate
        self.b_x_init = self.player.get_transform().location.x
        self.a_y_init = self.player.get_transform().location.y
        
        #### +16 and swap x, y axis
        gyaw = np.deg2rad(90-self.target_yaw)-self.ab_yaw_init  ### in the right hand
        gx = 16 + math.cos(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.sin(self.ab_yaw_init)*(self.target_x- self.b_x_init)  
        gy = 16 - math.sin(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.cos(self.ab_yaw_init)*(self.target_x - self.b_x_init)  


        plt.cla()
        plt.plot(ox, oy, "sk", markersize=1)
        plot_car(sx, sy, syaw)
        plot_car(gx, gy, gyaw) 
        plt.legend()
        plt.title('Visulize the input')
        plt.show()
        plt.pause(5000)


        start_time = time.time()  
        print('Starting to find a path...')  

        #######################################################################
        # feng_points = [[i, j] for i, j in zip(ox, oy)]
        # atest = convex_hull_graham(feng_points) 
        # ox, oy = [], []
        # for item in atest:
        #     ox.append(item[0])
        #     oy.append(item[1])
        #######################################################################    

        result = hybrid_a_star.hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy)
        if result is not None:
            #########################################
            if len(result[0]) <= 100:
                print('It takes {} seconds to find a path'.format((time.time()-start_time)), 'total waypoints: ', len(result[0]))
                return result, ox_real, oy_real, gx, gy, gyaw 
            else:
                print('It takes {} seconds but still cannot find an optimal path'.format((time.time()-start_time)))  
                return None, ox_real, oy_real, gx, gy, gyaw 
            ##########################################
        else:
            print('It takes {} seconds but still cannot find a path'.format((time.time()-start_time)))  
            return None, ox_real, oy_real, gx, gy, gyaw  
            ### choose a safe action, say moving forward with a small step 
    #################################################################################################        

    def get_model_data(self, data_frame):
        data = {}
        #########################################################################################################################
        ####### semantic camera always rotate with ego so it is always using ego coordinate system.
        #print('semantic data: ', type(data_frame['BEV_semantic']))
        # data_frame['feng_BEV_semantic'].convert(carla.ColorConverter.CityScapesPalette)
        # feng = np.copy(np.frombuffer(data_frame['feng_BEV_semantic'].raw_data, dtype=np.dtype("uint8")))
        # feng = np.reshape(feng, (data_frame['feng_BEV_semantic'].height, data_frame['feng_BEV_semantic'].width, 4))
        # ### convert BGRA to RGB
        # feng = feng[:, :, :3]
        # feng = feng[:,:, ::-1] ## since B & R channels share the same value, we can save the reverse process
        # data['feng'] = feng  
        # # cv2.imshow('semantic', feng[160:480, 160:480, :]) 
        # # cv2.waitKey(0)
        # # filename = str(time.time()) + '.png'
        # # cv2.imwrite('log/'+filename, feng)
        data['feng'] = None ### I disabled the semantic sensor 
        #########################################################################################################################
        ##### lidar
               
        feng2 = np.copy(np.frombuffer(data_frame['feng_lidar_occ'].raw_data, dtype=np.dtype('f4'))) 
        feng2 = np.reshape(feng2, (int(feng2.shape[0]/4), 4))
        intensity = feng2[:, -1] 
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = feng2[:, :-1] 

        #####filter out ego and z < 0
        mask = np.all((points >= self.bbox_min) & (points <= self.bbox_max), axis=1)
        filtered_points = points[~mask]

        local_lidar_points = filtered_points.T
        local_lidar_points2 = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        world_points = np.dot(data_frame['feng_lidar_occ'].transform.get_matrix(), local_lidar_points2)
        filtered_points = (np.dot(np.array(self.player.get_transform().get_inverse_matrix()), world_points)).T[:,:-1] 

        #print('Point cloud z range: ', min(filtered_points[:,2]), max(filtered_points[:,2]), '\n')
        filtered_points = filtered_points[filtered_points[:,2]>0.2] ###-height ##originall for lidar coordinate system
        filtered_points[:,2] = 0.05  ### flatten all points to the same z

        point_list = o3d.geometry.PointCloud()           ######## change left handed to righ handed
        filtered_points[:, [0,1]] = filtered_points[:, [1,0]]
        point_list.points = o3d.utility.Vector3dVector(filtered_points) ### right handed  ##
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        data['point_list'] = point_list
        ##########################################################################################################################
        return data

     

     
