import carla
import math
import pathlib
import yaml
import torch
import logging
import time
import pygame
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict, deque
from copy import deepcopy

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic, detokenize_waypoint, convert_veh_coord
from data_generation.network_evaluator import NetworkEvaluator
from data_generation.tools import encode_npy_to_pil
from model.parking_model import ParkingModel

try:
    from agent.hybrid_A_star_TF_Dec_13 import hybrid_astar_planning
except ImportError:
    print("Error importing hybrid_A_star_TF_Dec_13. Make sure the cython module is compiled.")
    hybrid_astar_planning = None

from agent.speed_dynamics_model import SpeedDynamicsModel
from agent.path_collector_TF_Dec_13 import VehiclePIDController


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def analyze_and_split_path(path):
    """
    Analyze the path and split it into forward and reverse segments.
    
    Args:
        path: The planned path [x_coords, y_coords]
    
    Returns:
        forward_path: Path segment for forward movement
        reverse_path: Path segment for reverse movement
        split_index: Index where the path changes from forward to reverse
    """
    if path is None or len(path[0]) == 0:
        return None, None, -1
    
    path_x = path[0]
    path_y = path[1]
    path_length = len(path_x)
    
    if path_length < 3:
        # Path too short to analyze, treat as forward only
        return path, None, -1
    
    # Method 1: Find the point where X coordinate starts decreasing significantly
    # This indicates the vehicle needs to start reversing
    split_index = -1
    max_x = -float('inf')
    max_x_index = 0
    
    # Find the maximum X coordinate (furthest forward point)
    for i in range(path_length):
        if path_x[i] > max_x:
            max_x = path_x[i]
            max_x_index = i
    
    # If the maximum X is not at the goal (index 0), we likely have a reverse segment
    if max_x_index > path_length // 4:  # At least 25% of path is forward
        split_index = max_x_index
    else:
        # Method 2: Look for direction changes in the path
        # Calculate the general direction of movement
        forward_count = 0
        reverse_count = 0
        
        for i in range(path_length - 1):
            dx = path_x[i] - path_x[i + 1]  # Remember: index 0 is goal
            if dx > 0.1:  # Moving forward (toward higher X)
                forward_count += 1
            elif dx < -0.1:  # Moving backward (toward lower X)
                reverse_count += 1
        
        # If we have both forward and reverse movements, find the transition
        if forward_count > 0 and reverse_count > 0:
            # Find the transition point
            for i in range(path_length - 5, 0, -1):  # Start from near the beginning
                # Check if this is a transition from forward to reverse
                window_size = min(3, i)
                forward_motion = 0
                reverse_motion = 0
                
                for j in range(i - window_size, i):
                    if j >= 0 and j < path_length - 1:
                        dx = path_x[j] - path_x[j + 1]
                        if dx > 0.1:
                            forward_motion += 1
                        elif dx < -0.1:
                            reverse_motion += 1
                
                if forward_motion > reverse_motion:
                    split_index = i
                    break
    
    # Create path segments
    if split_index > 0:
        # Forward path: from start (high index) to split point
        forward_path = [path_x[split_index:], path_y[split_index:]]
        # Reverse path: from split point to goal (index 0)
        reverse_path = [path_x[:split_index + 1], path_y[:split_index + 1]]
        
        # logging.info(f"Path split at index {split_index}: Forward={len(forward_path[0])} points, Reverse={len(reverse_path[0])} points")
        return forward_path, reverse_path, split_index
    else:
        # No clear split found, treat entire path as forward
        # logging.info("No clear forward/reverse split found, treating entire path as forward")
        return path, None, -1


def get_target_waypoint_from_path(path, vehicle_transform, ego_xy, current_target_index=None, initial_ego_pos=None, direction=1):
    """
    Get the target waypoint from the planned path with monotonic progression.
    Ensures target_index only decreases (moves towards goal at index 0).
    
    Args:
        path: The planned path [x_coords, y_coords]
        vehicle_transform: Current vehicle transform
        ego_xy: Integrated ego position [x, y]
        current_target_index: Current target index (should only decrease)
        initial_ego_pos: Initial position when path was planned
        direction: Current driving direction (1 for forward, -1 for reverse)
    
    Returns:
        waypoint: Target waypoint for PID control
        new_target_index: Updated target index (monotonically decreasing)
    """
    if path is None or len(path[0]) == 0:
        return None, current_target_index
    
    path_x = path[0]  # Forward coordinates in initial ego frame
    path_y = path[1]  # Lateral coordinates in initial ego frame
    path_length = len(path_x)
    
    # Initialize target index if first call
    if current_target_index is None:
        # Start with a lookahead. Instead of starting at the very last point (path_length - 1),
        # which is right under the vehicle, start a few points ahead to encourage forward movement.
        # This prevents the target from appearing behind the vehicle at the very start.
        initial_lookahead = 5
        current_target_index = max(0, path_length - 1 - initial_lookahead)
    
    # Calculate current vehicle REAR position relative to initial position when path was planned
    vehicle_x, vehicle_y = _get_rear_axle_in_initial_ego_frame(vehicle_transform, initial_ego_pos, ego_xy)
    
    # Check if we should advance to next waypoint
    # Calculate distance to current target
    current_target_x = path_x[current_target_index]
    current_target_y = path_y[current_target_index]
    distance_to_current_target = math.sqrt((current_target_x - vehicle_x)**2 + (current_target_y - vehicle_y)**2)
    # logging.info(f"current_target_x: {current_target_x}, current_target_y: {current_target_y}")
    
    # If we're close enough to current target, advance to next waypoint (lower index)
    # Use different thresholds for forward and reverse, since reverse is slower.
    advance_threshold = 2.0 if direction == 1 else 1  # meters
    if distance_to_current_target < advance_threshold and current_target_index > 0:
        # Move towards goal (decrease index)
        current_target_index = max(0, current_target_index - 1)
        # logging.info(f"🎯 Advanced to next waypoint: index {current_target_index}")
    
    # Apply lookahead based on direction: a longer lookahead for faster forward travel,
    # and a shorter lookahead for more precise reverse maneuvers.
    if direction == 1:  # Forward
        lookahead_points = 3  # Look ahead by 3 waypoints
    else:  # Reverse
        lookahead_points = 1  # Shorter lookahead for reverse
    
    target_index = max(0, current_target_index - lookahead_points)
    
    # If we're near the goal, just target the goal directly
    if current_target_index <= 5:
        target_index = 0  # Go directly to goal
    
    # Get target point coordinates
    target_x = path_x[target_index]
    target_y = path_y[target_index]
    
    # Transform target from initial ego frame to current world coordinates
    if initial_ego_pos is not None:
        initial_world_x, initial_world_y, initial_world_yaw = initial_ego_pos
        initial_world_yaw = np.deg2rad(initial_world_yaw)
        
        world_x = initial_world_x + (target_x * np.cos(initial_world_yaw) - target_y * np.sin(initial_world_yaw))
        world_y = initial_world_y + (target_x * np.sin(initial_world_yaw) + target_y * np.cos(initial_world_yaw))
    else:
        # Fallback: use current vehicle transform
        world_x = ego_xy[0] + (target_x * math.cos(math.radians(vehicle_transform.rotation.yaw)) - 
                                                 target_y * math.sin(math.radians(vehicle_transform.rotation.yaw)))
        world_y = ego_xy[1] + (target_x * math.sin(math.radians(vehicle_transform.rotation.yaw)) + 
                                                 target_y * math.cos(math.radians(vehicle_transform.rotation.yaw)))
    
    # Create a waypoint
    waypoint_location = carla.Location(x=world_x, y=world_y, z=vehicle_transform.location.z)
    waypoint_transform = carla.Transform(waypoint_location, vehicle_transform.rotation)
    
    # Create a simple waypoint object
    class SimpleWaypoint:
        def __init__(self, location, transform):
            self.location = location
            self.transform = transform
    
    waypoint = SimpleWaypoint(waypoint_location, waypoint_transform)
    
    # logging.info(f"Vehicle pos: ({vehicle_x:.2f}, {vehicle_y:.2f}), "
    #            f"Current target: {current_target_index},"
    #            f"Distance to current: {distance_to_current_target:.2f}m")
    
    return waypoint, current_target_index


def _get_rear_axle_in_initial_ego_frame(vehicle_transform, initial_ego_pos, ego_xy):
    """
    Calculates the current rear axle position in the initial ego-centric frame of reference.
    This is the single source of truth for the vehicle's position relative to the planned path.
    The previous implementation had a mathematical error in the coordinate transformation,
    resulting in a constant +1.37m error in the forward direction (x-axis). This function corrects that.

    Args:
        vehicle_transform: Current CARLA transform of the vehicle.
        initial_ego_pos: The vehicle's pose [x, y, yaw] when the path was planned.
        ego_xy: Integrated ego position [x, y]
        
    Returns:
        (rear_x_ego, rear_y_ego): Position of the rear axle in the initial frame.
    """
    if initial_ego_pos is None:
        # Fallback if initial position is not set
        return 0.0, 0.0
        
    rear_axle_offset = 1.37

    # Get current vehicle center and yaw in world coordinates
    current_center_x = ego_xy[0]
    current_center_y = ego_xy[1]
    current_world_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
    
    # Get initial vehicle center and yaw in world coordinates
    initial_center_x, initial_center_y, initial_center_yaw_deg = initial_ego_pos
    initial_world_yaw_rad = math.radians(initial_center_yaw_deg)
    
    # 1. Calculate vehicle CENTER displacement in world coordinates
    dx_center_world = current_center_x - initial_center_x
    dy_center_world = current_center_y - initial_center_y
    
    # 2. Transform CENTER displacement to the initial ego frame to get the center's current position
    center_x_ego = dx_center_world * math.cos(initial_world_yaw_rad) + dy_center_world * math.sin(initial_world_yaw_rad)
    center_y_ego = -dx_center_world * math.sin(initial_world_yaw_rad) + dy_center_world * math.cos(initial_world_yaw_rad)
    
    # 3. Calculate current vehicle YAW relative to the initial ego frame's orientation
    yaw_in_ego_frame = current_world_yaw_rad - initial_world_yaw_rad
    
    # 4. Calculate REAR axle position in the initial ego frame by offsetting from the center position
    # This is the correct formulation.
    rear_x_ego = center_x_ego - rear_axle_offset * math.cos(yaw_in_ego_frame)
    rear_y_ego = center_y_ego - rear_axle_offset * math.sin(yaw_in_ego_frame)
    
    return rear_x_ego, rear_y_ego


# convert location in ego centric to world frame
def get_vehicle_rear_position(vehicle_transform, ego_xy):
    """
    Calculate the rear axle position of the vehicle.
    For parking control, we should use the rear axle as the control reference point.
    This matches the implementation in path_collector_TF_Dec_13.py
    
    Args:
        vehicle_transform: CARLA Transform of the vehicle center
        ego_xy: Integrated ego position [x, y]
    
    Returns:
        rear_x, rear_y: World coordinates of the rear axle position
    """
    # Vehicle dimensions - matches path_collector_TF_Dec_13.py
    # Distance from vehicle center to rear axle
    rear_axle_offset = 1.37  # meters (same as in path_collector_TF_Dec_13.py)
    
    # Vehicle center position and orientation
    center_x = ego_xy[0]
    center_y = ego_xy[1]
    yaw_rad = math.radians(vehicle_transform.rotation.yaw)
    
    # Calculate rear axle position - same formula as path_collector_TF_Dec_13.py
    # Move backwards from center by rear_axle_offset
    rear_x = center_x - rear_axle_offset * math.cos(yaw_rad)
    rear_y = center_y - rear_axle_offset * math.sin(yaw_rad)
    
    return rear_x, rear_y


def convert_to_world(delta_x, delta_y, delta_yaw, ego_trans):

    Ex = ego_trans.location.x
    Ey = ego_trans.location.y
    Eyaw = ego_trans.rotation.yaw

    theta = math.radians(Eyaw)


    Wx = Ex + (delta_x * math.cos(theta) - delta_y * math.sin(theta))
    Wy = Ey + (delta_x * math.sin(theta) + delta_y * math.cos(theta))


    Wyaw = Eyaw + delta_yaw
    if Wyaw > 180:
        Wyaw -= 360
    elif Wyaw < -180:
        Wyaw += 360

    return [Wx, Wy, Wyaw]
def show_control_info(window, control, steering_wheel_image, width, height, font):
    histogram_width = 15

    t_x = width - 30
    t_y = height - 50

    b_x = t_x - 30
    b_y = t_y

    s_x = t_x - 80
    s_y = t_y - 40

    r_x = t_x - 140
    r_y = t_y

    # throttle max = 0.5 in data gen
    throttle_height = (control['throttle'] * 200) * 0.8
    throttle_rect = pygame.Rect(t_x, t_y - throttle_height, histogram_width, throttle_height)
    pygame.draw.rect(window, (0, 255, 0), throttle_rect)

    brake_height = (control['brake'] * 100) * 0.8
    brake_rect = pygame.Rect(b_x, b_y - brake_height, histogram_width, brake_height)
    pygame.draw.rect(window, (255, 0, 0), brake_rect)

    steer = -control['steer'] * 90
    rotated_steering_wheel = pygame.transform.rotate(steering_wheel_image, steer)
    rotated_rect = rotated_steering_wheel.get_rect(center=(s_x, s_y))
    window.blit(rotated_steering_wheel, rotated_rect)

    reverse = bool(control['reverse'])
    rect = pygame.Rect((r_x, r_y - 10), (10, 10))
    pygame.draw.rect(window, (0, 0, 0), rect, 0 if reverse else 1)

    # show text
    throttle_text = font.render("T", True, (0, 255, 0))
    brake_text = font.render("B", True, (255, 0, 0))
    steer_text = font.render("S", True, (0, 0, 0))
    reverse_text = font.render("R", True, (0, 0, 0))

    window.blit(throttle_text, (t_x + 2, t_y + 10))
    window.blit(brake_text, (b_x + 2, b_y + 10))
    window.blit(steer_text, (s_x - 4, s_y + 50))
    window.blit(reverse_text, (r_x, r_y + 10))


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def grid_show(to_shows, cols):
    it = iter(to_shows)
    fig, axs = plt.subplots(1, cols, figsize=(cols * 2, cols))
    for j in range(cols):
        try:
            image, title = next(it)
        except StopIteration:
            image = np.zeros_like(to_shows[0][0])
            title = 'pad'
        axs[j].imshow(image)
        axs[j].set_title(title)
        axs[j].set_yticks([])
        axs[j].set_xticks([])
    plt.show()


def visualize_heads(att_map):
    to_shows = []
    att_map = att_map.squeeze()
    cols = att_map.shape[0] + 1
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def get_atten_avg_map(att_map, grid_index, image, grid_size=16):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    att_map = att_map.squeeze()
    average_att_map = att_map.mean(axis=0)
    atten_avg = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    atten_avg = Image.fromarray(atten_avg.numpy()).resize(image.size)
    return grid_image, atten_avg


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=16, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        mask = att_map[i][grid_index].reshape(grid_size[0], grid_size[1])
        mask = Image.fromarray(mask.numpy()).resize(image.size)
        to_shows.append((mask, f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    average_mask = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    average_mask = Image.fromarray(average_mask.numpy()).resize(image.size)
    to_shows.append((average_mask, 'Head Average'))

    plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
    rows = 1
    cols = 7

    it = iter(to_shows)
    for j in range(cols):
        try:
            mask, title = next(it)
        except StopIteration:
            mask = np.zeros_like(to_shows[0][0])
            title = 'pad'
        ax_attem = plt.subplot(rows, cols, j + 1)
        ax_attem.axis('off')
        ax_attem.set_title(title, fontsize=10)
        ax_attem.imshow(grid_image)
        ax_attem.imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')

    plt.pause(0.1)
    plt.clf()


class ParkingAgent:
    def __init__(self, network_evaluator: NetworkEvaluator, args):

        self.show_eva_imgs = args.show_eva_imgs

        self.atten_avg = None
        self.grid_image = None
        self.point_cloud = None
        self.path = None

        self.camera_front = None
        self.camera_front_left = None
        self.camera_front_right = None
        self.camera_back = None
        self.camera_back_left = None
        self.camera_back_right = None
        self.seg_bev = None
        self.target_bev = None

        self.pre_target_point = None

        self.model = None
        self.device = None
        self.speed_dynamics_model = None

        self.cfg = Configuration()
        self.load_cfg(args)

        self.log_path = pathlib.Path(self.cfg.log_dir)
        if not self.log_path.exists():
            self.log_path.mkdir()

        self.BOS_token = self.cfg.token_nums - 3

        self.hist_frame_nums = self.cfg.hist_frame_nums

        self.net_eva = network_evaluator
        self.world = network_evaluator.world
        self.player = network_evaluator.world.player

        self.is_init = False
        self.intrinsic_crop = None
        self.extrinsic = None
        self.image_process = None
        self.semantic_process = ProcessSemantic(self.cfg)

        self.process_frequency = 3  # process sensor data for every 3 steps 0.1s
        self.step = -1

        self.prev_xy_thea = None

        self.trans_control = carla.VehicleControl()
        self.gru_control = carla.VehicleControl()

        self.save_output = SaveOutput()
        self.hook_handle = None
        self.load_model(args.model_path)

        if hasattr(args, 'speed_model_path') and args.speed_model_path:
            try:
                self.speed_dynamics_model = SpeedDynamicsModel(model_path=args.speed_model_path, device=self.device)
                logging.info("Speed dynamics model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load speed dynamics model: {e}")
                self.speed_dynamics_model = None

        self.stop_count = 0
        self.boost = False
        self.boot_step = 0

        self.relative_target = [0,0]
        self.ego_xy = []
        self.ego_xy_dynamic = []
        self.parking_goal_world = None

        # Add flag to track if path planning window has been created
        self.path_window_created = False
        
        # Track ego movement for path visualization
        self.initial_ego_pos = None  # Store initial position when path is planned
        self.accumulated_distance = 0.0  # Track distance traveled
        
        # Store fixed goal position for visualization (once path is planned)
        self.fixed_goal_ego = None

        # Initialize PID controller
        self.pid_controller = None
        self.current_target_index = None  # Track current target waypoint index
        self.continuous_forward_mode = False  # Enable continuous forward when target too far
        
        # Path segmentation for forward/reverse phases
        self.forward_path = None  # Forward path segment
        self.reverse_path = None  # Reverse path segment
        self.current_phase = 'forward'  # Current execution phase: 'forward' or 'reverse'
        self.forward_completed = False  # Flag to track if forward phase is completed
        
        # Timestep counter for measurements saving
        self.timestep_counter = 0
        
        self.init_agent()
        # self.visualize = True

        plt.ion()

    def load_cfg(self, args):

        with open(args.model_config_path, 'r') as config_file:
            try:
                cfg_yaml = (yaml.safe_load(config_file))
            except yaml.YAMLError:
                logging.exception('Invalid YAML Config file {}', args.config)
        self.cfg = get_cfg(cfg_yaml)

    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModel(self.cfg)
        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        patch_attention(self.model.feature_fusion.tf_encoder.layers[-1].self_attn)
        self.hook_handle = self.model.feature_fusion.tf_encoder.layers[-1].self_attn.register_forward_hook(
            self.save_output)

        logging.info('Load E2EParkingModel from %s', parking_pth_path)

    def save_seg_img(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]
        # image_file = pathlib.Path(self.cfg.log_dir) / ('%04d.png' % self.step)
        # Image.fromarray(np.uint8(pred_seg_img), mode='L').save(image_file)
        self.seg_bev = pred_seg_img

    def plot_hybrid_astar_path(self, goal_ego, path_result, current_ego_pos=None):
        """
        Plots the hybrid A* planning result in a separate figure.
        Shows: planned path, point cloud obstacles, ego positions, target points, 
        path segments, and current target waypoint.
        current_ego_pos: [x, y, yaw] current position of ego vehicle in egocentric coordinates
        """
        # Create or switch to the specific figure for path planning
        fig = plt.figure("Hybrid A* Path", figsize=(12, 10))
        plt.figure(fig.number)  # Make sure we're working on this specific figure
        plt.clf()

        # Plot obstacles from the point cloud
        if self.point_cloud is not None and len(self.point_cloud) > 0:
            # y-axis is lateral, x-axis is longitudinal
            plt.scatter(self.point_cloud[:, 1], self.point_cloud[:, 0], s=10, c='k', marker='.', label="Obstacles")

        # Plot ego positions (current, center, and start positions)
        rear_axle_offset = 1.37  # meters (same as path_collector_TF_Dec_13.py)
        
        if current_ego_pos is not None:
            ego_x, ego_y, ego_yaw = current_ego_pos
            # Plot current ego center position
            plt.plot(ego_y, ego_x, 'ko', markersize=8, label='Ego Center (Current)', alpha=0.7)
            
            # Calculate and plot rear axle position in ego frame
            rear_ego_x = ego_x - rear_axle_offset * np.cos(ego_yaw)
            rear_ego_y = ego_y - rear_axle_offset * np.sin(ego_yaw)
            plt.plot(rear_ego_y, rear_ego_x, 'ro', markersize=8, label='Ego Rear (Current)')
            
            # Arrow pointing in the current heading direction from rear axle
            arrow_length = 1.0
            arrow_dx = arrow_length * np.sin(ego_yaw)  # lateral component
            arrow_dy = arrow_length * np.cos(ego_yaw)  # longitudinal component
            plt.arrow(rear_ego_y, rear_ego_x, arrow_dx, arrow_dy, head_width=0.3, head_length=0.3, fc='r', ec='r')
            
            # Also plot the original starting positions
            plt.plot(0, 0, 'co', markersize=6, label='Start Center', alpha=0.7)
            plt.plot(0, -rear_axle_offset, 'mo', markersize=6, label='Start Rear (Path Start)', alpha=0.7)
        else:
            # Default: plot at origin
            plt.plot(0, 0, 'ko', markersize=8, label='Ego Center', alpha=0.7)
            plt.plot(0, -rear_axle_offset, 'ro', markersize=8, label='Ego Rear (Start)')
            # Arrow pointing forward from rear axle
            plt.arrow(0, -rear_axle_offset, 0, 1.0, head_width=0.3, head_length=0.3, fc='r', ec='r')

        # Plot the target goal (use fixed goal if available)
        if self.fixed_goal_ego is not None:
            plt.plot(self.fixed_goal_ego[1], self.fixed_goal_ego[0], 'g*', markersize=12, label='Goal (Fixed)')
        else:
            plt.plot(goal_ego[1], goal_ego[0], 'g*', markersize=12, label='Goal')

        # Plot the planned path segments if they exist
        if path_result:
            # The planner returns path with x and y attributes
            path_x = path_result[0]
            path_y = path_result[1]
            
            # Plot the complete path as a thin gray line for reference
            plt.plot(path_y, path_x, 'gray', linewidth=1, alpha=0.3, label='Complete Path')
            
            # Plot forward path segment
            if hasattr(self, 'forward_path') and self.forward_path is not None:
                forward_x = self.forward_path[0]
                forward_y = self.forward_path[1]
                plt.plot(forward_y, forward_x, 'g-', linewidth=3, label='Forward Path', alpha=0.8)
                plt.scatter(forward_y, forward_x, s=30, c='green', marker='o', alpha=0.6, 
                           label=f'Forward Points ({len(forward_x)})', zorder=5)
            
            # Plot reverse path segment
            if hasattr(self, 'reverse_path') and self.reverse_path is not None:
                reverse_x = self.reverse_path[0]
                reverse_y = self.reverse_path[1]
                plt.plot(reverse_y, reverse_x, 'r-', linewidth=3, label='Reverse Path', alpha=0.8)
                plt.scatter(reverse_y, reverse_x, s=30, c='red', marker='s', alpha=0.6, 
                           label=f'Reverse Points ({len(reverse_x)})', zorder=5)
            
            # Highlight current execution phase
            if hasattr(self, 'current_phase'):
                phase_text = f"Current Phase: {self.current_phase.upper()}"
                if hasattr(self, 'forward_completed') and self.forward_completed:
                    phase_text += " (Forward Completed)"
                plt.text(0.02, 0.98, phase_text, transform=plt.gca().transAxes, 
                        fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                        verticalalignment='top')
            
            # Highlight current target waypoint if available
            if hasattr(self, 'current_target_index') and self.current_target_index is not None:
                # Get current path based on phase
                current_path_x, current_path_y = None, None
                if hasattr(self, 'current_phase'):
                    if self.current_phase == 'forward' and hasattr(self, 'forward_path') and self.forward_path is not None:
                        current_path_x = self.forward_path[0]
                        current_path_y = self.forward_path[1]
                    elif self.current_phase == 'reverse' and hasattr(self, 'reverse_path') and self.reverse_path is not None:
                        current_path_x = self.reverse_path[0]
                        current_path_y = self.reverse_path[1]
                
                if current_path_x is not None and current_path_y is not None:
                    # Show current target waypoint
                    current_idx = self.current_target_index
                    if current_idx < len(current_path_x):
                        plt.plot(current_path_y[current_idx], current_path_x[current_idx], 'mo', markersize=12, 
                                label=f'Current Target ({current_idx})', alpha=0.9, zorder=10)
            
            #logging.info(f"Path plotted with {len(path_x)} total waypoints")

        plt.xlabel('Lateral (Right) [m]')
        plt.ylabel('Longitudinal (Forward) [m]')
        plt.title('Hybrid A* Path Planning - Detailed View')
        plt.axis("equal")
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=8)
        
        # Keep the window persistent
        if not self.path_window_created:
            self.path_window_created = True
            plt.show(block=False)  # Show window without blocking
        else:
            plt.draw()  # Just update the existing window
        
        plt.pause(0.1)

    def save_target_bev_img(self, target_bev):
        target_bev = target_bev[0]
        target_bev = target_bev.detach().cpu().numpy()
        target_bev[target_bev == 1] = 255
        target_bev_img = target_bev[0, :, :][::-1]
        self.target_bev = target_bev_img

    def save_prev_target(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]

        h, w = pred_seg_img.shape
        target_slot_x = []
        target_slot_y = []
        for row_idx in range(h):
            for col_idx in range(w):
                if pred_seg_img[row_idx, col_idx] == 255:
                    target_slot_x.append(row_idx)
                    target_slot_y.append(col_idx)

        # target point in bev
        if (len(target_slot_x) > 0) and (len(target_slot_y) > 0):
            new_target_x = int(np.average(target_slot_x))
            new_target_y = int(np.average(target_slot_y))
            self.pre_target_point = self.get_target_point_ego_coord(pred_seg_img, [new_target_x, new_target_y])

    def generate_point_cloud_from_segmentation(self, pred_segmentation):
        """
        Generates a 2D point cloud of obstacles from a predicted segmentation map.
        Obstacles are considered to be class 1 (vehicles).
        The coordinates are in the egocentric frame (x: forward, y: right).
        """
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=False)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()

        # Class 1 is vehicle/obstacle
        obstacle_pixels = np.argwhere(pred_segmentation == 1)

        bev_h, bev_w = pred_segmentation.shape

        point_cloud = []
        # The coordinates are (row, col)
        # row -> x, col -> y
        for p in obstacle_pixels:
            row, col = p
            # Convert pixel coordinates to egocentric coordinates
            # This conversion is based on `get_target_point_ego_coord`
            # The y-axis (lateral) is correct.
            # The x-axis (forward) was incorrect. Based on the BEV representation,
            # a larger row index corresponds to a larger forward distance.
            # The car is at the bottom-center of the BEV map.
            ego_x = (row - bev_h / 2.0) * self.cfg.bev_x_bound[2]
            ego_y = (col - bev_w / 2.0) * self.cfg.bev_y_bound[2]
            point_cloud.append([ego_x, ego_y])

        return np.array(point_cloud)

    def get_target_point_ego_coord(self, pred_seg_img, target_point_pixel_idx):
        bev_shape = pred_seg_img.shape[0]
        x = -(target_point_pixel_idx[0] - bev_shape / 2)
        y = target_point_pixel_idx[1] - bev_shape / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2]]
        return target_point_ego_coord

    def init_agent(self):
        # Reset timestep counter for new task
        self.timestep_counter = 0
        
        w = self.world.cam_config['width']
        h = self.world.cam_config['height']

        self.intrinsic_crop = update_intrinsics(
            torch.from_numpy(self.world.intrinsic).float(),
            (h - self.cfg.image_crop) / 2,
            (w - self.cfg.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic_crop = self.intrinsic_crop.unsqueeze(0).expand(6, 3, 3)

        veh2cam_dict = self.world.veh2cam_dict
        front_to_ego = torch.from_numpy(veh2cam_dict['camera_front']).float().unsqueeze(0)
        front_left_to_ego = torch.from_numpy(veh2cam_dict['camera_front_left']).float().unsqueeze(0)
        front_right_to_ego = torch.from_numpy(veh2cam_dict['camera_front_right']).float().unsqueeze(0)
        back_to_ego = torch.from_numpy(veh2cam_dict['camera_back']).float().unsqueeze(0)
        back_left_to_ego = torch.from_numpy(veh2cam_dict['camera_back_left']).float().unsqueeze(0)
        back_right_to_ego = torch.from_numpy(veh2cam_dict['camera_back_right']).float().unsqueeze(0)

        self.extrinsic = torch.cat([front_to_ego, front_left_to_ego, front_right_to_ego, back_to_ego,back_left_to_ego,back_right_to_ego], dim=0)

        self.image_process = ProcessImage(self.cfg.image_crop)

        self.step = -1
        self.pre_target_point = None
        self.ego_xy=[]
        self.ego_xy_dynamic=[]
        self.path_window_created = False  # Reset window flag on agent initialization
        self.initial_ego_pos = None  # Reset initial position
        self.accumulated_distance = 0.0  # Reset distance tracking
        self.current_target_index = None  # Reset target index tracking
        self._last_direction = 1  # Initialize direction to forward
        self.continuous_forward_mode = False  # Reset continuous forward mode
        
        # Clear previous path and related data for new task
        self.path = None  # Force new path planning
        self.point_cloud = None  # Clear previous obstacle data
        self.fixed_goal_ego = None  # Clear previous goal visualization
        
        # Reset path segmentation variables
        self.forward_path = None
        self.reverse_path = None
        self.current_phase = 'forward'
        self.forward_completed = False
        
        #logging.info("Agent reinitialized - all path data cleared for new task")

        args_lateral = {'K_P': 1.95, 'K_D': 0.01, 'K_I': 1.4, 'dt': 0.03}
        args_longitudinal = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
        self.pid_controller = VehiclePIDController(
            self.player, 
            args_lateral, 
            args_longitudinal, 
            offset=0, 
            max_throttle=0.75,  # Use same as path_collector
            max_brake=0.3,      # Use same as path_collector
            max_steering=0.6    
        )

    def save_atten_avg_map(self, data):
        atten = self.save_output.outputs[0].detach().cpu()
        # visualize_heads(atten)

        bev = data['segmentation']
        bev = bev.convert("RGB")
        # visualize_grid_to_grid(atten, 136, bev)
        grid_image, atten_avg = get_atten_avg_map(atten, 136, bev)
        grid_image = np.asarray(grid_image)[::-1, ...]
        atten_avg = np.asarray(atten_avg)[::-1, ...]
        return grid_image, atten_avg

    def _save_timestep_measurement(self, measurements_file):
        """Save single measurement to timestep-specific JSON file organized by epoch, task, and trial"""
        # Get current task info from network evaluator
        eva_result_path = self.net_eva._eva_result_path
        current_epoch = self.net_eva._eva_epoch_idx + 1
        current_task = self.net_eva._eva_task_idx
        current_trial = self.net_eva._eva_parking_idx + 1  # Convert 0-based to 1-based for readability
        
        # Create organized folder structure: epoch{epoch}/task{task}/trial{trial}/
        epoch_dir = eva_result_path / f"epoch{current_epoch}"
        task_dir = epoch_dir / f"task{current_task}"
        trial_dir = task_dir / f"trial{current_trial}"
        
        # Create directories if they don't exist
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: measurements_timestep{timestep}.json
        filename = f"measurements_timestep{self.timestep_counter}.json"
        filepath = trial_dir / filename
        
        # Save single measurement to JSON
        try:
            with open(filepath, 'w') as f:
                json.dump(measurements_file, f, indent=2, default=str)
            # logging.info(f"Saved measurement timestep {self.timestep_counter} to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save measurement to {filepath}: {e}")

    def tick(self):
        if self.net_eva.agent_need_init:
            self.init_agent()
            self.net_eva.agent_need_init = False

        self.step += 1

        # stop 1s for new eva
        if self.step < 30:
            self.player.apply_control(carla.VehicleControl())
            self.player.set_transform(self.net_eva.ego_transform)
            return

        if self.step % self.process_frequency == 0:
            data_frame = self.world.sensor_data_frame

            if not data_frame:
                return

            vehicle_transform = data_frame['veh_transfrom']
            imu_data = data_frame['imu']
            gnss_data = data_frame['gnss']
            vehicle_velocity = data_frame['veh_velocity']
            vehicle_control = data_frame['veh_control']

            measurements_file = {
            'x': vehicle_transform.location.x,
            'y': vehicle_transform.location.y,
            'z': vehicle_transform.location.z,
            'pitch': vehicle_transform.rotation.pitch,
            'yaw': vehicle_transform.rotation.yaw,
            'roll': vehicle_transform.rotation.roll,
            'speed': (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)),
            'speed_x':3.6 * vehicle_velocity.x,
            'speed_y':3.6 * vehicle_velocity.y,
            'speed_z':3.6 * vehicle_velocity.z,
            'Throttle': vehicle_control.throttle,
            'Steer': vehicle_control.steer,
            'Brake': vehicle_control.brake,
            'Reverse': vehicle_control.reverse,
            'Hand brake': vehicle_control.hand_brake,
            'Manual': vehicle_control.manual_gear_shift,
            'Gear': {-1: 'R', 0: 'N'}.get(vehicle_control.gear, vehicle_control.gear),
            'acc_x': imu_data.accelerometer.x,
            'acc_y': imu_data.accelerometer.y,
            'acc_z': imu_data.accelerometer.z,
            'gyr_x': imu_data.gyroscope.x,
            'gyr_y': imu_data.gyroscope.y,
            'gyr_z': imu_data.gyroscope.z,
            'compass': imu_data.compass,
            'lat': gnss_data.latitude,
            'lon': gnss_data.longitude
            }

            # Save measurements_file immediately and increment timestep counter
            #self._save_timestep_measurement(measurements_file)
            self.timestep_counter += 1

            data = self.get_model_data(data_frame)

            self.model.eval()
            with torch.no_grad():
                start_time = time.time()

                pred_controls, pred_waypoints, pred_segmentation, _, target_bev = self.model.predict(data)

                if self.path is None:
                    # 1. Generate obstacle point cloud from segmentation
                    self.point_cloud = self.generate_point_cloud_from_segmentation(pred_segmentation)
                    #logging.info(f"Generated point cloud with {len(self.point_cloud)} points on step {self.step}.")

                    if hybrid_astar_planning is not None:
                        # 2. Define start and end states for the planner
                        # Use rear axle position as start point (consistent with control reference)
                        rear_axle_offset = 1.37  # Same as in get_vehicle_rear_position
                        sx = -rear_axle_offset  # Rear axle is behind vehicle center in ego frame
                        sy = 0.0  # No lateral offset
                        syaw = 0.0  # Same orientation as vehicle

                        # The goal for the planner should be the desired location of the vehicle's REAR AXLE,
                        # not the vehicle's center, to be consistent with the start point (sx, sy) and the PID controller.

                        #Get the world coordinates of the goal (vehicle center).
                        goal_center_x = self.net_eva.eva_parking_goal[0]
                        goal_center_y = self.net_eva.eva_parking_goal[1]
                        goal_yaw_deg = self.net_eva.eva_parking_goal[2]
                        goal_yaw_rad = np.deg2rad(goal_yaw_deg)

                        # Calculate the desired world coordinates of the rear axle at the goal position.
                        rear_goal_world_x = goal_center_x - rear_axle_offset * math.cos(goal_yaw_rad)
                        rear_goal_world_y = goal_center_y - rear_axle_offset * math.sin(goal_yaw_rad)

                        #Convert these world coordinates to the vehicle's current ego-centric frame.
                        current_x = data["ego_xy_dynamic"][0]
                        current_y = data["ego_xy_dynamic"][1]
                        current_yaw_rad = np.deg2rad(vehicle_transform.rotation.yaw)

                        dx = rear_goal_world_x - current_x
                        dy = rear_goal_world_y - current_y

                        ex = dx * math.cos(current_yaw_rad) + dy * math.sin(current_yaw_rad)
                        ey = -dx * math.sin(current_yaw_rad) + dy * math.cos(current_yaw_rad)

                        #The goal yaw remains the relative difference in orientation.
                        ego_yaw_deg = goal_yaw_deg - vehicle_transform.rotation.yaw
                        if ego_yaw_deg > 180:
                            ego_yaw_deg -= 360
                        if ego_yaw_deg < -180:
                            ego_yaw_deg += 360
                        
                        eyaw = np.deg2rad(ego_yaw_deg)

                        # For visualization, use dynamic model position for consistency
                        vis_dynamic_transform = carla.Transform(
                            carla.Location(x=data["ego_xy_dynamic"][0], y=data["ego_xy_dynamic"][1], z=vehicle_transform.location.z),
                            vehicle_transform.rotation
                        )
                        goal_ego_for_vis = convert_slot_coord(vis_dynamic_transform, self.net_eva.eva_parking_goal)

                        # Get obstacle coordinates. The point cloud is already in the egocentric frame.
                        if self.point_cloud is not None and len(self.point_cloud) > 0:
                           ox = (self.point_cloud[:, 0]).tolist()
                           oy = (self.point_cloud[:, 1]).tolist()
                        else:
                           ox = []
                           oy = []
                        path_result = hybrid_astar_planning(ex, ey, eyaw, sx, sy, syaw, ox, oy)

                        if path_result:
                            # Save initial position when path is first planned (always use current position)
                            self.initial_ego_pos = [data["ego_xy_dynamic"][0], data["ego_xy_dynamic"][1], vehicle_transform.rotation.yaw]
                            # Save fixed goal position for visualization (use center-based goal for plotting)
                            self.fixed_goal_ego = goal_ego_for_vis.copy()
                               
                            # Check path direction - verify start and end points
                            path_x = path_result[0]
                            path_y = path_result[1]
                            
                            if len(path_x) > 0:
                                path_start_x, path_start_y = path_x[0], path_y[0]
                                path_end_x, path_end_y = path_x[-1], path_y[-1]
                                
                                
                                # Check if path start matches our intended start
                                start_error = math.sqrt((path_start_x - sx)**2 + (path_start_y - sy)**2)
                                end_error = math.sqrt((path_end_x - ex)**2 + (path_end_y - ey)**2)
                                
                                # With corrected parameter order, path should be correct now
                                if start_error < 1.0 and end_error < 1.0:
                                    self.path = path_result
                                else:
                                    self.path = path_result  # Use it anyway
                            else:
                                self.path = path_result
                            
                            # Analyze and split path into forward and reverse segments
                            self.forward_path, self.reverse_path, split_index = analyze_and_split_path(self.path)
                            
                            # Initialize phase tracking
                            if len(self.forward_path[0]) > 4:
                                self.current_phase = 'forward'
                                self.forward_completed = False
                                self.current_target_index = None  # Reset target index
                            else:
                                self.current_phase = 'reverse'
                                self.forward_completed = True
                                self.current_target_index = None  # Reset target index

                            # Log path segmentation info
                            if self.forward_path is not None:
                                # logging.info(f"🟢 Forward path: {len(self.forward_path[0])} waypoints")
                                pass
                            if self.reverse_path is not None:
                                # logging.info(f"🔴 Reverse path: {len(self.reverse_path[0])} waypoints")
                                pass
                            else:
                                logging.info("🟢 Forward-only path (no reverse segment)")

                        # Call the plotting function to visualize the result
                        if self.show_eva_imgs:
                            self.plot_hybrid_astar_path(goal_ego_for_vis, path_result)
                
                # Update path visualization with current ego position even if path already exists
                if self.path is not None and self.show_eva_imgs and self.initial_ego_pos is not None:
                    # Calculate current position relative to initial position
                    current_world_x = data["ego_xy_dynamic"][0]
                    current_world_y = data["ego_xy_dynamic"][1]
                    current_world_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
                    
                    initial_world_x, initial_world_y, initial_world_yaw = self.initial_ego_pos
                    initial_world_yaw = np.deg2rad(initial_world_yaw)
                    
                    # Calculate displacement in egocentric coordinates (relative to initial position)
                    dx_world = current_world_x - initial_world_x
                    dy_world = current_world_y - initial_world_y
                    
                    # Transform to initial ego frame
                    ego_x = dx_world * np.cos(initial_world_yaw) + dy_world * np.sin(initial_world_yaw)
                    ego_y = -dx_world * np.sin(initial_world_yaw) + dy_world * np.cos(initial_world_yaw)
                    ego_yaw = current_world_yaw - initial_world_yaw
                    
                    # Normalize yaw angle
                    while ego_yaw > np.pi:
                        ego_yaw -= 2 * np.pi
                    while ego_yaw < -np.pi:
                        ego_yaw += 2 * np.pi
                    
                    current_ego_pos = [ego_x, ego_y, ego_yaw]
                    # Use dynamic model position for visualization consistency
                    vis_dynamic_transform = carla.Transform(
                        carla.Location(x=current_world_x, y=current_world_y, z=vehicle_transform.location.z),
                        vehicle_transform.rotation
                    )
                    goal_ego = convert_slot_coord(vis_dynamic_transform, self.net_eva.eva_parking_goal)
                    self.plot_hybrid_astar_path(goal_ego, self.path, current_ego_pos)

                end_time = time.time()
                self.net_eva.inference_time.append(end_time - start_time)

                if self.show_eva_imgs:
                    self.save_prev_target(pred_segmentation)
                
                # Check if we should use continuous forward mode
                if self.continuous_forward_mode:
                    # Continuous forward movement when target is too far
                    self.trans_control.throttle = 0.3  # Moderate throttle
                    self.trans_control.brake = 0.0
                    self.trans_control.steer = 0.0  # Go straight
                    self.trans_control.reverse = False
                    
                    # Check if we're getting closer to target - if so, try planning again
                    # Use dynamic model position for consistency
                    forward_dynamic_transform = carla.Transform(
                        carla.Location(x=data["ego_xy_dynamic"][0], y=data["ego_xy_dynamic"][1], z=vehicle_transform.location.z),
                        vehicle_transform.rotation
                    )
                    goal_ego = convert_slot_coord(forward_dynamic_transform, self.net_eva.eva_parking_goal)
                    current_distance = math.sqrt(goal_ego[0]**2 + goal_ego[1]**2)
                    
                    if current_distance < 12.0:  # If we're now closer, try planning again
                        self.continuous_forward_mode = False
                        self.path = None  # Force replanning on next tick
                        self.initial_ego_pos = None  # Clear old initial position to use current position for new planning
                    
                
                # Use PID control with planned path segments
                elif self.pid_controller is not None and (self.forward_path is not None or self.reverse_path is not None):
                    
                    # Determine which path segment to follow based on current phase
                    if self.current_phase == 'forward' and self.forward_path is not None:
                        current_path = self.forward_path
                        expected_direction = 1  # Forward
                        phase_name = "Forward"
                    elif self.current_phase == 'reverse' and self.reverse_path is not None:
                        current_path = self.reverse_path
                        expected_direction = -1  # Reverse
                        phase_name = "Reverse"
                    else:
                        # No valid path for current phase, stop
                        self.trans_control.throttle = 0.0
                        self.trans_control.brake = 0.3
                        self.trans_control.steer = 0.0
                        self.trans_control.reverse = False
                        # logging.info(f"No valid path for phase '{self.current_phase}', stopping vehicle")
                        current_path = None
                    
                    if current_path is not None:
                        # Get target waypoint from current path segment
                        waypoint, self.current_target_index = get_target_waypoint_from_path(
                            current_path, vehicle_transform, data["ego_xy_dynamic"], self.current_target_index, self.initial_ego_pos,
                            direction=expected_direction
                        )
                        
                        if waypoint is not None:
                            # Check if we need to transition to the next phase
                            if self.current_phase == 'forward' and not self.forward_completed:
                                # To transition from forward to reverse, we check if the REAR AXLE has reached
                                # the end of the forward path. This is the correct reference point.
                                end_of_forward_path_x = current_path[0][0]
                                end_of_forward_path_y = current_path[1][0]

                                # Get current vehicle REAR position in the initial ego frame
                                vehicle_rear_x_ego, vehicle_rear_y_ego = _get_rear_axle_in_initial_ego_frame(vehicle_transform, self.initial_ego_pos, data["ego_xy_dynamic"])

                                distance_to_forward_end = math.sqrt((end_of_forward_path_x - vehicle_rear_x_ego)**2 + 
                                                                    (end_of_forward_path_y - vehicle_rear_y_ego)**2)
                                
                                forward_completion_threshold = 0.3 # meters
                                if distance_to_forward_end < forward_completion_threshold:
                                    self.forward_completed = True
                                    #logging.info(f"🟢 Forward phase completed! Rear axle distance to end: {distance_to_forward_end:.2f}m")

                                    # Transition to reverse phase if reverse path exists
                                    if self.reverse_path is not None:
                                        self.current_phase = 'reverse'
                                        self.current_target_index = None  # Reset for reverse path
                                        #logging.info("🔄 Transitioning to reverse phase")
                                        
                                        # Stop briefly for phase transition
                                        self.trans_control.throttle = 0.0
                                        self.trans_control.brake = 0.8
                                        self.trans_control.steer = 0.0
                                        self.trans_control.reverse = False
                                        return  # Skip PID control this frame
                                    else:
                                        logging.info("✅ All path segments completed!")
                            
                            # Use expected direction for the current phase
                            direction = expected_direction
                            
                            # Set speed based on phase and proximity to goal
                            if self.current_phase == 'forward':
                                target_speed = 9.0
                                if self.current_target_index is not None and self.current_target_index <= 8:
                                    target_speed = min(target_speed, 8.0)
                            else:  # reverse phase
                                target_speed = 3.0
                                if self.current_target_index is not None and self.current_target_index <= 8:
                                    target_speed = min(target_speed, 2.0)
                            
                            # Get PID control using expected direction
                            pid_control = self.pid_controller.run_step(target_speed, waypoint, direction)
                            
                            self.trans_control.throttle = pid_control.throttle
                            self.trans_control.brake = pid_control.brake
                            self.trans_control.steer = pid_control.steer
                            self.trans_control.reverse = (direction == -1)

                            # If in final reversing stage, adjust steering based on yaw difference for final alignment.
                            if self.current_phase == 'reverse' and self.current_target_index is not None and self.current_target_index <= 8:
                                
                                # Calculate signed yaw difference to the final goal orientation
                                current_yaw = vehicle_transform.rotation.yaw
                                goal_yaw = self.net_eva.eva_parking_goal[2]
                                
                                signed_yaw_diff = goal_yaw - current_yaw
                                # Normalize angle to [-180, 180]
                                if signed_yaw_diff > 180:
                                    signed_yaw_diff -= 360
                                if signed_yaw_diff < -180:
                                    signed_yaw_diff += 360
                                
                                # If yaw difference is very small, force steering to be straight.
                                if abs(signed_yaw_diff) < 1.0:
                                    self.trans_control.steer = 0.0

                                # logging.info(f"Final alignment: YawDiff={signed_yaw_diff:.2f} deg, SteerCmd={self.trans_control.steer:.3f}")

                                # FINAL GOAL REACHED CHECK
                                # Get unsigned yaw difference for stop check
                                yaw_diff = abs(signed_yaw_diff)
                                
                                # Override PID if we are at the very end of the reverse path and close enough to stop.
                                final_goal_x = current_path[0][0]
                                final_goal_y = current_path[1][0]

                                # Calculate current rear position in the initial ego frame to compare with the path
                                vehicle_rear_x_ego, vehicle_rear_y_ego = _get_rear_axle_in_initial_ego_frame(vehicle_transform, self.initial_ego_pos, data["ego_xy_dynamic"])

                                distance_to_final_goal = math.sqrt((final_goal_x - vehicle_rear_x_ego)**2 + (final_goal_y - vehicle_rear_y_ego)**2)

                                stop_threshold = 0.5 # 50 cm
                                yaw_stop_threshold = 10.0 # degrees
                                if distance_to_final_goal < stop_threshold and yaw_diff < yaw_stop_threshold:
                                    #logging.info(f"✅ FINAL GOAL REACHED! Distance: {distance_to_final_goal:.2f}m, Yaw Diff: {yaw_diff:.2f}deg. Stopping.")
                                    self.trans_control.throttle = 0.0
                                    self.trans_control.brake = 0.8  # Apply firm brake to stop
                                    self.trans_control.steer = 0.0
                            
                            # Force straight steering during forward phase
                            if self.current_phase == 'forward':
                                self.trans_control.steer = 0.0
                            
                            # Limit throttle when reversing for safety
                            if direction == -1:
                                self.trans_control.throttle = min(self.trans_control.throttle, 0.5)
                            
                            # logging.info(f"{phase_name} Phase - Direction: {'REV' if direction == -1 else 'FWD'}, "
                            #            f"Target Index: {self.current_target_index}, Speed: {target_speed:.1f}km/h, "
                            #            f"Throttle: {self.trans_control.throttle:.3f}, Brake: {self.trans_control.brake:.3f}, Steer: {self.trans_control.steer:.3f}")
                        else:
                            # No valid waypoint, stop the vehicle
                            self.trans_control.throttle = 0.0
                            self.trans_control.brake = 0.3
                            self.trans_control.steer = 0.0
                            self.trans_control.reverse = False
                            # logging.info(f"No valid waypoint found for {phase_name} phase, stopping vehicle")
                else:
                    # No path, no PID controller, and not in continuous forward mode
                    # Check if we should enable continuous forward mode
                    if not self.continuous_forward_mode:
                        # Use dynamic model position for consistency
                        check_dynamic_transform = carla.Transform(
                            carla.Location(x=data["ego_xy_dynamic"][0], y=data["ego_xy_dynamic"][1], z=vehicle_transform.location.z),
                            vehicle_transform.rotation
                        )
                        goal_ego = convert_slot_coord(check_dynamic_transform, self.net_eva.eva_parking_goal)
                        target_distance = math.sqrt(goal_ego[0]**2 + goal_ego[1]**2)
                        
                        if target_distance > 10.0:
                            #logging.info(f"🚗 Target distance {target_distance:.2f}m > 15m, enabling continuous forward mode")
                            self.continuous_forward_mode = True
                            self.initial_ego_pos = None  # Clear old position data
                        # else:
                        #     # Stop the vehicle if target is close but no path available
                        #     self.trans_control.throttle = 0.0
                        #     self.trans_control.brake = 0.3
                        #     self.trans_control.steer = 0.0
                        #     self.trans_control.reverse = False
                        #     logging.info("No path available and target not too far, stopping vehicle")
                    else:
                        # This shouldn't happen, but just in case
                        self.trans_control.throttle = 0.0
                        self.trans_control.brake = 0.3
                        self.trans_control.steer = 0.0
                        self.trans_control.reverse = False
                        #logging.info("Unexpected state, stopping vehicle")

                self.speed_limit(data_frame)

                if self.show_eva_imgs:
                    self.grid_image, self.atten_avg = self.save_atten_avg_map(data)
                    self.save_seg_img(pred_segmentation)
                    self.save_target_bev_img(target_bev)
                    self.display_imgs()
                self.save_output.clear()

            self.prev_xy_thea = [data["ego_xy_dynamic"][0],
                                 data["ego_xy_dynamic"][1],
                                 imu_data.compass if np.isnan(imu_data.compass) else 0]
        
        self.player.apply_control(self.trans_control)

    def speed_limit(self, data_frame):
        # if vehicle stops at initialization, give throttle until Gear turns to 1
        if data_frame['veh_control'].gear == 0:
            self.trans_control.throttle = 0.5

        speed = (3.6 * math.sqrt(
            data_frame['veh_velocity'].x ** 2 + data_frame['veh_velocity'].y ** 2 + data_frame['veh_velocity'].z ** 2))

        # limit the vehicle speed within 15km/h when reverse is False
        if not self.trans_control.reverse and speed >= 12:
            self.trans_control.throttle = 0.0

        # limit the vehicle speed within 10km/h when reverse is True
        if self.trans_control.reverse and speed >= 10:
            self.trans_control.throttle = 0.0

        # if brake and throttle both not on, and speed < 2 for more than 2 seconds, give it a small throttle for 1
        # second
        if self.trans_control.throttle < 1e-5 and self.trans_control.brake < 1e-5 and speed < 2.0:
            self.stop_count += 1
        else:
            self.stop_count = 0.0

        if self.stop_count > 10:  # 1s
            self.boost = True

        if self.boost:
            self.trans_control.throttle = 0.3
            self.boot_step += 1

        if self.boot_step > 10 or self.trans_control.brake > 1e-5:  # 1s
            self.boot_step = 0
            self.boost = False

    def world_to_ego_velocity(self, speed_x, speed_y, speed_z, roll, pitch, yaw):
        """
        Convert velocity from the world coordinate frame to the vehicle's egocentric frame.

        Parameters:
            speed_x, speed_y, speed_z : float
                Velocity components in the world frame [m/s].
            roll, pitch, yaw : float
                Vehicle orientation angles in degrees.

        Returns:
            v_ego : ndarray, shape (3,)
                Velocity in the vehicle frame:
                [v_forward, v_right, v_up] in m/s.
        """
        # 1. Convert angles from degrees to radians
        r = np.deg2rad(roll)
        p = np.deg2rad(pitch)
        y = np.deg2rad(yaw)

        # 2. Build rotation matrices for each axis
        # Rotation around X-axis (roll)
        Rx = np.array([
            [1,          0,           0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r),  np.cos(r)]
        ])
        # Rotation around Y-axis (pitch)
        Ry = np.array([
            [ np.cos(p), 0, np.sin(p)],
            [         0, 1,         0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        # Rotation around Z-axis (yaw)
        Rz = np.array([
            [ np.cos(y), -np.sin(y), 0],
            [ np.sin(y),  np.cos(y), 0],
            [         0,          0, 1]
        ])

        # 3. Combine rotations: from vehicle (ego) frame to world frame
        R_ego_to_world = Rz @ Ry @ Rx

        # 4. World-frame velocity vector
        v_world = np.array([speed_x, speed_y, speed_z])

        # 5. Transform world velocity into vehicle frame (inverse rotation)
        v_ego = R_ego_to_world.T @ v_world

        return  v_ego

    def get_model_data(self, data_frame):

        vehicle_transform = data_frame['veh_transfrom'] # world frame
        imu_data = data_frame['imu'] # ego frame
        vehicle_velocity = data_frame['veh_velocity'] #m/s
        #print('vx',vehicle_velocity.x)
        #print('vy',vehicle_velocity.y)
        #print('ax',imu_data.accelerometer.x)
        #print('ay',imu_data.accelerometer.y)
        #print("\n")
        data = {}

        # Compute using speed - initialize first
        if not self.ego_xy_dynamic: # read only once after initilization
            self.ego_xy = [vehicle_transform.location.x, vehicle_transform.location.y]
            self.ego_xy_dynamic = [vehicle_transform.location.x, vehicle_transform.location.y]
            #print('self.ego_xy initialization:', self.ego_xy)
            
            # Calculate initial target_point using dynamic model position
            dynamic_transform = carla.Transform(
                carla.Location(x=self.ego_xy_dynamic[0], y=self.ego_xy_dynamic[1], z=vehicle_transform.location.z),
                vehicle_transform.rotation  # Keep CARLA's yaw - this is acceptable
            )
            target_point = convert_slot_coord(dynamic_transform, self.net_eva.eva_parking_goal)
            
        parking_goal_world = self.net_eva.eva_parking_goal[:2]
        self.parking_goal_world = parking_goal_world
        speed = 3.6*np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        #print(speed)
        #print('This is target under global frame', parking_goal_world)
        dt = 0.1  

        yaw = np.deg2rad(vehicle_transform.rotation.yaw)  # Convert yaw to radians

        accel_x_world = imu_data.accelerometer.x * np.cos(yaw) + imu_data.accelerometer.y * np.sin(yaw)
        accel_y_world = -imu_data.accelerometer.x * np.sin(yaw) + imu_data.accelerometer.y * np.cos(yaw)

        displacement_x_world = vehicle_velocity.x * dt + 0.5 * accel_x_world * dt**2
        displacement_y_world = vehicle_velocity.y * dt + 0.5 * accel_y_world * dt**2
        self.ego_xy[0] += displacement_x_world
        self.ego_xy[1] += displacement_y_world
        #print('self.ego_xy:', self.ego_xy)

        # Note: relative_target now computed using dynamic model below
        
        ##############Dynamic ####################
        ego_pos_torch = deepcopy(self.ego_xy_dynamic)
        ego_pos_torch.append(vehicle_transform.rotation.yaw)
        ego_pos_torch = torch.tensor(ego_pos_torch).to(self.device)
        ego_motion_torch = torch.tensor([3.6*vehicle_velocity.x, 3.6*vehicle_velocity.y, 3.6*vehicle_velocity.z, imu_data.accelerometer.x, imu_data.accelerometer.y],
                                        dtype=torch.float).to(self.device)
        raw_control_torch = torch.tensor([data_frame['veh_control'].throttle, data_frame['veh_control'].brake, data_frame['veh_control'].steer, data_frame['veh_control'].reverse], dtype=torch.float).to(self.device)

        input_data = {
            # 'yaw'
            'ego_pos': ego_pos_torch.unsqueeze(0),  # Shape: (1, 3)
            'ego_motion': ego_motion_torch.view(1,-1),  # Shape: (1, 4)
            'raw_control': raw_control_torch.view(1,-1), # Shape: (4,)
        }
        delta_mean, log_var, x_displacement_track, y_displacement_track = self.speed_dynamics_model.predict(input_data)
        self.ego_xy_dynamic[0] += delta_mean.squeeze(0)[0].detach().item()
        self.ego_xy_dynamic[1] += delta_mean.squeeze(0)[1].detach().item()
        #print('self.ego_xy_dynamic:', self.ego_xy_dynamic)
        
        # Calculate target_point using dynamic model position (avoid CARLA location dependency)
        # Create a transform using dynamic model position but keeping CARLA's rotation
        dynamic_transform = carla.Transform(
            carla.Location(x=self.ego_xy_dynamic[0], y=self.ego_xy_dynamic[1], z=vehicle_transform.location.z),
            vehicle_transform.rotation  # Keep CARLA's yaw - this is acceptable
        )
        target_point = convert_slot_coord(dynamic_transform, self.net_eva.eva_parking_goal)
        
        relative_x_world_dynamic = parking_goal_world[0] - self.ego_xy_dynamic[0]
        relative_y_world_dynamic = parking_goal_world[1] - self.ego_xy_dynamic[1]

        relative_x_ego_dynamic = (relative_x_world_dynamic * np.cos(yaw) + relative_y_world_dynamic * np.sin(yaw))
        relative_y_ego_dynamic = (-relative_x_world_dynamic * np.sin(yaw) + relative_y_world_dynamic * np.cos(yaw))
        ###########################################

        data['relative_target'] = torch.tensor([relative_x_ego_dynamic,relative_y_ego_dynamic], dtype=torch.float).unsqueeze(0)

        #print("This is relative_target:", data['relative_target'])
        front_final, self.camera_front = self.image_process(data_frame['camera_front'])
        front_left_final, self.camera_front_left = self.image_process(data_frame['camera_front_left'])
        front_right_final, self.camera_front_right = self.image_process(data_frame['camera_front_right'])
        back_final, self.camera_back = self.image_process(data_frame['camera_back'])
        back_left_final, self.camera_back_left = self.image_process(data_frame['camera_back_left'])
        back_right_final, self.camera_back_right = self.image_process(data_frame['camera_back_right'])

        images = [front_final, front_left_final, front_right_final,back_final,back_left_final,back_right_final]
        images = torch.cat(images, dim=0)
        data['image'] = images.unsqueeze(0)

        data['extrinsics'] = self.extrinsic.unsqueeze(0)
        data['intrinsics'] = self.intrinsic_crop.unsqueeze(0)

        #velocity = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)) #km/h

        roll = data_frame["veh_transfrom"].rotation.roll
        pitch = data_frame["veh_transfrom"].rotation.pitch
        yaw = data_frame["veh_transfrom"].rotation.yaw

        velocity_ego = self.world_to_ego_velocity(vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z, roll,
                                                  pitch, yaw)
        acc_ego = self.world_to_ego_velocity(data_frame["imu"].accelerometer.x, data_frame["imu"].accelerometer.y,
                                             data_frame["imu"].accelerometer.z,
                                             roll,pitch, yaw)

        velocity = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)) #km/h
        data['ego_motion'] = torch.tensor([velocity, imu_data.accelerometer.x, imu_data.accelerometer.y],
                                          dtype=torch.float).unsqueeze(0).unsqueeze(0)


        target_types = ["gt","predicted","tracking","dynamics"]
        target_type = target_types[1]
        if target_type =="tracking":
            data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)
            data["target_point"][0][0] = data["relative_target"][0][0]
            data["target_point"][0][1] = data["relative_target"][0][1]
        elif target_type == "gt":
            data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)
        elif target_type == "predicted":
            if self.pre_target_point is not None:
                target_point = [self.pre_target_point[0], self.pre_target_point[1], target_point[2]]
                data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)
            else:
                data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)
        elif target_type == "dynamics":
            data["target_point"][0][0] = torch.tensor(relative_x_ego_dynamic, dtype=torch.float).unsqueeze(0)
            data["target_point"][0][1] = torch.tensor(relative_y_ego_dynamic, dtype=torch.float).unsqueeze(0)
        data['gt_control'] = torch.tensor([self.BOS_token], dtype=torch.int64).unsqueeze(0)
        data['gt_waypoint'] = torch.tensor([self.BOS_token], dtype=torch.int64).unsqueeze(0)
        if self.show_eva_imgs:
            img = encode_npy_to_pil(np.asarray(data_frame['topdown'].squeeze().cpu()))
            img = np.moveaxis(img, 0, 2)
            img = Image.fromarray(img)
            seg_gt = self.semantic_process(image=img, scale=0.5, crop=300, target_slot=target_point) # 训练的时候是200，但是inference的时候老报错超出范围
            seg_gt[seg_gt == 1] = 128
            seg_gt[seg_gt == 2] = 255
            data['segmentation'] = Image.fromarray(seg_gt)
        data["ego_trans"] = vehicle_transform
        data["ego_xy_dynamic"] = self.ego_xy_dynamic

        # logging.info(
        #     f"Position Update - Ground Truth: ({vehicle_transform.location.x:.2f}, {vehicle_transform.location.y:.2f}), "
        #     f"Dynamic Model Prediction: ({self.ego_xy_dynamic[0]:.2f}, {self.ego_xy_dynamic[1]:.2f})"
        # )

        return data

    def draw_control_info(self, trans_control):
        ego_t = carla.Transform(carla.Location(x=302.0, y=-248.239487, z=0.32682),
                                carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))

        throttle_loc = carla.Location(x=ego_t.location.x - 1.5, y=ego_t.location.y + 6.5, z=0.20)
        throttle = int(trans_control.throttle * 1000) / 1000
        self.world.world.debug.draw_string(throttle_loc, ' throttle: ' + str(throttle),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        brake_loc = carla.Location(x=ego_t.location.x - 2.5, y=ego_t.location.y + 6.5, z=0.20)
        brake = int(trans_control.brake * 1000) / 1000
        self.world.world.debug.draw_string(brake_loc, '    brake: ' + str(brake),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        steer_loc = carla.Location(x=ego_t.location.x - 3.5, y=ego_t.location.y + 6.5, z=0.20)
        steer = int(trans_control.steer * 1000) / 1000
        self.world.world.debug.draw_string(steer_loc, '     steer: ' + str(steer),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

        reverse_loc = carla.Location(x=ego_t.location.x - 4.5, y=ego_t.location.y + 6.5, z=0.20)
        self.world.world.debug.draw_string(reverse_loc, 'reverse: ' + str(trans_control.reverse),
                                           draw_shadow=True, color=carla.Color(255, 0, 0))

    def get_gru_control(self, control_output):
        control = control_output[0][0].tolist()
        if control[0] > 0:
            throttle = control[0]
            brake = 0.0
        else:
            throttle = 0.0
            brake = -control[0]
        steer = control[1]
        reverse = (True if control[2] > 0 else False)
        return [throttle, brake, steer, reverse] 

    def carla_to_nparray(self, carla_img):
        image_array = np.reshape(np.copy(carla_img.raw_data), (carla_img.height, carla_img.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]
        return image_array

    def display_imgs(self):
        # Create or get the main display figure
        main_fig = plt.figure("Main Display", figsize=(12, 6))
        plt.figure(main_fig.number)  # Switch to main display figure
        plt.clf()  # Only clear the main display figure
        
        plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
        rows = 2
        cols = 4
        ax_ctl = plt.subplot(rows, cols, 4)
        ax_ctl.axis('off')
        t_x = 0.2
        t_y = 0.8
        
        throttle_val = self.trans_control.throttle
        if math.isnan(throttle_val):
            throttle_val = 0.0
        throttle_show = int(throttle_val * 1000) / 1000

        steer_val = self.trans_control.steer
        if math.isnan(steer_val):
            steer_val = 0.0
        steer_show = int(steer_val * 1000) / 1000

        brake_val = self.trans_control.brake
        if math.isnan(brake_val):
            brake_val = 0.0
        brake_show = int(brake_val * 1000) / 1000

        reverse_show = self.trans_control.reverse
        ax_ctl.text(t_x, t_y, 'Throttle: ' + str(throttle_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.2, '    Steer: ' + str(steer_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.4, '   Brake: ' + str(brake_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.6, 'Reverse: ' + str(reverse_show), fontsize=10, color='red')
        if hasattr(self, 'continuous_forward_mode') and self.continuous_forward_mode:
            ax_ctl.text(t_x, t_y - 0.8, 'Control: Continuous Forward', fontsize=10, color='orange')
        else:
            # Show current phase information
            if hasattr(self, 'current_phase'):
                phase_color = 'green' if self.current_phase == 'forward' else 'red'
                phase_status = ""
                if hasattr(self, 'forward_completed') and self.forward_completed:
                    phase_status = " (F-Done)"
                ax_ctl.text(t_x, t_y - 0.8, f'Phase: {self.current_phase.upper()}{phase_status}', 
                           fontsize=10, color=phase_color)
            else:
                ax_ctl.text(t_x, t_y - 0.8, 'Control: PID', fontsize=10, color='blue')

        ax_pc = plt.subplot(rows, cols, 3)
        ax_pc.set_title('2D Point Cloud', fontsize=10)
        if self.point_cloud is not None and len(self.point_cloud) > 0:
            ax_pc.scatter(self.point_cloud[:, 1], self.point_cloud[:, 0], s=1, c='k')

        # Plot ego vehicle
        ax_pc.plot(0, 0, 'ro', markersize=5, label='Ego')
        ax_pc.arrow(0, 0, 0, 2, head_width=0.5, head_length=0.5, fc='r', ec='r')

        ax_pc.set_xlabel('Right (m)')
        ax_pc.set_ylabel('Forward (m)')
        ax_pc.set_aspect('equal', adjustable='box')
        ax_pc.grid(True)
        ax_pc.set_xlim([-20, 20])
        ax_pc.set_ylim([-5, 35])

        ax_atten = plt.subplot(rows, cols, 7)
        ax_atten.axis('off')
        ax_atten.set_title('atten(output)', fontsize=10)
        ax_atten.imshow(self.grid_image)
        ax_atten.imshow(self.atten_avg / np.max(self.atten_avg), alpha=0.6, cmap='rainbow')

        ax_bev = plt.subplot(rows, cols, 8)
        ax_bev.axis('off')
        ax_bev.set_title('seg_bev(output)', fontsize=10)
        ax_bev.imshow(self.seg_bev)

        plt.pause(0.1)
        # Don't call plt.clf() here as it would clear all figures including the path planning window

    def get_eva_control(self):
        control = {
            'throttle': self.trans_control.throttle,
            'steer': self.trans_control.steer,
            'brake': self.trans_control.brake,
            'reverse': self.trans_control.reverse,
        }
        return control
    

