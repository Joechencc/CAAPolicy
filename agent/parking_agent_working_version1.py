import carla
import math
import pathlib
import yaml
import torch
import logging
import time
import pygame

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict

import cv2 
import time, sys
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla')
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla/carla/PythonAPI')
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla/carla/PythonAPI/carla')

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic
from dataset.carla_dataset import detokenize
from data_generation.network_evaluator import NetworkEvaluator
from data_generation.tools import encode_npy_to_pil
from model.parking_model import ParkingModel

import math
import agent.Fast_hybrid_A_star.hybrid_A_star as hybrid_a_star
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import agent.Fast_hybrid_A_star.draw as draw
from functools import reduce
import open3d as o3d #### for lidar point clouds
from matplotlib import cm

from controller.controller import VehiclePIDController 

lateral_par = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.05, 'dt': 0.03} # {'K_P': 1.5, 'K_D': 0.5, 'K_I': 0.1, 'dt': 0.03}
longitudinal_par = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.2, 'dt':0.03}
max_steering = 0.77 #0.8 ##0.75

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [16.0, 0.0, 0.0],
        [0.0, 16.0, 0.0],
        [0.0, 0.0, 2.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='Carla Lidar', width=960, height=540, left=480, top=270)
# vis.get_render_option().background_color = [0.05, 0.05, 0.05]
# vis.get_render_option().point_size = 1
# vis.get_render_option().show_coordinate_frame = True
# add_open3d_axis(vis)

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


K_theta = 1.0 
K_e = 0.5

class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct
    def update(self, a, delta, direct, dt):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.direct = direct
        self.v += self.direct * a * dt   
    def update2(self, x, y, yaw, direct, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direct = direct
        self.v = v     

def calc_curvature(x, y, directions):
    c= []
    for i in range(1, len(x) - 1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = math.hypot(dxn, dyn)
        dp = math.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        d = (dn + dp) / 2.0

        if np.isnan(curvature):
            curvature = 0.0

        if directions[i] <= 0.0:
            curvature = -curvature

        if len(c) == 0:
            c.append(curvature)
        c.append(curvature) 
    c.append(c[-1])
    return c

def pid_control(target_v, v, dist, direct):
    """
    using LQR as lateral controller, PID as longitudinal controller (speed control)
    :param target_v: target speed
    :param v: current speed
    :param dist: distance to end point
    :param direct: current direction of vehicle, 1.0: forward, -1.0: backward
    :return: acceleration
    """

    a = 0.3 * (target_v - direct * v)
    if dist < 10.0:
        if v > 2:
            a = -3.0
        elif v < -2:
            a = -1.0
    return a

class PATH_control:
    def __init__(self, cx, cy, cyaw, ccurv):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ccurv = ccurv
        self.len = len(self.cx)
        self.s0 = 1

    def calc_theta_e_and_er(self, node):
        """
        calc theta_e and er.
        theta_e = theta_car - theta_path
        er = lateral distance in frenet frame

        :param node: current information of vehicle
        :return: theta_e and er
        """

        ind = self.nearest_index(node)

        k = self.ccurv[ind]
        yaw = self.cyaw[ind]

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[node.x - self.cx[ind]],
                                      [node.y - self.cy[ind]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        theta_e = pi_2_pi(node.yaw - self.cyaw[ind])

        return theta_e, er, k, yaw, ind

    def nearest_index(self, node):
        """
        find the index of the nearest point to current position.
        :param node: current information
        :return: nearest index
        """

        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        dist = np.hypot(dx, dy)
        self.s0 += np.argmin(dist[self.s0:self.len]) 

        return min(self.s0, self.len-1) ##self.s0 #### do we need to consider the case when s0 >= len
    

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
        return min(self.s0+1, self.len-1) ##self.s0 #### do we need to consider the case when s0 >= len    

def rear_wheel_feedback_control(node, ref_path):
    """
    rear wheel feedback controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :return: optimal steering angle
    """

    theta_e, er, k, yaw, ind = ref_path.calc_theta_e_and_er(node)
    vr = node.v

    omega = vr*k*math.cos(theta_e) / (1.0 - k * er) - \
            K_theta*abs(vr) * theta_e - K_e * vr * math.sin(theta_e) * er / (theta_e+1e-8)

    delta = math.atan2(WB * omega, vr)

    return delta, ind, omega


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

def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta    


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

        self.rgb_rear = None
        self.rgb_right = None
        self.rgb_left = None
        self.rgb_front = None
        self.seg_bev = None
        self.target_bev = None

        self.pre_target_point = None

        self.model = None
        self.device = None

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

        ################# bounding box info #####################
        ################ #get_world_vertices(self.player.get_transform())
        self.bbox = self.player.bounding_box.get_local_vertices() 
        ##########################################################
        ### str(self.player.bounding_box.extent)
        ### 'Vector3D(x=2.395890, y=1.081725, z=0.744160)'
        ############ I make lidar yaw =-90, and x => -x ######
        # self.bbox_min = [self.bbox[0].y, self.bbox[0].x, -1.6+self.bbox[0].z] ### with respect to lidar z
        # self.bbox_max = [self.bbox[7].y, self.bbox[7].x, -1.6+self.bbox[7].z]

        self.bbox_min = [self.bbox[0].x, self.bbox[0].y, -1.6+self.bbox[0].z] ### with respect to lidar z
        self.bbox_max = [self.bbox[7].x, self.bbox[7].y, -1.6+self.bbox[7].z]
        
        self.solution_required = True
        self.current_target = self.net_eva._eva_parking_goal.copy()

        ##### get target with respect to the Carla world coordinate (vehicle center)
        self.target_x, self.target_y, self.target_yaw = self.net_eva._eva_parking_goal[0], self.net_eva._eva_parking_goal[1], self.net_eva._eva_parking_goal[2]
        ### shift the center to rear center in world coordinate system
        self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
        self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))

        ########################################################
        
        self.pid_controller = VehiclePIDController(self.player, lateral_par, longitudinal_par, max_throttle=1.0, max_steering=max_steering) ##0.75
        self.path_stage = None ### Breaking the path into multiple segments and using this index to access the segment
        self.current_stage = None
        self.r_trajectory = None
        self.my_control = carla.VehicleControl()
        self.prev_dist = None
                

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
        # self.load_model(args.model_path)

        self.stop_count = 0
        self.boost = False
        self.boot_step = 0

        self.init_agent()

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
        self.model.load_state_dict(state_dict)
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

    def get_target_point_ego_coord(self, pred_seg_img, target_point_pixel_idx):
        bev_shape = pred_seg_img.shape[0]
        x = -(target_point_pixel_idx[0] - bev_shape / 2)
        y = target_point_pixel_idx[1] - bev_shape / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2]]
        return target_point_ego_coord

    def init_agent(self):
        w = self.world.cam_config['width']
        h = self.world.cam_config['height']

        self.intrinsic_crop = update_intrinsics(
            torch.from_numpy(self.world.intrinsic).float(),
            (h - self.cfg.image_crop) / 2,
            (w - self.cfg.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic_crop = self.intrinsic_crop.unsqueeze(0).expand(4, 3, 3)

        veh2cam_dict = self.world.veh2cam_dict
        front_to_ego = torch.from_numpy(veh2cam_dict['rgb_front']).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(veh2cam_dict['rgb_left']).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(veh2cam_dict['rgb_right']).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(veh2cam_dict['rgb_rear']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

        self.image_process = ProcessImage(self.cfg.image_crop)

        self.step = -1
        self.pre_target_point = None

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

    def tick(self, carla_world):
        if self.current_target != self.net_eva._eva_parking_goal or self.net_eva.agent_need_init: ### we also need to consider the initial position change for ego
            print('Parking task changes, hence we need to find a new path')
            self.current_target = self.net_eva._eva_parking_goal.copy()
            self.solution_required = True
            ##### get target with respect to the Carla world coordinate (vehicle center)
            self.target_x, self.target_y, self.target_yaw = self.net_eva._eva_parking_goal[0], self.net_eva._eva_parking_goal[1], self.net_eva._eva_parking_goal[2]
            ### shift the center to rear center in world coordinate system
            self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
            self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))
            self.r_trajectory = None
            self.current_stage = None

        if self.net_eva.agent_need_init:
            self.init_agent()
            self.net_eva.agent_need_init = False

        self.step += 1

        # stop 1s for new eva
        if self.step < 30: ### 30 frames per second
            self.player.apply_control(carla.VehicleControl())
            self.player.set_transform(self.net_eva.ego_transform)
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
            
            if self.solution_required:                                               ## semantic ## lidar pc
                result, ox_real, oy_real, gx, gy, gyaw = self.hybrid_A_star_solution(data['feng'], voxelGrid)
                sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ## ego rear center pos in BEV coordinate system (right hand)

                if result is not None:
                    self.solution_required = False
                    self.path_stage = 0 ### go through the path segments one by one, starts with the first segment
                    self.ego_traject = [] ### to log ego trajectory under the controller
                    self.hist_use = 10000 ### a large number
                   
     
                    #### in current ego frame coordinate system
                    x, y, yaw, direction = result

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
                        # ###Get the center in world coordinate
                        # x_world += 1.37*math.cos(yaw_world)
                        # y_world += 1.37*math.sin(yaw_world)
                        location = carla.Location(x=x_world, y= y_world, z = 0.2)
                        carla_world.debug.draw_point(location, size=0.02,  color=carla.Color(255,0,0),life_time=30)
                        self.positions.append([x_world, y_world, yaw_world, direction[k]])
                        ###### np.savetxt('rear_path.txt', np_vec)  ##np.loadtxt('rear_path.txt') ==> a np array

                    oy_real_w, ox_real_w = [], []
                    for ele1, ele2 in zip(ox_real, oy_real):
                        oy_real_w.append(math.cos(self.ab_yaw_init)*(ele1-16) - math.sin(self.ab_yaw_init)*(ele2-16)+ self.a_y_init)
                        ox_real_w.append(math.sin(self.ab_yaw_init)*(ele1-16) + math.cos(self.ab_yaw_init)*(ele2-16)+ self.b_x_init)

                    np_vec = np.array(self.positions)
                    plt.cla()
                    #plt.scatter(self.player.get_transform().location.y, self.player.get_transform().location.x, label='ego center')
                    plt.plot(oy_real_w, ox_real_w, "sk", markersize=1)
                    plt.scatter(self.player.get_transform().location.y-1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)), self.player.get_transform().location.x-1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)), label='ego rear start')
                    plt.scatter(self.target_y, self.target_x, label='target rear')
                    plt.plot(np_vec[:, 1],np_vec[:,0], label='planned path rear') ### using UE coordinate system
                    plt.legend()
                    plt.title('Visulize the planned path in world coordinate for 5 seconds')
                    plt.show()
                    plt.pause(5)
                    plt.close()  
                    

                    print('break the path into multiple segments based on directions')
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

                    ########  add final goal  ###### not necessary when we use rear point plot
                    ## tmp.append([self.target_x, self.target_y, np.deg2rad(self.target_yaw), self.positions[i][-1]])  
                    self.mul_pos.append(tmp) #[::-1]   

                    
                    # print('\nVisualize the found path in simulated plot in the BEV frame...')
                    # for k in range(0, len(x), 1):
                    #     plt.cla()
                    #     plt.plot(ox_real, oy_real, "sk", markersize=1)
                    #     plt.plot(x, y, linewidth=1.5, color='r')
                    #     if k < len(x) - 2:
                    #         dy = (yaw[k + 1] - yaw[k]) / MOVE_STEP
                    #         steer = pi_2_pi(math.atan(-WB * dy / direction[k]))
                    #     else:
                    #         steer = 0.0
                    #     draw.draw_car(sx, sy, syaw, 0.0, LB, LF, W, TR, TW, WB, WD, 'dimgray')    
                    #     draw.draw_car(gx, gy, gyaw, 0.0, LB, LF, W, TR, TW, WB, WD, 'dimgray')
                    #     draw.draw_car(x[k], y[k], yaw[k], steer, LB, LF, W, TR, TW, WB, WD)

                    #     plt.title("Hybrid A*")
                    #     plt.axis("equal")
                    #     plt.pause(0.0001)
                    # plt.close()

                    

                    # # #### This block manually assign the next position for ego
                    # for k in range(0, len(x), 1):
                    #     y_graph = x[k]-16.0  #Rear center in ego coordinate (since this path uses rear center)
                    #     x_graph = y[k]-16.0  
                    #     yaw_graph = 90-np.rad2deg(yaw[k])  #np.deg2rad(90)-(yaw[k])  ## right hand to left hand
                    #     ###first convert this point from ego coordinate to world coordinate
                    #     y_world = math.cos(self.ab_yaw_init)*y_graph - math.sin(self.ab_yaw_init)*x_graph + self.a_y_init
                    #     x_world = math.sin(self.ab_yaw_init)*y_graph + math.cos(self.ab_yaw_init)*x_graph + self.b_x_init
                    #     yaw_world = np.deg2rad(yaw_graph)-self.ab_yaw_init
                    
                    #     ###Get the center in world coordinate
                    #     x_world += 1.37*math.cos(yaw_world)
                    #     y_world += 1.37*math.sin(yaw_world)
                    #     ### Assign position
                    #     transform = self.player.get_transform()
                    #     transform.location.x = x_world  
                    #     transform.location.y = y_world  
                    #     transform.rotation.yaw = np.rad2deg(yaw_world)
                    #     self.player.set_transform(transform) 
                    #     carla_world.tick()

                else:
                    print('\nHybrid A* fails to find a path, soft movement to trigger a new search position...')
                    plt.cla()
                    plt.plot(ox_real, oy_real, "sk", markersize=1)
                    draw.draw_car(sx, sy, syaw, 0.0, LB, LF, W, TR, TW, WB, WD, 'dimgray')    
                    draw.draw_car(gx, gy, gyaw, 0.0, LB, LF, W, TR, TW, WB, WD, 'dimgray')
                    plt.title("Hybrid A* fails to find a path")
                    plt.axis("equal")
                    plt.show()        
                    #### set ego new position as a steady throttle
                    self.trans_control.throttle = 1  
                    self.trans_control.brake = 0  
                    self.trans_control.steer = 0 
                    self.player.apply_control(self.trans_control)
                return    
            else:
                #print('Tracking the generated path for one step...') 
                ego_center_current_x = np.round(self.player.get_transform().location.x,1)
                ego_center_current_y = np.round(self.player.get_transform().location.y,1)

                ego_rear_x = np.round(ego_center_current_x - 1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)),1)
                ego_rear_y = np.round(ego_center_current_y - 1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)),1)
                dist = np.round(math.hypot( ego_rear_x - self.mul_pos[self.path_stage][-1][0],   ego_rear_y - self.mul_pos[self.path_stage][-1][1]),2)
                #### update path stage
                #print('dist {} and self.hist_use {}'.format(dist, self.hist_use))
                if (dist > self.hist_use and dist < 0.3): #or dist < 0.1
                    print('path segment completed and switch to the next path...')
                    self.my_control.brake = 1.0 #### stop vehicle moving after reaching the target position
                    self.my_control.steer = 0.0
                    self.player.apply_control(self.my_control)
                    carla_world.tick()
                    ### reinitilize the controll buffer
                    self.pid_controller = VehiclePIDController(self.player, lateral_par, longitudinal_par, max_throttle=1.0, max_steering=max_steering) ##0.75
        
                    if self.path_stage == len(self.mul_pos)-1:
                        print('We completed all paths') 
                        self.trans_control.brake = 1.0
                        return
                        
                    self.path_stage += 1  
                    self.path_stage = min(self.path_stage, len(self.mul_pos)-1) 
                    self.hist_use = 10000
                     
                else:
                    self.hist_use = dist    
                     
                    
                if self.current_stage != self.path_stage:
                    self.current_stage = self.path_stage
                    path_lists = np.array(self.mul_pos[self.path_stage][::2])
                    self.r_trajectory = Real_control(path_lists[:,0], path_lists[:,1], path_lists[:,2])
                waypoint_index = self.r_trajectory.nearest_index(ego_rear_x, ego_rear_y)
                
                print('waypoint_index: ', waypoint_index, ' dist to end target: ', dist) ## self.r_trajectory.cx[waypoint_index], self.r_trajectory.cy[waypoint_index],
                #time.sleep(0.1)
                ### convert path point into carla.transform
                location = carla.Location(x=self.r_trajectory.cx[waypoint_index], y= self.r_trajectory.cy[waypoint_index], z=0)
                rotation = carla.Rotation(pitch=0.0, yaw=np.rad2deg(self.r_trajectory.cyaw[waypoint_index]), roll=0.0) 
                waypoint = carla.Transform(location, rotation)

                target_speed = 2.24 ### similar to 5 miles/hour #4.2 #0.21 #3
                # position_vector = [self.r_trajectory.cx[waypoint_index]-self.player.get_transform().location.x, self.r_trajectory.cy[waypoint_index]-self.player.get_transform().location.y]
                # vehicle_forward_vector = [self.player.get_transform().get_forward_vector().x, self.player.get_transform().get_forward_vector().y] 
                # # print('vehicle_forward_vector: ', vehicle_forward_vector) np.dot(vehicle_forward_vector,position_vector) >=0: 
                if self.mul_pos[self.path_stage][-1][-1] == 1: ### forward 2 m/s target speed
                    #print('driving forward')
                    self.my_control = self.pid_controller.run_step(target_speed, waypoint) #5
                    self.my_control.gear = 1
                    #self.my_control.reverse = False
                    ##self.my_control.gear = 2
                else:
                    print('driving backward')
                    #### we have modified the controller to take the reverse 
                    #self.my_control = self.pid_controller.run_step(target_speed, waypoint, -1)

                    
                    ### change the ego heading
                    # self.my_control = self.pid_controller.run_step(target_speed, waypoint, -1)
                    # self.my_control.steer = -self.my_control.steer

                    self.my_control = self.pid_controller.run_step(target_speed, waypoint)

                    self.my_control.reverse = True  
                    
                print(str(self.my_control)) #print(str(self.trans_control))        
        
        self.player.apply_control(self.my_control) ###self.player.apply_control(self.trans_control)
        #self.player.get_transform().location.z = 0.2
        #carla_world.debug.draw_point(self.player.get_transform().location, size=0.03,  color=carla.Color(0,255,0),life_time=10)

        rear_x_ego = self.player.get_transform().location.x - 1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)
        rear_y_ego = self.player.get_transform().location.y - 1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)) #np.round(,1)

        self.ego_traject.append([rear_x_ego, rear_y_ego])




        np_vec = np.array(self.positions) ##[::2]
        np_ego = np.array(self.ego_traject)
        plt.cla()
        plt.scatter(np_ego[-1,1], np_ego[-1,0], label='ego rear')
        plt.plot(np_vec[:,1], np_vec[:,0], label='planned path') ##scatter
        try:
            plt.scatter(self.r_trajectory.cy[waypoint_index], self.r_trajectory.cx[waypoint_index], label='current target') 
        except:
            pass    
        plt.plot(np_ego[:,1], np_ego[:,0], label='ego path')
        plt.axes().set_xticks(np.arange(int(min(np.round(np_vec[:,1]))), int(max(np.round(np_vec[:,1]))), 0.1), minor=True)
        plt.axes().set_yticks(np.arange(int(min(np.round(np_vec[:,0]))), int(max(np.round(np_vec[:,0]))), 0.1), minor=True)
        # plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)
        # plt.tick_params(axis='y', which='minor', bottom=False, top=False, labelbottom=False)
        plt.grid()
        plt.grid(which='minor', alpha=0.3)
        plt.legend()
        plt.pause(0.001)
        

    def speed_limit(self, data_frame):
        # if vehicle stops at initialization, give throttle until Gear turns to 1
        if data_frame['veh_control'].gear == 0:
            self.trans_control.throttle = 0.5

        speed = (3.6 * math.sqrt(
            data_frame['veh_velocity'].x ** 2 + data_frame['veh_velocity'].y ** 2 + data_frame['veh_velocity'].z ** 2))

        # limit the vehicle speed within 15km/h when reverse is False
        if not self.trans_control.reverse and speed >= 12:
            self.trans_control.throttle = 0.0

        # limit the vehicle speed within 8km/h when reverse is True
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
         

        # plt.cla()
        # plt.plot(ox, oy, "sk", markersize=1, label='obstacles')
        # plt.scatter(sx, sy, label='ego rear center')
        # plt.scatter(gx, gy, label='ego rear center')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()


        start_time = time.time()  
        print('Starting to find a path...')  
        result = hybrid_a_star.hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy)
        if result is not None:
            print('It takes {} seconds to find a path'.format((time.time()-start_time)), 'total waypoints: ', len(result[0]))
            return result, ox_real, oy_real, gx, gy, gyaw 
        else:
            print('It takes {} seconds but still cannot find a path'.format((time.time()-start_time)))  
            return None, ox_real, oy_real, gx, gy, gyaw  
            ### choose a safe action, say moving forward with a small step 
    #################################################################################################        

    def get_model_data(self, data_frame):

        data = {}

        vehicle_transform = data_frame['veh_transfrom']
        imu_data = data_frame['imu']
        vehicle_velocity = data_frame['veh_velocity']

        #########################################################################################################################
        ####### semantic camera always rotate with ego so it is always using ego coordinate system.
        #print('semantic data: ', type(data_frame['BEV_semantic']))
        # data_frame['BEV_semantic'].convert(carla.ColorConverter.CityScapesPalette)
        # feng = np.copy(np.frombuffer(data_frame['BEV_semantic'].raw_data, dtype=np.dtype("uint8")))
        # feng = np.reshape(feng, (data_frame['BEV_semantic'].height, data_frame['BEV_semantic'].width, 4))
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
        ### lidar has its own rotation matrix for different time due to rotating of data collection?
        ### https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor
        # print('lidar transform: ', data_frame['lidar'].transform) 
        # print('ego transform: ', str(self.player.get_transform()))

       
        feng2 = np.copy(np.frombuffer(data_frame['lidar'].raw_data, dtype=np.dtype('f4'))) 
        feng2 = np.reshape(feng2, (int(feng2.shape[0]/4), 4))
        intensity = feng2[:, -1] 
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = feng2[:, :-1] 

        # ##### lidar use global coordinate system we convert the pc to ego coordinate system
        # #points[:, 0] = -points[:, 0] ### not knowing why but refer from the lidar example code


        # plt.cla()
        # plt.plot(points[:,1], points[:,0], "sk", markersize=1, label='lidar pc')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()

        # #####method 1
        # rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # ego_lidar_point = (rotation_matrix @ points.T).T  ### here we get lidar point with respect to ego using Carla_garage and ignore the translation in Z
        # plt.cla()
        # plt.plot(ego_lidar_point[:,1], ego_lidar_point[:,0], "sk", markersize=1, label='lidar pc')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()

        # # #####methold 2  ### open3d_lidar.py
        # # # We're negating the y to correclty visualize a world that matches
        # # # what we see in Unreal since Open3D uses a right-handed coordinate system
        # # points2 = points[:,:]
        # # points2 = np.append(points, np.ones((points2.shape[0], 1)), axis=1)
        # # points2 = np.dot(self.player.get_transform().get_matrix(), points2.T).T
        # # plt.cla()
        # # plt.plot(points2[:,1], points2[:,0], "sk", markersize=1, label='lidar pc')
        # # plt.plot(points2[:,0], points2[:,1], "sk", markersize=1, label='lidar pc')
        # # plt.legend()
        # # plt.show()
        # # import pdb; pdb.set_trace()


        # #####method 3 ### lidar_to_camera.py
        # local_lidar_points = feng2[:, :-1].T
        # local_lidar_points2 = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        # world_points = np.dot(data_frame['lidar'].transform.get_matrix(), local_lidar_points2)
        # ego_points = np.dot(np.array(self.player.get_transform().get_inverse_matrix()), world_points) 
        # plt.cla()
        # plt.plot(ego_points[1, :], ego_points[0, :], "sk", markersize=1, label='lidar pc')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()


        # plt.cla()
        # plt.plot(points[:,1], points[:,0], "sk", markersize=1, label='lidar pc-method0')
        # plt.plot(ego_lidar_point[:,1], ego_lidar_point[:,0], ".", markersize=1, label='lidar pc-method1')
        # #plt.plot(points2[:,1], points2[:,0], "sk", markersize=1, label='lidar pc-method2')
        # plt.plot(ego_points[1,:], ego_points[0, :], "<", markersize=1, label='lidar pc-method3')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()

        #####filter out ego and z < 0
        mask = np.all((points >= self.bbox_min) & (points <= self.bbox_max), axis=1)
        filtered_points = points[~mask]

        local_lidar_points = filtered_points.T
        local_lidar_points2 = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        world_points = np.dot(data_frame['lidar'].transform.get_matrix(), local_lidar_points2)
        filtered_points = (np.dot(np.array(self.player.get_transform().get_inverse_matrix()), world_points)).T[:,:-1] 


        height = 1.6+abs(vehicle_transform.location.z)
        #print('Point cloud z range: ', min(filtered_points[:,2]), max(filtered_points[:,2]), '\n')
        filtered_points = filtered_points[filtered_points[:,2]>0.1] ###-height ##originall for lidar coordinate system
        filtered_points[:,2] = 0.05  ### flatten all points to the same z

        # #print(len(filtered_points))
        # ###filtered_points = filtered_points[np.lexsort(np.fliplr(filtered_points).T)]
        # ### np.min(filtered_points, axis=0)

        # plt.cla()
        # plt.plot(filtered_points[:,1], filtered_points[:,0], "sk", markersize=1, label='lidar pc')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()
        
 
        point_list = o3d.geometry.PointCloud()           ######## change left handed to righ handed
        filtered_points[:, [0,1]] = filtered_points[:, [1,0]]
        point_list.points = o3d.utility.Vector3dVector(filtered_points) ### right handed  ##
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        ###### visualize 
        #o3d.visualization.draw_geometries([point_list])
       
        # voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_list, 0.1)

        # [for voxel in voxelGrid.get_voxels()]
        # print('voxel size of the Voxel Grid: ', voxelGrid.voxel_size) 
        # print('Voxel Grid Origin: ', voxelGrid.origin) 
        # print('total voxel number: ', len(voxelGrid.get_voxels()))
        # voxels = voxelGrid.get_voxels()
        # if len(voxels) > 0:
        #     print('grid index check if it is (i,j,k): ', voxels[0].grid_index, voxels[1].grid_index, voxels[2].grid_index)
        #import pdb; pdb.set_trace()
        data['point_list'] = point_list
        ##########################################################################################################################

        

        target_point = convert_slot_coord(vehicle_transform, self.net_eva.eva_parking_goal)

        front_final, self.rgb_front = self.image_process(data_frame['rgb_front'])
        left_final, self.rgb_left = self.image_process(data_frame['rgb_left'])
        right_final, self.rgb_right = self.image_process(data_frame['rgb_right'])
        rear_final, self.rgb_rear = self.image_process(data_frame['rgb_rear'])

        images = [front_final, left_final, right_final, rear_final]
        images = torch.cat(images, dim=0)
        data['image'] = images.unsqueeze(0)

        data['extrinsics'] = self.extrinsic.unsqueeze(0)
        data['intrinsics'] = self.intrinsic_crop.unsqueeze(0)

        velocity = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2))
        data['ego_motion'] = torch.tensor([velocity, imu_data.accelerometer.x, imu_data.accelerometer.y],
                                          dtype=torch.float).unsqueeze(0).unsqueeze(0)

        if self.pre_target_point is not None:
            target_point = [self.pre_target_point[0], self.pre_target_point[1], target_point[2]]
        data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)

        data['gt_control'] = torch.tensor([self.BOS_token], dtype=torch.int64).unsqueeze(0)

        if self.show_eva_imgs:
            img = encode_npy_to_pil(np.asarray(data_frame['topdown'].squeeze().cpu()))
            img = np.moveaxis(img, 0, 2)
            img = Image.fromarray(img)
            seg_gt = self.semantic_process(image=img, scale=0.5, crop=200, target_slot=target_point)
            seg_gt[seg_gt == 1] = 128
            seg_gt[seg_gt == 2] = 255
            data['segmentation'] = Image.fromarray(seg_gt)


         

        return data

    def draw_waypoints(self, waypoints):
        ego_t = self.world.player.get_transform()
        ego_loc = carla.Location(x=ego_t.location.x, y=ego_t.location.y, z=0.20)
        self.world.world.debug.draw_string(ego_loc, 'O', draw_shadow=True, color=carla.Color(255, 0, 0))

        wp_list = waypoints[0].tolist()
        for wp in wp_list:
            logging.info('wp: dx: %4f; dy: %4f;', wp[0], wp[1])
            loc = carla.Location(x=ego_t.location.x + wp[0], y=ego_t.location.y + wp[1], z=0.20)
            self.world.world.debug.draw_string(loc, 'O', draw_shadow=True, color=carla.Color(0, 255, 0))

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
        plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
        rows = 2
        cols = 4
        ax_ctl = plt.subplot(rows, cols, 4)
        ax_ctl.axis('off')
        t_x = 0.2
        t_y = 0.8
        throttle_show = int(self.trans_control.throttle * 1000) / 1000
        steer_show = int(self.trans_control.steer * 1000) / 1000
        brake_show = int(self.trans_control.brake * 1000) / 1000
        reverse_show = self.trans_control.reverse
        ax_ctl.text(t_x, t_y, 'Throttle: ' + str(throttle_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.2, '    Steer: ' + str(steer_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.4, '   Brake: ' + str(brake_show), fontsize=10, color='red')
        ax_ctl.text(t_x, t_y - 0.6, 'Reverse: ' + str(reverse_show), fontsize=10, color='red')

        ax_front = plt.subplot(rows, cols, 1)
        ax_front.axis('off')
        ax_front.set_title('front', fontsize=10)
        ax_front.imshow(self.rgb_front)

        ax_rear = plt.subplot(rows, cols, 2)
        ax_rear.axis('off')
        ax_rear.set_title('rear', fontsize=10)
        ax_rear.imshow(self.rgb_rear)

        ax_atten = plt.subplot(rows, cols, 7)
        ax_atten.axis('off')
        ax_atten.set_title('atten(output)', fontsize=10)
        ax_atten.imshow(self.grid_image)
        ax_atten.imshow(self.atten_avg / np.max(self.atten_avg), alpha=0.6, cmap='rainbow')

        ax_left = plt.subplot(rows, cols, 5)
        ax_left.axis('off')
        ax_left.set_title('left', fontsize=10)
        ax_left.imshow(self.rgb_left)

        ax_right = plt.subplot(rows, cols, 6)
        ax_right.axis('off')
        ax_right.set_title('right', fontsize=10)
        ax_right.imshow(self.rgb_right)

        ax_bev = plt.subplot(rows, cols, 3)
        ax_bev.axis('off')
        ax_bev.set_title('target_bev(input)', fontsize=10)
        ax_bev.imshow(self.target_bev)

        ax_bev = plt.subplot(rows, cols, 8)
        ax_bev.axis('off')
        ax_bev.set_title('seg_bev(output)', fontsize=10)
        ax_bev.imshow(self.seg_bev)

        plt.pause(0.1)
        plt.clf()

    def get_eva_control(self):
        control = {
            'throttle': self.trans_control.throttle,
            'steer': self.trans_control.steer,
            'brake': self.trans_control.brake,
            'reverse': self.trans_control.reverse,
        }
        return control
