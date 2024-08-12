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
import open3d as o3d

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic
from dataset.carla_dataset import detokenize
from data_generation.network_evaluator import NetworkEvaluator
from data_generation.tools import encode_npy_to_pil
from model.parking_model import ParkingModel
from collections import Counter
from copy import deepcopy
import cv2

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

        self.rgb_back = None
        self.rgb_back_left = None
        self.rgb_back_right = None
        self.rgb_front_right = None
        self.rgb_front_left = None
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
        self.load_model(args.model_path, args.model_path_conet)

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

    def load_model(self, parking_pth_path, conet_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModel(self.cfg)
        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        ckpt_conet = torch.load(conet_pth_path, map_location='cuda:0')

        if self.cfg.feature_encoder == "conet":
            state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
            state_dict = OrderedDict([(k, v) for k, v in state_dict.items() if not (k.startswith('bev_model') or k.startswith('bev_encoder'))])
            # Change later
            # state_dict = OrderedDict([(k.replace('bev_model', 'conet_model').replace('bev_encoder', 'conet_encoder'), v) for k, v in state_dict.items()])
            parts_to_include = ['img_backbone', 'img_neck', 'img_view_transformer', 'occ_encoder', 'occ_encoder_neck']
            state_dict.update({k: v for k, v in ckpt_conet['state_dict'].items() if any(part in k for part in parts_to_include)})
            # state_dict.update({k: v for k, v in ckpt_conet['state_dict'].items() if k.startswith('img_backbone') or k.startswith('img_neck')})
        else:
            state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if self.cfg.feature_encoder == "conet":
            patch_attention(self.model.conet_fusion.tf_encoder.layers[-1].self_attn)
            self.hook_handle = self.model.conet_fusion.tf_encoder.layers[-1].self_attn.register_forward_hook(
                self.save_output)
        elif self.cfg.feature_encoder == "bev":
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
    
    def save_prev_target_conet(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_seg_img = pred_segmentation[0, :, :, :][::-1]

        h, w, d = pred_seg_img.shape
        target_slot_x = []
        target_slot_y = []
        target_slot_z = []
        for row_idx in range(h):
            for col_idx in range(w):
                for dep_idx in range(d):
                    if pred_seg_img[row_idx, col_idx, dep_idx] != 0:
                        target_slot_x.append(row_idx)
                        target_slot_y.append(col_idx)
                        target_slot_z.append(dep_idx)

        # target point in bev
        if (len(target_slot_x) > 0):
            new_target_x = int(np.average(target_slot_x))
            new_target_y = int(np.average(target_slot_y))
            new_target_z = int(np.average(target_slot_z))
            self.pre_target_point = self.get_target_point_ego_coord_conet(pred_seg_img, [new_target_x, new_target_y, new_target_z])

    def get_target_point_ego_coord(self, pred_seg_img, target_point_pixel_idx):
        bev_shape_x, bev_shape_y = pred_seg_img.shape
        x = -(target_point_pixel_idx[0] - bev_shape_x / 2)
        y = target_point_pixel_idx[1] - bev_shape_y / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2]]
        return target_point_ego_coord
    
    def get_target_point_ego_coord_conet(self, pred_seg_img, target_point_pixel_idx):
        bev_shape_x, bev_shape_y, bev_shape_z = pred_seg_img.shape
        x = -(target_point_pixel_idx[0] - bev_shape_x / 2)
        y = target_point_pixel_idx[1] - bev_shape_y / 2
        z = target_point_pixel_idx[2] - bev_shape_z / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2], z * self.cfg.bev_z_bound[2]]
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
        self.intrinsic_crop = self.intrinsic_crop.unsqueeze(0).expand(6, 3, 3)

        veh2cam_dict = self.world.veh2cam_dict
        front_to_ego = torch.from_numpy(veh2cam_dict['rgb_front']).float().unsqueeze(0)
        front_left_to_ego = torch.from_numpy(veh2cam_dict['rgb_front_left']).float().unsqueeze(0)
        front_right_to_ego = torch.from_numpy(veh2cam_dict['rgb_front_right']).float().unsqueeze(0)
        back_to_ego = torch.from_numpy(veh2cam_dict['rgb_back']).float().unsqueeze(0)
        back_left_to_ego = torch.from_numpy(veh2cam_dict['rgb_back_left']).float().unsqueeze(0)
        back_right_to_ego = torch.from_numpy(veh2cam_dict['rgb_back_right']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, front_left_to_ego, front_right_to_ego, back_to_ego,back_left_to_ego,back_right_to_ego], dim=0)

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

            data = self.get_model_data(data_frame)

            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                import pdb; pdb.set_trace()
                pred_controls, pred_segmentation, _, target_bev = self.model.predict(data)

                end_time = time.time()
                self.net_eva.inference_time.append(end_time - start_time)
                if self.cfg.feature_encoder == 'bev':
                    self.save_prev_target(pred_segmentation)
                elif self.cfg.feature_encoder == 'conet':
                    self.save_prev_target_conet(pred_segmentation)

                control_signal = detokenize(pred_controls[0].tolist()[1:], self.cfg.token_nums)

                self.trans_control.throttle = control_signal[0]
                self.trans_control.brake = control_signal[1]
                self.trans_control.steer = control_signal[2]
                self.trans_control.reverse = control_signal[3]

                self.speed_limit(data_frame)

                if self.show_eva_imgs:
                    self.grid_image, self.atten_avg = self.save_atten_avg_map(data)
                    self.save_seg_img(pred_segmentation)
                    self.save_target_bev_img(target_bev)
                    self.display_imgs()

                self.save_output.clear()

            self.prev_xy_thea = [vehicle_transform.location.x,
                                 vehicle_transform.location.y,
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

    def get_model_data(self, data_frame):
        vehicle_transform = data_frame['veh_transfrom']
        imu_data = data_frame['imu']
        vehicle_velocity = data_frame['veh_velocity']

        data = {}

        target_point = convert_slot_coord(vehicle_transform, self.net_eva.eva_parking_goal)

        front_final, self.rgb_front = self.image_process(data_frame['rgb_front'])
        front_left_final, self.rgb_front_left = self.image_process(data_frame['rgb_front_left'])
        front_right_final, self.rgb_front_right = self.image_process(data_frame['rgb_front_right'])
        back_right_final, self.rgb_back_right = self.image_process(data_frame['rgb_back_right'])
        back_left_final, self.rgb_bleft_right = self.image_process(data_frame['rgb_back_left'])
        back_final, self.rgb_back = self.image_process(data_frame['rgb_back'])

        images = [front_final, front_left_final,front_right_final,back_final,back_left_final,back_right_final]
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
            if self.cfg.feature_encoder == "bev":
                img = encode_npy_to_pil(np.asarray(data_frame['topdown'].squeeze().cpu()))
                img = np.moveaxis(img, 0, 2)
                img = Image.fromarray(img)
                seg_gt = self.semantic_process(image=img, scale=0.5, crop=200, target_slot=target_point)
                seg_gt[seg_gt == 1] = 128
                seg_gt[seg_gt == 2] = 255
                data['segmentation'] = Image.fromarray(seg_gt)
            elif self.cfg.feature_encoder == "conet":
                segmentation = self.semantic_process3D(data_frame['lidar'], visual=True)
                
        return data

    def semantic_process3D(self, lidar, visual=False):
        data = np.frombuffer(lidar.raw_data, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('cos_angle', np.float32),
            ('object_idx', np.uint32),
            ('object_tag', np.uint32)
        ]))
        points = np.stack((data['x'],data['y'],data['z']), axis=-1)
        points = self.lidar2ego(points,np.array([0,0,1.6]))
        categories = data['object_tag'].astype(int)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        resolution = (self.cfg.point_cloud_range[3] - self.cfg.point_cloud_range[0]) / self.cfg.occ_size[0]
        min_bound = self.cfg.point_cloud_range[0:3]
        max_bound = self.cfg.point_cloud_range[3:]
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, resolution, min_bound, max_bound)
        
        # Map each point to its corresponding voxel and category
        point_to_voxel_map = {}
        for point, category in zip(np.asarray(pcd.points), categories):
            if point[0]<max_bound[0] and point[0]>min_bound[0] and point[1]<max_bound[1] and point[1]>min_bound[1] and point[2]<max_bound[2] and point[2]>min_bound[2]:
                voxel_index = tuple(voxel_grid.get_voxel(point))
                if voxel_index in point_to_voxel_map:
                    point_to_voxel_map[voxel_index].append(category)
                else:
                    point_to_voxel_map[voxel_index] = [category]
        # Determine the majority category for each voxel
        voxel_categories = {voxel: self.most_common(categories) for voxel, categories in point_to_voxel_map.items()}
        for key in voxel_categories.keys():
            value = convert_semantic_label(voxel_categories[key])
            voxel_categories[key]=value
        if visual:
            NUSC_COLOR_MAP = {  # RGB.
                # 0: (0, 0, 0),  # Black. noise
                1: (112, 128, 144),  # Slategrey barrier
                2: (220, 20, 60),  # Crimson bicycle
                3: (255, 127, 80),  # Orangered bus
                4: (255, 158, 0),  # Orange car
                5: (233, 150, 70),  # Darksalmon construction
                6: (255, 61, 99),  # Red motorcycle
                7: (0, 0, 230),  # Blue pedestrian
                8: (47, 79, 79),  # Darkslategrey trafficcone
                9: (255, 140, 0),  # Darkorange trailer
                10: (255, 99, 71),  # Tomato truck
                11: (0, 207, 191),  # nuTonomy green driveable_surface
                12: (175, 0, 75),  # flat other
                13: (75, 0, 75),  # sidewalk
                14: (112, 180, 60),  # terrain
                15: (222, 184, 135),  # Burlywood mannade
                16: (0, 175, 0),  # Green vegetation
            }
            boxes = o3d.geometry.TriangleMesh()
            for coord_inds, values in voxel_categories.items():
                box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
                pose = np.array(coord_inds)
                pose[0] = 320 - pose[0]
                pose = np.array(min_bound)+ pose*0.2
                box.translate(pose)
                if values != 255:
                    # import pdb; pdb.set_trace()
                    # box.paint_uniform_color(np.array([255, 0, 0]) / 255.0)
                    box.paint_uniform_color(np.array(NUSC_COLOR_MAP[values]) / 255.0)
                else:
                    # box.paint_uniform_color((0.0, 0.0, 0.0))
                    continue
                boxes += box
            render_img_high_view([boxes], fname="./visual/voxel.png")
        # Color mapping normalized to [0, 1]
        import pdb; pdb.set_trace()
        return voxel_categories
    
    def lidar2ego(self, points,translation,rotation=None):
        # input should be (n,3)
        if(rotation is not None):
            print("lidar should be at same orientation with vehicle!!!!")
        translated_points = points + translation
        return translated_points

    def most_common(self, lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

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


def render_img(geometries, fname=None, view_point=-100, zoom=0.7, show_wireframe=True, point_size=5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1360, height=1080)
    for g in geometries:
        vis.add_geometry(g)
    renderoption = vis.get_render_option()
    # renderoption.load_from_json('/home/wru1syv/avp/data/renderoption.json')
    renderoption.mesh_show_wireframe = show_wireframe
    renderoption.point_size = point_size
    ctr = vis.get_view_control()
    ctr.rotate(0.0, view_point)
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer())
    if fname:
        # import pdb; pdb.set_trace()
        cv2.imwrite(fname, (img[:, :, ::-1] * 255).astype(np.uint8))
    vis.destroy_window()
    return img

def render_img_cam_param(geometries, cam_param, fname=None, show_wireframe=True, point_size=5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=cam_param.intrinsic.width, height=cam_param.intrinsic.height)
    for g in geometries:
        vis.add_geometry(g)
    renderoption = vis.get_render_option()
    renderoption.load_from_json('/home/wru1syv/avp/data/renderoption.json')
    renderoption.mesh_show_wireframe = show_wireframe
    renderoption.point_size = point_size
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer())
    if fname:
        cv2.imwrite(fname, (img[:, :, ::-1] * 255).astype(np.uint8))
    vis.destroy_window()
    return img


def render_img_high_view(geometries, **kwargs):
    kwargs_default = {'fname': None, 'view_point': -100, 'zoom': 0.7, 'show_wireframe': True, 'point_size': 5}
    kwargs = {**kwargs_default, **kwargs}
    # R = geometries[0].get_rotation_matrix_from_zyx((np.pi / 2, 0, 0))
    # R = np.array([[-1,0,0],[0,1,0],[0,0,1]])
    R = geometries[0].get_rotation_matrix_from_zyx((-np.pi / 2, 0, 0))
    # R = R@R_

    # import pdb; pdb.set_trace()
    geometries_rot = []
    for g in geometries:
        dg = deepcopy(g)
        if type(g) != o3d.geometry.AxisAlignedBoundingBox:
            dg.rotate(R, center=(0, 0, 0))
        geometries_rot.append(dg) 
    img = render_img(geometries_rot, **kwargs)
    return img


# convert from carla to nuscenes
def convert_semantic_label(category_index):
    carla_categories = {
        0: "Unlabeled",
        1: "Building",
        2: "Fence",
        3: "Other",
        4: "Pedestrian",
        5: "Pole",
        6: "RoadLine",
        7: "Road",
        8: "SideWalk",
        9: "Vegetation",
        10: "Vehicles",
        11: "Wall",
        12: "TrafficSign",
        13: "Sky",
        14: "Ground",
        15: "Bridge",
        16: "RailTrack",
        17: "GuardRail",
        18: "TrafficLight",
        19: "Static",
        20: "Dynamic",
        21: "Water",
        22: "Terrain"
    }
    categories = {
        1: "barrier",
        2: "bicycle",
        3: "bus",
        4: "car",
        5: "construction_vehicle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic_cone",
        9: "trailer",
        10: "truck",
        11: "driveable_surface",
        12: "other_flat",
        13: "sidewalk",
        14: "terrain",
        15: "manmade",
        16: "vegetation"
    }
    mapping = {
        0: 1,   # Unlabeled -> barrier
        1: 15,  # Building -> manmade
        2: 1,   # Fence -> barrier
        3: 1,   # Other -> barrier
        4: 7,   # Pedestrian -> pedestrian
        5: 1,   # Pole -> barrier
        6: 12,  # RoadLine -> other_flat
        7: 11,  # Road -> driveable_surface
        8: 13,  # SideWalk -> sidewalk
        9: 16,  # Vegetation -> vegetation
        10: 4,  # Vehicles -> car
        11: 1,  # Wall -> barrier
        12: 1,  # TrafficSign -> barrier
        13: 12, # Sky -> other_flat
        14: 12, # Ground -> other_flat
        15: 12, # Bridge -> other_flat
        16: 12, # RailTrack -> other_flat
        17: 1,  # GuardRail -> barrier
        18: 1,  # TrafficLight -> barrier
        19: 1,  # Static -> barrier
        20: 1,  # Dynamic -> barrier
        21: 14, # Water -> terrain
        22: 14  # Terrain -> terrain
    }
    return mapping.get(category_index, 1) 