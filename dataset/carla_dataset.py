import json
import os
import carla
import torch.utils.data
import numpy as np
import torchvision.transforms

from PIL import Image
from loguru import logger
from data_generation.world import cam_specs_
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
import torch.nn.functional as F

def convert_slot_coord(ego_trans, target_point):
    """
    Convert target parking slot from world frame into self_veh frame
    :param ego_trans: veh2world transform
    :param target_point: target parking slot in world frame [x, y, yaw]
    :return: target parking slot in veh frame [x, y, yaw]
    """
    target_point_self_veh = convert_veh_coord(target_point[0], target_point[1], 1.0, ego_trans)

    yaw_diff = target_point[2] - ego_trans.rotation.yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    target_point = [target_point_self_veh[0], target_point_self_veh[1], yaw_diff]

    return target_point

def convert_slot_coord3D(ego_trans, target_point):
    """
    Convert target parking slot from world frame into self_veh frame
    :param ego_trans: veh2world transform
    :param target_point: target parking slot in world frame [x, y, yaw]
    :return: target parking slot in veh frame [x, y, yaw]
    """
    target_point_self_veh = convert_veh_coord3D(target_point[0], target_point[1], target_point[2], ego_trans)

    yaw_diff = target_point[3] - ego_trans.rotation.yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360
    target_point = [target_point_self_veh[0], target_point_self_veh[1], target_point_self_veh[2], yaw_diff]

    return target_point


def convert_veh_coord(x, y, z, ego_trans):
    """
    Convert coordinate (x,y,z) in world frame into self-veh frame
    :param x:
    :param y:
    :param z:
    :param ego_trans: veh2world transform
    :return: coordinate in self-veh frame
    """

    world2veh = np.array(ego_trans.get_inverse_matrix())
    target_array = np.array([x, y, z, 1.0], dtype=float)
    target_point_self_veh = world2veh @ target_array
    return target_point_self_veh

def convert_veh_coord3D(x, y, z, ego_trans):
    """
    Convert coordinate (x,y,z) in world frame into self-veh frame
    :param x:
    :param y:
    :param z:
    :param ego_trans: veh2world transform
    :return: coordinate in self-veh frame
    """

    world2veh = np.array(ego_trans.get_inverse_matrix())
    target_array = np.array([x, y, z, 1.0], dtype=float)
    target_point_self_veh = world2veh @ target_array
    return target_point_self_veh

def scale_and_crop_image(image, scale=1.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array
    :param image: original image
    :param scale: scale factor
    :param crop: crop size
    :return: cropped image
    """

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), resample=Image.NEAREST)
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop].copy()
    return cropped_image


def tokenize(throttle, brake, steer, reverse, token_nums=200):
    """
    Tokenize control signal
    :param throttle: [0,1]
    :param brake: [0,1]
    :param steer: [-1,1]
    :param reverse: {0,1}
    :param token_nums: size of token
    :return: tokenized control range [0, token_nums-4]
    """

    valid_token = token_nums - 4
    half_token = valid_token / 2

    if brake != 0.0:
        throttle_brake_token = int(half_token * (-brake + 1))
    else:
        throttle_brake_token = int(half_token * (throttle + 1))
    steer_token = int((steer + 1) * half_token)
    reverse_token = int(reverse * valid_token)
    return [throttle_brake_token, steer_token, reverse_token]


def detokenize(token_list, token_nums=200):
    """
    Detokenize control signals
    :param token_list: [throttle_brake, steer, reverse]
    :param token_nums: size of token number
    :return: control signal values
    """

    valid_token = token_nums - 4
    half_token = float(valid_token / 2)

    if token_list[0] > half_token:
        throttle = token_list[0] / half_token - 1
        brake = 0.0
    else:
        throttle = 0.0
        brake = -(token_list[0] / half_token - 1)

    steer = (token_list[1] / half_token) - 1
    reverse = (True if token_list[2] > half_token else False)

    return [throttle, brake, steer, reverse]


def get_depth(depth_image_path, crop):
    """
    Convert carla RGB depth image into single channel depth in meters
    :param depth_image_path: carla depth image in RGB format
    :param crop: crop size
    :return: numpy array of depth image in meters
    """
    depth_image = Image.open(depth_image_path).convert('RGB')

    data = np.array(scale_and_crop_image(depth_image, scale=1.0, crop=crop))

    data = data.astype(np.float32)

    normalized = np.dot(data, [1.0, 256.0, 65536.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return torch.from_numpy(in_meters).unsqueeze(0)


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    update_intrinsic = intrinsics.clone()

    update_intrinsic[0, 0] *= scale_width
    update_intrinsic[0, 2] *= scale_width
    update_intrinsic[1, 1] *= scale_height
    update_intrinsic[1, 2] *= scale_height

    update_intrinsic[0, 2] -= left_crop
    update_intrinsic[1, 2] -= top_crop

    return update_intrinsic


def add_raw_control(data, throttle_brake, steer, reverse):
    if data['Brake'] != 0.0:
        throttle_brake.append(-data['Brake'])
    else:
        throttle_brake.append(data['Throttle'])
    steer.append(data['Steer'])
    reverse.append(int(data['Reverse']))


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, config):
        super(CarlaDataset, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums - 3
        self.EOS_token = self.BOS_token + 1
        self.PAD_token = self.EOS_token + 1

        self.root_dir = root_dir
        self.is_train = is_train

        # camera configs
        self.image_crop = self.cfg.image_crop
        self.intrinsic = None
        self.veh2cam_dict = {}
        self.extrinsic = None
        self.image_process = ProcessImage(self.image_crop)
        # if self.cfg.feature_encoder == "bev":
        # self.semantic_process = ProcessSemantic(self.cfg)
        # elif self.cfg.feature_encoder == "conet":
        self.semantic_process3D = ProcessSemantic3D(self.cfg)
        self.init_camera_config()

        # data
        self.front = []
        self.front_left = []
        self.front_right = []
        self.back = []
        self.back_left = []
        self.back_right = []

        self.front_depth = []
        self.front_left_depth = []
        self.front_right_depth = []
        self.back_depth = []
        self.back_left_depth = []
        self.back_right_depth = []

        self.control = []

        self.velocity = []
        self.acc_x = []
        self.acc_y = []

        self.throttle_brake = []
        self.steer = []
        self.reverse = []

        self.target_point = []

        self.topdown = []
        self.voxel = []

        self.get_data()

    def init_camera_config(self):
        cam_config = {'width': 400, 'height': 300, 'fov': 100}
        cam_specs = cam_specs_

        # intrinsic
        w = cam_config['width']
        h = cam_config['height']
        fov = cam_config['fov']
        f = w / (2 * np.tan(fov * np.pi / 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic_original = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        self.intrinsic = update_intrinsics(
            torch.from_numpy(intrinsic_original).float(),
            (h - self.image_crop) / 2,
            (w - self.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic = self.intrinsic.unsqueeze(0).expand(6, 3, 3)
        keys = ['rgb_front', 'rgb_front_left', 'rgb_front_right', 'rgb_back', 'rgb_back_left', 'rgb_back_right']
        sensor2egos = []
        for key in keys:
            cam_spec = cam_specs[key]
            ego2sensor = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                    carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                    roll=cam_spec['roll']))
        # sensor2ego = cam2pixel @ np.array(ego2sensor.get_inverse_matrix())
            sensor2ego = np.array(ego2sensor.get_inverse_matrix())
            sensor2egos.append(torch.from_numpy(sensor2ego).float().unsqueeze(0))
        sensor2egos = torch.cat(sensor2egos)
        # .unsqueeze(0).repeat(self.cfg.batch_size,1,1,1)
        self.extrinsic = sensor2egos

    def plot_grid_2D(self, twoD_map, save_path=None, vmax=None, layer=None):
        H, W = twoD_map.shape

        # twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        cmap = plt.cm.viridis # viridis color projection

        if vmax is None:
            vmax=np.max(twoD_map)*1.2
        plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

        color_legend = plt.colorbar()
        color_legend.set_label('Color Legend') # legend

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def get_data_2D(self):
        val_towns = self.cfg.validation_map
        train_towns = self.cfg.training_map
        train_data = os.path.join(self.root_dir, train_towns)
        val_data = os.path.join(self.root_dir, val_towns)

        town_dir = train_data if self.is_train == 1 else val_data

        # collect all parking data tasks
        root_dirs = os.listdir(town_dir)
        all_tasks = []
        for root_dir in root_dirs:
            root_path = os.path.join(town_dir, root_dir)
            for task_dir in os.listdir(root_path):
                task_path = os.path.join(root_path, task_dir)
                all_tasks.append(task_path)

        for task_path in all_tasks:
            total_frames = len(os.listdir(task_path + "/measurements/"))
            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                # collect data at current frame
                # image
                filename = f"{str(frame).zfill(4)}.png"
                self.front.append(task_path + "/rgb_front/" + filename)
                self.front_left.append(task_path + "/rgb_left/" + filename)
                self.front_right.append(task_path + "/rgb_right/" + filename)
                self.back.append(task_path + "/rgb_rear/" + filename)
                # self.back_left.append(task_path + "/camera_back_left/" + filename)
                # self.back_right.append(task_path + "/camera_back_right/" + filename)

                # depth
                self.front_depth.append(task_path + "/depth_front/" + filename)
                self.front_left_depth.append(task_path + "/depth_left/" + filename)
                self.front_right_depth.append(task_path + "/depth_right/" + filename)
                self.back_depth.append(task_path + "/depth_rear/" + filename)
                # self.back_left_depth.append(task_path + "/depth_back/" + filename)
                # self.back_right_depth.append(task_path + "/depth_back/" + filename)

                # BEV Semantic
                self.topdown.append(task_path + "/topdown/encoded_" + filename)
                self.voxel.append(task_path + "/voxel/" + filename.split(".")[0]+"_info.npy")
                # import pdb; pdb.set_trace()
                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # ego position
                ego_trans = carla.Transform(carla.Location(x=data['x'], y=data['y'], z=data['z']),
                                            carla.Rotation(yaw=data['yaw'], pitch=data['pitch'], roll=data['roll']))

                # motion
                self.velocity.append(data['speed'])
                self.acc_x.append(data['acc_x'])
                self.acc_y.append(data['acc_y'])

                # control
                controls = []
                throttle_brakes = []
                steers = []
                reverse = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + 1 + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                    controls.append(
                        tokenize(data['Throttle'], data["Brake"], data["Steer"], data["Reverse"], self.cfg.token_nums))
                    add_raw_control(data, throttle_brakes, steers, reverse)

                controls = [item for sublist in controls for item in sublist]
                controls.insert(0, self.BOS_token)
                controls.append(self.EOS_token)
                controls.append(self.PAD_token)
                self.control.append(controls)

                self.throttle_brake.append(throttle_brakes)
                self.steer.append(steers)
                self.reverse.append(reverse)

                # target point
                # if self.cfg.only_3d_perception == True:
                #     data = {"x":0,"y":0,"z":0,"yaw":0}
                # elif self.cfg.only_3d_perception == False:
                with open(task_path + f"/parking_goal/0001.json", "r") as read_file:
                    data = json.load(read_file)
                # import pdb; pdb.set_trace()
                parking_goal = [data['x'], data['y'], data['yaw']]
                #todo fix z
                parking_goal = convert_slot_coord(ego_trans, parking_goal)
                self.target_point.append(parking_goal)

        self.front = np.array(self.front).astype(np.string_)
        self.front_left = np.array(self.front_left).astype(np.string_)
        self.front_right = np.array(self.front_right).astype(np.string_)
        self.back = np.array(self.back).astype(np.string_)
        # self.back_left = np.array(self.back_left).astype(np.string_)
        # self.back_right = np.array(self.back_right).astype(np.string_)

        self.front_depth = np.array(self.front_depth).astype(np.string_)
        self.front_left_depth = np.array(self.front_left_depth).astype(np.string_)
        self.front_right_depth = np.array(self.front_right_depth).astype(np.string_)
        self.back_depth = np.array(self.back_depth).astype(np.string_)
        # self.back_left_depth = np.array(self.back_left_depth).astype(np.string_)
        # self.back_right_depth = np.array(self.back_right_depth).astype(np.string_)

        self.topdown = np.array(self.topdown).astype(np.string_)
        self.voxel = np.array(self.voxel).astype(np.string_)

        self.velocity = np.array(self.velocity).astype(np.float32)
        self.acc_x = np.array(self.acc_x).astype(np.float32)
        self.acc_y = np.array(self.acc_y).astype(np.float32)

        self.control = np.array(self.control).astype(np.int64)

        self.throttle_brake = np.array(self.throttle_brake).astype(np.float32)
        self.steer = np.array(self.steer).astype(np.float32)
        self.reverse = np.array(self.reverse).astype(np.int64)

        self.target_point = np.array(self.target_point).astype(np.float32)

        logger.info('Preloaded {} sequences', str(len(self.front)))

    def get_data(self):
        val_towns = self.cfg.validation_map
        train_towns = self.cfg.training_map
        train_data = os.path.join(self.root_dir, train_towns)
        val_data = os.path.join(self.root_dir, val_towns)

        town_dir = train_data if self.is_train == 1 else val_data

        # collect all parking data tasks
        root_dirs = os.listdir(town_dir)
        all_tasks = []
        for root_dir in root_dirs:
            root_path = os.path.join(town_dir, root_dir)
            for task_dir in os.listdir(root_path):
                task_path = os.path.join(root_path, task_dir)
                all_tasks.append(task_path)

        for task_path in all_tasks:
            total_frames = len(os.listdir(task_path + "/measurements/"))
            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                # collect data at current frame
                # image
                filename = f"{str(frame).zfill(4)}.png"
                self.front.append(task_path + "/camera_front/" + filename)
                self.front_left.append(task_path + "/camera_front_left/" + filename)
                self.front_right.append(task_path + "/camera_front_right/" + filename)
                self.back.append(task_path + "/camera_back/" + filename)
                self.back_left.append(task_path + "/camera_back_left/" + filename)
                self.back_right.append(task_path + "/camera_back_right/" + filename)

                # depth
                self.front_depth.append(task_path + "/depth_front/" + filename)
                self.front_left_depth.append(task_path + "/depth_front_left/" + filename)
                self.front_right_depth.append(task_path + "/depth_front_right/" + filename)
                self.back_depth.append(task_path + "/depth_back/" + filename)
                self.back_left_depth.append(task_path + "/depth_back/" + filename)
                self.back_right_depth.append(task_path + "/depth_back/" + filename)

                # BEV Semantic
                self.topdown.append(task_path + "/topdown/encoded_" + filename)
                self.voxel.append(task_path + "/voxel/" + filename.split(".")[0]+"_info.npy")
                # import pdb; pdb.set_trace()
                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # ego position
                ego_trans = carla.Transform(carla.Location(x=data['x'], y=data['y'], z=data['z']),
                                            carla.Rotation(yaw=data['yaw'], pitch=data['pitch'], roll=data['roll']))

                # motion
                self.velocity.append(data['speed'])
                self.acc_x.append(data['acc_x'])
                self.acc_y.append(data['acc_y'])

                # control
                controls = []
                throttle_brakes = []
                steers = []
                reverse = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + 1 + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                    data["Steer"] = np.clip(data["Steer"], -1, 1)
                    controls.append(
                        tokenize(data['Throttle'], data["Brake"], data["Steer"], data["Reverse"], self.cfg.token_nums))
                    add_raw_control(data, throttle_brakes, steers, reverse)

                controls = [item for sublist in controls for item in sublist]
                controls.insert(0, self.BOS_token)
                controls.append(self.EOS_token)
                controls.append(self.PAD_token)
                self.control.append(controls)

                self.throttle_brake.append(throttle_brakes)
                self.steer.append(steers)
                self.reverse.append(reverse)

                # target point
                # if self.cfg.only_3d_perception == True:
                #     data = {"x":0,"y":0,"z":0,"yaw":0}
                # elif self.cfg.only_3d_perception == False:
                with open(task_path + f"/parking_goal/0001.json", "r") as read_file:
                    data = json.load(read_file)
                # import pdb; pdb.set_trace()
                # parking_goal = [data['x'], data['y'], data['yaw']]
                if 'z' in data:
                    parking_goal = [data['x'], data['y'], data['z'], data['yaw']]
                else:
                    parking_goal = [data['x'], data['y'], 0, data['yaw']]
                #todo fix z
                parking_goal = convert_slot_coord3D(ego_trans, parking_goal)
                self.target_point.append(parking_goal)

        self.front = np.array(self.front).astype(np.string_)
        self.front_left = np.array(self.front_left).astype(np.string_)
        self.front_right = np.array(self.front_right).astype(np.string_)
        self.back = np.array(self.back).astype(np.string_)
        self.back_left = np.array(self.back_left).astype(np.string_)
        self.back_right = np.array(self.back_right).astype(np.string_)

        self.front_depth = np.array(self.front_depth).astype(np.string_)
        self.front_left_depth = np.array(self.front_left_depth).astype(np.string_)
        self.front_right_depth = np.array(self.front_right_depth).astype(np.string_)
        self.back_depth = np.array(self.back_depth).astype(np.string_)
        self.back_left_depth = np.array(self.back_left_depth).astype(np.string_)
        self.back_right_depth = np.array(self.back_right_depth).astype(np.string_)

        self.topdown = np.array(self.topdown).astype(np.string_)
        self.voxel = np.array(self.voxel).astype(np.string_)

        self.velocity = np.array(self.velocity).astype(np.float32)
        self.acc_x = np.array(self.acc_x).astype(np.float32)
        self.acc_y = np.array(self.acc_y).astype(np.float32)

        self.control = np.array(self.control).astype(np.int64)

        self.throttle_brake = np.array(self.throttle_brake).astype(np.float32)
        self.steer = np.array(self.steer).astype(np.float32)
        self.reverse = np.array(self.reverse).astype(np.int64)

        self.target_point = np.array(self.target_point).astype(np.float32)

        logger.info('Preloaded {} sequences', str(len(self.front)))

    def __len__(self):
        return len(self.front)

    def __getitem__(self, index):
        data = {}
        keys = ['image', 'depth', 'extrinsics', 'intrinsics', 'target_point', 'ego_motion', 'segmentation',
                'gt_control', 'gt_acc', 'gt_steer', 'gt_reverse']
        for key in keys:
            data[key] = []

        # image & extrinsics & intrinsics
        images = [self.image_process(self.front[index])[0], self.image_process(self.front_left[index])[0],
                  self.image_process(self.front_right[index])[0], self.image_process(self.back[index])[0],
                  self.image_process(self.back_left[index])[0],self.image_process(self.back_right[index])[0]]

        # images = [self.image_process(self.front[index])[0], self.image_process(self.front_left[index])[0],
        #           self.image_process(self.front_right[index])[0], self.image_process(self.back[index])[0]]

        images = torch.cat(images, dim=0)
        data['image'] = images

        data['extrinsics'] = self.extrinsic
        data['intrinsics'] = self.intrinsic

        # depth
        depths = [get_depth(self.front_depth[index], self.image_crop),
                  get_depth(self.front_left_depth[index], self.image_crop),
                  get_depth(self.front_right_depth[index], self.image_crop),
                  get_depth(self.back_depth[index], self.image_crop),
                  get_depth(self.back_left_depth[index], self.image_crop),
                  get_depth(self.back_right_depth[index], self.image_crop)]
        # depths = [get_depth(self.front_depth[index], self.image_crop),
        #           get_depth(self.front_left_depth[index], self.image_crop),
        #           get_depth(self.front_right_depth[index], self.image_crop),
        #           get_depth(self.back_depth[index], self.image_crop)]
        depths = torch.cat(depths, dim=0)
        data['depth'] = depths

        # segmentation
        # if self.cfg.feature_encoder == "bev":
        # segmentation = self.semantic_process(self.topdown[index], scale=0.5, crop=200,
        #                                     target_slot=self.target_point[index])
        # elif self.cfg.feature_encoder == "conet":
        segmentation = self.semantic_process3D(self.voxel[index],
                                            target_slot=self.target_point[index])
        
        
        # self.plot_grid_2D(segmentation, os.path.join("visual", "gt.png"))
        # print("segmentation:::",segmentation.shape)
        # self.plot_grid_2D(segmentation, os.path.join("visual", "gt_fine_car.png"))
        # import pdb; pdb.set_trace()             
        data['segmentation'] = torch.from_numpy(segmentation).long().unsqueeze(0)

        # target_point
        data['target_point'] = torch.from_numpy(self.target_point[index])

        # ego_motion
        ego_motion = np.column_stack((self.velocity[index], self.acc_x[index], self.acc_y[index]))
        data['ego_motion'] = torch.from_numpy(ego_motion)

        # gt control token
        data['gt_control'] = torch.from_numpy(self.control[index])

        # gt control raw
        data['gt_acc'] = torch.from_numpy(self.throttle_brake[index])
        data['gt_steer'] = torch.from_numpy(self.steer[index])
        data['gt_reverse'] = torch.from_numpy(self.reverse[index])
        return data


class ProcessSemantic:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image, scale, crop, target_slot):
        """
        Process original BEV ground truth image; return cropped image with target slot
        :param image: PIL Image or path to image
        :param scale: scale factor
        :param crop: image crop size
        :param target_slot: center location of the target parking slot in meters; vehicle frame
        :return: processed BEV semantic ground truth
        """

        # read image from disk
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert('L')

        # crop image
        cropped_image = scale_and_crop_image(image, scale, crop)

        # draw target slot on BEV semantic
        cropped_image = self.draw_target_slot(cropped_image, target_slot)

        # create a new BEV semantic GT
        h, w = cropped_image.shape
        vehicle_index = cropped_image == 75
        target_index = cropped_image == 255
        semantics = np.zeros((h, w))
        semantics[vehicle_index] = 1
        semantics[target_index] = 2
        # LSS method vehicle toward positive x-axis on image
        semantics = semantics[::-1]

        return semantics.copy()

    def plot_grid_2D(self, twoD_map, save_path=None, vmax=None, layer=None):
        H, W = twoD_map.shape

        # twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        cmap = plt.cm.viridis # viridis color projection

        if vmax is None:
            vmax=np.max(twoD_map)*1.2
        plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

        color_legend = plt.colorbar()
        color_legend.set_label('Color Legend') # legend

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def draw_target_slot(self, image, target_slot):

        size = image.shape[0]

        # convert target slot position into pixels
        x_pixel = target_slot[0] / self.cfg.bev_x_bound[2]
        y_pixel = target_slot[1] / self.cfg.bev_y_bound[2]
        target_point = np.array([size / 2 - x_pixel, size / 2 + y_pixel], dtype=int)

        # draw the whole parking slot
        slot_points = []
        for x in range(-27, 28):
            for y in range(-15, 16):
                slot_points.append(np.array([x, y, 1, 1], dtype=int))

        # rotate parking slots points

        slot_trans = np.array(
            carla.Transform(carla.Location(), carla.Rotation(yaw=float(-target_slot[2]))).get_matrix())
        slot_points = np.vstack(slot_points).T
        slot_points_ego = (slot_trans @ slot_points)[0:2].astype(int)

        # get parking slot points on pixel frame
        slot_points_ego[0] += target_point[0]
        slot_points_ego[1] += target_point[1]

        image[tuple(slot_points_ego)] = 255

        return image

class ProcessSemantic3D:
    def __init__(self, cfg):
        self.cfg = cfg
    def indices2coor(self,indices,min,resolution):
        return np.array(indices)*resolution+np.array(min)
    def __call__(self, voxel, target_slot):
        """
        Process original BEV ground truth image; return cropped image with target slot
        :param voxel:
        :param target_slot: center location of the target parking slot in meters; vehicle frame
        :return: processed BEV semantic ground truth
        """

        # read image from disk
        data = np.load(voxel, allow_pickle=True).item()
        gt_occ, resolution, min_bound, max_bound = data['gt_occ'], data['resolution'], data['min_bound'], data['max_bound']
        grid_size = np.ceil((np.array(max_bound) - np.array(min_bound)) / resolution).astype(int)
        voxels = np.zeros(grid_size, dtype=int)
        for indices, value in gt_occ.items():
            # GT invert x-axis
            # indice_0 = int(- np.array(indices)[0])

            tmp_coor = self.indices2coor(indices,min_bound,resolution)
            if (min_bound[0] < tmp_coor[0] and  tmp_coor[0] < max_bound[0] and
                min_bound[1] < tmp_coor[1] and  tmp_coor[1]< max_bound[1] and
                min_bound[2] < tmp_coor[2] and  tmp_coor[2]< max_bound[2]):
                indices = (indices[0], -indices[1], indices[2])

                voxels[(indices)] = value

        voxels = self.draw_target_slot3D(voxels, target_slot, min_bound, max_bound, resolution)

        min_index = np.floor((np.array(self.cfg.point_cloud_range[:3]) - np.array(min_bound)) / resolution).astype(int)
        max_index = np.ceil((np.array(self.cfg.point_cloud_range[3:]) - np.array(max_bound)) / resolution).astype(int)
        cropped_voxels = voxels[min_index[0]:max_index[0], min_index[1]:max_index[1], min_index[2]:max_index[2]]
        #Exclude the car itself
        H, W, D = cropped_voxels.shape
        cropped_voxels[int(H/2)-12:int(H/2)+12,int(W/2)-12:int(W/2)+12,:] = 0
        mask = np.isin(cropped_voxels, [11, 12, 13, 14])
        cropped_voxels[mask] = 0

        if self.cfg.seg_classes == 3:
            map_segmentation = np.zeros_like(cropped_voxels)
            map_segmentation[(cropped_voxels != 4) & (cropped_voxels != 6) & (cropped_voxels != 17) & (cropped_voxels != 0)] = 3
            map_segmentation[(cropped_voxels == 4) | (cropped_voxels == 6)] = 1
            map_segmentation[(cropped_voxels == 17)] = 2

            ############## Here #############
            # segmentation = np.max(map_segmentation, axis=2)
            # segmentation = segmentation[:,::-1]

            # self.plot_grid_2D(segmentation, os.path.join("visual", "gt_test.png"))
            
            # interpolated_segmentation = np.max(interpolated_segmentation, axis=2)
            # interpolated_segmentation = interpolated_segmentation[:,::-1].astype(np.int64)
            # self.plot_grid_2D(segmentation, os.path.join("visual", "gt_nods.png"))
            # self.plot_grid_2D(interpolated_segmentation, os.path.join("visual", "interpolate_gt.png"))
            # import pdb; pdb.set_trace()

            ####################################
            downsampled_segmentation = np.zeros((40, 40, 5), dtype=int)
            for i in range(40):
                for j in range(40):
                    for k in range(5):
                        # 获取每个 4x4x4 区域
                        block = map_segmentation[i*4:(i+1)*4, j*4:(j+1)*4, k*4:(k+1)*4]
                        # 计算该块的众数并存入 downsampled_segmentation
                        downsampled_segmentation[i, j, k] = np.max(block)
            
            downsampled_segmentation_tensor = torch.tensor(downsampled_segmentation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            interpolated_segmentation_tensor = F.interpolate(downsampled_segmentation_tensor, size=(160, 160, 20), mode='nearest')
            interpolated_segmentation = interpolated_segmentation_tensor.squeeze().numpy().astype(np.int64)
            # map_segmentation = interpolated_segmentation.astype(np.int64)

            combined_array = np.zeros_like(interpolated_segmentation)
            combined_array[(map_segmentation == 1) | (map_segmentation == 2)] = map_segmentation[(map_segmentation == 1) | (map_segmentation == 2)]
            combined_array[(interpolated_segmentation == 3)] = interpolated_segmentation[(interpolated_segmentation == 3)]
            combined_array[combined_array==3] = 1
            map_segmentation = combined_array
        #####################################

        elif self.cfg.seg_classes == 18:
            map_segmentation = cropped_voxels.copy()

        segmentation = np.max(map_segmentation, axis=2)
        segmentation = segmentation[:,::-1]

        # self.plot_grid_2D(segmentation, os.path.join("visual", "gt_test.png"))

        return segmentation.copy()

    def draw_target_slot3D(self, voxel, target_slot, min_bound, max_bound, resolution):
        size_h, size_w, size_d = voxel.shape
        # convert target slot position into pixels
        x_pixel = target_slot[0] / resolution
        y_pixel = target_slot[1] / resolution
        z_pixel = target_slot[2] / resolution
        target_point = np.array([size_h / 2 + x_pixel, size_w / 2 - y_pixel, size_d / 2 + z_pixel], dtype=int)
    
        # draw the whole parking slot
        slot_points = []
        for x in range(-15, 16):
            for y in range(-9, 10):
                for z in range(-3, 4):
                    slot_points.append(np.array([x, y, z, 1], dtype=int))
    
        # rotate parking slots points
    
        slot_trans = np.array(
            carla.Transform(carla.Location(), carla.Rotation(yaw=float(-target_slot[3]))).get_matrix())
        slot_points = np.vstack(slot_points).T
        slot_points_ego = (slot_trans @ slot_points)[0:3].astype(int)
    
        # get parking slot points on pixel frame
        slot_points_ego[0] += target_point[0]
        slot_points_ego[1] += target_point[1]
        slot_points_ego[2] += target_point[2]
    
        voxel[tuple(slot_points_ego)] = 17
    
        return voxel
    
    def plot_grid_2D(self, twoD_map, save_path=None, vmax=None, layer=None):
        if self.cfg.seg_classes == 3:
            H, W = twoD_map.shape

            # twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
            # twoD_map = threeD_grid[:,:,7]
            cmap = plt.cm.viridis # viridis color projection

            if vmax is None:
                vmax=np.max(twoD_map)*1.2
            plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

            color_legend = plt.colorbar()
            color_legend.set_label('Color Legend') # legend

        elif self.cfg.seg_classes == 18:
            NUSC_COLOR_MAP = {
                1: (112, 128, 144),  # Slategrey barrier
                2: (220, 20, 60),    # Crimson bicycle
                3: (255, 127, 80),   # Orangered bus
                4: (255, 158, 0),    # Orange car
                5: (233, 150, 70),   # Darksalmon construction
                6: (255, 61, 99),    # Red motorcycle
                7: (0, 0, 230),      # Blue pedestrian
                8: (47, 79, 79),     # Darkslategrey trafficcone
                9: (255, 140, 0),    # Darkorange trailer
                10: (255, 99, 71),   # Tomato truck
                11: (0, 207, 191),   # nuTonomy green driveable_surface
                12: (175, 0, 75),    # flat other
                13: (75, 0, 75),     # sidewalk
                14: (112, 180, 60),  # terrain
                15: (222, 184, 135), # Burlywood mannade
                16: (0, 175, 0),     # Green vegetation
                17: (128, 128, 128)  # target point
            }
        
            # Normalize RGB values to [0, 1]
            color_list = [(0, 0, 0)] + [(r / 255, g / 255, b / 255) for r, g, b in NUSC_COLOR_MAP.values()]
            cmap = ListedColormap(color_list)

            # Plot using the custom color map
            plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=0, vmax=len(color_list) - 1)
            
            # Add color bar with specific ticks for each label
            color_legend = plt.colorbar(ticks=range(len(color_list)))
            color_legend.set_label('Semantic Labels')
            color_legend.ax.set_yticklabels(['0'] + list(NUSC_COLOR_MAP.keys()))

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def visual_voxel(self, gt_occ, voxels, grid_size=0.2, save_path=os.path.join("visual","voxel.png")):
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
            17: (128,128,128) # target point
        }

        boxes = open3d.geometry.TriangleMesh()
        for coord_inds, values in gt_occ.items():
            box = open3d.geometry.TriangleMesh.create_box(grid_size, grid_size, grid_size)
            p = list(coord_inds)
            box.translate(p)
            if values != 255:
                box.paint_uniform_color(np.array(NUSC_COLOR_MAP[values]) / 255.0)
            else:
                # box.paint_uniform_color((0.0, 0.0, 0.0))
                continue
            boxes += box
        self.render_img_high_view([boxes], fname=save_path)
        import pdb; pdb.set_trace()

    def render_img_high_view(self, geometries, **kwargs):
        kwargs_default = {'fname': None, 'view_point': -100, 'zoom': 0.7, 'show_wireframe': True, 'point_size': 5}
        kwargs = {**kwargs_default, **kwargs}
        R = geometries[0].get_rotation_matrix_from_zyx((np.pi / 2, 0, 0))

        img = self.render_img(geometries, **kwargs)
        return img

    def render_img(self, geometries, fname=None, view_point=-100, zoom=0.7, show_wireframe=True, point_size=5):
        vis = open3d.visualization.Visualizer()
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

class ProcessImage:
    def __init__(self, crop):
        self.crop = crop

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

    def __call__(self, image):
        if isinstance(image, carla.Image):
            image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            image = image[:, :, :3]
            image = image[:, :, ::-1]
            image = Image.fromarray(image)
        else:
            image = Image.open(image).convert('RGB')
            image = image.resize((400, 300))

        crop_image = scale_and_crop_image(image, scale=1.0, crop=self.crop)

        return self.normalise_image(np.array(crop_image)).unsqueeze(0), crop_image
