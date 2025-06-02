import json
import os
import carla
import torch.utils.data
import numpy as np
import torchvision.transforms
import yaml

from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt

class CarlaDatasetDynamic(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, config):
        super(CarlaDatasetDynamic, self).__init__()
        self.cfg = config

        self.root_dir = root_dir
        self.is_train = is_train

        self.raw_control = []

        self.ego_motion = [] # [speed_x,speed_y, acc_x,acc_y]

        self.ego_pos = []

        self.ego_pos_last = []

        self.task_offsets = []

        self.get_data()

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
        start_index = 0  # Initialize the start index for the first task

        for task_path in all_tasks:
            total_frames = len(os.listdir(task_path + "/measurements/"))
            # Record the number of valid frames for offset calculation
            num_valid_frames = total_frames - self.cfg.hist_frame_nums - self.cfg.future_frame_nums

            # Store the end indice for this task
            self.task_offsets.append((start_index, start_index + num_valid_frames))

            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                
                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # current ego pose
                cur_x = data['x']
                cur_y = data['y']
                cur_yaw = data['yaw']
                ego_pos = [cur_x, cur_y, cur_yaw]
                
                # Save Waypoints
                self.ego_pos.append(ego_pos)
                # ego_motion
                self.ego_motion.append([data['speed_x'], data['speed_y'], data['acc_x'], data['acc_y']])
                # current control will be used to predict next ego_pose
                self.raw_control.append([data['Throttle'],data['Brake'],data['Steer'], data['Reverse']])

            ### We miss the last element due to future_frame_nums
            with open(task_path + f"/measurements/{str(frame+1).zfill(4)}.json", "r") as read_file:
                data_last = json.load(read_file)
            last_x, last_y, last_yaw = data_last['x'], data_last['y'], data_last['yaw']
            ego_pos_last = [last_x, last_y, last_yaw]
            self.ego_pos_last.append(ego_pos_last)

            # Update the start index for the next task
            start_index += num_valid_frames
            
        self.ego_motion = np.array(self.ego_motion).astype(np.float32)
        self.ego_pos = np.array(self.ego_pos).astype(np.float32)
        self.ego_pos_last = np.array(self.ego_pos_last).astype(np.float32)
        # Convert lists to tensors
        self.task_offsets = np.array(self.task_offsets).astype(np.int32)
        logger.info('Preloaded {} sequences', str(len(self.ego_motion)))

        # Plot and save IMU statistics
        # self.plot_imu_statistics()

    def __len__(self):

        return len(self.ego_motion)

    def __getitem__(self, index):
        data = {}
        keys = [ 'ego_motion', 'segmentation','task_offsets',
                'ego_pos','ego_pos_next','raw_control']
        for key in keys:
            data[key] = []

        # ego_motion
        data['ego_motion'] =  torch.from_numpy(self.ego_motion[index]) # [speed_x,speed_y, acc_x, acc_y]

        # gt untokenized control
        data['raw_control'] = torch.from_numpy(np.array(self.raw_control[index]))

        # gt ego_pos current & next frame
        data['ego_pos'] = torch.from_numpy(self.ego_pos[index])

        # Determine if the current index is at the end of a task
        is_end_of_task = any(index == offset[1] - 1 for offset in self.task_offsets)

        if is_end_of_task:
            # If at the end of a task, set ego_pos_next to the current ego_pos
            for i, offset in enumerate(self.task_offsets):
                if index == offset[1] - 1:
                    task_index = i
                    break
            data['ego_pos_next'] = torch.from_numpy(self.ego_pos_last[task_index])
        else:
            # Otherwise, set ego_pos_next to the next frame's ego_pos
            data['ego_pos_next'] = torch.from_numpy(self.ego_pos[index + 1])

        data['task_offsets'] = torch.from_numpy(self.task_offsets)
        # print("dataloading order check: ", index)
        return data