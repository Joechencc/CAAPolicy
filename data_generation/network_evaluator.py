import math
import random
import logging
import sys
import pathlib
import carla
import numpy as np
import pandas as pd
import glob
import cv2
from datetime import datetime
from data_generation.tools import encode_npy_to_pil
from data_generation import parking_position
from data_generation.world import World
import json
from threading import Thread

from utils.ply2voxel import voxelization_save


class NetworkEvaluator:
    def __init__(self, carla_world, args):
        self._task_index = 0
        self._seed = args.random_seed
        self._init_seed = args.random_seed
        random.seed(args.random_seed)

        self._world = World(carla_world, args)

        # only eva odd slot: 1, 3, 5, ..., 15
        # 2-1, index 15 + 1 = 16
        self._parking_goal_index = 16
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator = parking_position.EgoPosTown04()

        now = datetime.now()
        result_dir = '_'.join(map(lambda x: '%02d' % x,
                                  (now.year, now.month, now.day, now.hour, now.minute, now.second)))
        self._eva_result_path = pathlib.Path(args.eva_result_path) / result_dir
        self._eva_result_path.mkdir(parents=True, exist_ok=False)

        self._render_bev = args.show_eva_imgs

        self._eva_epochs = args.eva_epochs
        self._eva_task_nums = args.eva_task_nums
        self._eva_parking_nums = args.eva_parking_nums
        self._eva_epoch_idx = 0
        self._eva_task_idx = 0
        self._eva_parking_idx = 0

        # parking goal diff, init with max float value
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = sys.float_info.max

        self._goal_reach_x_diff = 1.0              # meter
        self._goal_reach_y_diff = 0.6              # meter
        self._goal_reach_orientation_diff = 10.0   # degree

        # 1s = 30HZ for our carla setting
        self._frames_per_second = 30

        # metric frames
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_nearby_no_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0
        self._num_frames_in_goal_needed = 2 * self._frames_per_second         # 2s
        self._num_frames_nearby_goal_needed = 2 * self._frames_per_second     # 2s
        self._num_frames_nearby_no_goal_needed = 2 * self._frames_per_second  # 2s
        self._num_frames_outbound_needed = 10 * self._frames_per_second       # 10s
        self._num_frames_total_needed = 30 * self._frames_per_second          # 30s

        # metric for 1 slot
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0
        self._position_error = []
        self._orientation_error = []
        self._parking_time = []
        self._inference_time = []

        # metric rate for 16 slot
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_parking_time = []
        self._average_inference_time = []

        self._epoch_metric_info = {}

        # In paper: NTR = NTSR + NTFR, TR = TR + OR
        self._metric_names = {
            "target_success_rate": "TSR",
            "target_fail_rate": "TFR",
            "no_target_success_rate": "NTSR",
            "no_target_fail_rate": "NTFR",
            "collision_rate": "CR",
            "outbound_rate": "OR",
            "timeout_rate": "TR",
            "average_position_error": "APE",
            "average_orientation_error": "AOE",
            "average_parking_time": "APT",
            "average_inference_time": "AIT",
        }

        self._ego_transform = None
        self._eva_parking_goal = None
        self._agent_need_init = None
        self._start_time = None

        self.init()
        self.start_eva_epoch()
        self._batch_data_frames = []

    def init(self):
        logging.info("***************** Start init eva environment *****************")

        self._ego_transform = self._ego_transform_generator.get_init_ego_transform()
        self._world.init_ego_vehicle(self._ego_transform)
        logging.info("Init ego vehicle success!")

        self._world.init_sensors()
        logging.info("Init sensors success!")

        self._world.next_weather()
        logging.info("Init weather success!")

        logging.info("*****************   End init eva environment *****************")

    def tick(self, clock):
        # update diff to world
        self._world.distance_diff_to_goal = self._distance_diff_to_goal
        self._world.rotation_diff_to_goal = self._orientation_diff_to_goal
        self._world.x_diff_to_goal = self._x_diff_to_goal
        self._world.y_diff_to_goal = self._y_diff_to_goal

        self._num_frames_total += 1

        # detect collision
        is_collision = self._world.tick(clock, self._parking_goal_index)
        if is_collision:
            #self.save_sensor_data([0,0])
            #print("saved")
            self._collision_nums += 1
            logging.info("parking collision for task %s-%d, collision_num: %d",
                         parking_position.slot_id[self._eva_task_idx],
                         self._eva_parking_idx + 1, self._collision_nums)
            self._batch_data_frames.clear()
            self.start_next_parking()
            return

        if self._render_bev:
            sensor_data_frame = self._world.sensor_data_frame
            sensor_data_frame['topdown'] = self._world.render_BEV()

        # detect timeout
        if self._num_frames_total > self._num_frames_total_needed:
            self._timeout_nums += 1
            logging.info("parking timeout for task %s-%d, timeout_num: %d",
                         parking_position.slot_id[self._eva_task_idx],
                         self._eva_parking_idx + 1, self._timeout_nums)
            self._batch_data_frames.clear()
            self.start_next_parking()
            return

        # detect out of bound
        ego_loc = self._world.ego_transform.location
        if self.is_out_of_bound(ego_loc):
            self._num_frames_outbound += 1
        else:
            self._num_frames_outbound = 0

        if self._num_frames_outbound > self._num_frames_outbound_needed:
            self._outbound_nums += 1
            logging.info("parking outbound for task %s-%d, outbound_num: %d",
                         parking_position.slot_id[self._eva_task_idx],
                         self._eva_parking_idx + 1, self._outbound_nums)
            self._batch_data_frames.clear()
            self.start_next_parking()
            return

        self.eva_check_goal()

    def start_eva_epoch(self):
        logging.info("***************** Start eva epoch %d *****************", self._eva_epoch_idx + 1)

        self._start_time = datetime.now()

        self.soft_destroy()

        self._seed = self._init_seed

        self._parking_goal_index = 17
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y,
                                                        self._eva_parking_nums,
                                                        self._eva_parking_idx)
        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform(self._eva_parking_nums,
                                                                                  self._eva_parking_idx)
        self._world.player.set_transform(self._ego_transform)
        self._world.init_static_npc(self._seed, self._parking_goal_index)
        #self._world.init_single_pergola([290.9, -232.73, 0.3])


        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]

        self._eva_task_idx = 0
        self.clear_metric_rate()
        self._epoch_metric_info = {}
        logging.info("***************** Start eva task %s *****************",
                     parking_position.slot_id[self._eva_task_idx])

    def start_next_parking(self):
        self._agent_need_init = True

        self._eva_parking_idx += 1
        if self.is_complete_slot(self._eva_parking_idx):
            logging.info("*****************   End eva task %s *****************",
                         parking_position.slot_id[self._eva_task_idx])
            self.save_slot_metric()
            self.start_next_slot()
            return

        self.clear_metric_frame()

        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform(self._eva_parking_nums,
                                                                                  self._eva_parking_idx)
        self._world.player.set_transform(self._ego_transform)
        self._world.player.apply_control(carla.VehicleControl())

    def is_complete_slot(self, eva_parking_idx):
        return eva_parking_idx >= self._eva_parking_nums

    def start_next_slot(self):
        self._eva_task_idx += 1
        if self.is_complete_epoch(self._eva_task_idx):
            logging.info("*****************   End eva epoch %d *****************", self._eva_epoch_idx + 1)
            self.save_epoch_metric_csv()
            self._eva_epoch_idx += 1
            if self._eva_epoch_idx >= self._eva_epochs:
                self.save_mean_std_csv()
                exit(0)
            else:
                self.start_eva_epoch()
            return

        if self._eva_task_idx < 16:
            self._parking_goal_index += 2
        else:
            self._parking_goal_index = 16

        self.soft_destroy()

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y,
                                                        self._eva_parking_nums,
                                                        self._eva_parking_idx)
        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform(self._eva_parking_nums,
                                                                                  self._eva_parking_idx)
        self._world.player.set_transform(self._ego_transform)

        self._seed += 1
        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]
        self._world.restart(self._seed, self._parking_goal_index, self._ego_transform)

        logging.info("***************** Start eva task %s *****************",
                     parking_position.slot_id[self._eva_task_idx])

    def is_complete_epoch(self, eva_task_idx):
        return eva_task_idx >= self._eva_task_nums

    def clear_metric_num(self):
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0
        self._position_error = []
        self._orientation_error = []
        self._inference_time = []
        self._parking_time = []

    def clear_metric_frame(self):
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0

    def clear_metric_rate(self):
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_inference_time = []
        self._average_parking_time = []

    def soft_destroy(self):
        self._eva_parking_idx = 0
        self.clear_metric_num()
        self.clear_metric_frame()
        self._world.soft_destroy()

    def destroy(self):
        self._world.destroy()

    def eva_check_goal(self):
        # get ego current state
        player = self._world.player
        t = player.get_transform().location
        r = player.get_transform().rotation
        v = player.get_velocity()
        c = player.get_control()
        speed = (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # find the closest goal
        closest_goal = [0.0, 0.0]
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = min(abs(r.yaw), 180 - abs(r.yaw))
        for parking_goal in self._world.all_parking_goals:
            if t.distance(parking_goal) < self._distance_diff_to_goal:
                self._distance_diff_to_goal = t.distance(parking_goal)
                self._x_diff_to_goal = abs(t.x - parking_goal.x)
                self._y_diff_to_goal = abs(t.y - parking_goal.y)
                closest_goal[0] = parking_goal.x
                closest_goal[1] = parking_goal.y

        # check stop
        is_stop = (c.throttle == 0.0) and (speed < 1e-3) and c.reverse
        if not is_stop:
            self._num_frames_in_goal = 0
            self._num_frames_nearby_goal = 0
            self._num_frames_nearby_no_goal = 0
            return

        # check success parking
        if self.check_success_slot(closest_goal, t):
            self.save_sensor_data(closest_goal)
            self.start_next_parking()
            return

        # check fail parking
        if self.check_fail_slot(closest_goal, t):
            #self.save_sensor_data(closest_goal)
            #print("saved")
            self._batch_data_frames.clear()
            self.start_next_parking()
            return

    def check_success_slot(self, closest_goal, ego_transform):
        x_in_slot = (abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff)
        y_in_slot = (abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff)
        r_in_slot = (self._orientation_diff_to_goal <= self._goal_reach_orientation_diff)

        if x_in_slot and y_in_slot and r_in_slot:
            self._num_frames_in_goal += 1

        if self._num_frames_in_goal > self._num_frames_in_goal_needed:
            if (self._eva_parking_goal[0] == closest_goal[0]) and (self._eva_parking_goal[1] == closest_goal[1]):
                self._target_success_nums += 1
                self._position_error.append(self._distance_diff_to_goal)
                self._orientation_error.append(self._orientation_diff_to_goal)
                self._parking_time.append(self._num_frames_total / self._frames_per_second)
                logging.info("parking target success for task %s-%d, target_success_nums: %d",
                             parking_position.slot_id[self._eva_task_idx],
                             self._eva_parking_idx + 1, self._target_success_nums)
            else:
                self._no_target_success_nums += 1
                logging.info("parking no target success for task %s-%d, no_target_success_nums: %d",
                             parking_position.slot_id[self._eva_task_idx],
                             self._eva_parking_idx + 1, self._no_target_success_nums)
            return True

        return False

    def check_fail_slot(self, closest_goal, ego_transform):
        x_not_in_slot = \
            (self._goal_reach_x_diff < abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff * 2)
        y_not_in_slot = \
            (self._goal_reach_y_diff < abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff * 2)
        r_not_in_slot = (self._goal_reach_orientation_diff <
                         self._orientation_diff_to_goal <=
                         self._goal_reach_orientation_diff * 2)

        if x_not_in_slot or y_not_in_slot or r_not_in_slot:
            if (self._eva_parking_goal[0] == closest_goal[0]) and (self._eva_parking_goal[1] == closest_goal[1]):
                self._num_frames_nearby_goal += 1
            else:
                self._num_frames_nearby_no_goal += 1

        if self._num_frames_nearby_goal > self._num_frames_nearby_goal_needed:
            self._target_fail_nums += 1
            logging.info("parking target fail for task %s-%d, target_fail_nums: %d",
                         parking_position.slot_id[self._eva_task_idx],
                         self._eva_parking_idx + 1, self._target_fail_nums)
            return True

        if self._num_frames_nearby_no_goal > self._num_frames_nearby_no_goal_needed:
            self._no_target_fail_nums += 1
            logging.info("parking no target fail for task %s-%d, no_target_fail_nums: %d",
                         parking_position.slot_id[self._eva_task_idx],
                         self._eva_parking_idx + 1, self._no_target_fail_nums)
            return True

        return False

    def save_slot_metric(self):
        TSR = (self._target_success_nums / float(self._eva_parking_nums)) * 100.0
        TFR = (self._target_fail_nums / float(self._eva_parking_nums)) * 100.0
        NTSR = (self._no_target_success_nums / float(self._eva_parking_nums)) * 100.0
        NTFR = (self._no_target_fail_nums / float(self._eva_parking_nums)) * 100.0
        CR = (self._collision_nums / float(self._eva_parking_nums)) * 100.0
        OR = (self._outbound_nums / float(self._eva_parking_nums)) * 100.0
        TR = (self._timeout_nums / float(self._eva_parking_nums)) * 100.0
        APE = np.mean(self._position_error)
        AOE = np.mean(self._orientation_error)
        APT = np.mean(self._parking_time)
        AIT = np.mean(self._inference_time)

        slot_id = parking_position.slot_id[self._eva_task_idx]
        self._epoch_metric_info[slot_id] = {
            'target_success_rate': TSR,
            'target_fail_rate': TFR,
            'no_target_success_rate': NTSR,
            'no_target_fail_rate': NTFR,
            'collision_rate': CR,
            'outbound_rate': OR,
            'timeout_rate': TR,
            'average_position_error': APE,
            'average_orientation_error': AOE,
            'average_parking_time': APT,
            'average_inference_time': AIT,
        }

        self._target_success_rate.append(TSR)
        self._target_fail_rate.append(TFR)
        self._no_target_success_rate.append(NTSR)
        self._no_target_fail_rate.append(NTFR)
        self._collision_rate.append(CR)
        self._outbound_rate.append(OR)
        self._timeout_rate.append(TR)
        self._average_position_error.append(APE)
        self._average_orientation_error.append(AOE)
        self._average_parking_time.append(APT)
        self._average_inference_time.append(AIT)

    def save_epoch_metric_csv(self):
        self._epoch_metric_info['Avg'] = {
            'target_success_rate': np.mean(self._target_success_rate),
            'target_fail_rate': np.mean(self._target_fail_rate),
            'no_target_success_rate': np.mean(self._no_target_success_rate),
            'no_target_fail_rate': np.mean(self._no_target_fail_rate),
            'collision_rate': np.mean(self._collision_rate),
            'outbound_rate': np.mean(self._outbound_rate),
            'timeout_rate': np.mean(self._timeout_rate),
            'average_position_error': np.mean(self._average_position_error),
            'average_orientation_error': np.mean(self._average_orientation_error),
            'average_inference_time': np.mean(self._average_inference_time),
            'average_parking_time': np.mean(self._average_parking_time),
        }

        info_df = pd.DataFrame(self._epoch_metric_info)
        csv_name = 'eva_epoch_' + str(self._eva_epoch_idx + 1) + '_result.csv'
        self.save_csv(info_df, csv_name)
        logging.info(f"eva epoch {self._eva_epoch_idx + 1} total time: {datetime.now() - self._start_time}")

    def save_csv(self, info_df, csv_name):
        info_df = info_df.T
        info_df.rename(columns=self._metric_names, inplace=True)
        pd.set_option('display.max_columns', 1000)
        pd.options.display.float_format = '{:,.3f}'.format
        print(info_df)
        info_df.to_csv(self._eva_result_path / csv_name)

    def save_mean_std_csv(self):
        df_mean = pd.DataFrame()
        df_std = pd.DataFrame()
        csv_files = glob.glob(f'{self._eva_result_path}/*_result.csv')
        for i in range(self._eva_task_nums):
            df_row_i = pd.DataFrame()
            for csv in csv_files:
                df_csv = pd.read_csv(csv)
                row_i = df_csv.iloc[[i]]
                df_row_i = df_row_i.append(row_i)

            row_i_mean = df_row_i.mean(axis=0).to_frame().T
            row_i_std = (df_row_i.std(axis=0, ddof=0).to_frame().T / math.sqrt(6))

            df_mean = pd.concat([df_mean, row_i_mean], axis=0)
            df_std = pd.concat([df_std, row_i_std], axis=0)

        row_mean = df_mean.mean(axis=0).to_frame().T
        row_std = df_std.mean(axis=0).to_frame().T
        df_mean = pd.concat([df_mean, row_mean], axis=0)
        df_std = pd.concat([df_std, row_std], axis=0)

        all_name = ['2-1', '2-3', '2-5', '2-7', '2-9', '2-11', '2-13', '2-15',
                    '3-1', '3-3', '3-5', '3-7', '3-9', '3-11', '3-13', '3-15']
        name = all_name[:self._eva_task_nums]
        name.append('Avg')

        df_mean.index = name
        df_std.index = name

        pd.set_option('display.max_columns', 1000)
        pd.options.display.float_format = '{:,.3f}'.format

        logging.info('Mean')
        print(df_mean)

        logging.info('Std')
        print(df_std)

        df_mean.to_csv(self._eva_result_path / 'result_mean.csv')
        df_std.to_csv(self._eva_result_path / 'result_std.csv')

    def is_out_of_bound(self, ego_loc):
        x_out_bound = ((ego_loc.x < parking_position.town04_bound['x_min']) or
                       (ego_loc.x > parking_position.town04_bound['x_max']))
        y_out_bound = ((ego_loc.y < parking_position.town04_bound['y_min']) or
                       (ego_loc.y > parking_position.town04_bound['y_max']))
        return x_out_bound or y_out_bound

    def world_tick(self):
        step = self._world.step
        if step % 3 == 0:
            sensor_data_frame = self._world.sensor_data_frame
            sensor_data_frame['bev_state'] = self._world.bev_state
            self._batch_data_frames.append(sensor_data_frame.copy())
        self._world.world_tick()

    def render(self, display):
        self._world.render(display)

    @property
    def world(self):
        return self._world

    @property
    def agent_need_init(self):
        return self._agent_need_init

    @agent_need_init.setter
    def agent_need_init(self, need_init):
        self._agent_need_init = need_init

    @property
    def inference_time(self):
        return self._inference_time

    @property
    def eva_parking_goal(self):
        return self._eva_parking_goal

    @property
    def ego_transform(self):
        return self._ego_transform
    def save_sensor_data(self, parking_goal):

        # create dirs
        self._save_path = "./e2e_parking/eval/"+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        cur_save_path = pathlib.Path(self._save_path) / ('task' + str(self._task_index))
        cur_save_path.mkdir(parents=True, exist_ok=False)
        (cur_save_path / 'measurements').mkdir()
        (cur_save_path / 'lidar').mkdir()
        (cur_save_path / 'parking_goal').mkdir()
        (cur_save_path / 'topdown').mkdir()
        (cur_save_path / 'voxel').mkdir()
        for sensor in self._batch_data_frames[0].keys():
            if  sensor.startswith('camera'):
                (cur_save_path / sensor).mkdir()

        total_frames = len(self._batch_data_frames)
        thread_num = 16
        frames_for_thread = total_frames // thread_num
        thread_list = []
        for t_idx in range(thread_num):
            start = t_idx * frames_for_thread
            if t_idx == thread_num - 1:
                end = total_frames
            else:
                end = (t_idx + 1) * frames_for_thread
            t = Thread(target=self.save_unit_data, args=(start, end, cur_save_path))
            t.start()
            thread_list.append(t)

        for thread in thread_list:
            thread.join()

        # save Parking Goal
        measurements_file = cur_save_path / 'parking_goal' / '0001.json'
        with open(measurements_file, 'w') as f:
            data = {'x': parking_goal[0],
                    'y': parking_goal[1],
                    'z':0 
                    }
            json.dump(data, f, indent=4)

        # save vehicle video
        self._world.save_video(cur_save_path)

        logging.info('saved sensor data for task %d in %s', self._task_index, str(cur_save_path))

    def save_unit_data(self, start, end, cur_save_path):
        for index in range(start, end):
            data_frame = self._batch_data_frames[index]

            # save camera / lidar
            for sensor in data_frame.keys():
                # if sensor.startswith('rgb'):
                #     # _, image = self.image_process(self.target_parking_goal, cam_id=sensor, image=data_frame[sensor])
                #     # image = Image.fromarray(image)
                #     # image.save(str(cur_save_path / sensor / ('%04d.png' % index)))
                #     data_frame[sensor].save_to_disk(
                #         str(cur_save_path / sensor / ('%04d.png' % index)))
                if sensor.startswith('depth') and not sensor.startswith("depth_e2e"):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                if sensor.startswith('camera'):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor / ('%04d.png' % index)))
                elif sensor.startswith('lidar'):
                    data_frame[sensor].save_to_disk(
                        str(cur_save_path / sensor  / ('%04d.ply' % index)))
                    voxelization_save(str(cur_save_path / sensor / ('%04d.ply' % index)),
                                      str(cur_save_path / "voxel" / ('%04d' % index)), )

            # save measurements
            imu_data = data_frame['imu']
            gnss_data = data_frame['gnss']
            vehicle_transform = data_frame['veh_transfrom']
            vehicle_velocity = data_frame['veh_velocity']
            vehicle_control = data_frame['veh_control']

            data = {
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z,
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw,
                'roll': vehicle_transform.rotation.roll,
                'speed': (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)),
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

            measurements_file = cur_save_path / 'measurements' / ('%04d.json' % index)
            with open(measurements_file, 'w') as f:
                json.dump(data, f, indent=4)

            def save_img(image, keyword=""):
                img_save = np.moveaxis(image, 0, 2)
                save_path = str(cur_save_path / 'topdown' / ('encoded_%04d' % index + keyword + '.png'))
                cv2.imwrite(save_path, img_save)

            keyword = ""
            bev_view1 = self._world.render_BEV_from_state(data_frame['bev_state'])
            img1 = encode_npy_to_pil(np.asarray(bev_view1.squeeze().cpu()))
            save_img(img1, keyword)