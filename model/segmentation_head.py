import math

import torch
import torch.nn.functional as F

from torch import nn
from tool.config import Configuration
from model.conet_model import OccNet
import carla
import numpy as np
from data_generation.world import cam_specs_, cam2pixel_


class SegmentationHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegmentationHead, self).__init__()
        self.cfg = cfg

        self.in_channel = self.cfg.bev_encoder_out_channel
        self.out_channel = self.cfg.bev_encoder_in_channel
        self.seg_classes = self.cfg.seg_classes

        self.relu = nn.ReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5_conv = nn.Conv2d(self.in_channel, self.out_channel, (1, 1))
        self.up_conv5 = nn.Conv2d(self.out_channel, self.out_channel, (1, 1))
        self.up_conv4 = nn.Conv2d(self.out_channel, self.out_channel, (1, 1))
        self.up_conv3 = nn.Conv2d(self.out_channel, self.out_channel, (1, 1))

        self.cd5_conv = nn.Conv2d(self.out_channel, self.in_channel, (1, 1))
        self.down_conv5 = nn.Conv2d(self.in_channel, self.in_channel, (1, 1))
        self.down_conv4 = nn.Conv2d(self.in_channel, self.in_channel, (1, 1))
        self.down_conv3 = nn.Conv2d(self.in_channel, self.in_channel, (1, 1))

        self.OccNet = OccNet(**self.cfg.OccNet_cfg)

        # self.segmentation_head = nn.Sequential(
        #     nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.out_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.out_channel, self.seg_classes, kernel_size=1, padding=0)
        # )

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.up_sample(p5)))
        p3 = self.relu(self.up_conv4(self.up_sample(p4)))
        p2 = self.relu(self.up_conv3(self.up_sample(p3)))
        x_intepolate = int((self.cfg.bev_x_bound[1] - self.cfg.bev_x_bound[0])/self.cfg.bev_x_bound[2])
        y_intepolate = int((self.cfg.bev_y_bound[1] - self.cfg.bev_y_bound[0])/self.cfg.bev_y_bound[2])
        p1 = F.interpolate(p2, size=(x_intepolate, y_intepolate), mode="bilinear", align_corners=False)
        return p1

    def down_top(self, x):
        p5 = self.relu(self.cd5_conv(x))
        p4 = self.relu(self.down_conv5(self.down_sample(p5)))  # 下采样一次
        p3 = self.relu(self.down_conv4(self.down_sample(p4)))  # 再下采样
        p2 = self.relu(self.down_conv3(self.down_sample(p3)))  # 再下采样
        
        p1 = F.interpolate(p2, size=(16, 16), mode="bilinear", align_corners=False)

        return p1

    def forward(self, fuse_feature, images, intrinsics):
        fuse_feature_t = fuse_feature.transpose(1, 2)
        b, c, s = fuse_feature_t.shape
        fuse_bev = torch.reshape(fuse_feature_t, (b, c, int(math.sqrt(s)), -1))
        fuse_bev = self.top_down(fuse_bev)

        ###### refined
        B, I = images.shape[:2]
        img_metas = self.construct_metas()
        rot, trans, cam2ego, post_rots, post_trans, bda_rot, img_shape, gt_depths = self.transform_spec(cam_specs_, cam2pixel_, B, I, images.shape, images.device)
        img = [images.clone(), rot, trans, intrinsics, post_rots, post_trans, bda_rot, img_shape, gt_depths, cam2ego]
        res = self.OccNet(coarse_feat=fuse_bev, img_metas=img_metas, img_inputs=img)

        # pred_segmentation_coarse = self.segmentation_head(fuse_bev)
        pred_segmentation_coarse = res['pred_c']
        pred_segmentation = res['pred_f']
        fine_feature = res['fine_feature']
        fine_bev = self.down_top(fine_feature)
        fine_bev = torch.reshape(fine_bev, (b, c, -1)).transpose(1, 2)

        return pred_segmentation_coarse, pred_segmentation, fine_bev

    def construct_metas(self):
        metas = {}
        metas['pc_range'] = np.array(self.cfg.point_cloud_range)
        metas['occ_size'] = np.array(self.cfg.occ_size)
        metas['scene_token'] = ''
        metas['lidar_token'] = ''
        metas['prev_idx'] = ''
        metas['next_idx'] = ''

        return metas

    def transform_spec(self, cam_specs, cam2pixel, B, I, img_shape, device):
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
        sensor2egos = torch.cat(sensor2egos).unsqueeze(0).repeat(B,1,1,1).to(device)
        rot, trans = sensor2egos[:,:,:2,:2], sensor2egos[:,:,:2,3]
        post_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, I, 1, 1).to(device)
        post_trans = torch.tensor([0.,-4.,0.]).unsqueeze(0).unsqueeze(0).repeat(B, I, 1).to(device)
        bda_rot = torch.eye(2).unsqueeze(0).repeat(B, 1, 1).to(device)
        gt_depths = torch.zeros(1).unsqueeze(0).unsqueeze(0).repeat(B, I, 1).to(device)
        img_shape = torch.tensor(img_shape[-2:]).to(device).unsqueeze(0).repeat(B,1)
        return rot, trans, sensor2egos, post_rots, post_trans, bda_rot, img_shape, gt_depths
