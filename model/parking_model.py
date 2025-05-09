import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead
from model.waypoint_predict import WaypointPredict

import numpy as np
import os
import matplotlib.pyplot as plt

class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        self.waypoint_predict = WaypointPredict(self.cfg)

        self.segmentation_head = SegmentationHead(self.cfg)

    def adjust_target_bev(self, bev_feature, target_point):
        b, c, h, w = bev_feature.shape
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)

        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise
        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
                             target_point_batch[1] - 4:target_point_batch[1] + 4] = 1.0
        return bev_feature, bev_target

    def add_target_bev(self, bev_feature, target_point):
        b, c, h, w = bev_feature.shape
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)

        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise

        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
                             target_point_batch[1] - 4:target_point_batch[1] + 4] = 1.0

        bev_feature = torch.cat([bev_feature, bev_target], dim=1)
        return bev_feature, bev_target

    def detokenize_waypoint(self, token_list, token_nums=204):
        """
        Detokenize waypoint values
        :param token_list: [x_token, y_token, yaw_token]
        :param token_nums: size of token number
        :return: original x, y, yaw values
        """
        token_nums -= 4  # Adjusting for the valid range of tokens

        # Helper function to detokenize a single value
        def detokenize_single_value(token, min_value, max_value):
            # Scale token from [0, token_nums] to [0, 1]
            normalized_value = token / token_nums
            # Scale and shift the normalized value to its original range
            original_value = (normalized_value * (max_value - min_value)) + min_value
            return original_value

        # Detokenize each parameter
        x = detokenize_single_value(token_list[0], -6, 6)
        y = detokenize_single_value(token_list[1], -6, 6)
        yaw = detokenize_single_value(token_list[2], -40, 40)

        return [x, y, yaw]
    
    def plot_grid_2D(self, twoD_map, waypoints=None, save_path=None, vmax=None, layer=None):
        # print(twoD_map.shape) # (200, 200)
        H, W = twoD_map.shape

        # twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        cmap = plt.cm.viridis # viridis color projection

        if vmax is None:
            vmax=np.max(twoD_map)*1.2
        plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

        color_legend = plt.colorbar()
        color_legend.set_label('Color Legend') # legend

        # Plot GT waypoints if provided
        if waypoints is not None:
            waypoints_ego = []
            for i in range(0, self.cfg.future_frame_nums):
                wp = self.detokenize_waypoint(waypoints.tolist()[i*3+1:i*3+4], self.cfg.token_nums)
                waypoints_ego.append(wp)
            for idx, wp in enumerate(waypoints_ego):
                # Convert ego-coordinates to pixel coordinatess
                x_pix = W / 2 + wp[0] / self.cfg.bev_x_bound[2]
                y_pix = H / 2 + wp[1] / self.cfg.bev_y_bound[2]
                plt.plot(y_pix, x_pix, 'ro')  # red dot
                plt.text(y_pix + 2, x_pix - 2, f'WP{idx + 1}', color='yellow', fontsize=8)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)#已经是相对车的位置了
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)
        # print("bev_feature: ", bev_feature.shape) # torch.Size([32, 64, 200, 200])

        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)

        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)
        # print("bev_down_sample: ", bev_down_sample.shape) # torch.Size([32, 256, 256])

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion, target_point)
        # print("fuse_feature: ", fuse_feature.shape) # torch.Size([32, 256, 264])

        pred_segmentation = self.segmentation_head(fuse_feature)
        # print("pred_segmentation: ", pred_segmentation.shape) # torch.Size([32, 3, 200, 200])

        # pred_c = torch.argmax(pred_segmentation[0], dim=0).cpu().numpy()
        # self.plot_grid_2D(pred_c, save_path=os.path.join("visual", "pred.png"))
        # self.plot_grid_2D(data['segmentation'][0][0].cpu().numpy(),
        #                 waypoints=data['gt_waypoint'][0].cpu().numpy(),
        #                 save_path=os.path.join("visual", "gt.png"))

        return fuse_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        fuse_feature_copy = fuse_feature.clone()
        # print(data['gt_control'].shape) # torch.Size([32, 15])
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        # print(data['gt_waypoint'].shape) # torch.Size([32, 15])
        pred_waypoint = self.waypoint_predict(fuse_feature_copy, data['gt_waypoint'].cuda())

        # pred_c = torch.argmax(pred_segmentation[0], dim=0).cpu().numpy()
        return pred_control, pred_waypoint, pred_segmentation, pred_depth

    def predict(self, data):
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        for i in range(3):
            pred_control = self.control_predict.predict(fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        for i in range(3*self.cfg.future_frame_nums):
            pred_waypoint = self.waypoint_predict.predict(fuse_feature_copy, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)
        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target
