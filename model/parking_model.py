import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead
from model.waypoint_predict import WaypointPredict
import matplotlib.pyplot as plt
import numpy as np

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

        x_pixel = (h / 2 + target_point[:, 0] / (self.cfg.bev_x_bound[2]/self.cfg.scale)).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / (self.cfg.bev_x_bound[2]/self.cfg.scale)).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise
        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
            target_point_batch[1] - 4:target_point_batch[1] + 4] = 1.0
        return bev_feature, bev_target


    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)#已经是相对车的位置了
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)
        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)
        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion, target_point)

        pred_segmentation, fuse_bev = self.segmentation_head(fuse_feature)

        return fuse_feature, fuse_bev, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        fuse_feature, fuse_bev, pred_segmentation, pred_depth, _ = self.encoder(data)
        fuse_feature_copy = fuse_feature.clone()
        pred_waypoint_token = self.waypoint_predict(fuse_feature_copy,data['gt_waypoint'].cuda())
        # pred_waypoints = self.detokenize_waypoints(pred_waypoint_token, self.cfg)
        pred_waypoints = self.detokenize_waypoints_gt(data['gt_waypoint'][:,1:], self.cfg)
        # self.plot_waypoints(pred_waypoints, pred_segmentation)
        # import pdb; pdb.set_trace()
        ### Refined segmentation taking pred_waypoint and fuse_bev as input, and output new fuse_feature, pred_segmentation
        ### fuse_feature: B, 256, 264
        ### pred_segmentation: B, 3, 200, 200
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        return pred_control, pred_waypoint_token, pred_segmentation, pred_depth

    def predict(self, data):
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        for i in range(3):
            pred_control = self.control_predict.predict(fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(fuse_feature_copy, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)
        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target

    def detokenize_waypoints(self,pred_waypoint_token, cfg):
        """
        Detokenize a batch of waypoint values from tokens.
        
        :param pred_waypoint_token: Tensor of shape (batch_size, 14, 204), where 204 is the probability dimension.
        :param cfg: Config containing min/max values for x, y, yaw.
        :return: Tensor of shape (batch_size, 4, 3), where 4 is the time steps and 3 is (x, y, yaw).
        """
        token_nums = cfg.token_nums - 4  # Adjust token range

        # 找到最大概率对应的 token index (batch_size, 14)
        token_indices = torch.argmax(pred_waypoint_token, dim=-1)

        # Detokenization function
        def detokenize_single_value(token, min_value, max_value):
            normalized_value = token / token_nums
            return (normalized_value * (max_value - min_value)) + min_value

        # 重新整理 token 结构，每 3 维对应一个时间步 (batch_size, 4, 3)
        token_indices = token_indices[:,:12].view(token_indices.shape[0], 4, 3)  # (batch_size, 4, 3)

        # Detokenize x, y, yaw 分量
        x_values = detokenize_single_value(token_indices[:, :, 0], cfg.x_min, cfg.x_max)
        y_values = detokenize_single_value(token_indices[:, :, 1], cfg.y_min, cfg.y_max)
        yaw_values = detokenize_single_value(token_indices[:, :, 2], cfg.yaw_min, cfg.yaw_max)

        # 拼接最终输出 (batch_size, 4, 3)
        waypoints = torch.stack([x_values, y_values, yaw_values], dim=-1)

        return waypoints

    def detokenize_waypoints_gt(self,pred_waypoint_gt_token, cfg):
        """
        Detokenize a batch of waypoint values from tokens.
        
        :param pred_waypoint_gt_token: Tensor of shape (batch_size, 14, 204), where 204 is the probability dimension.
        :param cfg: Config containing min/max values for x, y, yaw.
        :return: Tensor of shape (batch_size, 4, 3), where 4 is the time steps and 3 is (x, y, yaw).
        """
        token_nums = cfg.token_nums - 4  # Adjust token range

        # 找到最大概率对应的 token index (batch_size, 14)
        # token_indices = torch.argmax(pred_waypoint_gt_token, dim=-1)

        # Detokenization function
        def detokenize_single_value(token, min_value, max_value):
            normalized_value = token / token_nums
            return (normalized_value * (max_value - min_value)) + min_value

        # 重新整理 token 结构，每 3 维对应一个时间步 (batch_size, 4, 3)
        token_indices = pred_waypoint_gt_token[:,:12].view(pred_waypoint_gt_token.shape[0], 4, 3)  # (batch_size, 4, 3)

        # Detokenize x, y, yaw 分量
        x_values = detokenize_single_value(token_indices[:, :, 0], cfg.x_min, cfg.x_max)
        y_values = detokenize_single_value(token_indices[:, :, 1], cfg.y_min, cfg.y_max)
        yaw_values = detokenize_single_value(token_indices[:, :, 2], cfg.yaw_min, cfg.yaw_max)

        # 拼接最终输出 (batch_size, 4, 3)
        waypoints = torch.stack([x_values, y_values, yaw_values], dim=-1)

        return waypoints

    def plot_waypoints(self, pred_waypoints, pred_segmentation, save_path="waypoints_plot.png"):
        """
        Plot waypoints on the segmentation map and save the figure.
        
        :param pred_waypoints: (4, 3) numpy array, each row is (x, y, yaw).
        :param pred_segmentation: (3, 200, 200) numpy array, segmentation map.
        :param save_path: Path to save the plot.
        """
        pred_waypoints = pred_waypoints[0].cpu().detach().numpy()
        pred_segmentation = pred_segmentation[0].cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))

        # 计算 grid 范围
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        resolution = 0.1

        # 反转 y 轴（符合图像坐标系）
        extent = [x_min, x_max, y_max, y_min]

        # 取 segmentation 的最大通道值进行可视化 (使用 argmax)
        segmentation_vis = np.argmax(pred_segmentation, axis=0)

        # 显示 segmentation
        ax.imshow(segmentation_vis, cmap='gray', extent=extent, origin='upper')

        # 绘制 waypoints
        for i, (x, y, yaw) in enumerate(pred_waypoints):
            y = -y
            ax.scatter(x, y, color='red', marker='o', label="Waypoint" if i == 0 else "")
            yaw += 90
            
            yaw_rad = np.radians(yaw)

            # 画 yaw 方向箭头
            dx = np.cos(yaw_rad) * 0.5
            dy = np.sin(yaw_rad) * 0.5
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc='blue', ec='blue')

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Predicted Waypoints & Segmentation")
        ax.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭 plt 避免内存泄露