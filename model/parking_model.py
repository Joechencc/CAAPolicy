import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict, ControlPredictRL
from diffuser.models.diffusion import GaussianDiffusion
from model.segmentation_head import SegmentationHead
from model.waypoint_predict import WaypointPredict
from model.gradient_approx import GradientApproximator
from model.film_modulator import FiLMModulator

import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

import numpy as np
from diffuser.utils.visualizer import plot_trajectory_with_yaw, invert_trajectory_2D, deg2rad_trajectory_2D


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        # self.control_predict_rl = ControlPredictRL(self.cfg)

        self.waypoint_predict = WaypointPredict(self.cfg)

        self.grad_approx = GradientApproximator(self.cfg.bev_encoder_out_channel+3)

        self.film_modulate = FiLMModulator(self.cfg)

        # self.seg_modulate = 

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

    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True) #已经是相对车的位置了
        zero_target_point = torch.zeros_like(target_point).to(self.cfg.device, non_blocking=True).unsqueeze(1)
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        zero_ego_motion = torch.zeros_like(ego_motion).to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)
        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, target_point)

        # pred_segmentation = self.segmentation_head(fuse_feature)
        filmed_fuse_feature = self.film_modulate(fuse_feature, target_point)
        # self.visualize_fused_feature(filmed_fuse_feature, method="pca")
        pred_segmentation = self.segmentation_head(filmed_fuse_feature)

        # Step 1: Downsample segmentation to 16x16
        seg_down = F.interpolate(pred_segmentation, size=(16, 16), mode='bilinear', align_corners=False)  # [1, 3, 16, 16]
        # Step 2: Rearrange segmentation to [1, 256, 3]
        seg_down_flat = seg_down.permute(0, 2, 3, 1).reshape(pred_segmentation.shape[0], filmed_fuse_feature.shape[1], 3)  # [1, 256, 3]
        # Step 3: Concatenate along the feature dimension
        concat_feature = torch.cat([filmed_fuse_feature, seg_down_flat], dim=-1)  # [1, 256, 267]

        return concat_feature, pred_segmentation, pred_depth, bev_target


    def visualize_fused_feature(self, fused_feature: torch.Tensor, method: str = 'mean'):
        """
        Visualize a fused feature of shape (1, 256, 264) as a 2D image.

        Args:
            fused_feature (torch.Tensor): Tensor of shape [1, 256, 264]
            method (str): Aggregation method across channels ('mean', 'max', 'pca', 'umap')
        """
        assert fused_feature.shape == (1, 256, 264), "Expected shape [1, 256, 264]"
        
        # Reshape to (16, 16, 264)
        _, S, C = fused_feature.shape
        H = W = int(math.sqrt(S))
        feat_img = fused_feature.reshape(H, W, C)  # (16, 16, 264)
        feat_flat = feat_img.reshape(-1, C).cpu().numpy()  # (256, 264)

        # Reduce to 1D using different methods
        if method == 'mean':
            feat_vis = fused_feature.mean(dim=-1).reshape(H, W)
        elif method == 'max':
            feat_vis = fused_feature.max(dim=-1)[0].reshape(H, W)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            feat_pca = PCA(n_components=1).fit_transform(feat_flat)
            feat_vis = torch.tensor(feat_pca).reshape(H, W)
        elif method == 'umap':
            from umap import UMAP
            feat_umap = UMAP(n_components=1).fit_transform(feat_flat)
            feat_vis = torch.tensor(feat_umap).reshape(H, W)
        else:
            raise ValueError("method must be one of ['mean', 'max', 'pca', 'umap']")

        # Normalize to [0, 1]
        feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)

        # Plot
        plt.imshow(feat_vis.cpu().numpy(), cmap='viridis')
        plt.title(f"Fused Feature ({method})")
        plt.colorbar()
        plt.axis('off')
        plt.show()


    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        # if not self.training:
        #     fuse_feature = fuse_feature.clone().detach().requires_grad_(True)
        #     fuse_feature_copy = fuse_feature.clone().detach().requires_grad_(True)
        # else:
        fuse_feature.requires_grad_(True)
        fuse_feature_copy = fuse_feature.clone()
        approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        # original control_predict
        # pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        pred_control = self.control_predict(pred_segmentation, data)
        pred_waypoint = self.waypoint_predict(pred_segmentation, data['gt_waypoint'].cuda())

        return pred_control, pred_waypoint, pred_segmentation, pred_depth, fuse_feature, approx_grad

    # def forward_eval_twice(self, refined_fuse_feature, pred_multi_controls, pred_multi_waypoints):
    #     refined_fuse_feature_copy = refined_fuse_feature.clone()
    #     _, pred_control = self.control_predict.predict(refined_fuse_feature, pred_multi_controls)
    #     pred_waypoint = self.waypoint_predict.predict(refined_fuse_feature_copy, pred_multi_waypoints)
    #     return pred_control, pred_waypoint
    def forward_twice(self, refined_feature, data):
        refined_feature.requires_grad_(True)
        refined_feature_copy = refined_feature.clone()
        pred_control = self.control_predict(refined_feature, data)
        pred_waypoint = self.waypoint_predict(refined_feature_copy, data['gt_waypoint'].cuda())

        return pred_control, pred_waypoint
        
    def predict(self, data):
        # with torch.enable_grad():
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        pred_tgt_logits = []

        # INFO: Original control
        for i in range(3):
            pred_control = self.control_predict.predict(pred_segmentation, pred_multi_controls, data)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)

        # INFO: New Control
        # logits = self.control_policy_rl(approx_grad*fuse_feature)  # [1, 3, 200]
        # dists = [torch.distributions.Categorical(logits=logits[:, i]) for i in range(3)]
        # actions = torch.stack([d.sample() for d in dists], dim=1)  # [1, 3]
        # pred_multi_controls = torch.cat([pred_multi_controls, actions], dim=1)

        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(pred_segmentation, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)
        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target

    def predict_control_logits(self, data):
        # with torch.enable_grad():
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        pred_tgt_logits = []

        for i in range(3):
            pred_control, pred_controls_f = self.control_predict.predict_logits(approx_grad*fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
            pred_tgt_logits.append(pred_controls_f)
        pred_tgt_logits = torch.concat(pred_tgt_logits, dim=0).cuda()

        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(approx_grad*fuse_feature_copy, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)

        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target, fuse_feature, pred_tgt_logits

    def fuse_feature_to_logits(self, fuse_feature, pred_multi_controls):
        approx_grad = self.grad_approx(fuse_feature.transpose(1,2)).transpose(1,2)
        pred_tgt_logits = []
        # for i in range(3):
        pred_control, pred_controls_f = self.control_predict.predict_logits(approx_grad*fuse_feature, pred_multi_controls)
        # pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        # pred_tgt_logits.append(pred_controls_f.cuda())
        # pred_tgt_logits = torch.concat(pred_tgt_logits, dim=0).cuda()
        return pred_controls_f


class ParkingModelDiffusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        # revise this
        self.trajectory_predict = GaussianDiffusion(self.cfg)

        self.film_modulate = FiLMModulator(self.cfg)

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

    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True) # in car ego frame
        zero_target_point = torch.zeros_like(target_point).to(self.cfg.device, non_blocking=True).unsqueeze(1)
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        zero_ego_motion = torch.zeros_like(ego_motion).to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)
        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, target_point)

        # pred_segmentation = self.segmentation_head(fuse_feature)
        filmed_fuse_feature = self.film_modulate(fuse_feature, target_point)
        # self.visualize_fused_feature(filmed_fuse_feature, method="pca")
        pred_segmentation = self.segmentation_head(filmed_fuse_feature)

        # Step 1: Downsample segmentation to 16x16
        seg_down = F.interpolate(pred_segmentation, size=(16, 16), mode='bilinear', align_corners=False)  # [1, 3, 16, 16]
        # Step 2: Rearrange segmentation to [1, 256, 3]
        seg_down_flat = seg_down.permute(0, 2, 3, 1).reshape(pred_segmentation.shape[0], filmed_fuse_feature.shape[1], 3)  # [1, 256, 3]
        # Step 3: Concatenate along the feature dimension
        concat_feature = torch.cat([filmed_fuse_feature, seg_down_flat], dim=-1)  # [1, 256, 267]

        return concat_feature, pred_segmentation, pred_depth, bev_target


    def visualize_fused_feature(self, fused_feature: torch.Tensor, method: str = 'mean'):
        """
        Visualize a fused feature of shape (1, 256, 264) as a 2D image.

        Args:
            fused_feature (torch.Tensor): Tensor of shape [1, 256, 264]
            method (str): Aggregation method across channels ('mean', 'max', 'pca', 'umap')
        """
        assert fused_feature.shape == (1, 256, 264), "Expected shape [1, 256, 264]"
        
        # Reshape to (16, 16, 264)
        _, S, C = fused_feature.shape
        H = W = int(math.sqrt(S))
        feat_img = fused_feature.reshape(H, W, C)  # (16, 16, 264)
        feat_flat = feat_img.reshape(-1, C).cpu().numpy()  # (256, 264)

        # Reduce to 1D using different methods
        if method == 'mean':
            feat_vis = fused_feature.mean(dim=-1).reshape(H, W)
        elif method == 'max':
            feat_vis = fused_feature.max(dim=-1)[0].reshape(H, W)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            feat_pca = PCA(n_components=1).fit_transform(feat_flat)
            feat_vis = torch.tensor(feat_pca).reshape(H, W)
        elif method == 'umap':
            from umap import UMAP
            feat_umap = UMAP(n_components=1).fit_transform(feat_flat)
            feat_vis = torch.tensor(feat_umap).reshape(H, W)
        else:
            raise ValueError("method must be one of ['mean', 'max', 'pca', 'umap']")

        # Normalize to [0, 1]
        feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)

        # Plot
        plt.imshow(feat_vis.cpu().numpy(), cmap='viridis')
        plt.title(f"Fused Feature ({method})")
        plt.colorbar()
        plt.axis('off')
        plt.show()


    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        if self.cfg.ego_centric_traj:
            gt_target_point_traj = self.world_to_ego0(data["ego_trans_traj"])
        else:
            gt_target_point_traj = data["gt_target_point_traj"]

        if self.cfg.normalize_traj:
            gt_target_point_traj = self.normalize_trajectories(gt_target_point_traj, device = "cuda")
            target_point = self.normalize_trajectories(data["target_point"], device = "cuda")
        else:
            gt_target_point_traj = gt_target_point_traj
            target_point = target_point

        # if not self.training:
        #     fuse_feature = fuse_feature.clone().detach().requires_grad_(True)
        #     fuse_feature_copy = fuse_feature.clone().detach().requires_grad_(True)
        # else:
        fuse_feature.requires_grad_(True)
        fuse_feature_copy = fuse_feature.clone()
        # approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        # original control_predict
        # pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((gt_target_point_traj[:,0:1,:], gt_target_point_traj[:,-1:,:]), dim=1)
        else:
            start_end_relative_point = gt_target_point_traj[:,0:1,:]
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass
        # pred_control = self.trajectory_predict(seg_egoMotion_tgtPose, start_end_relative_point)
        # pred_waypoint = self.waypoint_predict(pred_segmentation, data['gt_waypoint'].cuda())

        return pred_segmentation, pred_depth, fuse_feature

    def diffusion_loss(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)

        if self.cfg.ego_centric_traj:
            gt_target_point_traj = self.world_to_ego0(data["ego_trans_traj"])
        else:
            gt_target_point_traj = data["gt_target_point_traj"]

        if self.cfg.normalize_traj:
            gt_target_point_traj = self.normalize_trajectories(gt_target_point_traj, device = "cuda")
            target_point = self.normalize_trajectories(data["target_point"], device = "cuda")
        else:
            gt_target_point_traj = gt_target_point_traj
            target_point = target_point



        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((gt_target_point_traj[:,0:1,:], gt_target_point_traj[:,-1:,:]), dim=1)
        else:
            start_end_relative_point = gt_target_point_traj[:,0:1,:]
            
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass
        loss = self.trajectory_predict.loss(gt_target_point_traj, seg_egoMotion_tgtPose, start_end_relative_point)[0]
        return loss

    # def forward_eval_twice(self, refined_fuse_feature, pred_multi_controls, pred_multi_waypoints):
    #     refined_fuse_feature_copy = refined_fuse_feature.clone()
    #     _, pred_control = self.control_predict.predict(refined_fuse_feature, pred_multi_controls)
    #     pred_waypoint = self.waypoint_predict.predict(refined_fuse_feature_copy, pred_multi_waypoints)
    #     return pred_control, pred_waypoint
    def forward_twice(self, refined_feature, data):
        refined_feature.requires_grad_(True)
        refined_feature_copy = refined_feature.clone()
        pred_control = self.control_predict(refined_feature, data)
        pred_waypoint = self.waypoint_predict(refined_feature_copy, data['gt_waypoint'].cuda())

        return pred_control, pred_waypoint
        
    def predict(self, data, final_steps = False):
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)

        if self.cfg.normalize_traj:
            target_point = self.normalize_trajectories(data["target_point"], device = "cpu")
        else:
            target_point = target_point

        fuse_feature.requires_grad_(True)
        fuse_feature_copy = fuse_feature.clone()

        if self.cfg.ego_centric_traj:
            end_relative_point = target_point.unsqueeze(1)
            if self.cfg.normalize_traj:
                end_relative_point = (end_relative_point)
            start_relative_point = torch.zeros_like(end_relative_point)
        else:
            start_relative_point = target_point.unsqueeze(1)
            if self.cfg.normalize_traj:
                start_relative_point = (start_relative_point)
            end_relative_point = torch.zeros_like(start_relative_point)

        if "global" in self.cfg.planner_type:
            start_end_relative_point = torch.cat((start_relative_point, end_relative_point), dim=1)
        else:
            start_end_relative_point = start_relative_point
            if final_steps:
                start_end_relative_point = torch.cat((start_relative_point, end_relative_point), dim=1)
        # start_end_relative_point = torch.cat((start_relative_point, end_relative_point), dim=1)
        # torch.cat((data["gt_target_point_traj"][:,0:1,:], data["gt_target_point_traj"][:,-1:,:]), dim=1)
        if self.cfg.motion_head == "embedding":
            seg_egoMotion_tgtPose = {"pred_segmentation": fuse_feature, "target_point": target_point}
        elif self.cfg.motion_head == "segmentation":
            seg_egoMotion_tgtPose = {"pred_segmentation": pred_segmentation, "target_point": target_point}
        else:
            pass        
        
        pred_traj = self.trajectory_predict(seg_egoMotion_tgtPose, start_end_relative_point)
        if self.cfg.normalize_traj:
            pred_traj = self.denormalize_target_point(pred_traj, device="cuda")
        # pred_control = self.denormalize_target_point(pred_control, mean=torch.Tensor([[-2.4473171, -0.39712235, 0.10734732]]).cuda(), std=torch.Tensor([[2.7895596, 3.3346443, 1.0033212]]).cuda())

        if self.cfg.ego_centric_traj:
            pred_traj = pred_traj.squeeze(0)
        else:
            pred_traj = self.world_to_ego0(pred_traj)

        # plot_trajectory_with_yaw(invert_trajectory_2D(deg2rad_trajectory_2D(pred_control.squeeze(0))), invert_y=True)

        return pred_traj, pred_segmentation, pred_depth, bev_target

    def normalize_trajectories(self, traj, device = "cpu"):
        normalized_traj = traj / torch.tensor([[10.0, 10.0, 180.0]], device=device)
        return normalized_traj

    def denormalize_target_point(self, traj, device):
        if device == "cuda":
            traj = traj * torch.Tensor([[10.0, 10.0, 180.0]]).cuda()
        elif device == "cpu":
            traj = traj * torch.Tensor([[10.0, 10.0, 180.0]])
        else:
            pass
        return traj

    def world_to_ego0(self, ego_trans_world):
        """
        ego_trans_world: [B, T, 3] -> (x_w, y_w, yaw_deg_w)
        returns:         [B, T, 3] -> (x_0, y_0, yaw_deg_rel) in ego_0 frame
        """
        def wrap_deg(d):
            # wrap to (-180, 180]
            return (d + 180.0) % 360.0 - 180.0

        xw  = ego_trans_world[..., 0]
        yw  = ego_trans_world[..., 1]
        yaw = ego_trans_world[..., 2]          # degrees

        x0   = xw[..., :1]                      # [B, 1]
        y0   = yw[..., :1]                      # [B, 1]
        yaw0 = yaw[..., :1]                     # [B, 1]

        th0 = torch.deg2rad(yaw0)
        c0, s0 = torch.cos(th0), torch.sin(th0)

        # translate then rotate by -yaw0
        dx = xw - x0
        dy = yw - y0
        x_rel =  c0 * dx + s0 * dy
        y_rel = -s0 * dx + c0 * dy

        yaw_rel = wrap_deg(yaw - yaw0)

        return torch.stack([x_rel, y_rel, yaw_rel], dim=-1)

    def predict_control_logits(self, data):
        # with torch.enable_grad():
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        pred_tgt_logits = []

        for i in range(3):
            pred_control, pred_controls_f = self.control_predict.predict_logits(approx_grad*fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
            pred_tgt_logits.append(pred_controls_f)
        pred_tgt_logits = torch.concat(pred_tgt_logits, dim=0).cuda()

        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(approx_grad*fuse_feature_copy, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)

        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target, fuse_feature, pred_tgt_logits

    def fuse_feature_to_logits(self, fuse_feature, pred_multi_controls):
        approx_grad = self.grad_approx(fuse_feature.transpose(1,2)).transpose(1,2)
        pred_tgt_logits = []
        # for i in range(3):
        pred_control, pred_controls_f = self.control_predict.predict_logits(approx_grad*fuse_feature, pred_multi_controls)
        # pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        # pred_tgt_logits.append(pred_controls_f.cuda())
        # pred_tgt_logits = torch.concat(pred_tgt_logits, dim=0).cuda()
        return pred_controls_f



