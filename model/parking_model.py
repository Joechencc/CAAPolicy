import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead
from model.waypoint_predict import WaypointPredict
from model.gradient_approx import GradientApproximator
from model.temporal_fusion import TemporalFusion
from model.dynamics_model import DynamicsModel
from model.hybrid_dynamics_model import HybridDynamicsModel

class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        self.waypoint_predict = WaypointPredict(self.cfg)

        self.grad_approx = GradientApproximator(self.cfg.bev_encoder_out_channel)

        self.segmentation_head = SegmentationHead(self.cfg)

        self.dynamics_model = DynamicsModel()

        self.temporal_fusion = TemporalFusion(self.cfg)

        self.hybrid_dynamics_model = HybridDynamicsModel()

        checkpoint = torch.load(self.cfg.hybrid_ckpt_path, map_location=self.cfg.device)
        # Remove "model." prefix from keys in the state_dict
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.hybrid_dynamics_model.load_state_dict(new_state_dict)

        # Freeze weights of hybrid_dynamics_model
        for param in self.hybrid_dynamics_model.parameters():
            param.requires_grad = False

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
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)#已经是相对车的位置了
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)
        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion, target_point)

        pred_segmentation = self.segmentation_head(fuse_feature)

        return fuse_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        # import pdb; pdb.set_trace()
        # TODO: every time goes through the encoder, memory reserved increase 2GB
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        fuse_feature_copy = fuse_feature.clone()
        # TODO: Train with ground truth, only use dynamics model for inference
        delta_xy = data['delta_xy'].cuda()
        # TODO: couuld also use ground truth for inference.
        # with torch.no_grad():
        #     _,_, delta_xy = self.hybrid_dynamics_model(data)
        # Temporal Fusion: return the fused feature and bev cache for next-step's input
        fuse_feature_copy = self.temporal_fusion(fuse_feature_copy, delta_xy, data['ego_pos'][:, 2].unsqueeze(1))

        approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        pred_waypoint = self.waypoint_predict(fuse_feature_copy,data['gt_waypoint'].cuda())
        
        return pred_control, pred_waypoint, pred_segmentation, pred_depth, fuse_feature, approx_grad

    # def forward_eval_twice(self, refined_fuse_feature, pred_multi_controls, pred_multi_waypoints):
    #     refined_fuse_feature_copy = refined_fuse_feature.clone()
    #     _, pred_control = self.control_predict.predict(refined_fuse_feature, pred_multi_controls)
    #     pred_waypoint = self.waypoint_predict.predict(refined_fuse_feature_copy, pred_multi_waypoints)
    #     return pred_control, pred_waypoint
    def forward_twice(self, refined_feature, data):
        refined_feature_copy = refined_feature.clone()
        pred_control = self.control_predict(refined_feature, data['gt_control'].cuda())
        pred_waypoint = self.waypoint_predict(refined_feature_copy,data['gt_waypoint'].cuda())

        return pred_control, pred_waypoint
        
    def predict(self, data):
        # with torch.enable_grad():
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        pred_multi_waypoints = data['gt_waypoint'].cuda()
        fuse_feature_copy = fuse_feature.clone()
        approx_grad = self.grad_approx(fuse_feature_copy.transpose(1,2)).transpose(1,2)
        pred_tgt_logits = []

        for i in range(3):
            pred_control = self.control_predict.predict(approx_grad*fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)

        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(approx_grad*fuse_feature_copy, pred_multi_waypoints)
            pred_multi_waypoints = torch.cat([pred_multi_waypoints, pred_waypoint], dim=1)
        return pred_multi_controls, pred_multi_waypoints, pred_segmentation, pred_depth, bev_target
