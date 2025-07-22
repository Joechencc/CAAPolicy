import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict, ControlPredictRL
from model.segmentation_head import SegmentationHead
from model.waypoint_predict import WaypointPredict
from model.gradient_approx import GradientApproximator


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

        self.grad_approx = GradientApproximator(self.cfg.bev_encoder_out_channel)

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
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        noise_ego_motion = torch.randn_like(ego_motion).to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)
        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion, target_point)

        pred_segmentation = self.segmentation_head(fuse_feature)

        return fuse_feature, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        # if not self.training:
        #     fuse_feature = fuse_feature.clone().detach().requires_grad_(True)
        #     fuse_feature_copy = fuse_feature.clone().detach().requires_grad_(True)
        # else:
        fuse_feature_copy = fuse_feature.clone()
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

        # INFO: Original control
        for i in range(3):
            pred_control = self.control_predict.predict(approx_grad*fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)

        # INFO: New Control
        # logits = self.control_policy_rl(approx_grad*fuse_feature)  # [1, 3, 200]
        # dists = [torch.distributions.Categorical(logits=logits[:, i]) for i in range(3)]
        # actions = torch.stack([d.sample() for d in dists], dim=1)  # [1, 3]
        # pred_multi_controls = torch.cat([pred_multi_controls, actions], dim=1)

        for i in range(12):
            pred_waypoint = self.waypoint_predict.predict(approx_grad*fuse_feature_copy, pred_multi_waypoints)
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


