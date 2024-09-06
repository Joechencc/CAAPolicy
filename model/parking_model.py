import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.conet_model import OccNet
from model.conet_encoder import ConetEncoder
from model.feature_fusion import FeatureFusion
from model.conet_fusion import CONetFusion
from model.control_predict import ControlPredict
from model.control_conet import ControlCONet
from model.segmentation_head import SegmentationHead
from model.seg3d_head import Seg3dHead
from data_generation.world import World
from data_generation.world import cam_specs_, cam2pixel_
import numpy as np
import carla
import os
import matplotlib.pyplot as plt


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg
        if self.cfg.feature_encoder == "bev":
            self.bev_model = BevModel(self.cfg)
            self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)
            self.feature_fusion = FeatureFusion(self.cfg)
            self.segmentation_head = SegmentationHead(self.cfg)
            self.control_predict = ControlPredict(self.cfg)

        elif self.cfg.feature_encoder == "conet":
            self.OccNet = OccNet(**self.cfg.OccNet_cfg)
            self.conet_encoder = ConetEncoder(self.cfg.conet_encoder_in_channel)
            self.conet_fusion = CONetFusion(self.cfg)
            self.seg3D_head = Seg3dHead(self.cfg)
            self.control_conet = ControlCONet(self.cfg)
        
    def add_target_bev(self, bev_feature, target_point):
        # Create a batch bev_feature
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

    def add_target_conet(self, bev_feature, target_point):
        # Create a batch bev_feature
        b, c, h, w, d = bev_feature.shape
        bev_target = torch.zeros((b, 1, h, w, d), dtype=torch.float).to(self.cfg.device, non_blocking=True)
        occ_size = (self.cfg.point_cloud_range[3] - self.cfg.point_cloud_range[0]) / h
        x_pixel = (h / 2 + target_point[:, 0] / occ_size).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / occ_size).unsqueeze(0).T.int()
        z_pixel = (d / 2 + target_point[:, 2] / occ_size).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel, z_pixel], dim=1)

        noise = (torch.rand_like(target_point, dtype=torch.float) * 10 - 5).int()
        target_point += noise

        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            bev_target_batch[target_point_batch[0] - 4:target_point_batch[0] + 4,
                             target_point_batch[1] - 4:target_point_batch[1] + 4,
                             target_point_batch[2] - 4:target_point_batch[2] + 4] = 1.0

        bev_feature = torch.cat([bev_feature, bev_target], dim=1)
        return bev_feature, bev_target

    def encoder(self, data):
        images = data['image'].to(self.cfg.device, non_blocking=True) #[1, 6, 3, 900, 1600]
        B, I = images.shape[:2]
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True) # [1, 3]
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        if self.cfg.feature_encoder == "bev":
            bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics) #bev_feature:[1, 64, 200, 200], pred_depth:[4, 48, 32, 32]
            bev_feature, bev_target = self.add_target_bev(bev_feature, target_point) #bev_feature:[1, 65, 200, 200], target_point:[1, 1, 200, 200]
            bev_down_sample = self.bev_encoder(bev_feature)
            fuse_feature = self.feature_fusion(bev_down_sample, ego_motion)
            pred_segmentation = self.segmentation_head(fuse_feature)
            return fuse_feature, pred_segmentation, pred_depth, bev_target

        elif self.cfg.feature_encoder == "conet":
            if self.cfg.only_3d_perception == False:
                rot, trans, cam2ego, post_rots, post_trans, bda_rot, img_shape, gt_depths = self.transform_spec(cam_specs_, cam2pixel_, B, I, images.shape, images.device)
                img_metas = self.construct_metas()
                img = [images, rot, trans, intrinsics, post_rots, post_trans, bda_rot, img_shape, gt_depths, cam2ego]
                res = self.OccNet(img_metas=img_metas,img_inputs=img) 
                coarse_semantic, conet_feature, pred_depth = res['pred_c'], res['fine_feature'], res['depth'] 
                # coarse_semantic (2,18,40,40,5), conet_feature (2,192,160,160,20), pred_depth(12,96,16,16)
                conet_feature, conet_target = self.add_target_conet(conet_feature, target_point) 
                # conet_feature (2,193,160,160,20), conet_target(2,1,160,160,20)
                conet_down_sample = self.conet_encoder(conet_feature)
                # conet_down_sample(2,512,128)
                fuse_feature = self.conet_fusion(conet_down_sample, ego_motion)
                # fuse_feature(2,128,256)
                pred_segmentation = self.seg3D_head(fuse_feature)
                #pred_segmentation (2, 18, 160, 160, 20)
                return fuse_feature, coarse_semantic, pred_segmentation, pred_depth, conet_target   
            elif self.cfg.only_3d_perception == True:
                rot, trans, cam2ego, post_rots, post_trans, bda_rot, img_shape, gt_depths = self.transform_spec(cam_specs_, cam2pixel_, B, I, images.shape, images.device)
                img_metas = self.construct_metas()
                img = [images, rot, trans, intrinsics, post_rots, post_trans, bda_rot, img_shape, gt_depths, cam2ego]
                res = self.OccNet(img_metas=img_metas,img_inputs=img) 
                coarse_semantic, fine_semantic, pred_depth = res['pred_c'], res['pred_f'], res['depth']
                # coarse_semantic (2,18,40,40,5), conet_feature (2,18,160,160,20), pred_depth(12,96,16,16)

                return torch.randn(2, 128, 256).to("cuda:0"), coarse_semantic,fine_semantic,pred_depth,torch.randn(2,1,160,160,20).to("cuda:0")


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
        rot, trans = sensor2egos[:,:,:3,:3], sensor2egos[:,:,:3,3]
        post_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, I, 1, 1).to(device)
        post_trans = torch.tensor([0.,-4.,0.]).unsqueeze(0).unsqueeze(0).repeat(B, I, 1).to(device)
        bda_rot = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
        gt_depths = torch.zeros(1).unsqueeze(0).unsqueeze(0).repeat(B, I, 1).to(device)
        img_shape = torch.tensor(img_shape[-2:]).to(device).unsqueeze(0).repeat(B,1)
        return rot, trans, sensor2egos, post_rots, post_trans, bda_rot, img_shape, gt_depths

    def forward(self, data):
        
        if self.cfg.feature_encoder == 'bev':
            fuse_feature, coarse_segmentation, fine_segmentation, pred_depth, _ = self.encoder(data)
            pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        elif self.cfg.feature_encoder == 'conet':
            fuse_feature, coarse_segmentation, fine_segmentation, pred_depth, _ = self.encoder(data)
            self.plot_grid(fine_segmentation, os.path.join("visual", "pred_fine.png"))
            self.plot_grid(coarse_segmentation, os.path.join("visual", "pred_coarse.png"))
            #pred_control = self.control_conet(fuse_feature, data['gt_control'].cuda())
            pred_control = torch.randn(2, 5, 3)
        return pred_control, coarse_segmentation, fine_segmentation, pred_depth

    # def predict(self, data):
    #     fuse_feature, coarse_segmentation, fine_segmentation, pred_depth, bev_target = self.encoder(data)
    #     breakpoint()
    #     self.plot_grid(fine_segmentation, os.path.join("visual", "pred_fine.png"))
    #     self.plot_grid(coarse_segmentation, os.path.join("visual", "pred_coarse.png"))

    #     assert()
    #     pred_multi_controls = data['gt_control'].cuda()
    #     for i in range(3):
    #         if self.cfg.feature_encoder == 'bev':
    #             pred_control = self.control_predict.predict(fuse_feature, pred_multi_controls)
    #         elif self.cfg.feature_encoder == 'conet':
    #             pred_control = self.control_conet.predict(fuse_feature, pred_multi_controls)
    #         pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
    #     return pred_multi_controls, coarse_segmentation, fine_segmentation, pred_depth, bev_target

    def plot_grid(self, threeD_grid, save_path=None, vmax=None, layer=None):
        # import pdb; pdb.set_trace()
        threeD_grid = torch.argmax(threeD_grid[0], dim=0).cpu().numpy()
        H, W, D = threeD_grid.shape


        threeD_grid[threeD_grid==4]=1
        threeD_grid[threeD_grid==17]=2
        twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = twoD_map[::-1,::-1]
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