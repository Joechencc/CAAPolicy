import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.core.utils import reduce_mean
from mmcv.models.builder import HEADS
from mmcv.models.bricks import build_conv_layer, build_norm_layer
from .lovasz_softmax import lovasz_softmax
from model.utils import coarse_to_fine_coordinates, project_points_on_img
from model.utils.nusc_param import nusc_class_frequencies, nusc_class_names
from model.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from scipy.spatial.transform import Rotation as R

@HEADS.register_module()
class CONetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        num_img_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        fine_topk=20000,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        baseline_mode=None,
        final_occ_size=[256, 256, 20],
        empty_idx=0,
        visible_loss=False,
        balance_cls_weight=True,
        cascade_ratio=1,
        sample_from_voxel=False,
        sample_from_img=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super(CONetHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.fine_topk = fine_topk
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()
        self.baseline_mode = baseline_mode
        self.final_occ_size = final_occ_size
        self.visible_loss = visible_loss
        self.cascade_ratio = cascade_ratio
        self.sample_from_voxel = sample_from_voxel
        self.sample_from_img = sample_from_img

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=128, 
                                out_channels=64, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, 64)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=64, 
                                out_channels=out_channel, kernel_size=1, stride=1, padding=0)
            )

        if self.cascade_ratio != 1: 
            if self.sample_from_voxel or self.sample_from_img:
                fine_mlp_input_dim = 0 if not self.sample_from_voxel else 128
                if sample_from_img:
                    self.img_mlp_0 = nn.Sequential(
                        nn.Conv2d(512, 128, 1, 1, 0),
                        nn.GroupNorm(16, 128),
                        nn.ReLU(inplace=True)
                    )
                    self.img_mlp = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.GroupNorm(16, 64),
                        nn.ReLU(inplace=True),
                    )
                    fine_mlp_input_dim += 64

                self.fine_mlp = nn.Sequential(
                    nn.Linear(fine_mlp_input_dim, 64),
                    nn.GroupNorm(16, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, out_channel)
            )

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)

        self.soft_weights = soft_weights
        self.num_img_level = num_img_level
        
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        
     
    def forward(self, coarse_feat, img_feats=None, transform=None, **kwargs):
        
        # forward voxel 
        out_voxel_feats = coarse_feat
        coarse_occ = self.occ_pred_conv(out_voxel_feats)
        out_mask = None
        output = {}

        if self.cascade_ratio != 1:
            if self.sample_from_img or self.sample_from_voxel:
                coarse_occ_mask = coarse_occ.argmax(1) != self.empty_idx
                
                assert coarse_occ_mask.sum() > 0, 'no foreground in coarse voxel'
                B, W, H = coarse_occ_mask.shape
                coarse_coord_x, coarse_coord_y= torch.meshgrid(torch.arange(W).to(coarse_occ.device),
                            torch.arange(H).to(coarse_occ.device), indexing='ij')
                ##### We need to change other mode later ##### 
                if self.baseline_mode == "NearRefine": # Provide near_range_mask
                    h,w = self.final_occ_size
                    w_mask = (torch.arange(W) >= int(w/2)-40) & (torch.arange(W) < int(w/2)+40)
                    h_mask = (torch.arange(H) >= int(h/2)-40) & (torch.arange(H) < int(h/2)+40)
                    w_mask = w_mask.view(1, W, 1, 1)  # Shape (1, W, 1, 1)
                    h_mask = h_mask.view(1, 1, H, 1)  # Shape (1, 1, H, 1)
                    w_h_mask = w_mask & h_mask
                    w_h_mask = w_h_mask.expand(B, W, H, D)
                    w_h_mask = w_h_mask.bool().to(coarse_occ_mask.device)
                    out_mask = w_h_mask.clone()
                    coarse_occ_mask = coarse_occ_mask & w_h_mask
                elif self.baseline_mode == "Trajectory": # Optimize along the trajectory
                    #calculate delta translation and delta rotation
                    delta_translation = [kwargs['ego2global_translation_next'][i] - kwargs['ego2global_translation'][i] for i in range(3)]
                    q1 = [kwargs['ego2global_rotation'][i] for i in range(4)]
                    q2 = [kwargs['ego2global_rotation_next'][i] for i in range(4)]
                    delta_translation, q1, q2 = torch.stack(delta_translation).transpose(0,1), torch.stack(q1).transpose(0,1), torch.stack(q2).transpose(0,1)
                    
                    r1 = R.from_quat(q1.cpu())
                    r2 = R.from_quat(q2.cpu())
                    delta_rotation = r2 * r1.inv() 
                    delta_rotation = torch.tensor(delta_rotation.as_matrix()).to(coarse_occ_mask.device)
                    # Calculate 5 waypoints
                    trajectory_waypoints = self.create_trajectory(r1, delta_translation, delta_rotation, coarse_occ_mask.device, n_step=5)
                    traj_mask = self.create_waypoint_mask(trajectory_waypoints, kwargs['img_metas'][0]['pc_range'], (B, W, H, D), coarse_occ_mask.device)
                    out_mask = traj_mask.clone()
                    coarse_occ_mask = coarse_occ_mask & traj_mask
                elif self.baseline_mode == "Zonotope": # Optimize along the trajectory
                    #calculate delta translation and delta rotation
                    delta_translation = [kwargs['ego2global_translation_next'][i] - kwargs['ego2global_translation'][i] for i in range(3)]
                    q1 = [kwargs['ego2global_rotation'][i] for i in range(4)]
                    q2 = [kwargs['ego2global_rotation_next'][i] for i in range(4)]
                    delta_translation, q1, q2 = torch.stack(delta_translation).transpose(0,1), torch.stack(q1).transpose(0,1), torch.stack(q2).transpose(0,1)
                    
                    r1 = R.from_quat(q1.cpu())
                    r2 = R.from_quat(q2.cpu())
                    delta_rotation = r2 * r1.inv() 
                    delta_rotation = torch.tensor(delta_rotation.as_matrix()).to(coarse_occ_mask.device)
                    # Calculate 5 waypoints
                    trajectory_waypoints = self.create_trajectory(r1, delta_translation, delta_rotation, coarse_occ_mask.device, n_step=5)
                    zonotopes = self.generate_zonotopes(trajectory_waypoints, coarse_occ_mask.device)
                    zono_mask = self.create_waypoint_mask(zonotopes, kwargs['img_metas'][0]['pc_range'], (B, W, H, D), coarse_occ_mask.device)
                    out_mask = zono_mask.clone()
                    coarse_occ_mask = coarse_occ_mask & zono_mask

                output['fine_output'] = []
                output['fine_feature'] = []
                output['fine_coord'] = []

                if self.sample_from_img and img_feats is not None:
                    img_feats_ = img_feats[0]
                    B_i,N_i,C_i, W_i, H_i = img_feats_.shape
                    img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                    img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]

                for b in range(B):
                    append_feats = []
                    this_coarse_coord = torch.stack([coarse_coord_x[coarse_occ_mask[b]],
                                                    coarse_coord_y[coarse_occ_mask[b]]], dim=0)  # 2, N
                    
                    if self.training:
                        this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio, topk=self.fine_topk)  # 2, 8N/64N
                    else:
                        this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio)  # 2, 8N/64N
                    
                    output['fine_coord'].append(this_fine_coord)
                    new_coord = this_fine_coord[None].permute(0,2,1).float().contiguous()  # x y z

                    if self.sample_from_voxel:
                        this_fine_coord = this_fine_coord.float()
                        this_fine_coord[0, :] = (this_fine_coord[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                        this_fine_coord[1, :] = (this_fine_coord[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                        # this_fine_coord[2, :] = (this_fine_coord[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                        this_fine_coord = this_fine_coord[None,None].permute(0,3,1,2).float()
                        # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 2]; output: [B, C, N, 1, 1]
                        new_feat = F.grid_sample(out_voxel_feats[b:b+1].permute(0,1,3,2), this_fine_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
                        append_feats.append(new_feat[0,:,:,0].permute(1,0))
                        assert torch.isnan(new_feat).sum().item() == 0
                    # image branch
                    if img_feats is not None and self.sample_from_img:
                        W_new, H_new = W * self.cascade_ratio, H * self.cascade_ratio
                        img_uv, img_mask = project_points_on_img(new_coord, rots=transform[0][b:b+1], trans=transform[1][b:b+1],
                                    intrins=transform[2][b:b+1], post_rots=transform[3][b:b+1],
                                    post_trans=transform[4][b:b+1], bda_mat=transform[5][b:b+1],
                                    W_img=transform[6][b:b+1][0][1], H_img=transform[6][b:b+1][0][0],
                                    pts_range=self.point_cloud_range, W_occ=W_new, H_occ=H_new)  # 1 N n_cam 2
                        for img_feat in img_feats:
                            sampled_img_feat = F.grid_sample(img_feat[b].contiguous(), img_uv.contiguous(), align_corners=True, mode='bilinear', padding_mode='zeros')
                            sampled_img_feat = sampled_img_feat * img_mask.permute(2,1,0)[:,None]
                            sampled_img_feat = self.img_mlp(sampled_img_feat.sum(0)[:,:,0].permute(1,0))
                            append_feats.append(sampled_img_feat)  # N C
                            assert torch.isnan(sampled_img_feat).sum().item() == 0
                    output['fine_output'].append(self.fine_mlp(torch.concat(append_feats, dim=1)))
                    output['fine_feature'].append(torch.concat(append_feats, dim=1))

        res = {
            'output_feature': out_voxel_feats,
            'output_feature_fine': output['fine_feature'],
            'output_voxels': coarse_occ,
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
            'coarse_occ_mask': out_mask,
        }
        
        return res
    
    def create_transformation_matrix(self, trans, rot, device):
        transformation_matrix = torch.eye(4).to(device).unsqueeze(0).repeat(trans.shape[0], 1, 1)
        transformation_matrix[:, :3, :3] = rot
        transformation_matrix[:, :3, 3] = trans
        return transformation_matrix
    
    def create_trajectory(self, r1, delta_translation, delta_rotation, device, n_step=5):
        transformation = self.create_transformation_matrix(delta_translation, delta_rotation, device)
        trajectory_waypoints = []
        for i in range(n_step):
            if i == 0:
                waypoint = self.create_transformation_matrix(torch.tensor([0,0,0]).to(device).unsqueeze(0).repeat(r1.as_matrix().shape[0], 1), torch.tensor(r1.as_matrix()).to(device), device)
            else:
                waypoint = torch.matmul(transformation, waypoint)
            trajectory_waypoints.append(waypoint[:,:3,3])
        return trajectory_waypoints

    def create_waypoint_mask(self, trajectory_waypoints, pc_range, occ_size, device):
        waypoint_mask = torch.zeros(occ_size).to(device).bool()
        grid_size = (pc_range[3]-pc_range[0]) / occ_size[1] 
        B, W, H, D = occ_size
        for waypoint in trajectory_waypoints:
            x_idx, y_idx, z_idx = self.point_to_grid_index(waypoint, pc_range, grid_size)
            x_idx -= int(W/2)
            y_idx -= int(H/2)
            w_mask = (torch.arange(W).unsqueeze(0).repeat(x_idx.shape[0],1) >= (60+torch.tensor(x_idx)).unsqueeze(1)) & (torch.arange(W).unsqueeze(0).repeat(x_idx.shape[0],1) < (68+torch.tensor(x_idx)).unsqueeze(1))
            h_mask = (torch.arange(H).unsqueeze(0).repeat(x_idx.shape[0],1) >= (60+torch.tensor(y_idx)).unsqueeze(1)) & (torch.arange(H).unsqueeze(0).repeat(x_idx.shape[0],1) < (68+torch.tensor(y_idx)).unsqueeze(1))
            w_mask = w_mask.view(B, W, 1, 1)  # Shape (B, W, 1, 1)
            h_mask = h_mask.view(B, 1, H, 1)  # Shape (B, 1, H, 1)
            w_h_mask = w_mask & h_mask
            w_h_mask = w_h_mask.expand(B, W, H, D)
            w_h_mask = w_h_mask.bool().to(device)
            waypoint_mask = waypoint_mask | w_h_mask
        return waypoint_mask
    
    def generate_zonotopes(self, trajectory_waypoints, device):
        zonotopes = []
        for waypoint in trajectory_waypoints:
            center = waypoint.cpu().numpy()
            # Example: Use identity matrix as generators for simplicity
            generators = np.eye(3)
            zonotope_vertices = self.calculate_zonotope(center, generators)
            zonotopes.append(torch.tensor(zonotope_vertices).to(device))
        zonotopes = torch.concat(zonotopes)
        return zonotopes
    
    def calculate_zonotope(self, center, generators):
        num_generators = generators.shape[1]
        vertices = []
        
        # Iterate over all combinations of generators
        for i in range(1 << num_generators):
            combination = np.array([1 if (i & (1 << j)) else -1 for j in range(num_generators)])
            vertex = center + np.dot(generators, combination)
            vertices.append(vertex)
        return np.array(vertices)
    
    def point_to_grid_index(self, point, pc_range, grid_size=0.8):
        """
        points to voxel grid
        """
        x_idx = (point[:,0].cpu().numpy() - pc_range[0]) / grid_size
        y_idx = (point[:,1].cpu().numpy() - pc_range[1]) / grid_size
        z_idx = (point[:,2].cpu().numpy() - pc_range[2]) / grid_size
        return x_idx.astype(int), y_idx.astype(int), z_idx.astype(int)

    def loss_voxel(self, output_voxels, target_voxels, tag):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict

    def loss_point(self, fine_coord, fine_output, target_voxels, tag):

        selected_gt = target_voxels[:, fine_coord[0,:], fine_coord[1,:], fine_coord[2,:]].long()[0]
        assert torch.isnan(selected_gt).sum().item() == 0, torch.isnan(selected_gt).sum().item()
        assert torch.isnan(fine_output).sum().item() == 0, torch.isnan(fine_output).sum().item()

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(fine_output, selected_gt, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(fine_output, dim=1), selected_gt, ignore=255)


        return loss_dict

    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, visible_mask=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='c_{}'.format(index)))
        if self.cascade_ratio != 1:
            loss_batch_dict = {}
            if self.sample_from_voxel or self.sample_from_img:
                for index, (fine_coord, fine_output) in enumerate(zip(output_coords_fine, output_voxels_fine)):
                    this_batch_loss = self.loss_point(fine_coord, fine_output, target_voxels, tag='fine')
                    for k, v in this_batch_loss.items():
                        if k not in loss_batch_dict:
                            loss_batch_dict[k] = v
                        else:
                            loss_batch_dict[k] = loss_batch_dict[k] + v
                for k, v in loss_batch_dict.items():
                    loss_dict[k] = v / len(output_coords_fine)
            
        return loss_dict
    
        
