import torch
import collections 
import torch.nn.functional as F
import torch.nn as nn

from mmcv.models import DETECTORS, build_backbone, build_neck
from mmcv.utils import auto_fp16, force_fp32
from mmcv.models.backbones.base_module import BaseModule
from .bevdepth import BEVDepth
from .conet_head import CONetHead
from model.mmdirs.ViewTransformerLSSVoxel import ViewTransformerLiftSplatShootVoxel

import numpy as np
import time
import copy, os, cv2
from mmcv.models.bricks import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.models.builder import BACKBONES, NECKS
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

@DETECTORS.register_module()
class OccNet(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            occ_fuser=None,
            loss_norm=False,
            occ_size=None,
            **kwargs):
        super().__init__(**kwargs) #### uncomment it later when merging
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.occ_fuser = None
        self.occ_size = occ_size
        self.maxpool = nn.AdaptiveMaxPool3d((160, 160, 1))
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
            
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'img_feats': [x.clone()]}
    
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_enc_feats = self.image_encoder(img[0])
        img_feats = img_enc_feats['img_feats']
        
        return img_feats

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats
    
    def plot_grid(self, twoD_grid, save_path=None, vmax=None, layer=None):
        H, W = twoD_grid.shape

        # twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        twoD_map = twoD_grid
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

    def forward(self,
            coarse_feat=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        return self.forward_once(coarse_feat, img_metas, img_inputs, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def forward_once(self, coarse_feat, img_metas, img=None, rescale=False, 
            gt_occ=None, visible_mask=None, visual=False, **kwargs):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            coarse_feat=coarse_feat,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
            **kwargs,
        )

        fine_feature = None
        B, C, H, W = coarse_feat.shape
        device = coarse_feat.device
        
        ########        
        pred_f = None
        # if output['output_voxels_fine'] is not None:
        #     if output['output_coords_fine'] is not None and output['output_feature_fine'] is not None:
        # if gt_occ is not None:
        #     fine_feature = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, output['output_feature_fine'][0].shape[1], 1, 1).float()
        #     pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, output['output_feature_fine'][0].shape[1], 1, 1).float()
        # else:
        fine_feature = self.empty_idx * torch.ones((B, H*4, W*4), device=device)[:, None].repeat(1, output['output_feature_fine'][0].shape[1], 1, 1).float()
        pred_f = self.empty_idx * torch.ones((B, H*4, W*4), device=device)[:, None].repeat(1, output['output_voxels_fine'][0].shape[1], 1, 1).float()
        for i in range(len(output['output_feature_fine'])):
            fine_pred_feature = output['output_feature_fine'][i]  # N feats
            fine_coord = output['output_coords_fine'][i]  # 2 N
            fine_pred = output['output_voxels_fine'][i]
            fine_feature[i, :, fine_coord[0], fine_coord[1]] = fine_pred_feature.permute(1, 0)[None]
            pred_f[i, :, fine_coord[0], fine_coord[1]] = fine_pred.permute(1, 0)[None]
    # else:
    #     assert()
        pred_c = output['output_voxels']
        # visual = True
        # if visual:
        #     if gt_occ is not None:
        #         H, W, D = self.occ_size
        #         pred_c = F.interpolate(pred_c, size=[H, W], mode='trilinear', align_corners=False).contiguous()
        #         pred_c = torch.argmax(pred_c[0], dim=0).cpu().numpy()
        #         self.plot_grid(pred_c, os.path.join("visual", "pred.png"))
        #         self.plot_grid(gt_occ[0][0].unsqueeze(-1), os.path.join("visual", "gt.png"))
        #     else:
        #         H, W, D = self.occ_size
        #         pred_c = F.interpolate(pred_c, size=[H, W], mode='trilinear', align_corners=False).contiguous()
        #         pred_c = torch.argmax(pred_c[0], dim=0).cpu().numpy()
        #         self.plot_grid(pred_c, os.path.join("visual", "pred.png"))

        coarse_occ_mask = output['coarse_occ_mask']
        fine_feature = self.conv1(fine_feature)
        fine_feature = F.interpolate(fine_feature, size=(200, 200), mode='bilinear', align_corners=False)

        # coarse_feature = self.maxpool(output['output_feature']).squeeze(-1)  
        coarse_feature = self.conv2(coarse_feat)
        coarse_feature = F.interpolate(coarse_feature, size=(200, 200), mode='bilinear', align_corners=False)

        fine_feature = torch.cat((coarse_feature, fine_feature), dim=1)
        # if gt_occ is not None:
        #     test_output = {
        #         'pred_c': pred_c,
        #         'pred_f': pred_f,
        #         'coarse_feature': coarse_feat,
        #         'fine_feature': fine_feature,
        #         'coarse_occ_mask': output['coarse_occ_mask'],
        #     }
        # else:
        test_output = {
            'pred_c': pred_c,
            'pred_f': pred_f,
            'coarse_feature': coarse_feat,
            'fine_feature': fine_feature,
            'coarse_occ_mask': output['coarse_occ_mask'],
        }

        return test_output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)

@BACKBONES.register_module()
class CustomResNet3D(BaseModule):
    def __init__(self,
                 depth,
                 block_inplanes=[64, 128, 256, 512],
                 block_strides=[1, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 n_input_channels=3,
                 shortcut_type='B',
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 widen_factor=1.0):
        super().__init__()
        
        layer_metas = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }
        
        if depth in [10, 18, 34]:
            block = BasicBlock
        else:
            assert depth in [50, 101]
            block = Bottleneck
        
        layers = layer_metas[depth]
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.out_indices = out_indices
        
        # replace the first several downsampling layers with the channel-squeeze layers
        self.input_proj = nn.Sequential(
            nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=False),
            build_norm_layer(norm_cfg, self.in_planes)[1],
            nn.ReLU(inplace=True),
        )
        
        self.layers = nn.ModuleList()

        for i in range(len(block_inplanes)):
            self.layers.append(self._make_layer(block, block_inplanes[i], layers[i], 
                                shortcut_type, block_strides[i], norm_cfg=norm_cfg))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_cfg=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  norm_cfg=norm_cfg))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        res = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            
            if index in self.out_indices:
                res.append(x)
            
        return res

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

@NECKS.register_module()
class FPN3D(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """
    def __init__(self,
                 in_channels=[80, 160, 320, 640],
                 out_channels=256,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=dict(type='Conv3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 upsample_cfg=dict(mode='trilinear'),
                 init_cfg=None):
        super(FPN3D, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        
        self.num_out = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.num_out):
            # Expand percetion field
            l_conv = nn.Sequential(
                ConvModule(in_channels[i], out_channels, 
                    kernel_size=1, padding=0,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                    act_cfg=act_cfg, bias=False, 
                    inplace=True),
            )
            
            fpn_conv = nn.Sequential(
                ConvModule(out_channels, out_channels, 
                    kernel_size=3, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                    act_cfg=act_cfg, bias=False, 
                    inplace=True),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if self.with_cp:
                lateral_i = torch.utils.checkpoint.checkpoint(lateral_conv, inputs[i])
            else:
                lateral_i = lateral_conv(inputs[i])
            laterals.append(lateral_i)

        # build down-top path
        for i in range(self.num_out - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], 
                    size=prev_shape, align_corners=False, **self.upsample_cfg)
        
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(self.num_out)
        # ]
        
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            if self.with_cp:
                out_i = torch.utils.checkpoint.checkpoint(fpn_conv, laterals[i])
            else:
                out_i = fpn_conv(laterals[i])
            outs.append(out_i)
        
        return outs

@NECKS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]