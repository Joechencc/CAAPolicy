import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BEVUpsample(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super(BEVUpsample, self).__init__()
        # 主分支
        self.main_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            bias=False
        )
        # 用来保证跳跃分支和输出形状一致（维度匹配）
        self.shortcut = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            bias=False
        )
        # 可自行选择合适的激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 主分支
        out = self.main_conv(x)
        # 跳跃分支
        identity = self.shortcut(x)
        # 残差连接
        out += identity
        # 激活
        out = self.act(out)
        return out

    
class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

        self.segmentation_head = SegmentationHead(self.cfg)

        self.bev_upsample = BEVUpsample()

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
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)


        # image_seg = data["segmentation"].squeeze(0).squeeze(0).cpu()
        
        # image_occ_w_target = data["gt_occ_w_target"].squeeze(0).cpu().numpy()
        # image_occ = data["gt_occ"].squeeze(0)[1].cpu().numpy()
        
        # topdown = data["topdown"][0].cpu().numpy()

        # plt.imshow(topdown, cmap='gray')  # 使用灰度图
        # plt.colorbar()  # 显示颜色条
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('output_image_topdown.png', bbox_inches='tight', pad_inches=0)  # 保存图像
        # plt.close()  # 关闭图像窗口

        # plt.imshow(image_seg, cmap='gray')  # 使用灰度图
        # plt.colorbar()  # 显示颜色条
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('output_image_seg.png', bbox_inches='tight', pad_inches=0)  # 保存图像
        # plt.close()  # 关闭图像窗口

        # plt.imshow(image_occ, cmap='gray')  # 使用灰度图
        # plt.colorbar()  # 显示颜色条
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('output_image_occ.png', bbox_inches='tight', pad_inches=0)  # 保存图像
        # plt.close()  # 关闭图像窗口

        # plt.imshow(image_occ_w_target, cmap='gray')  # 使用灰度图
        # plt.colorbar()  # 显示颜色条
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('output_image_occ_w_target.png', bbox_inches='tight', pad_inches=0)  # 保存图像
        # plt.close()  # 关闭图像窗口
        #bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)

        bev_feature = self.bev_upsample(data["gt_occ"].float())

        bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        fuse_feature = self.feature_fusion(bev_down_sample, ego_motion)

        pred_segmentation = self.segmentation_head(fuse_feature)

        #return fuse_feature, pred_segmentation, pred_depth, bev_target
        return fuse_feature, pred_segmentation, None, bev_target

    def forward(self, data):
        fuse_feature, pred_segmentation, pred_depth, _ = self.encoder(data)
        pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda())
        return pred_control, pred_segmentation, pred_depth

    def predict(self, data):
        fuse_feature, pred_segmentation, pred_depth, bev_target = self.encoder(data)
        pred_multi_controls = data['gt_control'].cuda()
        for i in range(3):
            pred_control = self.control_predict.predict(fuse_feature, pred_multi_controls)
            pred_multi_controls = torch.cat([pred_multi_controls, pred_control], dim=1)
        return pred_multi_controls, pred_segmentation, pred_depth, bev_target
