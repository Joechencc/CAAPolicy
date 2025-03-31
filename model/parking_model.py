import torch
from torch import nn

from tool.config import Configuration
from model.bev_model import BevModel
from model.bev_encoder import BevEncoder
from model.feature_fusion import FeatureFusion
from model.control_predict import ControlPredict
from model.segmentation_head import SegmentationHead
from model.DINOv2_extractor import FeatureExtractor


class ParkingModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        self.bev_model = BevModel(self.cfg)

        self.BOS_token = torch.tensor([self.cfg.BOS_token], dtype=torch.int64).unsqueeze(0)
        
        # DINOv2 for image tokenization
        self.dinov2 = FeatureExtractor(
            model_name="resnet18",
            output_dim=512,
            image_size=256  # Must match actual input size
        )

        self.bev_encoder = BevEncoder(self.cfg.bev_encoder_in_channel)

        self.feature_fusion = FeatureFusion(self.cfg)

        self.control_predict = ControlPredict(self.cfg)

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
        # image & extrinsics & intrinsics
        # images = [self.image_process(self.front[index])[0], self.image_process(self.front_left[index])[0],self.image_process(self.front_right[index])[0],
        #           self.image_process(self.back[index])[0], self.image_process(self.back_left[index])[0],self.image_process(self.back_right[index])[0],]
        # images = torch.cat(images, dim=0)
        # data['image'] = images
        images = data['image'].to(self.cfg.device, non_blocking=True)
        images_future = data['future_images'].to(self.cfg.device, non_blocking=True) #use image for debugigng purpose
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        target_point = data['target_point'].to(self.cfg.device, non_blocking=True)#已经是相对车的位置了
        ego_motion = data['ego_motion'].to(self.cfg.device, non_blocking=True)
        bev_feature, pred_depth = self.bev_model(images, intrinsics, extrinsics)
        # TODO: Change 6 images to 3 images per frame
        img_feature = self.dinov2(images[:,3:,:,:,:])
        # TODO: Change 6 images to 3 images per frame
        img_feature_future = self.dinov2(images_future[:,3:,:,:,:])
        # bev_feature, bev_target = self.add_target_bev(bev_feature, target_point)

        bev_target = self.adjust_target_bev(bev_feature, target_point)

        bev_down_sample = self.bev_encoder(bev_feature)

        target_point = target_point.unsqueeze(1)
        fuse_feature = self.feature_fusion(bev_down_sample, img_feature, ego_motion, target_point)

        pred_segmentation = self.segmentation_head(fuse_feature[:,:self.cfg.tf_en_bev_length,:])

        return fuse_feature, img_feature_future, pred_segmentation, pred_depth, bev_target

    def forward(self, data):
        fuse_feature, img_feature_future, pred_segmentation, pred_depth, _ = self.encoder(data)
        pred_image_feature, pred_control = self.control_predict(fuse_feature, data['gt_control'].cuda(),img_feature_future.cuda())
        return pred_image_feature, pred_control, pred_segmentation, pred_depth

    def predict(self, data):
        # how should I initialize the pred_image_feature for the first inference?
        fuse_feature, _, pred_segmentation, pred_depth, bev_target = self.encoder(data) # ignore the img_feature_future as it is unavailale for inference
        BOS_token = self.BOS_token.cuda()
        _, pred_control = self.control_predict.predict(fuse_feature, BOS_token) #autoregressive token prediction inside control_predict

        return pred_control, pred_segmentation, pred_depth, bev_target
