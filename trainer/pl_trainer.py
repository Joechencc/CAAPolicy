import torch
import pytorch_lightning as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, ModelSummary
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.waypoint_loss import WaypointLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from model.parking_model import ParkingModel
import torch.nn.functional as F


def setup_callbacks(cfg):
    callbacks = []

    ckpt_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=40,
                                    mode='min',
                                    filename='E2EParking-{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    callbacks.append(ckpt_callback)

    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)

    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    return callbacks

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration, model_path=None):
        super(ParkingTrainingModule, self).__init__()
        self.save_hyperparameters(ignore=['model_path'])

        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.waypoint_loss_func = WaypointLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        self.segmentation_loss_func = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.seg_vehicle_weights)
        )

        self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = ParkingModel(self.cfg)

        self.perception_training_steps = 15


    def on_train_epoch_start(self):

        dataloader = self.trainer.datamodule.train_dataloader()
        dataset = dataloader.dataset

        if self.current_epoch < self.perception_training_steps:
            # Freeze backbone
            # for p in self.backbone.parameters():
            #     p.requires_grad = False

            if self.current_epoch % 3 != 0:
                if hasattr(dataset, 'relabel_goals'):
                    dataset.relabel_goals(self.current_epoch)
                    print("Training with relabeled target.")
            else:
                dataset.keep_goals(self.current_epoch)
                print("Training with original target.")

        else:
            dataset.keep_goals(self.current_epoch)
            print("Train perception with low lr and motion with normal lr.")
            # freeze_module(self.parking_model.bev_model)
            # freeze_module(self.parking_model.bev_encoder)
            # freeze_module(self.parking_model.feature_fusion)
            # freeze_module(self.parking_model.film_modulate)
            # freeze_module(self.parking_model.segmentation_head)


    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_control, pred_waypoint, pred_segmentation, pred_depth, fuse_feature, approx_grad = self.parking_model(batch)

        control_loss = self.control_loss_func(pred_control, batch)
        loss_dict.update({
            "control_loss": control_loss*0.0 if self.current_epoch < self.perception_training_steps else control_loss
        })

        waypoint_loss = self.waypoint_loss_func(pred_waypoint, batch)
        loss_dict.update({
            "waypoint_loss": waypoint_loss*0.0 if self.current_epoch < self.perception_training_steps else waypoint_loss
        })


        # grads = []
        # for b in range(pred_control.shape[0]): # B,12,256
        #     grad = torch.autograd.grad(pred_control[b][1:13,:].mean(), fuse_feature, create_graph=True)[0]
        #     grads.append(grad[b])
        # grad_gt = torch.stack(grads, dim=0)
        # refined_feature = approx_grad*fuse_feature
        # pred_control_2, pred_waypoint_2 = self.parking_model.forward_twice(refined_feature, batch)
        # control_loss_2 = self.control_loss_func(pred_control_2, batch)

        # loss_dict.update({
        #     "control_loss": control_loss_2*0.0 if self.current_epoch < self.perception_training_steps else control_loss_2
        # })

        # grad_loss = F.mse_loss(approx_grad, grad_gt.detach())
        # loss_dict.update({
        #     "grad_loss": grad_loss*0.0 if self.current_epoch < self.perception_training_steps else grad_loss
        # })

        # waypoint_loss_2 = self.waypoint_loss_func(pred_waypoint_2, batch)
        # loss_dict.update({
        #     "waypoint_loss": waypoint_loss_2*0.0 if self.current_epoch < self.perception_training_steps else waypoint_loss_2
        # })

        segmentation_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        loss_dict.update({
            "segmentation_loss": segmentation_loss
        })

        depth_loss = self.depth_loss_func(pred_depth, batch['depth'])
        loss_dict.update({
            "depth_loss": depth_loss
        })
        train_loss = sum(loss_dict.values())
        loss_dict.update({
            "train_loss": train_loss
        })
        self.log_dict(loss_dict)
        # self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation')
        # self.log_depth(pred_depth, batch['depth'], 'depth')

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        with torch.enable_grad():
            pred_control, pred_waypoint, pred_segmentation, pred_depth, fuse_feature, approx_grad = self.parking_model(batch)

            control_loss = self.control_loss_func(pred_control, batch)
            waypoint_loss = self.waypoint_loss_func(pred_waypoint, batch)
        #     grads = []
        #     for b in range(pred_control.shape[0]):
        #         grad = torch.autograd.grad(pred_control[b][1:13,:].mean(), fuse_feature, create_graph=True)[0]
        #         grads.append(grad[b])
        #     grad_gt = torch.stack(grads, dim=0)
        #     refined_feature = approx_grad*fuse_feature
        # pred_control_2, pred_waypoint_2 = self.parking_model.forward_twice(refined_feature, batch)

        acc_steer_val_loss, reverse_val_loss = self.control_val_loss_func(pred_control, batch)
        val_loss_dict.update({
            "acc_steer_val_loss": acc_steer_val_loss*0.0 if self.current_epoch < self.perception_training_steps else acc_steer_val_loss,
            "reverse_val_loss": reverse_val_loss*0.0 if self.current_epoch < self.perception_training_steps else reverse_val_loss
        })
        # grad_loss = F.mse_loss(approx_grad, grad_gt.detach())
        # val_loss_dict.update({
        #     "grad_loss": grad_loss*0.0 if self.current_epoch < self.perception_training_steps else grad_loss
        # })
        waypoint_loss = self.waypoint_loss_func(pred_waypoint, batch)
        val_loss_dict.update({
            "waypoint_val_loss": waypoint_loss*0.0 if self.current_epoch < self.perception_training_steps else waypoint_loss,
        })

        segmentation_val_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        val_loss_dict.update({
            "segmentation_val_loss": segmentation_val_loss
        })

        depth_val_loss = self.depth_loss_func(pred_depth, batch['depth'])
        val_loss_dict.update({
            "depth_val_loss": depth_val_loss
        })

        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })
        self.log_dict(val_loss_dict)
        # self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation_val')
        # self.log_depth(pred_depth, batch['depth'], 'depth_val')

        return val_loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(),
    #                                  lr=self.cfg.learning_rate,
    #                                  weight_decay=self.cfg.weight_decay)
    #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
    #     return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_optimizers(self):
    
        base_lr = self.cfg.learning_rate
        dino_lr = 0.1 * self.cfg.learning_rate  # ← add this to your config
        perception_lr = self.cfg.learning_rate if self.current_epoch < self.perception_training_steps else 0.1 * self.cfg.learning_rate 
        weight_decay = self.cfg.weight_decay

        # Separate DINOv2 parameters (in DinoCamEncoder) and the rest
        dino_params = self.parking_model.bev_model.cam_encoder.parameters()

        named_params = list(self.named_parameters())

        perception_params = (p for n, p in named_params 
        if n.startswith("parking_model.bev_model") or n.startswith("parking_model.bev_encoder") 
        or n.startswith("parking_model.feature_fusion") or n.startswith("parking_model.film_modulate") 
        or n.startswith("parking_model.segmentation_head"))

        other_params = (p for n, p in named_params
            if not (n.startswith("parking_model.bev_model") or n.startswith("parking_model.bev_encoder") 
            or n.startswith("parking_model.feature_fusion") or n.startswith("parking_model.film_modulate") 
            or n.startswith("parking_model.segmentation_head"))
        )

        # Parameter groups
        param_groups = [
            # {"params": dino_params, "lr": dino_lr, "weight_decay": weight_decay},
            {"params": perception_params, "lr": perception_lr, "weight_decay": weight_decay},
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        ]

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        # Scheduler (applies to both groups uniformly)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cfg.epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


    def log_segmentation(self, pred_segmentation, gt_segmentation, name):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("GT Seg")
        ax[1].set_title("Pred Seg")

        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]

        gt_segmentation = gt_segmentation[0]
        gt_segmentation = gt_segmentation.detach().cpu().numpy()
        gt_segmentation[gt_segmentation == 1] = 128
        gt_segmentation[gt_segmentation == 2] = 255
        gt_seg_img = gt_segmentation[0, :, :][::-1]

        norm = mpl.colors.Normalize(vmin=0, vmax=255)
        ax[0].imshow(gt_seg_img, norm=norm)
        ax[1].imshow(pred_seg_img, norm=norm)

        tensorboard = self.logger.experiment
        tensorboard.add_figure(figure=fig, tag=name)
        plt.close(fig)

    def log_depth(self, pred_depth, gt_depth, name):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("GT Depth")
        ax[1].set_title("Pred Depth")

        pred_depth = pred_depth[1]
        pred_depth = torch.argmax(pred_depth, dim=0)
        pred_depth = pred_depth.detach().cpu().numpy()
        pred_depth = pred_depth * self.cfg.d_bound[2] + self.cfg.d_bound[0]

        gt_depth = gt_depth[0][1]
        gt_depth = gt_depth.detach().cpu().numpy()

        norm = mpl.colors.Normalize()
        ax[0].imshow(gt_depth)
        ax[1].imshow(pred_depth, norm=norm)

        tensorboard = self.logger.experiment
        tensorboard.add_figure(figure=fig, tag=name)
        plt.close(fig)

