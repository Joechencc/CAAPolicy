import os
import torch

from datetime import datetime


class Configuration:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = None
    log_dir = None
    checkpoint_dir = None
    log_every_n_steps = None
    check_val_every_n_epoch = None

    epochs = None
    learning_rate = None
    weight_decay = None
    batch_size = None

    training_map = None
    validation_map = None
    future_frame_nums = None
    hist_frame_nums = None
    token_nums = None
    image_crop = None

    feature_encoder = None
    voxel_out_indices = None
    point_cloud_range = None
    occ_size = None

    bev_encoder_in_channel = None
    bev_encoder_out_channel = None

    conet_encoder_in_channel = None
    conet_encoder_out_channel = None

    bev_x_bound = None
    bev_y_bound = None
    bev_z_bound = None
    d_bound = None
    final_dim = None
    bev_down_sample = None
    use_depth_distribution = None
    backbone = None

    seg_classes = None
    seg_classes_conet = None
    seg_vehicle_weights = None
    seg_dim = None

    tf_en_dim = None
    tf_en_conet_dim = None
    tf_en_heads = None
    tf_en_layers = None
    tf_en_dropout = None
    tf_en_bev_length = None
    tf_en_conet_length = None
    tf_en_motion_length = None

    tf_de_dim = None
    tf_de_conet_dim = None
    tf_de_heads = None
    tf_de_layers = None
    tf_de_dropout = None
    tf_de_tgt_dim = None


def get_cfg(cfg_yaml: dict):
    today = datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)

    config = cfg_yaml['parking_model']
    cfg = Configuration()

    cfg.data_dir = config['data_dir']
    cfg.log_dir = os.path.join(config['log_dir'], exp_name)
    cfg.checkpoint_dir = os.path.join(config['checkpoint_dir'], exp_name)
    cfg.log_every_n_steps = config['log_every_n_steps']
    cfg.check_val_every_n_epoch = config['check_val_every_n_epoch']

    cfg.epochs = config['epochs']
    cfg.learning_rate = config['learning_rate']
    cfg.weight_decay = config['weight_decay']
    cfg.batch_size = config['batch_size']

    cfg.training_map = config['training_map']
    cfg.validation_map = config['validation_map']
    cfg.future_frame_nums = config['future_frame_nums']
    cfg.hist_frame_nums = config['hist_frame_nums']
    cfg.token_nums = config['token_nums']
    cfg.image_crop = config['image_crop']
    cfg.feature_encoder = config['feature_encoder']
    cfg.voxel_out_indices = config['voxel_out_indices']
    cfg.point_cloud_range = config['point_cloud_range']
    cfg.occ_size = config['occ_size']

    ####### CONET Config ##########
    cfg.OccNet_cfg = config['OccNet_cfg']
    occ_size, point_cloud_range, lss_downsample = config['occ_size'], config['point_cloud_range'], config['lss_downsample']
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
    }
    cfg.OccNet_cfg['img_backbone']['out_indices'] = eval(cfg.OccNet_cfg['img_backbone']['out_indices'])
    cfg.OccNet_cfg['img_view_transformer']['data_config']['input_size'] = eval(cfg.OccNet_cfg['img_view_transformer']['data_config']['input_size'])
    cfg.OccNet_cfg['img_view_transformer']['data_config']['src_size'] = eval(cfg.OccNet_cfg['img_view_transformer']['data_config']['src_size'])
    cfg.OccNet_cfg['img_view_transformer']['data_config']['resize'] = eval(cfg.OccNet_cfg['img_view_transformer']['data_config']['resize'])
    cfg.OccNet_cfg['img_view_transformer']['data_config']['rot'] = eval(cfg.OccNet_cfg['img_view_transformer']['data_config']['rot'])
    cfg.OccNet_cfg['img_view_transformer']['data_config']['crop_h'] = eval(cfg.OccNet_cfg['img_view_transformer']['data_config']['crop_h'])
    
    cfg.OccNet_cfg['img_view_transformer']['grid_config'] = grid_config
    voxel_out_indices = eval(cfg.voxel_out_indices)
    voxel_out_channel = 256
    cfg.OccNet_cfg['occ_encoder_backbone_cfg']['out_indices'] = eval(cfg.OccNet_cfg['occ_encoder_backbone_cfg']['out_indices'])
    cfg.OccNet_cfg['occ_encoder_backbone_cfg']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
    cfg.OccNet_cfg['occ_encoder_neck_cfg']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
    cfg.OccNet_cfg['pts_bbox_head']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
    cfg.OccNet_cfg['pts_bbox_head']['num_level'] = len(voxel_out_indices)
    cfg.OccNet_cfg['pts_bbox_head']['in_channels'] = [voxel_out_channel] * len(voxel_out_indices)
    cfg.OccNet_cfg['pts_bbox_head']['loss_weight_cfg'] = dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        )

    cfg.conet_encoder_in_channel = config['conet_encoder_in_channel']
    cfg.conet_encoder_out_channel = config['conet_encoder_out_channel']
    ###############################

    cfg.bev_encoder_in_channel = config['bev_encoder_in_channel']
    cfg.bev_encoder_out_channel = config['bev_encoder_out_channel']

    cfg.bev_x_bound = config['bev_x_bound']
    cfg.bev_y_bound = config['bev_y_bound']
    cfg.bev_z_bound = config['bev_z_bound']
    cfg.d_bound = config['d_bound']
    cfg.final_dim = config['final_dim']
    cfg.bev_down_sample = config['bev_down_sample']
    cfg.use_depth_distribution = config['use_depth_distribution']
    cfg.backbone = config["backbone"]

    cfg.seg_classes = config['seg_classes']
    cfg.seg_classes_conet = config['seg_classes_conet']
    cfg.seg_dim = config['Segdim']
    cfg.seg_vehicle_weights = config['seg_vehicle_weights']

    cfg.tf_en_dim = config['tf_en_dim']
    cfg.tf_en_conet_dim = config['tf_en_conet_dim']
    cfg.tf_en_heads = config['tf_en_heads']
    cfg.tf_en_layers = config['tf_en_layers']
    cfg.tf_en_dropout = config['tf_en_dropout']
    cfg.tf_en_bev_length = config['tf_en_bev_length']
    cfg.tf_en_conet_length = config['tf_en_conet_length']
    cfg.tf_en_motion_length = config['tf_en_motion_length']

    cfg.tf_de_dim = config['tf_de_dim']
    cfg.tf_de_conet_dim = config['tf_de_conet_dim']
    cfg.tf_de_heads = config['tf_de_heads']
    cfg.tf_de_layers = config['tf_de_layers']
    cfg.tf_de_dropout = config['tf_de_dropout']
    cfg.tf_de_tgt_dim = config['tf_de_tgt_dim']

    return cfg
