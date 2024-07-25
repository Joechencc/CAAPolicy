# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util
import mcubes
import open3d
import pdb
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results
from projects.occ_plugin.core import save_occ
import torch.nn.functional as F

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    
    SC_metric = 0
    SSC_metric = 0
    SSC_metric_fine = 0
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    logger.info(parameter_count_table(model))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            if show:
                save_occ(result['pred_c'], result['pred_f'], data['img_metas'], out_dir, data['visible_mask'], data['gt_occ'])
            
            # only support semantic voxel segmentation
            if 'SC_metric' in result.keys():
                SC_metric += result['SC_metric']
            if 'SSC_metric' in result.keys():
                SSC_metric += result['SSC_metric']
            if 'SSC_metric_fine' in result.keys():
                SSC_metric_fine += result['SSC_metric_fine']
            batch_size = 1

        
        # logging evaluation_semantic
        if 'SC_metric' in result.keys():
            mean_ious = cm_to_ious(SC_metric)
            print(format_SC_results(mean_ious[1:]))
        if 'SSC_metric' in result.keys():
            mean_ious = cm_to_ious(SSC_metric)
            print(format_SSC_results(mean_ious))
        if 'SSC_metric_fine' in result.keys():
            mean_ious = cm_to_ious(SSC_metric_fine)
            print(format_SSC_results(mean_ious))
        

        prog_bar.update()


    res = {
        'SC_metric': SC_metric,
        'SSC_metric': SSC_metric,
        'SSC_metric_fine': SSC_metric_fine,
    }

    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None, baseline_mode=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    
    # init predictions
    SC_metric = []
    SSC_metric = []
    SSC_metric_fine = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    batch_size=1

    for i, data in enumerate(data_loader):
        # if data['img_metas'].data[0][0]['scene_token'] != '2ed0fcbfc214478ca3b3ce013e7723ba' and data['img_metas'].data[0][0]['lidar_token'] != '64d1e962af7b46faae591eb1e3cba6f5':
        #     continue
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            if show:
                save_occ(result['pred_c'], result['pred_f'], data['img_metas'], out_dir, data['visible_mask'], data['gt_occ'])
            
            if 'SC_metric' in result.keys():
                SC_metric.append(result['SC_metric'])
            if 'SSC_metric' in result.keys():
                SSC_metric.append(result['SSC_metric'])
            if 'SSC_metric_fine' in result.keys():
                SSC_metric_fine.append(result['SSC_metric_fine'])
            batch_size = 1
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

        pc_range = data['img_metas'].data[0][0]['pc_range']
        if baseline_mode=='Basic':
            # Interpolation
            _, _, H, W, D = result['pred_f'].shape
            pred = F.interpolate(result['pred_c'], size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            pred = torch.argmax(pred[0], dim=0).cpu().numpy()
            # Get those points index and labels
            nonzero_indices = np.nonzero(pred)
            nonzero_values = pred[nonzero_indices]
            grid_size = (pc_range[3] - pc_range[0]) / H
            x_coords = nonzero_indices[0] * grid_size + pc_range[0] + grid_size / 2
            y_coords = nonzero_indices[1] * grid_size + pc_range[1] + grid_size / 2
            z_coords = nonzero_indices[2] * grid_size + pc_range[2] + grid_size / 2
            points = np.column_stack((x_coords, y_coords, z_coords))

            # np.save("results/points_basic.npy",points)
            # np.save("results/nonzero_values_basic.npy",nonzero_values)

            # pointclouds_mesh(points, nonzero_values, grid_size, bound_min=(pc_range[0], pc_range[1], pc_range[2]), bound_max=(pc_range[3], pc_range[4], pc_range[5]))
        else:
            # Interpolation
            _, _, H, W, D = result['pred_f'].shape
            pred = F.interpolate(result['pred_c'], size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            pred_c = torch.argmax(pred[0], dim=0).cpu().numpy()
            pred_f = torch.argmax(result['pred_f'][0], dim=0).cpu().numpy()
            
            coarse_occ_mask = result['coarse_occ_mask'].unsqueeze(1).float()
            coarse_occ_mask = F.interpolate(coarse_occ_mask, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            coarse_occ_mask = coarse_occ_mask > 0.5
            coarse_occ_mask = coarse_occ_mask.squeeze(1)
            
            pred_c = pred_c * ~coarse_occ_mask.cpu().numpy().squeeze(0)
            pred_f = pred_f * coarse_occ_mask.cpu().numpy().squeeze(0)
            # Get those points index and labels
            nonzero_indices_c = np.nonzero(pred_c)
            nonzero_values_c = pred_c[nonzero_indices_c]
            nonzero_indices_f = np.nonzero(pred_f)
            nonzero_values_f = pred_f[nonzero_indices_f]

            grid_size = (pc_range[3] - pc_range[0]) / H
            x_coords_c = nonzero_indices_c[0] * grid_size + pc_range[0] + grid_size / 2
            y_coords_c = nonzero_indices_c[1] * grid_size + pc_range[1] + grid_size / 2
            z_coords_c = nonzero_indices_c[2] * grid_size + pc_range[2] + grid_size / 2
            points_c = np.column_stack((x_coords_c, y_coords_c, z_coords_c))

            x_coords_f = nonzero_indices_f[0] * grid_size + pc_range[0] + grid_size / 2
            y_coords_f = nonzero_indices_f[1] * grid_size + pc_range[1] + grid_size / 2
            z_coords_f = nonzero_indices_f[2] * grid_size + pc_range[2] + grid_size / 2
            points_f = np.column_stack((x_coords_f, y_coords_f, z_coords_f))

            # np.save("results/points_c_"+baseline_mode+".npy",points_c)
            # np.save("results/nonzero_values_c_"+baseline_mode+".npy",nonzero_values_c)
            # np.save("results/points_f_"+baseline_mode+".npy",points_f)
            # np.save("results/nonzero_values_f_"+baseline_mode+".npy",nonzero_values_f)
            # assert()
            # pointclouds_mesh_nonuniform(points_c, nonzero_values_c, points_f, nonzero_values_f, grid_size, bound_min, bound_max, scales=2)

    # collect lists from multi-GPUs
    res = {}
    if 'SC_metric' in result.keys():
        SC_metric = [sum(SC_metric)]
        SC_metric = collect_results_cpu(SC_metric, len(dataset), tmpdir)
        res['SC_metric'] = SC_metric

    if 'SSC_metric' in result.keys():
        SSC_metric = [sum(SSC_metric)]
        SSC_metric = collect_results_cpu(SSC_metric, len(dataset), tmpdir)
        res['SSC_metric'] = SSC_metric

    if 'SSC_metric_fine' in result.keys():
        SSC_metric_fine = [sum(SSC_metric_fine)]
        SSC_metric_fine = collect_results_cpu(SSC_metric_fine, len(dataset), tmpdir)
        res['SSC_metric_fine'] = SSC_metric_fine

    return res


def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    # collect all parts
    if rank == 0:
    
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        # sort the results
        if type == 'list':
            ordered_results = []
            for res in part_list:  
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
        
        else:
            raise NotImplementedError
        
        # remove tmp dir
        shutil.rmtree(tmpdir)
    
    # 因为我们是分别eval SC和SSC,如果其他rank提前return,开始评测SSC
    # 而rank0的shutil.rmtree可能会删除其他rank正在写入SSC metric的文件
    dist.barrier()

    if rank != 0:
        return None
    
    return ordered_results

def pointclouds_mesh(points, labels, voxel_size, bound_min=(-51.2, -51.2, -5.0), bound_max=(51.2, 51.2, 3.0)):
    lidarseg_idx2name_mapping = {0: 'noise', 1: 'animal', 2: 'human.pedestrian.adult', 3: 'human.pedestrian.child', 4: 'human.pedestrian.construction_worker', 5: 'human.pedestrian.personal_mobility', 6: 'human.pedestrian.police_officer', 7: 'human.pedestrian.stroller', 8: 'human.pedestrian.wheelchair', 9: 'movable_object.barrier', 10: 'movable_object.debris', 11: 'movable_object.pushable_pullable', 12: 'movable_object.trafficcone', 13: 'static_object.bicycle_rack', 14: 'vehicle.bicycle', 15: 'vehicle.bus.bendy', 16: 'vehicle.bus.rigid', 17: 'vehicle.car', 18: 'vehicle.construction', 19: 'vehicle.emergency.ambulance', 20: 'vehicle.emergency.police', 21: 'vehicle.motorcycle', 22: 'vehicle.trailer', 23: 'vehicle.truck', 24: 'flat.driveable_surface', 25: 'flat.other', 26: 'flat.sidewalk', 27: 'flat.terrain', 28: 'static.manmade', 29: 'static.other', 30: 'static.vegetation', 31: 'vehicle.ego'}
    colormap = {'noise': (0, 0, 0), 'animal': (70, 130, 180), 'human.pedestrian.adult': (0, 0, 230), 'human.pedestrian.child': (135, 206, 235), 'human.pedestrian.construction_worker': (100, 149, 237), 'human.pedestrian.personal_mobility': (219, 112, 147), 'human.pedestrian.police_officer': (0, 0, 128), 'human.pedestrian.stroller': (240, 128, 128), 'human.pedestrian.wheelchair': (138, 43, 226), 'movable_object.barrier': (112, 128, 144), 'movable_object.debris': (210, 105, 30), 'movable_object.pushable_pullable': (105, 105, 105), 'movable_object.trafficcone': (47, 79, 79), 'static_object.bicycle_rack': (188, 143, 143), 'vehicle.bicycle': (220, 20, 60), 'vehicle.bus.bendy': (255, 127, 80), 'vehicle.bus.rigid': (255, 69, 0), 'vehicle.car': (255, 158, 0), 'vehicle.construction': (233, 150, 70), 'vehicle.emergency.ambulance': (255, 83, 0), 'vehicle.emergency.police': (255, 215, 0), 'vehicle.motorcycle': (255, 61, 99), 'vehicle.trailer': (255, 140, 0), 'vehicle.truck': (255, 99, 71), 'flat.driveable_surface': (0, 207, 191), 'flat.other': (175, 0, 75), 'flat.sidewalk': (75, 0, 75), 'flat.terrain': (112, 180, 60), 'static.manmade': (222, 184, 135), 'static.other': (255, 228, 196), 'static.vegetation': (0, 175, 0), 'vehicle.ego': (255, 240, 245)}
    color_bank = []
    for i in range(len(lidarseg_idx2name_mapping.keys())):
        color_bank.append(colormap[lidarseg_idx2name_mapping[i]])
    
    color_bank = np.array(color_bank) / 255.0
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd.colors = open3d.Vector3dVector(color_bank[labels])
    rot = rotation_matrix_from_zyx((np.pi / 2, 0, 0))
    points = np.asarray(pcd.points)
    rotated_points = points @ rot.T  # Apply rotation
    pcd.points = open3d.Vector3dVector(rotated_points)
    voxel_grid = open3d.geometry.create_surface_voxel_grid_from_point_cloud(pcd, voxel_size)     
    import pdb; pdb.set_trace()
    open3d.write_voxel_grid("results/voxel_grid.ply", voxel_grid)
    return None
    # # cross_positions = [
    # #     [5.5, 5.0, 0.0],
    # #     [5.5, 10.4, 0.0],
    # #     [7.4, 5.0, 0.0],
    # #     [7.4, 10.4, 0.0]
    # # ]
    # # import pdb; pdb.set_trace()
    # # cross_markers = [create_cross_marker(np.array(pos)) for pos in cross_positions]
    # vis = open3d.visualization.Visualizer()
    # # import pdb; pdb.set_trace()
    # # vis.create_window(width=1360, height=1080)
    # vis.add_geometry(voxel_grid)
    
    # renderoption = vis.get_render_option()
    # # renderoption.mesh_show_wireframe = True
    # # renderoption.point_size = 5
    # ctr = vis.get_view_control()
    # # ctr.rotate(0.0, view_point)
    # # ctr.set_zoom(zoom)
    # vis.poll_events()
    # vis.update_renderer()
    # img = np.asarray(vis.capture_screen_float_buffer())
    # fname = "/scratch/cc7287/PlanWithUncertainty/img.png"
    # if fname:
    #     # import pdb; pdb.set_trace()
    #     cv2.imwrite(fname, (img[:, :, ::-1] * 255).astype(np.uint8))
    # vis.destroy_window()
    # import pdb; pdb.set_trace()
    # open3d.visualization.draw_geometries([voxel_grid])
    # voxels = voxel_grid.get_voxels()
    # indices = np.array([v.grid_index for v in voxels], dtype=np.int32)
    # xyz = (indices * voxel_size + bound_min + voxel_size/2)
    # tree = KDTree(pcd.points)
    # dd, ind = tree.query(xyz, k=1)
    # indices_labeled = np.concatenate((indices[:, :], labels[ind, None]), axis=1)
    # np.save("87/uniform.npy",indices_labeled)
    # import pdb; pdb.set_trace()
    # return indices_labeled

def pointclouds_mesh_nonuniform(pcd, voxel_size, bound_min, bound_max, scales=2):
    lidarseg_idx2name_mapping = {0: 'noise', 1: 'animal', 2: 'human.pedestrian.adult', 3: 'human.pedestrian.child', 4: 'human.pedestrian.construction_worker', 5: 'human.pedestrian.personal_mobility', 6: 'human.pedestrian.police_officer', 7: 'human.pedestrian.stroller', 8: 'human.pedestrian.wheelchair', 9: 'movable_object.barrier', 10: 'movable_object.debris', 11: 'movable_object.pushable_pullable', 12: 'movable_object.trafficcone', 13: 'static_object.bicycle_rack', 14: 'vehicle.bicycle', 15: 'vehicle.bus.bendy', 16: 'vehicle.bus.rigid', 17: 'vehicle.car', 18: 'vehicle.construction', 19: 'vehicle.emergency.ambulance', 20: 'vehicle.emergency.police', 21: 'vehicle.motorcycle', 22: 'vehicle.trailer', 23: 'vehicle.truck', 24: 'flat.driveable_surface', 25: 'flat.other', 26: 'flat.sidewalk', 27: 'flat.terrain', 28: 'static.manmade', 29: 'static.other', 30: 'static.vegetation', 31: 'vehicle.ego'}
    colormap = {'noise': (0, 0, 0), 'animal': (70, 130, 180), 'human.pedestrian.adult': (0, 0, 230), 'human.pedestrian.child': (135, 206, 235), 'human.pedestrian.construction_worker': (100, 149, 237), 'human.pedestrian.personal_mobility': (219, 112, 147), 'human.pedestrian.police_officer': (0, 0, 128), 'human.pedestrian.stroller': (240, 128, 128), 'human.pedestrian.wheelchair': (138, 43, 226), 'movable_object.barrier': (112, 128, 144), 'movable_object.debris': (210, 105, 30), 'movable_object.pushable_pullable': (105, 105, 105), 'movable_object.trafficcone': (47, 79, 79), 'static_object.bicycle_rack': (188, 143, 143), 'vehicle.bicycle': (220, 20, 60), 'vehicle.bus.bendy': (255, 127, 80), 'vehicle.bus.rigid': (255, 69, 0), 'vehicle.car': (255, 158, 0), 'vehicle.construction': (233, 150, 70), 'vehicle.emergency.ambulance': (255, 83, 0), 'vehicle.emergency.police': (255, 215, 0), 'vehicle.motorcycle': (255, 61, 99), 'vehicle.trailer': (255, 140, 0), 'vehicle.truck': (255, 99, 71), 'flat.driveable_surface': (0, 207, 191), 'flat.other': (175, 0, 75), 'flat.sidewalk': (75, 0, 75), 'flat.terrain': (112, 180, 60), 'static.manmade': (222, 184, 135), 'static.other': (255, 228, 196), 'static.vegetation': (0, 175, 0), 'vehicle.ego': (255, 240, 245)}
    for i in range(len(lidarseg_idx2name_mapping.keys())):
        color_bank.append(colormap[lidarseg_idx2name_mapping[i]])
    color_bank = np.array(color_bank) / 255.0
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(color_bank[labels])
    rot = rotation_matrix_from_zyx((np.pi / 2, 0, 0))
    points = np.asarray(pcd.points)
    rotated_points = points @ rot.T  # Apply rotation
    pcd.points = open3d.utility.Vector3dVector(rotated_points)

    pcd_near = deepcopy(pcd)
    pcd_far = deepcopy(pcd)
    pcd_all_points = np.array(pcd_near.points)
    pcd_all_colors = np.array(pcd_near.colors)
    near_min, near_max = (-12.8, -12.8, -5.0), (12.8, 12.8, 3.0)
    mask = np.all((pcd_all_points >= near_min) & (pcd_all_points <= near_max), axis=1)
    pcd_far_points = pcd_all_points[~mask]
    pcd_near_points = pcd_all_points[mask]
    pcd_far_colors = pcd_all_colors[~mask]
    pcd_near_colors = pcd_all_colors[mask]
    pcd_near.points = open3d.utility.Vector3dVector(pcd_near_points)
    pcd_far.points = open3d.utility.Vector3dVector(pcd_far_points)
    pcd_near.colors = open3d.utility.Vector3dVector(pcd_near_colors)
    pcd_far.colors = open3d.utility.Vector3dVector(pcd_far_colors)
    voxel_grid_near = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_near, voxel_size, bound_min, bound_max)
    voxel_grid_far = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_far, scales*voxel_size, bound_min, bound_max)
    
    cross_positions = [
        [5.5, 5.0, 0.0],
        [5.5, 10.4, 0.0],
        [7.4, 5.0, 0.0],
        [7.4, 10.4, 0.0]
    ]
    # np.save("87/roi.npy",np.array(cross_positions))
    cross_markers = [create_cross_marker(np.array(pos)) for pos in cross_positions]

    open3d.visualization.draw_geometries([voxel_grid_near]+[voxel_grid_far]+cross_markers)
    voxels_near = voxel_grid_near.get_voxels()
    voxels_far = voxel_grid_far.get_voxels()
    indices_near = np.array([v.grid_index for v in voxels_near], dtype=np.int32)
    indices_far = np.array([v.grid_index for v in voxels_far], dtype=np.int32)
    xyz_near = (indices_near * voxel_size + bound_min + voxel_size/2)
    xyz_far = (indices_far * voxel_size*scales + bound_min + (voxel_size/2)*scales)
    tree_near = KDTree(pcd_near.points)
    dd_near, ind_near = tree_near.query(xyz_near, k=1)
    tree_far = KDTree(pcd_far.points)
    dd_far, ind_far = tree_far.query(xyz_far, k=1)
    indices_labeled_near = np.concatenate((indices_near[:, :], voxel_size*100 * np.ones_like(labels[ind_near, None]), labels[ind_near, None]), axis=1)
    indices_labeled_far = np.concatenate((indices_far[:, :], voxel_size*scales*100 * np.ones_like(labels[ind_far, None]), labels[ind_far, None]), axis=1)
    indices_labeled = np.concatenate((indices_labeled_near, indices_labeled_far), axis=0)

    np.save("87/nonuniform.npy",indices_labeled)
    return indices_labeled

def rotation_matrix_from_zyx(angles):
    z, y, x = angles
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    return Rz @ Ry @ Rx  # Matrix multiplication

def create_cross_marker(center, size=0.2, width=0.2):
    lines = []
    points = []
    # Define the lines of the cross marker
    directions = [
        ([size, 0, 0], [-size, 0, 0]),
        ([0, size, 0], [0, -size, 0]),
        ([0, 0, size], [0, 0, -size])
    ]
    
    offsets = np.linspace(-width / 2, width / 2, num=5)
    for start, end in directions:
        for offset in offsets:
            start_point = center + start + np.array([offset, 0, 0])
            end_point = center + end + np.array([offset, 0, 0])
            points.append(start_point)
            points.append(end_point)
            lines.append([len(points) - 2, len(points) - 1])
            
            start_point = center + start + np.array([0, offset, 0])
            end_point = center + end + np.array([0, offset, 0])
            points.append(start_point)
            points.append(end_point)
            lines.append([len(points) - 2, len(points) - 1])
            
            start_point = center + start + np.array([0, 0, offset])
            end_point = center + end + np.array([0, 0, offset])
            points.append(start_point)
            points.append(end_point)
            lines.append([len(points) - 2, len(points) - 1])
    import pdb; pdb.set_trace()
    line_set = open3d.geometry.LineSet(
        # points=open3d.Vector3dVector(points),
        lines=open3d.Vector2iVector(lines)
    )
    
    return line_set