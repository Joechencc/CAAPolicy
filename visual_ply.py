import open3d
import os, json
import numpy as np
from scipy.spatial import KDTree
from copy import deepcopy

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

    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines)
    )
    
    return line_set

def pointclouds_mesh(points, labels, voxel_size, bound_min, bound_max):
    lidarseg_idx2name_mapping = {0: 'noise', 1: 'animal', 2: 'human.pedestrian.adult', 3: 'human.pedestrian.child', 4: 'human.pedestrian.construction_worker', 5: 'human.pedestrian.personal_mobility', 6: 'human.pedestrian.police_officer', 7: 'human.pedestrian.stroller', 8: 'human.pedestrian.wheelchair', 9: 'movable_object.barrier', 10: 'movable_object.debris', 11: 'movable_object.pushable_pullable', 12: 'movable_object.trafficcone', 13: 'static_object.bicycle_rack', 14: 'vehicle.bicycle', 15: 'vehicle.bus.bendy', 16: 'vehicle.bus.rigid', 17: 'vehicle.car', 18: 'vehicle.construction', 19: 'vehicle.emergency.ambulance', 20: 'vehicle.emergency.police', 21: 'vehicle.motorcycle', 22: 'vehicle.trailer', 23: 'vehicle.truck', 24: 'flat.driveable_surface', 25: 'flat.other', 26: 'flat.sidewalk', 27: 'flat.terrain', 28: 'static.manmade', 29: 'static.other', 30: 'static.vegetation', 31: 'vehicle.ego'}
    colormap = {'noise': (0, 0, 0), 'animal': (70, 130, 180), 'human.pedestrian.adult': (0, 0, 230), 'human.pedestrian.child': (135, 206, 235), 'human.pedestrian.construction_worker': (100, 149, 237), 'human.pedestrian.personal_mobility': (219, 112, 147), 'human.pedestrian.police_officer': (0, 0, 128), 'human.pedestrian.stroller': (240, 128, 128), 'human.pedestrian.wheelchair': (138, 43, 226), 'movable_object.barrier': (112, 128, 144), 'movable_object.debris': (210, 105, 30), 'movable_object.pushable_pullable': (105, 105, 105), 'movable_object.trafficcone': (47, 79, 79), 'static_object.bicycle_rack': (188, 143, 143), 'vehicle.bicycle': (220, 20, 60), 'vehicle.bus.bendy': (255, 127, 80), 'vehicle.bus.rigid': (255, 69, 0), 'vehicle.car': (255, 158, 0), 'vehicle.construction': (233, 150, 70), 'vehicle.emergency.ambulance': (255, 83, 0), 'vehicle.emergency.police': (255, 215, 0), 'vehicle.motorcycle': (255, 61, 99), 'vehicle.trailer': (255, 140, 0), 'vehicle.truck': (255, 99, 71), 'flat.driveable_surface': (0, 207, 191), 'flat.other': (175, 0, 75), 'flat.sidewalk': (75, 0, 75), 'flat.terrain': (112, 180, 60), 'static.manmade': (222, 184, 135), 'static.other': (255, 228, 196), 'static.vegetation': (0, 175, 0), 'vehicle.ego': (255, 240, 245)}
    color_bank = []
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
    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, bound_min, bound_max)
    
    open3d.visualization.draw_geometries([voxel_grid])
    return None

def pointclouds_mesh_nonuniform(points, labels, points_f, labels_f, voxel_size, bound_min, bound_max, scales=4):
    lidarseg_idx2name_mapping = {0: 'noise', 1: 'animal', 2: 'human.pedestrian.adult', 3: 'human.pedestrian.child', 4: 'human.pedestrian.construction_worker', 5: 'human.pedestrian.personal_mobility', 6: 'human.pedestrian.police_officer', 7: 'human.pedestrian.stroller', 8: 'human.pedestrian.wheelchair', 9: 'movable_object.barrier', 10: 'movable_object.debris', 11: 'movable_object.pushable_pullable', 12: 'movable_object.trafficcone', 13: 'static_object.bicycle_rack', 14: 'vehicle.bicycle', 15: 'vehicle.bus.bendy', 16: 'vehicle.bus.rigid', 17: 'vehicle.car', 18: 'vehicle.construction', 19: 'vehicle.emergency.ambulance', 20: 'vehicle.emergency.police', 21: 'vehicle.motorcycle', 22: 'vehicle.trailer', 23: 'vehicle.truck', 24: 'flat.driveable_surface', 25: 'flat.other', 26: 'flat.sidewalk', 27: 'flat.terrain', 28: 'static.manmade', 29: 'static.other', 30: 'static.vegetation', 31: 'vehicle.ego'}
    colormap = {'noise': (0, 0, 0), 'animal': (70, 130, 180), 'human.pedestrian.adult': (0, 0, 230), 'human.pedestrian.child': (135, 206, 235), 'human.pedestrian.construction_worker': (100, 149, 237), 'human.pedestrian.personal_mobility': (219, 112, 147), 'human.pedestrian.police_officer': (0, 0, 128), 'human.pedestrian.stroller': (240, 128, 128), 'human.pedestrian.wheelchair': (138, 43, 226), 'movable_object.barrier': (112, 128, 144), 'movable_object.debris': (210, 105, 30), 'movable_object.pushable_pullable': (105, 105, 105), 'movable_object.trafficcone': (47, 79, 79), 'static_object.bicycle_rack': (188, 143, 143), 'vehicle.bicycle': (220, 20, 60), 'vehicle.bus.bendy': (255, 127, 80), 'vehicle.bus.rigid': (255, 69, 0), 'vehicle.car': (255, 158, 0), 'vehicle.construction': (233, 150, 70), 'vehicle.emergency.ambulance': (255, 83, 0), 'vehicle.emergency.police': (255, 215, 0), 'vehicle.motorcycle': (255, 61, 99), 'vehicle.trailer': (255, 140, 0), 'vehicle.truck': (255, 99, 71), 'flat.driveable_surface': (0, 207, 191), 'flat.other': (175, 0, 75), 'flat.sidewalk': (75, 0, 75), 'flat.terrain': (112, 180, 60), 'static.manmade': (222, 184, 135), 'static.other': (255, 228, 196), 'static.vegetation': (0, 175, 0), 'vehicle.ego': (255, 240, 245)}
    color_bank = []
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

    pcd_f = open3d.geometry.PointCloud()
    pcd_f.points = open3d.utility.Vector3dVector(points_f)
    pcd_f.colors = open3d.utility.Vector3dVector(color_bank[labels_f])
    rot = rotation_matrix_from_zyx((np.pi / 2, 0, 0))
    points_f = np.asarray(pcd_f.points)
    rotated_points_f = points_f @ rot.T  # Apply rotation
    pcd_f.points = open3d.utility.Vector3dVector(rotated_points_f)

    pcd_near = deepcopy(pcd)
    pcd_far = deepcopy(pcd_f)
    pcd_near_points = np.array(pcd_near.points)
    pcd_near_colors = np.array(pcd_near.colors)
    pcd_far_points = np.array(pcd_far.points)
    pcd_far_colors = np.array(pcd_far.colors)

    # near_min, near_max = (-12.8, -12.8, -5.0), (12.8, 12.8, 3.0)
    # mask = np.all((pcd_far_points >= near_min) & (pcd_far_points <= near_max), axis=1)
    # import pdb; pdb.set_trace()
    # pcd_far_points = pcd_far_points[~mask]
    # pcd_far_colors = pcd_far_points[~mask]
    # pcd_near_points = pcd_all_points[mask]
    # pcd_near_colors = pcd_all_colors[mask]
    pcd_near.points = open3d.utility.Vector3dVector(pcd_near_points)
    pcd_far.points = open3d.utility.Vector3dVector(pcd_far_points)
    pcd_near.colors = open3d.utility.Vector3dVector(pcd_near_colors)
    pcd_far.colors = open3d.utility.Vector3dVector(pcd_far_colors)
    voxel_grid_near = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_near, voxel_size/10, bound_min, bound_max)
    voxel_grid_far = open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_far, voxel_size, bound_min, bound_max)
    open3d.visualization.draw_geometries([voxel_grid_near]+[voxel_grid_far])
    # open3d.visualization.draw_geometries([voxel_grid_near])
    # cross_positions = [
    #     [5.5, 5.0, 0.0],
    #     [5.5, 10.4, 0.0],
    #     [7.4, 5.0, 0.0],
    #     [7.4, 10.4, 0.0]
    # ]
    # # np.save("87/roi.npy",np.array(cross_positions))
    # cross_markers = [create_cross_marker(np.array(pos)) for pos in cross_positions]

    # open3d.visualization.draw_geometries([voxel_grid_near]+[voxel_grid_far]+cross_markers)
    # voxels_near = voxel_grid_near.get_voxels()
    # voxels_far = voxel_grid_far.get_voxels()
    # indices_near = np.array([v.grid_index for v in voxels_near], dtype=np.int32)
    # indices_far = np.array([v.grid_index for v in voxels_far], dtype=np.int32)
    # xyz_near = (indices_near * voxel_size + bound_min + voxel_size/2)
    # xyz_far = (indices_far * voxel_size*scales + bound_min + (voxel_size/2)*scales)
    # tree_near = KDTree(pcd_near.points)
    # dd_near, ind_near = tree_near.query(xyz_near, k=1)
    # tree_far = KDTree(pcd_far.points)
    # dd_far, ind_far = tree_far.query(xyz_far, k=1)
    # indices_labeled_near = np.concatenate((indices_near[:, :], voxel_size*100 * np.ones_like(labels[ind_near, None]), labels[ind_near, None]), axis=1)
    # indices_labeled_far = np.concatenate((indices_far[:, :], voxel_size*scales*100 * np.ones_like(labels[ind_far, None]), labels[ind_far, None]), axis=1)
    # indices_labeled = np.concatenate((indices_labeled_near, indices_labeled_far), axis=0)

    # np.save("87/nonuniform.npy",indices_labeled)
    return None

# def visualize_point(voxel_grids, voxel_size, color_bank, bound_min, bound_max):
#     pcd = open3d.geometry.PointCloud()
#     voxel_coords = voxel_grids[:, :3]
#     import pdb; pdb.set_trace()
#     pcd.points = open3d.utility.Vector3dVector(voxel_coords * voxel_size + bound_min[::-1])
#     open3d.visualization.draw_geometries([pcd])
#     voxel_grid = open3d.geometry.VoxelGrid()
#     voxel_grid.voxel_size = voxel_size
#     for voxel in voxel_coords:
#         voxel_grid.create_from_point_cloud_within_bounds(pcd, voxel_size, bound_min[::-1], bound_max[::-1])
#     o3d.visualization.draw_geometries([voxel_grid])
mode = "zonotope"
if mode == "basic":
    points = np.load("points_basic.npy")
    labels = np.load("nonzero_values.npy")

    pointclouds_mesh(points, labels, 0.8, (-51.2,-51.2,-5), (51.2,51.2,3))
elif mode == "near":
    points_c = np.load("points_c_NearRefine.npy")
    labels_c = np.load("nonzero_values_c_NearRefine.npy")
    points_f = np.load("points_f_NearRefine.npy")
    labels_f = np.load("nonzero_values_f_NearRefine.npy")
    pointclouds_mesh_nonuniform(points_c, labels_c, points_f, labels_f, 0.8, (-51.2,-51.2,-5), (51.2,51.2,3))
elif mode == "trajectory":
    points_c = np.load("points_c_Trajectory.npy")
    labels_c = np.load("nonzero_values_c_Trajectory.npy")
    points_f = np.load("points_f_Trajectory.npy")
    labels_f = np.load("nonzero_values_f_Trajectory.npy")
    pointclouds_mesh_nonuniform(points_c, labels_c, points_f, labels_f, 0.8, (-51.2,-51.2,-5), (51.2,51.2,3))
elif mode == "zonotope":
    points_c = np.load("points_c_Zonotope.npy")
    labels_c = np.load("nonzero_values_c_Zonotope.npy")
    points_f = np.load("points_f_Zonotope.npy")
    labels_f = np.load("nonzero_values_f_Zonotope.npy")
    pointclouds_mesh_nonuniform(points_c, labels_c, points_f, labels_f, 0.8, (-51.2,-51.2,-5), (51.2,51.2,3))

# open3d.visualization.draw_geometries([loaded_voxel_grid])

# voxel_grids = pointclouds_mesh_nonuniform(pcd, voxel_size, bound_min, bound_max)
# # visualize_point(voxel_grids, voxel_size, color_bank, bound_min, bound_max)

# print(voxel_grids.shape)
