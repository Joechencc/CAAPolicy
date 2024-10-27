import numpy as np
import open3d as o3d
from collections import Counter
carla_categories_9_11 = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "SideWalk",
    9: "Vegetation",
    10: "Vehicles",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain"
}
categories = {
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "construction_vehicle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
    11: "driveable_surface",
    12: "other_flat",
    13: "sidewalk",
    14: "terrain",
    15: "manmade",
    16: "vegetation"
}
# convert from carla to nuscenes
def convert_semantic_label_9_11(category_index):
    mapping = {
        0: 1,   # Unlabeled -> barrier
        1: 15,  # Building -> manmade
        2: 1,   # Fence -> barrier
        3: 1,   # Other -> barrier
        4: 7,   # Pedestrian -> pedestrian
        5: 1,   # Pole -> barrier
        6: 12,  # RoadLine -> other_flat
        7: 11,  # Road -> driveable_surface
        8: 13,  # SideWalk -> sidewalk
        9: 16,  # Vegetation -> vegetation
        10: 4,  # Vehicles -> car
        11: 1,  # Wall -> barrier
        12: 1,  # TrafficSign -> barrier
        #13: 12, # Sky -> other_flat #should only exist in semantic camera
        14: 12, # Ground -> other_flat
        15: 12, # Bridge -> other_flat
        16: 12, # RailTrack -> other_flat
        17: 1,  # GuardRail -> barrier
        18: 1,  # TrafficLight -> barrier
        19: 1,  # Static -> barrier
        20: 1,  # Dynamic -> barrier
        21: 14, # Water -> terrain
        22: 14  # Terrain -> terrain
    }
    return mapping.get(category_index, 1)  # Default to 'barrier' if not found

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def lidar2ego(points,translation,rotation=None):
    # input should be (n,3)
    if(rotation is not None):
        print("lidar should be at same orientation with vehicle!!!!")
    translated_points = points + translation
    return translated_points

def convert2numpy(SemanticLidarMeasurement):
    point_list = [[d.point.x, d.point.y, d.point.z, d.object_tag] for d in SemanticLidarMeasurement]
    point_array = np.array(point_list)
    return point_array

def align_pcd_list(pcd_list,sensor_specs):
    all_points = []
    all_categories = []
    for index, key in enumerate(sensor_specs.keys()):
        data = pcd_list[key]
        tmp_points = data[:, 0:3]
        tmp_categories = data[:,3].astype(int)
        tmp_points = lidar2ego(tmp_points,[sensor_specs[key]["x"],sensor_specs[key]["y"],sensor_specs[key]["z"]])
        all_points.append(tmp_points)
        all_categories.append(tmp_categories)
    all_points = np.concatenate(all_points)
    all_categories = np.concatenate(all_categories)
    return all_points, all_categories

def voxelization(all_points, all_categories,min_bound = [-32, -32, -3],max_bound = [32, 32, 5], resolution = 0.2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Create voxel grid within the defined bounds
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, resolution, min_bound, max_bound)

    # Map each point to its corresponding voxel and category
    point_to_voxel_map = {}
    for point, category in zip(np.asarray(pcd.points), all_categories):
        if point[0] < max_bound[0] and point[0] > min_bound[0] and point[1] < max_bound[1] and point[1] > min_bound[
            1] and point[2] < max_bound[2] and point[2] > min_bound[2]:
            voxel_index = tuple(voxel_grid.get_voxel(point))
            if voxel_index in point_to_voxel_map:
                point_to_voxel_map[voxel_index].append(category)
            else:
                point_to_voxel_map[voxel_index] = [category]

    # Determine the majority category for each voxel
    voxel_categories = {voxel: most_common(categories) for voxel, categories in point_to_voxel_map.items()}
    voxel_categories = {key: convert_semantic_label_9_11(value) for key, value in voxel_categories.items()}
    dict = {"gt_occ": voxel_categories, "resolution": resolution, "min_bound": min_bound,
            "max_bound": max_bound}
    
    #o3d.io.write_voxel_grid("voxel.ply", voxel_grid)
    return dict