carla_categories = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail"
}

categories = {
    0: "noise",
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
def convert_semantic_label(category_index):
    mapping = {
        0: 1,   # Unlabeled -> barrier
        1: 11,  # Roads -> driveable_surface
        2: 13,  # SideWalks -> sidewalk
        3: 15,  # Building -> manmade
        4: 1,   # Wall -> barrier
        5: 1,   # Fence -> barrier
        6: 1,   # Pole -> barrier
        7: 1,  # TrafficLight -> barrier
        8: 1,   # TrafficSign -> barrier
        9: 16,  # Vegetation -> vegetation
        10: 14, # Terrain -> terrain
        11: 0, # Sky -> noise, shouldnt be any sky in semantic lidar, only for semantic camera
        12: 7,  # Pedestrian -> pedestrian
        13: 6,  # Rider -> motorcycle
        14: 4,  # Car -> car
        15: 10, # Truck -> truck
        16: 3,  # Bus -> bus
        17: 1, # Train -> barrier
        18: 6,  # Motorcycle -> motorcycle
        19: 2,  # Bicycle -> bicycle
        20: 1, # Static -> barrier
        21: 1, # Dynamic -> barrier
        22: 1,  # Other -> barrier
        23: 12, # Water -> other_flat
        24: 11, # RoadLine -> driveable surface
        25: 12, # Ground -> other_flat
        26: 1, # Bridge -> barrier Only the structure of the bridge.
        27: 12, # RailTrack -> other_flat
        28: 1   # GuardRail -> barrier
    }
    return mapping.get(category_index, 1)  # Default to 'barrier' if not found


def convert_semantic_label_vectorized(labels):
    # 基于上述映射定义转换数组
    max_label = 28  # 最大的类别标签
    conversion_array = np.full(max_label + 1, 1)  # 默认值为1 (barrier)

    # 根据原映射更新转换数组
    conversion_array[0] = 1
    conversion_array[1] = 11
    conversion_array[2] = 13
    conversion_array[3] = 15
    conversion_array[4] = 1
    conversion_array[5] = 1
    conversion_array[6] = 1
    conversion_array[7] = 1
    conversion_array[8] = 1
    conversion_array[9] = 16
    conversion_array[10] = 14
    conversion_array[11] = 0
    conversion_array[12] = 7
    conversion_array[13] = 6
    conversion_array[14] = 4
    conversion_array[15] = 10
    conversion_array[16] = 3
    conversion_array[17] = 1
    conversion_array[18] = 6
    conversion_array[19] = 2
    conversion_array[20] = 1
    conversion_array[21] = 1
    conversion_array[22] = 1
    conversion_array[23] = 12
    conversion_array[24] = 11
    conversion_array[25] = 12
    conversion_array[26] = 1
    conversion_array[27] = 12
    conversion_array[28] = 1

    # 使用转换数组进行快速转换
    converted_labels = conversion_array[labels.astype(int)]
    return converted_labels