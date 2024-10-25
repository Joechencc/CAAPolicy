import numpy as np
carla_categories = {
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
        1: 15,  # Building -> manmade
        2: 1,   # Fence -> barrier
        3: 1,  # Other -> barrier
        4: 7,   # Pedestrian -> pedestrian
        5: 1,   # Pole -> barrier
        6: 11,  # RoadLine -> driveable_surface
        7: 11,  # Road -> driveable_surface
        8: 13,  # SideWalk -> sidewalk
        9: 16,  # Vegetation -> vegetation
        10: 4,  # Vehicles -> car (generalization to include various vehicle types)
        11: 15, # Wall -> manmade
        12: 1,  # TrafficSign -> barrier
        13: 0,  # Sky -> noise
        14: 14, # Ground -> terrain
        15: 15, # Bridge -> manmade
        16: 1,  # RailTrack -> barrier
        17: 1,  # GuardRail -> barrier
        18: 1,  # TrafficLight -> barrier
        19: 1,  # Static -> barrier
        20: 1,  # Dynamic -> barrier
        21: 12, # Water -> other_flat
        22: 14  # Terrain -> terrain
    }
    return mapping.get(category_index, 1)  # defaulting to 'barrier' if category is not found


def convert_carla2nuScenes(old_dict, mapping):
    new_dict = {}
    for old_key, value in old_dict.items():

        new_key = mapping.get(old_key, 1)

        if new_key in new_dict:
            new_dict[new_key] += value
        else:
            new_dict[new_key] = value
    return new_dict
mapping = {
        0: 1,   # Unlabeled -> barrier
        1: 15,  # Building -> manmade
        2: 1,   # Fence -> barrier
        3: 1,  # Other -> barrier
        4: 7,   # Pedestrian -> pedestrian
        5: 1,   # Pole -> barrier
        6: 11,  # RoadLine -> driveable_surface
        7: 11,  # Road -> driveable_surface
        8: 13,  # SideWalk -> sidewalk
        9: 16,  # Vegetation -> vegetation
        10: 4,  # Vehicles -> car (generalization to include various vehicle types)
        11: 15, # Wall -> manmade
        12: 1,  # TrafficSign -> barrier
        13: 0,  # Sky -> noise
        14: 14, # Ground -> terrain
        15: 15, # Bridge -> manmade
        16: 1,  # RailTrack -> barrier
        17: 1,  # GuardRail -> barrier
        18: 1,  # TrafficLight -> barrier
        19: 1,  # Static -> barrier
        20: 1,  # Dynamic -> barrier
        21: 12, # Water -> other_flat
        22: 14  # Terrain -> terrain
    }