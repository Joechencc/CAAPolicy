import os
import numpy as np
from collections import defaultdict

from utils.convertSemanticLabel import convert_semantic_label_vectorized

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

def read_ply_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    end_header = next(i for i, line in enumerate(lines) if "end_header" in line)
    data_lines = lines[end_header + 1:]
    data = np.array([list(map(float, line.split())) for line in data_lines])
    return data[:, -1]  # Assuming labels are the last column

def scan_directory(base_dir):
    category_distribution = defaultdict(int)
    for root, dirs, files in os.walk(base_dir):
        if any(lidar in root for lidar in ["lidar_01", "lidar_02", "lidar_03", "lidar_04", "lidar_05"]):
            for file in files:
                if file.endswith('.ply'):
                    file_path = os.path.join(root, file)
                    labels = read_ply_file(file_path)
                    labels = convert_semantic_label_vectorized(labels)  # Vectorized label conversion
                    unique, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique, counts):
                        category_distribution[int(label)] += count
    return category_distribution


if __name__ == "__main__":
    base_dir = './output'
    distribution = scan_directory(base_dir)
    print(distribution)
    for k, v in sorted(distribution.items()):
        print(f"{categories[k]}: {v}")
