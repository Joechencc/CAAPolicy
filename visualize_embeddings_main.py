import carla
import math
import pathlib
import yaml
import torch
import logging
import time
import pygame
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic, detokenize_waypoint, convert_veh_coord
from dataset.carla_dataset import detokenize_control
from data_generation.network_evaluator import NetworkEvaluator
from data_generation.tools import encode_npy_to_pil
from model.parking_model import ParkingModel
from model.dynamics_model import DynamicsModel
from copy import deepcopy
from pathlib import Path
import umap.umap_ as umap
import glob
import cv2
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_parking_model(parking_pth_path, cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ParkingModel(cfg)
    ckpt = torch.load(parking_pth_path, map_location='cuda:0')
    state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
    model.load_state_dict(state_dict,strict=False)
    model.to(device)
    model.eval()
    return model

def load_cfg(config_file_path):

    with open(config_file_path, 'r') as config_file:
        try:
            cfg_yaml = (yaml.safe_load(config_file))
        except yaml.YAMLError:
            logging.exception('Invalid YAML Config file {}', config_file_path)
    cfg = get_cfg(cfg_yaml)
    return cfg

def generate_umap(embedding_tensor, umap_3d, fit_transform = True):
    width, height = int(embedding_tensor.shape[0]**0.5), int(embedding_tensor.shape[0]**0.5)
    embedding_np = embedding_tensor.detach().cpu().numpy()
    if fit_transform:
        embedding_umap = umap_3d.fit_transform(embedding_np)
    else:
        embedding_umap = umap_3d.transform(embedding_np)

    embedding_umap -= embedding_umap.min(0)
    embedding_umap /= embedding_umap.max(0)

    rgb_embedding = (embedding_umap * 255).astype(np.uint8)

    rgb_grid = rgb_embedding.reshape(width, height, 3)

    # Interpolate to original image resolution
    target_size = (256, 256)  # (W, H)
    rgb_smooth = cv2.resize(rgb_grid, target_size, interpolation=cv2.INTER_CUBIC)

    # Optional: apply Gaussian blur for additional smoothing
    rgb_smooth = cv2.GaussianBlur(rgb_smooth, (11, 11), sigmaX=5, sigmaY=5)

    return rgb_smooth

if __name__ == "__main__":
    # INFO: Load Parking Model
    parking_pth_path = "./milestone/leanable_attention_best_20.ckpt"
    cfg_file_path = "./config/training.yaml"
    cfg = load_cfg(cfg_file_path)
    parking_model = load_parking_model(parking_pth_path, cfg)
    print("Model has been loaded.")

    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda().eval()

    # INFO: Load camera images
    image_process_enet = ProcessImage(cfg.image_crop)
    # image_process_dino = ProcessImage(294)
    image_process_dino = transforms.Compose([
            transforms.Resize((294, 294)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    # Root directory that contains A/, B/, C/ ...
    root_dir = Path("./e2e_parking/Town_Opt_1000/time_12_17_12_16_08/task0/")

    # Get list of subfolders (e.g., A, B, C)
    subfolders = ["camera_front", "camera_front_left", "camera_front_right", "camera_back", "camera_back_left", "camera_back_right"]

    # Use the first subfolder to get reference filenames
    ref_subfolder = root_dir / subfolders[0]
    image_files = sorted(ref_subfolder.glob("*.png"))

    # Get every 10th image filename
    selected_filenames = image_files[50::10]  # These are full Paths

    # Project features to RGB using UMAP
    umap_3d_enet = umap.UMAP(n_components=3, random_state=42)
    umap_3d_dino = umap.UMAP(n_components=3, random_state=42)
    fit_transform_enet = True
    fit_transform_dino = True

    intrinsics = torch.Tensor(np.load("./representation_analyze/intrinsics.npy")).to(device)
    extrinsics = torch.Tensor(np.load("./representation_analyze/extrinsics.npy")).to(device)

    for filename_path in selected_filenames:
        
        filename = filename_path.name  # e.g., '0000.png'
        front_final_enet = image_process_enet(root_dir/subfolders[0]/filename)[0]
        front_left_final_enet = image_process_enet(root_dir/subfolders[1]/filename)[0]
        front_right_final_enet = image_process_enet(root_dir/subfolders[2]/filename)[0]
        back_final_enet = image_process_enet(root_dir/subfolders[3]/filename)[0]
        back_left_final_enet = image_process_enet(root_dir/subfolders[4]/filename)[0]
        back_right_final_enet = image_process_enet(root_dir/subfolders[5]/filename)[0]

        # front_final_dino = image_process_dino((root_dir/subfolders[0]/filename))[0]
        # front_left_final_dino = image_process_dino((root_dir/subfolders[1]/filename))[0]
        # front_right_final_dino = image_process_dino((root_dir/subfolders[2]/filename))[0]
        # back_final_dino = image_process_dino((root_dir/subfolders[3]/filename))[0]
        # back_left_final_dino = image_process_dino((root_dir/subfolders[4]/filename))[0]
        # back_right_final_dino = image_process_dino((root_dir/subfolders[5]/filename))[0]

        front_final_dino = image_process_dino(Image.open(root_dir/subfolders[0]/filename).convert('RGB')).unsqueeze(0)
        front_left_final_dino = image_process_dino(Image.open(root_dir/subfolders[1]/filename).convert('RGB')).unsqueeze(0)
        front_right_final_dino = image_process_dino(Image.open(root_dir/subfolders[2]/filename).convert('RGB')).unsqueeze(0)
        back_final_dino = image_process_dino(Image.open(root_dir/subfolders[3]/filename).convert('RGB')).unsqueeze(0)
        back_left_final_dino = image_process_dino(Image.open(root_dir/subfolders[4]/filename).convert('RGB')).unsqueeze(0)
        back_right_final_dino = image_process_dino(Image.open(root_dir/subfolders[5]/filename).convert('RGB')).unsqueeze(0)

        images_enet = [front_final_enet, front_left_final_enet, front_right_final_enet, back_final_enet, back_left_final_enet, back_right_final_enet]
        images_dino = [front_final_dino, front_left_final_dino, front_right_final_dino, back_final_dino, back_left_final_dino, back_right_final_dino]
        images_enet = torch.cat(images_enet, dim=0).unsqueeze(0).to(device)
        images_dino = torch.cat(images_dino, dim=0).to(device)

        cams_feature, depth, depth_bin_feature, bev_feature = parking_model.bev_model.get_intermidiate_layers(images_enet, intrinsics, extrinsics)
        with torch.no_grad():
            dino_features = dino_model.get_intermediate_layers(images_dino, n=1, reshape=False)[0]  # (1, num_patches, dim)

        for i, (cam_feature, dino_feature) in enumerate(zip(cams_feature, dino_features)):
            cam_feature, dino_feature = cam_feature.permute(1,2,0).reshape(-1,cam_feature.shape[0]), dino_feature
            enet_embedding_umap = generate_umap(cam_feature, umap_3d_enet, fit_transform=fit_transform_enet)
            dino_embedding_umap = generate_umap(dino_feature, umap_3d_dino, fit_transform=fit_transform_dino)
            fit_transform_enet = False
            fit_transform_dino = False

            # Plot results
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(images_enet[0, i].detach().cpu().numpy().transpose(1,2,0))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("ENet UMAP RGB")
            plt.imshow(enet_embedding_umap)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("DinoV2 UMAP RGB")
            plt.imshow(dino_embedding_umap)
            plt.axis("off")

            plt.tight_layout()
            plt.show()