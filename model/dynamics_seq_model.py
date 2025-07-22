import torch
from torch import nn
import numpy as np
import math

class DynamicsModel(nn.Module):  # Fixed typo in nn.module -> nn.Module

    def __init__(self, hidden_dim=128, output_dim=2, num_layers=1):
        '''
        This module is used to predict ego_motion
        Input: speed_x, speed_y, acc_x, acc_y, throttle, brake, steer, reverse, cos(yaw), sin(yaw))
        Output: delta_mean, log_var, displacement_x_world, displacement_y_world
        '''
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Define the MLP model with LayerNorm
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, dt=0.1):
        # Extract x, y, and yaw
        B, S, _ = data['ego_pos'].shape  # S = 4
        ego_pos = data['ego_pos']  # [x, y, yaw]
        ego_motion = data['ego_motion'] # vel, accel_x, accel_y
        x, y, yaw = ego_pos[:, :, 2], ego_pos[:, :, 2], ego_pos[:, :, 2]

        # Convert yaw to radians
        yaw = torch.deg2rad(yaw)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Transform accelerations to the world frame
        accel_x_world = ego_motion[:, :, 1] * cos_yaw + ego_motion[:, :, 2] * sin_yaw
        accel_y_world = -ego_motion[:, :, 1] * sin_yaw + ego_motion[:, :, 2] * cos_yaw

        # Convert speed from km/h to m/s
        reverse = data['raw_control'][:, :, 3]

        # Recover vehicle velocity components in the world frame
        vehicle_velocity = data['ego_motion'][:,:,0] / 3.6

        speed = vehicle_velocity
        vehicle_velocity_x = speed * cos_yaw
        vehicle_velocity_y = speed * sin_yaw

        # Compute displacements for reference only km/h -> m/s
        displacement_x_world_track = vehicle_velocity_x * dt + 0.5 * accel_x_world * dt**2
        displacement_y_world_track = vehicle_velocity_y * dt + 0.5 * accel_y_world * dt**2

        # torch.szie[B]
        throttle, brake, steer, reverse = data['raw_control'][:,:,0], data['raw_control'][:,:,1], data['raw_control'][:,:,2], data['raw_control'][:,:,3]
        # torch.size[B,3]
        cur_ego_pos = data['ego_pos']  # [x, y, yaw]
        dt_tensor = torch.full((B, S, 1), dt, device=throttle.device)
        # Concatenate all inputs into a single tensor
        # inputs.shape = torch.size[26,3]
        features = torch.cat([
            speed.unsqueeze(-1),
            accel_x_world.unsqueeze(-1),
            accel_y_world.unsqueeze(-1),
            throttle.unsqueeze(-1),
            brake.unsqueeze(-1),
            steer.unsqueeze(-1),
            reverse.unsqueeze(-1),
            cos_yaw.unsqueeze(-1),
            sin_yaw.unsqueeze(-1),
            dt_tensor
        ], dim=-1)
         # LSTM encode
        lstm_out, (h_n, _) = self.lstm(features)  # lstm_out: (B, S, H), h_n: (1, B, H)
        last_hidden = h_n[-1]  # (B, H)

        feat = self.mlp(last_hidden)
        delta_mean = self.mean_head(feat)
        logvar = torch.exp(self.std_head(feat))

        return delta_mean, logvar, displacement_x_world_track, displacement_y_world_track