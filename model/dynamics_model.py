import torch
from torch import nn

class DynamicsModel(nn.Module):  # Fixed typo in nn.module -> nn.Module

    def __init__(self):
        '''
        This module is used to predict ego_motion
        Input: velocity, acc_x, acc_y, throttle, brake, steer, current_ego_pos(x, y, yaw)
        Output: next_ego_pos
        '''
        super().__init__()
        
        # Define the MLP model with LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(10, 256),
            nn.LayerNorm(256),  # Replace BatchNorm1d with LayerNorm
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: 3 features (next x, y, yaw)
        )

    def forward(self, data):
        # Input
        #torch.szie[B,1]
        velocity, acc_x, acc_y = data['ego_motion'][:,:,0], data['ego_motion'][:,:,1], data['ego_motion'][:,:,2]
        # torch.szie[B]
        throttle, brake, steer, reverse = data['raw_control'][:,0], data['raw_control'][:,1], data['raw_control'][:,2], data['raw_control'][:,3]
        # torch.size[B,3]
        cur_ego_pos = data['ego_pos']  # [x, y, yaw]
        # Concatenate all inputs into a single tensor
        # inputs.shape = torch.size[26,3]
        inputs = torch.cat((velocity, acc_x, acc_y,
                            throttle.reshape(-1,1), brake.reshape(-1,1), steer.reshape(-1,1), reverse.reshape(-1,1),
                            cur_ego_pos), dim=1).float()
        # TODO: check if inputs needs to stored as float64
        # Pass through the MLP
        next_ego_pos = self.mlp(inputs)

        return next_ego_pos