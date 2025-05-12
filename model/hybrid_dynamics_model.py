import torch
from torch import nn

def physics_model(ego_pos, ego_motion, reverse, speed_xy, dt=0.1):
    """
    Predict the next ego position using the kinematic bicycle model.
    Args:
        ego_pos: Current position [x, y, yaw] (torch.Tensor of shape [B, 3])
        ego_motion: Current IMU measurement [speed, acc_x, acc_y] (torch.Tensor of shape [B, 3])
        reverse: Boolean tensor indicating whether the car is reversing (torch.Tensor of shape [B])
        dt: Time step (float)
    Returns:
        next_ego_pos: Predicted next position [x, y, yaw] (torch.Tensor of shape [B, 3])
    """
    # Extract x, y, and yaw
    x, y, yaw = ego_pos[:, 0], ego_pos[:, 1], ego_pos[:, 2]
    
    # Convert yaw to radians
    yaw = torch.deg2rad(yaw)

    # Transform accelerations to the world frame
    accel_x_world = ego_motion[:, 1] * torch.cos(yaw) - ego_motion[:, 2] * torch.sin(yaw)
    accel_y_world = ego_motion[:, 1] * torch.sin(yaw) + ego_motion[:, 2] * torch.cos(yaw)

    # Convert speed from km/h to m/s
    # speed = ego_motion[:, 0] / 3.6  # Convert speed to m/s

    # Adjust speed direction based on the reverse flag
    # speed = torch.where(reverse.bool(), -speed, speed)  # Negative speed when reversing

    # Recover vehicle velocity components in the world frame
    # vehicle_velocity_x = speed * torch.cos(yaw)
    # vehicle_velocity_y = speed * torch.sin(yaw)
    vehicle_velocity_x = speed_xy[:,0]
    vehicle_velocity_y = speed_xy[:,1]
    # Compute displacements
    displacement_x_world = vehicle_velocity_x * dt + 0.5 * accel_x_world * dt**2
    displacement_y_world = vehicle_velocity_y * dt + 0.5 * accel_y_world * dt**2

    # Update x and y positions
    x += displacement_x_world
    y += displacement_y_world
    return torch.stack((x, y), dim=1), displacement_x_world, displacement_y_world

class ResidualNetwork(nn.Module):
    """
    Neural network to learn the residual error between the kinematic model's prediction
    and the ground truth.
    """
    # TODO: remove x,y from the input 10 --> 8
    def __init__(self, hidden_dim=128, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm1d with LayerNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.std_head = nn.Linear(hidden_dim, output_dim)
        # self.mlp = nn.Sequential(
        #     # TODO: 3 layers are enough?
        #     # nn.Linear(10, 128),
        #     # nn.LayerNorm(128),
        #     # nn.ReLU(),
        #     # nn.Linear(128, 64),
        #     nn.Linear(8, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     # TODO: output 4 features (dx,dy,stdx,stdy)
        #     nn.Linear(32, 2)  # Residual correction [dx, dy]
        # )
        # TODO: add variance to the output
        # TODO: nl gt_x, gt_y 
    def forward(self, inputs):
        feat = self.mlp(inputs)
        delta_mean = self.mean_head(feat)
        logvar = torch.exp(self.std_head(feat)) 
        return delta_mean, logvar
class HybridDynamicsModel(nn.Module):
    """
    Hybrid model combining a physics model and a neural network for residual learning.
    """
    def __init__(self):
        super().__init__()
        self.residual_network = ResidualNetwork()

    def forward(self, data):
        """
        Forward pass of the hybrid model.
        Args:
            data: Dictionary containing the following keys:
                - ego_motion: IMU measurements [velocity, acc_x, acc_y] (torch.Tensor of shape [B, 1, 3])
                - raw_control: Control inputs [throttle, brake, steer, reverse] (torch.Tensor of shape [B, 4])
                - ego_pos: Current ego position [x, y, yaw] (torch.Tensor of shape [B, 3])
                - speed: Current speed [speed_x, speed_y] (torch.Tensor of shape [B, 2])
        Returns:
            final_prediction: Predicted next ego position [x, y] (torch.Tensor of shape [B, 3])
        """
        # Extract inputs
        ego_pos = data['ego_pos']  # [x, y, yaw]
        ego_motion = data['ego_motion'].squeeze(1)[:, :3]
        reverse = data['raw_control'][:, 3]
        
        # Coarse prediction using the kinematic model
        coarse_prediction, displacement_x_world, displacement_y_world = physics_model(ego_pos, ego_motion, reverse, data['speed'])
        # Prepare inputs for the residual network
        inputs = torch.cat((ego_motion[:, 1:],  # IMU measurements ignore speed
                            data['raw_control'].float(),           # Control inputs
                            ego_pos[:,2].unsqueeze(0),
                            data['speed']), dim=1).float() # [speed_x,speed_y]
                            # Only use yaw for input, x, y should not be good features to use

        # Residual correction
        # TODO: changed residual to end to end ego_motion prediction
        # residual = self.residual_network(inputs)
        delta_mean, logvar = self.residual_network(inputs)
        # Final prediction
        # final_prediction = coarse_prediction + residual
        # final_prediction = coarse_prediction

        # TODO: also return the prediction accuracy of coarse model(using physics only)
        return delta_mean, logvar, coarse_prediction, torch.stack((displacement_x_world, displacement_y_world), dim=1), 