import torch
from torch import nn

def physics_model(ego_pos, ego_motion, reverse, dt=0.1):
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
    speed = ego_motion[:, 0] / 3.6  # Convert speed to m/s

    # Adjust speed direction based on the reverse flag
    speed = torch.where(reverse.bool(), -speed, speed)  # Negative speed when reversing

    # Recover vehicle velocity components in the world frame
    vehicle_velocity_x = speed * torch.cos(yaw)
    vehicle_velocity_y = speed * torch.sin(yaw)

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
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Residual correction [dx, dy, dyaw]
        )

    def forward(self, inputs):
        return self.mlp(inputs)

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
        Returns:
            final_prediction: Predicted next ego position [x, y] (torch.Tensor of shape [B, 3])
        """
        # Extract inputs
        ego_pos = data['ego_pos']  # [x, y, yaw]
        ego_motion = data['ego_motion'].squeeze(1)[:, :3]
        reverse = data['raw_control'][:, 3]
        # Coarse prediction using the kinematic model
        coarse_prediction, displacement_x_world, displacement_y_world = physics_model(ego_pos, ego_motion, reverse)
        # Prepare inputs for the residual network
        inputs = torch.cat((ego_motion,  # IMU measurements
                            data['raw_control'].float(),           # Control inputs
                            ego_pos), dim=1).float()

        # Residual correction
        residual = self.residual_network(inputs)

        # Final prediction
        final_prediction = coarse_prediction + residual
        # final_prediction = coarse_prediction

        # TODO: also return the prediction accuracy of coarse model(using physics only)
        return final_prediction, coarse_prediction, torch.stack((displacement_x_world, displacement_y_world), dim=1), 