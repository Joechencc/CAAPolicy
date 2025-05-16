import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.carla_dataset_dynamic import CarlaDatasetDynamic
from dataset.carla_dataset import CarlaDataset
from model.dynamics_model import DynamicsModel as DynamicsModel 
from tool.config import get_cfg
import yaml
from loguru import logger
import numpy as np

def load_config(config_path):
    """
    Load the configuration file in the same way as in pl_train.py.
    """
    with open(config_path, 'r') as yaml_file:
        try:
            cfg_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError:
            logger.exception("Failed to open config file: {}", config_path)
            raise
    return get_cfg(cfg_yaml)

def test_hybrid_dynamics_model(ckpt_path, dataset_path, config, task_index=10):
    """
    Test the HybridDynamicsModel and visualize the predicted vs ground truth trajectories.

    Args:
        ckpt_path (str): Path to the trained model checkpoint.
        dataset_path (str): Path to the dataset.
        config: Configuration object for the dataset.
        task_index (int): Index of the task to visualize.
    """
    # Load the dataset
    dataset = CarlaDatasetDynamic(root_dir=dataset_path, is_train=False, config=config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Get the start and end indices for the specified task
    start_index, end_index = dataset.task_offsets[task_index]

    # Load the trained model
    model = DynamicsModel()
    checkpoint = torch.load("./ckpt/last.ckpt")
    # Remove "model." prefix from keys in the state_dict
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    
    model.eval()

    # Initialize lists to store trajectories
    predicted_trajectory = []
    ground_truth_trajectory = []
    tracked_trajectory = []

    # Iterate through the frames for the specified task
    for frame_index in range(start_index, end_index):

        # Extract initial ego position and ground truth waypoints
        if frame_index == start_index:
            ego_pos = dataset[frame_index]['ego_pos']  # Shape: (3,)
            track_pose = dataset[frame_index]['ego_pos']
            predicted_trajectory.append(ego_pos.cpu().numpy())

        # Prepare input data for the model
        input_data = {
            # 'yaw'
            'ego_pos': ego_pos.unsqueeze(0),  # Shape: (1, 3)
            'ego_motion': dataset[frame_index]['ego_motion'].view(1,-1),  # Shape: (1, 4)
            'raw_control': dataset[frame_index]['raw_control'].view(1,-1), # Shape: (4,)
        }
        # Predict the next ego position
        with torch.no_grad():
            # pred_ego_pos, _, _ = model(input_data) # (1,2)
            delta_mean, log_var, x_displacement, y_displacement = model(input_data) # delta_mean: (1, 2), log_var: (1, 2)
            pred_ego_pos = ego_pos[:2] + delta_mean.squeeze(0) # Shape: (1, 3)
            track_ego_pos = track_pose[:2] + torch.cat((x_displacement, y_displacement), dim = 0) # Shape: (1, 2)
        if frame_index < end_index - 2:
            # Update yaw for the next frame
            yaw = dataset[frame_index+1]['ego_pos'][2]  # Extract yaw from the ground truth ego_pos
            # Append gt_ego_pos
            gt_ego_pos = dataset[frame_index+1]['ego_pos'].view(1,-1)  # Shape: (3,)
            ground_truth_trajectory.append(gt_ego_pos.cpu().numpy())

        pred_ego_pos = torch.cat((pred_ego_pos, yaw.unsqueeze(0)), dim=0)
        track_ego_pos = torch.cat((track_ego_pos, yaw.unsqueeze(0)), dim=0)
        
        # Append the predicted position to the trajectory
        predicted_trajectory.append(pred_ego_pos.squeeze(0).cpu().numpy())
        tracked_trajectory.append(track_ego_pos.squeeze(0).cpu().numpy())

        # Update the ego position for the next step
        ego_pos = pred_ego_pos.squeeze(0)
        track_pose = track_ego_pos.squeeze(0)

    # Convert trajectories to numpy arrays
    predicted_trajectory = np.array(predicted_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory).squeeze(1)
    tracked_trajectory = np.array(tracked_trajectory)

    # Create the output directory if it doesn't exist
    output_dir = "./dynamic_prediction"
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], label='Ground Truth', color='green')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='Predicted', color='blue')
    plt.plot(tracked_trajectory[:, 0], tracked_trajectory[:, 1], label='Tracked', color='orange')
    # plot the yaw direciton at start for predicted trajectory and ground truth trajectory
    # it is simply just ground_truth_trajectory[0, 2] and predicted_trajectory[0, 2] the yaw is expressed in degree so use two different color arrow to indicate the yaw direction
    plt.quiver(ground_truth_trajectory[0, 0], ground_truth_trajectory[0, 1],
               np.cos(np.deg2rad(ground_truth_trajectory[0, 2])),
               np.sin(np.deg2rad(ground_truth_trajectory[0, 2])),
               angles='xy', scale_units='xy', scale=1.5, color='green', label='GT Yaw Direction')
    plt.quiver(predicted_trajectory[0, 0], predicted_trajectory[0, 1],
               np.cos(np.deg2rad(predicted_trajectory[0, 2])),
               np.sin(np.deg2rad(predicted_trajectory[0, 2])),
               angles='xy', scale_units='xy', scale=1, color='blue', label='Predicted Yaw Direction')

    plt.scatter(ground_truth_trajectory[0, 0], ground_truth_trajectory[0, 1], color='red', label='Start Point')
    plt.title(f'Trajectory Visualization for Task {task_index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()

    # Save the plot to the output directory
    plot_path = os.path.join(output_dir, f"task_{task_index}_trajectory.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Trajectory plot saved to {plot_path}")

if __name__ == '__main__':
    import argparse

    # Parse arguments
    arg_parser = argparse.ArgumentParser(description='Test Hybrid Dynamics Model')
    arg_parser.add_argument(
        '--config',
        default='./config/dynamics_training.yaml',
        type=str,
        help='Path to testing.yaml (default: ./config/dynamics_training.yaml)')
    arg_parser.add_argument(
        '--ckpt_path',
        default='./ckpt/last.ckpt',
        type=str,
        help='Path to the trained model checkpoint (default: ./ckpt/last.ckpt)')
    arg_parser.add_argument(
        '--dataset_path',
        default='/scratch/rs9193/dynamics_dataset/',
        type=str,
        help='Path to the dataset (default: /scratch/rs9193/dynamics_dataset/)')
    arg_parser.add_argument(
        '--task_index',
        default=0,
        type=int,
        help='Index of the task to visualize (default: 0)')
    args = arg_parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run the test
    test_hybrid_dynamics_model(
        ckpt_path=args.ckpt_path,
        dataset_path=args.dataset_path,
        config=config,
        task_index=args.task_index
    )

