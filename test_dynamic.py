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
    checkpoint = torch.load("./ckpt/exp_2025_5_9_0_25_18/last.ckpt")
    # Remove "model." prefix from keys in the state_dict
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    
    model.eval()

    # Initialize lists to store trajectories
    predicted_trajectory = []
    track_trajectory = []
    ground_truth_trajectory = []

    # Iterate through the frames for the specified task
    for frame_index in range(start_index, end_index):

        # Extract initial ego position and ground truth waypoints
        if frame_index == start_index:
            ego_pos = dataset[frame_index]['ego_pos']  # Shape: (3,)
            ego_pos_track = dataset[frame_index]['ego_pos']  # Shape: (3,)
            predicted_trajectory.append(ego_pos.cpu().numpy())
            ground_truth_trajectory.append(ego_pos.cpu().numpy().reshape(-1,3))

        # Prepare input data for the model
        input_data = {
            'ego_pos': ego_pos.unsqueeze(0),  # Shape: (1, 3)
            'ego_motion': dataset[frame_index]['ego_motion'],  # Shape: (1, 3)
            'raw_control': dataset[frame_index]['raw_control'].view(1,-1)  # Shape: (4,)
        }

        # Predict the next ego position
        with torch.no_grad():
            delta_mean, log_var, displacement_x_world, displacement_y_world  = model(input_data) # (1,2)
            pred_ego_pos = delta_mean + ego_pos.reshape(-1,3)[:,:2]
            track_ego_pos = torch.cat([displacement_x_world.reshape(-1,1), displacement_y_world.reshape(-1,1)],dim=1) + ego_pos_track.reshape(-1,3)[:,:2]
            
        if frame_index < end_index - 1:
            # Update yaw for the next frame
            yaw = dataset[frame_index+1]['ego_pos'][2]  # Extract yaw from the ground truth ego_pos
            # Append gt_ego_pos
            gt_ego_pos = dataset[frame_index+1]['ego_pos'].view(1,-1)  # Shape: (3,)
            ground_truth_trajectory.append(gt_ego_pos.cpu().numpy())
        pred_ego_pos = torch.cat((pred_ego_pos, yaw.unsqueeze(0).unsqueeze(0)), dim=1)
        track_ego_pos = torch.cat((track_ego_pos, yaw.unsqueeze(0).unsqueeze(0)), dim=1)
        
        # Append the predicted position to the trajectory
        predicted_trajectory.append(pred_ego_pos.squeeze(0).cpu().numpy())
        track_trajectory.append(track_ego_pos.squeeze(0).cpu().numpy())

        # Update the ego position for the next step
        ego_pos = pred_ego_pos.squeeze(0)
        ego_pos_track = track_ego_pos.squeeze(0)

    # Convert trajectories to numpy arrays
    predicted_trajectory = np.array(predicted_trajectory)
    track_trajectory = np.array(track_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory).squeeze(1)

    # Create the output directory if it doesn't exist
    output_dir = "./dynamic_prediction"
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], label='Ground Truth', color='green')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='Predicted', color='blue')
    plt.plot(track_trajectory[:, 0], track_trajectory[:, 1], label='Track', color='k')
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
        default='./config/training.yaml',
        type=str,
        help='Path to testing.yaml (default: ./config/training.yaml)')
    arg_parser.add_argument(
        '--ckpt_path',
        default='./ckpt/last.ckpt',
        type=str,
        help='Path to the trained model checkpoint (default: ./ckpt/last.ckpt)')
    arg_parser.add_argument(
        '--dataset_path',
        default='/scratch/rs9193/Town_Opt_1000_Val',
        type=str,
        help='Path to the dataset (default: /scratch/rs9193/Town_Opt_1000_Val)')
    arg_parser.add_argument(
        '--task_index',
        default=2,
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

