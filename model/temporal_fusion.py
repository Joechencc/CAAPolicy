import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TemporalFusion(nn.Module):  # Added nn.Module inheritance
    def __init__(self, cfg):
        super().__init__()
        # self.cfg = cfg
        self.bev_cache = None # Shape (B + T, 256, 264) e.g.: (30, 256, 264)
        self.delta_ego_pose_cache = None # Shape (B + T, 3) e.g.: (30,3)
        self.cur_idx = 0 # Global frame index for all tasks
        self.cur_start_idx = None # Global Start index of the current task
        self.task_offsets_iter = None # Shape: (B, 2)
        self.T = 10 # Number of time steps to align
        self.tasks_offsets = None
        self.is_sanity_checking = True # Flag to indicate if the model is in sanity checking mode
        self.reset_flag = False
    
    def forward(self, fuse_feature, data): # delta_xy, delta_yaw, restart_idx = True)
        '''
        Input:
        fuse_feature: torch.size(B, 256, 264)
        delta_ego_motion: [delta_x, delta_y, delta_yaw] (unit in meters)
        
        Output:
        Updated BEV cache with all T steps projected to the current ego-centric BEV range.
        '''
        restart_idx = data['restart'] # Shape: (B, 1)
        data['delta_ego_pos'] # Shape: (B, 3)
        temporal_bevs = []  # List to store the aligned BEV features for each time step and each data sample in the batch
        if not self.is_sanity_checking and not self.reset_flag:
            print("Resetting task counter")
            # Reset the iterator after sanity check
            self.task_offsets_iter = iter(data['task_offsets'][0,:,:].detach().tolist())
            # Reset the cur_idx counter
            self.cur_idx = 0
            print("task_offsets: ", data['task_offsets'][0,:,:].detach().tolist())
            self.reset_flag = True
            # import pdb; pdb.set_trace()

        if self.delta_ego_pose_cache == None:
            # Initialize the task offsets
            self.task_offsets_iter = iter(data['task_offsets'][0,:,:].detach().tolist())
            # Initialize the delta_ego_pose_cache with 10 zeros tensors cat with data['delta_ego_pos']
            self.delta_ego_pose_cache = torch.cat((torch.zeros((10, 3)).to(data['delta_ego_pos'].device), data['delta_ego_pos']), dim = 0)  # Shape: (B+T, 3), e.g.: (30, 3)
            
            # Initialize the BEV cache with 10 zeros tensors cat with fuse_feature
            self.bev_cache = torch.cat((torch.zeros((10, 256, 264)).to(fuse_feature.device), fuse_feature), dim=0)  # Shape: (B+T, 256, 264), e.g.: (30, 256, 264)
        else:
            self.delta_ego_pose_cache = torch.cat((self.delta_ego_pose_cache.to(data['delta_ego_pos'].device), data['delta_ego_pos']), dim = 0)
            self.bev_cache = torch.cat((self.bev_cache.to(fuse_feature.device), fuse_feature), dim=0)  # Shape: (B+T, 256, 264), e.g.: (30, 256, 264)
        
        # idx is the index of the current frame in the batch
        for idx in range(len(restart_idx)):
            # print("idx: ", idx)
            # if this is the first frame of the task, reinitialize the BEV cache
            if restart_idx[idx]: # No need to iterate if the task just starts
                print("Reinitializing BEV cache, idx = : ", self.cur_idx)
                # import pdb; pdb.set_trace()
                self.cur_start_idx = next(self.task_offsets_iter)[0]
            # Update the temporal BEV cache with the current frame and the valid past frames
            temporal_bevs.append(self.align_bev(idx))
            # Print the tracked current index and the start index of the current task
            # print(self.cur_idx, self.cur_start_idx)
            self.cur_idx += 1
            

        # convert from bev space to token sapce
        temporal_bevs = torch.stack(temporal_bevs)  # Convert list to tensor
        # Plot the aligned BEV features for debugging
        if self.cur_idx == 10:
            # import pdb; pdb.set_trace()
            self.plot_bev_features(temporal_bevs, title_prefix="Aligned BEV Feature")

        temporal_bevs = temporal_bevs.reshape(self.T, 10, 256, 264)  # Shape: (B, T=10, 16, 16, 264)
        B, T, L, C = temporal_bevs.shape  # B: batch size, L: feature length, T: time steps, C: channels

        # plot the BEV features for debugging
        # self.plot_bev_features(aligned_bev)
        # Update the delta_ego_pose_cache
        self.delta_ego_pose_cache = data['delta_ego_pos'][-self.T:,:].detach()  # Update the delta_ego_motion for next step temporal fusion
        # TODO: Accumulated Memory is sorted by detaching bev_cache
        # TODO: however, is this the right way to do it?
        # Update the BEV cache with the fuse_feature BEV
        self.bev_cache = (fuse_feature[-self.T:,:,:]).detach()  # Shape: (B + 10, 256, 264) 

        # Flatten the BEV features for self-attention
        flattened_bev = temporal_bevs.reshape(B, T, L * C)  # Shape: (B, T, L * C)

        # Compute self-attention scores
        attention_scores = torch.bmm(flattened_bev, flattened_bev.transpose(1, 2))  # Shape: (B, T, T)
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores along the time dimension

        # Apply attention weights to the BEV cache
        fused_bev = torch.bmm(attention_scores, flattened_bev)  # Shape: (B, T, L * C)

        # Reduce the temporal dimension to get a single fused feature
        fused_bev = fused_bev.mean(dim=1, keepdim=True)  # Shape: (B, 1, L * C)

        # Reshape back to (B, L, 1, C)
        fused_bev = fused_bev.view(B, L, 1, C)  # Shape: (B, L, 1, C)

        # Reshape to (B, 256, 264) for final output
        final_features = fused_bev.squeeze(2)  # Shape: (B, L, C)

        self.bev_cache = self.bev_cache.detach()
        # self.last_delta_xy = self.delta_xy.detach()  # Detach the last_ego_xy for next step
        return final_features

    def align_bev(self, idx):
        '''
        Align the egocentric BEV image at all T steps by aligning the first T-1 channels to the Tth channel.
        Fill blank areas with 0s. Fill invalid steps/frames with 0s. Use the delta_ego_motion cache to align the BEV features.
        '''
        # The num_past_frames is the number of past bev features to add to the BEV cache
        num_valid_past_frames = min(9, self.cur_idx - self.cur_start_idx)
        # Extract the relevant delta_ego_pose for alignment
        valid_delta_ego_pose = self.delta_ego_pose_cache[idx + self.T - num_valid_past_frames : idx + self.T]
        # [delta pose 1st to cur, delta pose 2nd to cur, ..., delta pose (T-1)th to cur]
        cumulative_ego_pose = -torch.flip(
            torch.cumsum(torch.flip(valid_delta_ego_pose, dims=[0]), dim=0), dims=[0]
        )  # Shape: (num_valid_past_frames, 3), e.g.: (9, 3)

        # Initialize the aligned BEV cache with zeros
        aligned_bev = torch.zeros((10, 16, 16, 264), device=self.bev_cache.device)  # Shape: (T=10, 16, 16, 264)

        # Align valid past frames
        # TODO: determine the order to align the past frames based on the cumulative ego pose
        # Iterate over the valid past frames in reverse order
        for t in range(num_valid_past_frames, -1, -1):
            if num_valid_past_frames == 0:
                break
            # import pdb; pdb.set_trace()
            # Downsample the BEV feature from (1, 256, 264) to (1, 16, 16, 264)
            # print("    t:", t)
            # TODO: Review the index calculation(for the BEV cache)
            bev_feature = self.bev_cache[idx + self.T - (num_valid_past_frames - t)].unsqueeze(0).squeeze(2)  # Shape: (256,1,264) --> (1, 256, 264)
            bev_feature_spatial = bev_feature.view(1, 16, 16, 264)  # Remove the added channel dimension: (1, 16, 16, 264)
            # import pdb; pdb.set_trace()
            # Extract cumulative ego pose for this frame
            delta_x, delta_y, delta_yaw = cumulative_ego_pose[t - 1]
            cos_yaw = torch.cos(delta_yaw)
            sin_yaw = torch.sin(delta_yaw)

            # Construct the affine transformation matrix
            theta = torch.tensor([
                [cos_yaw, sin_yaw, +delta_x],
                [-sin_yaw, cos_yaw, +delta_y]
            ], device=self.bev_cache.device).unsqueeze(0)  # Shape: (1, 2, 3)

            # Generate the sampling grid
            grid = F.affine_grid(theta, size=bev_feature_spatial.size(), align_corners=False)

            # Apply the affine transformation to the downsampled BEV feature
            aligned_bev[-t - 1] = F.grid_sample(
                bev_feature_spatial,  # Shape: (1, C, H, W)
                grid,
                padding_mode='zeros',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension

        # Add the current frame's BEV feature as the last frame in the aligned BEV cache
        current_bev_feature = self.bev_cache[idx].unsqueeze(0)  # Shape: (1, 256, 264)
        current_bev_spatial = current_bev_feature.unsqueeze(1).view(1, 16, 16, 264)
        aligned_bev[-1] = current_bev_spatial

        return aligned_bev  # Shape: (10, 16, 16, 264)

    def plot_bev_features(self, bev_features, title_prefix="BEV Feature 10 Time Step"):
        '''
        Plots all time steps' 16x16 BEV features for all samples in the batch.
        
        Input:
        bev_features: torch.Tensor of shape (B, T, H, W, C)
        title_prefix: Prefix for the plot title.
        '''
        print("Plotting aligned BEV features")
        import os
        import matplotlib.pyplot as plt

        B, T, H, W, C = bev_features.shape

        # Create a new folder under the current folder
        output_folder = os.path.join(os.getcwd(), "bev_feature_plots")
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # For each channel you want to visualize (here, channel 128)
        channel = 128

        # Create a figure with B rows and T columns
        fig, axes = plt.subplots(B, T, figsize=(T * 3, B * 3), squeeze=False)

        for b in range(B):
            for t in range(T):
                feature_map = bev_features[b, t, :, :, channel].cpu().detach().numpy()  # Shape: (H, W)
                ax = axes[b, t]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f"Timestep:{b} Feature:{B-t}")
                ax.set_xlabel("W")
                ax.set_ylabel("H")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(title_prefix)
        plot_path = os.path.join(output_folder, f"{title_prefix}_all_samples.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()