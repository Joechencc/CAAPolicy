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
        self.T = 20 # Number of time steps to align
        self.tasks_offsets = None
        self.reset_flag = True # Flag to indicate if the model needs reset
        self.is_train = False # Flag to indicate if the model is in training mode
        
        #########################
        self.gt_bev_cache = None # Shape (B + T, 200, 200, 1) e.g.: (30, 200, 200, 1)
        self.ego_pose_cache = None # Shape (B + T, 3) e.g.: (30,3)
    def reset_state(self):
        
        self.bev_cache = None
        self.delta_ego_pose_cache = None
        self.cur_idx = 0
        self.cur_start_idx = None
        self.task_offsets_iter = None
        self.tasks_offsets = None
        self.reset_flag = False

        ########################
        self.gt_bev_cache = None
        self.task_offsets_iter2 =None
        self.cur_idx2 = 0
        self.cur_start_idx2 = None
        self.ego_pose_cache = None # Shape (B + T, 3) e.g.: (30,3)

    def forward(self, fuse_feature, data): # delta_xy, delta_yaw, restart_idx = True)
        '''
        Input:
        fuse_feature: torch.size(B, 256, 264)
        delta_ego_motion: [delta_x, delta_y, delta_yaw] (unit in meters)
        
        Output:
        Updated BEV cache with all T steps projected to the current ego-centric BEV range.
        '''
        ################################################################################
        # plot the gt_bev_semantics for debugging
        # import pdb; pdb.set_trace()
        # Enale this for debugging
        if self.is_train:
            self.fuse_gt_bev_semantics(data)
            print("fuse_gt_bev_semantics done")

        
        ################################################################################
        restart_idx = data['restart'] # Shape: (B, 1)
        B = len(restart_idx) # Batch size
        data['delta_ego_pos'] # Shape: (B, 3)
        temporal_bevs = []  # List to store the aligned BEV features for each time step and each data sample in the batch
        if self.reset_flag:
            print("Resetting task counter, iterator")
            self.reset_state()
            print("task_offsets: ", data['task_offsets'][0,:,:].detach().tolist())
            self.reset_flag = False
            # import pdb; pdb.set_trace()

        if self.bev_cache == None:
            # Initialize the task offsets
            self.task_offsets_iter = iter(data['task_offsets'][0,:,:].detach().tolist())
            # Initialize the delta_ego_pose_cache with T zeros tensors cat with data['delta_ego_pos']
            self.delta_ego_pose_cache = torch.cat((torch.zeros((self.T, 3)).to(data['ego_pos'].device), (data['ego_pos_next'] - data['ego_pos'])), dim = 0)  # Shape: (B+T, 3), e.g.: (30, 3)
            
            # Initialize the BEV cache with T zeros tensors cat with fuse_feature
            self.bev_cache = torch.cat((torch.zeros((self.T, 256, 264)).to(fuse_feature.device), fuse_feature), dim=0)  # Shape: (B+T, 256, 264), e.g.: (30, 256, 264)
        else:
            self.delta_ego_pose_cache = torch.cat((self.delta_ego_pose_cache.to(data['ego_pos'].device), (data['ego_pos_next'] - data['ego_pos'])), dim = 0)
            self.bev_cache = torch.cat((self.bev_cache.to(fuse_feature.device), fuse_feature), dim=0)  # Shape: (B+T, 256, 264), e.g.: (30, 256, 264)
        
        # idx is the index of the current frame in the batch
        for idx in range(B):
            # print("idx: ", idx)
            # if this is the first frame of the task, reinitialize the BEV cache
            if restart_idx[idx]: # No need to iterate if the task just starts
                # import pdb; pdb.set_trace()
                print("Reinitializing BEV cache, idx = : ", self.cur_idx)
                # import pdb; pdb.set_trace()
                self.cur_start_idx = next(self.task_offsets_iter)[0]
            # Update the temporal BEV cache with the current frame and the valid past frames
            temporal_bevs.append(self.align_bev(idx))
            # Print the tracked current index and the start index of the current task
            print(self.cur_idx, self.cur_start_idx)
            self.cur_idx += 1
            

        # convert from bev space to token sapce
        temporal_bevs = torch.stack(temporal_bevs)  # Convert list to tensor
        # Plot the aligned BEV features for debugging
        if self.cur_idx == 10:
            # import pdb; pdb.set_trace()
            self.plot_bev_features(temporal_bevs, title_prefix="Aligned BEV Feature")

        temporal_bevs = temporal_bevs.reshape(B, self.T, 256, 264)  # Shape: (B, T=10, 16, 16, 264)
        B, T, L, C = temporal_bevs.shape  # B: batch size, L: feature length, T: time steps, C: channels

        # plot the BEV features for debugging
        # self.plot_bev_features(aligned_bev)
        # Update the delta_ego_pose_cache
        self.delta_ego_pose_cache = self.delta_ego_pose_cache[-self.T:,:].detach()  # Update the delta_ego_motion for next step temporal fusion
        # TODO: Accumulated Memory is sorted by detaching bev_cache
        # TODO: however, is this the right way to do it?
        # Update the BEV cache with the fuse_feature BEV
        self.bev_cache = (self.bev_cache[-self.T:,:,:]).detach()  # Shape: (B + T, 256, 264) 

        # Flatten the BEV features for self-attention
        flattened_bev = temporal_bevs.reshape(B, T, L * C)  # Shape: (B, T, L * C)
        temporal_bevs = temporal_bevs.detach()
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
        if self.cur_start_idx is None:
            print(f"[DEBUG] cur_idx={self.cur_idx}, cur_start_idx=None at idx={idx}")
        # The num_past_frames is the number of past bev features to add to the BEV cache
        num_valid_past_frames = min(self.T-1, self.cur_idx - self.cur_start_idx)
        # Extract the relevant delta_ego_pose for alignment
        valid_delta_ego_pose = self.delta_ego_pose_cache[idx + self.T - num_valid_past_frames : idx + self.T]
        # [delta pose 1st to cur, delta pose 2nd to cur, ..., delta pose (T-1)th to cur]
        # Compute reverse cumulative sum for each row (sum from i to end)
        cumulative_ego_pose = [valid_delta_ego_pose[i:].sum(dim=0) for i in range(valid_delta_ego_pose.shape[0])]

        # Initialize the aligned BEV cache with zeros
        aligned_bev = torch.zeros((self.T, 16, 16, 264), device=self.bev_cache.device)  # Shape: (T=10, 16, 16, 264)

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

        return aligned_bev  # Shape: (T, 16, 16, 264)

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
                ax.set_title(f"Batch_idx:{b} Timestep:{t}")
                ax.set_xlabel("W")
                ax.set_ylabel("H")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(title_prefix)
        plot_path = os.path.join(output_folder, f"{title_prefix}_all_samples.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()

    def plot_fused_bev_semantics(self, temporal_bevs_semantics, title_prefix="GT_BEV_Semantics"):
        import os
        # plot a single gt_bev_semantics for visualization
        # import pdb; pdb.set_trace()
        B, T, H, W, C = temporal_bevs_semantics.shape

        # Create a new folder under the current folder
        output_folder = os.path.join(os.getcwd(), "gt_bevs_semantics_plots")
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # For each channel you want to visualize (here, channel 128)
        channel = 0

        # Create a figure with B rows and T columns
        fig, axes = plt.subplots(B, T, figsize=(T * 3, B * 3), squeeze=False)

        for b in range(B):
            for t in range(T):
                feature_map = temporal_bevs_semantics[b, t, :, :, channel].cpu().detach().numpy()  # Shape: (H, W)
                ax = axes[b, t]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f"Step_idx:{self.cur_idx2-B+b} Timestep:t{t+1-self.T}") #TODO:
                ax.set_xlabel("W")
                ax.set_ylabel("H")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(title_prefix)
        plot_path = os.path.join(output_folder, f"{title_prefix}_all_samples.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()
        # Save the plot
        print(f"Saved GT BEV Segmentation plots for {B} samples to {plot_path}")
    def align_gt_bev_semantics(self, idx):
        if self.cur_start_idx2 is None:
            print(f"[DEBUG] cur_idx={self.cur_idx2}, cur_start_idx=None at idx={idx}")

        # The num_past_frames is the number of past bev features to add to the BEV cache
        num_valid_past_frames = min(self.T-1, self.cur_idx2 - self.cur_start_idx2)
        # Extract the relevant delta_ego_pose for alignment
        valid_delta_ego_pose = self.delta_ego_pose_cache[idx + self.T - num_valid_past_frames : idx + self.T]
        # [delta pose 1st to cur, delta pose 2nd to cur, ..., delta pose (T-1)th to cur]
        # Compute reverse cumulative sum for each row (sum from i to end)
        cumulative_ego_pose = [valid_delta_ego_pose[i:].sum(dim=0) for i in range(valid_delta_ego_pose.shape[0])]
        if self.cur_idx2 == 100:
            import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        # Initialize the aligned BEV cache with zeros
        aligned_bev_gt = torch.zeros((self.T, 200, 200, 1), device=self.gt_bev_cache.device)  # Shape: (self.T, 200, 200, 1)
        not_aligned_bev_gt = torch.zeros((self.T, 200, 200, 1), device=self.gt_bev_cache.device)  # Shape: (self.T, 200, 200, 1)
        assert len(cumulative_ego_pose) == num_valid_past_frames, f"len={len(cumulative_ego_pose)}, expected={num_valid_past_frames}, idx={idx}, cur_idx2={self.cur_idx2}"  
        # Align valid past frames
        
        # Iterate over the valid past frames in oldest-first order
        # import pdb; pdb.set_trace()
        for t in range(num_valid_past_frames, 0, -1):
            # append from  (- self.T) step to current step
            # import pdb; pdb.set_trace()
            # TODO: Review the index calculation(for the BEV cache)
            bev_feature = self.gt_bev_cache[idx + self.T - t].unsqueeze(0).squeeze(2) # torch.Size([1, 1, 200, 200])
            # Extract cumulative ego pose for this frame
            delta_x, delta_y, delta_yaw = cumulative_ego_pose[-t]
            import math
            delta_yaw *= (math.pi / 180)
            cos_yaw, sin_yaw = torch.cos(delta_yaw), torch.sin(delta_yaw)

            res = 0.1  # meters/pixel
            W, H = 200, 200

            half_w_m = (W * res) / 2  # (200 * 0.1) / 2 = 10 meters
            half_h_m = (H * res) / 2  # 10 meters

            t_x = -( cos_yaw*delta_x + sin_yaw*delta_y )
            t_y = -(-sin_yaw*delta_x + cos_yaw*delta_y )

            t_x = delta_x / half_w_m
            t_y = delta_y / half_h_m
            # Construct the affine transformation matrix
            theta = torch.tensor([
                [cos_yaw, sin_yaw, -t_x],
                [-sin_yaw, cos_yaw, -t_y]
            ], device=self.gt_bev_cache.device).unsqueeze(0)

            # Generate the sampling grid
            grid = F.affine_grid(theta, size=bev_feature.size(), align_corners=False)

            # Apply the affine transformation to the downsampled BEV feature
            aligned_bev_gt[-t-1] = F.grid_sample(
                bev_feature,  # Shape: (1, C, H, W)
                grid,
                padding_mode='zeros',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # Remove batch dimension

            not_aligned_bev_gt[-t-1] = bev_feature.squeeze(0).permute(1, 2, 0)

        # Add the current frame's BEV feature as the last frame in the aligned BEV cache
        current_bev_feature = self.gt_bev_cache[self.T+idx].unsqueeze(0)  # Shape: (1, 200, 200, 1)
        current_bev_spatial = current_bev_feature.unsqueeze(1).view(1, 200, 200, 1)
        aligned_bev_gt[-1] = current_bev_spatial
        not_aligned_bev_gt[-1] = current_bev_spatial
    
        return [aligned_bev_gt, not_aligned_bev_gt]  # Shape: (T, 200, 200, 1)
    
    # For debug only (global big BEV semantics)
    def padded_align_bev(self, idx, pad_size=300, res=0.1):
        
        import os
        import numpy as np
        import torch
        import torch.nn.functional as F
        import math
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        """
        Aligns past-frame BEV semantics to the current ego pose on a larger canvas
        so that information outside the 200×200 field of view is retained.

        Returns
        -------
        aligned_bev_gt  :  Tensor (T, pad_size, pad_size, 1)
                        each timestep’s padded & ego-aligned BEV
        global_bev_gt   :  Tensor (pad_size, pad_size, 1)
                        max-composite of all timesteps in current ego frame
        """
        # ------------------------------------------------------------------ #
        # (1) as before – work out how many past frames and cumulative poses #
        # ------------------------------------------------------------------ #
        num_valid_past_frames = min(self.T-1, self.cur_idx2 - self.cur_start_idx2)
        valid_delta_ego_pose  = self.delta_ego_pose_cache[
            idx + self.T - num_valid_past_frames : idx + self.T
        ]
        cumulative_ego_pose = [
            valid_delta_ego_pose[i:].sum(dim=0) for i in range(valid_delta_ego_pose.shape[0])
        ]
        print(f"num_valid_past_frames: {num_valid_past_frames}")

        # Now use self.ego_pose cache to align bev semantics to the current ego pose
        # valid_ego_pose is the corresponding ego pose for the bev caches to be aligned
        valid_ego_pose = self.ego_pose_cache[
            idx + self.T - num_valid_past_frames : idx + self.T + 1
        ]

        # ------------------------------------------------------------------ #
        # (2) prepare tensors                                                #
        # ------------------------------------------------------------------ #
        aligned_bev_gt = torch.zeros(
            (self.T, pad_size, pad_size, 1),
            device=self.gt_bev_cache.device
        )
        global_bev_gt  = torch.zeros(
            (pad_size, pad_size, 1),
            device=self.gt_bev_cache.device
        )

        # convenience constants for *padded* canvas
        half_w_m = (pad_size * res) / 2      # 25 m for 0.1 m/px & 500 px
        half_h_m = half_w_m

        # slice where the original 200×200 sits inside the 500×500
        inset = (pad_size - 200) // 2        # 150

        # ------------------------------------------------------------------ #
        # (3) iterate through valid past frames (oldest first)               #
        # ------------------------------------------------------------------ #
        for t in range(num_valid_past_frames, 0, -1):
            bev_small = self.gt_bev_cache[idx + self.T - t]          # (1,1,200,200)

            # --- (3a) pad to 500×500 -------------------------------------- #
            bev_pad   = torch.zeros_like(bev_small.new_zeros(1, 1, pad_size, pad_size))
            bev_pad[:, :, inset:inset+200, inset:inset+200] = bev_small

            # --- (3b) compute affine for padded canvas -------------------- #
            delta_x, delta_y, delta_yaw = cumulative_ego_pose[-t]
            delta_yaw = delta_yaw * math.pi / 180.0
            cos_yaw, sin_yaw = torch.cos(delta_yaw), torch.sin(delta_yaw)

            # translation normalised to [-1,1] coords
            t_x =  delta_x / half_w_m
            t_y =  delta_y / half_h_m

            theta = torch.tensor(
                [[ cos_yaw,  sin_yaw, -t_x],
                [-sin_yaw,  cos_yaw, -t_y]],
                device=bev_small.device
            ).unsqueeze(0)          # (1,2,3)

            grid  = F.affine_grid(theta, bev_pad.size(), align_corners=False)
            warped = F.grid_sample(
                bev_pad, grid, padding_mode='zeros', align_corners=False
            )                       # (1,1,500,500)

            aligned_bev_gt[-t-1] = warped.squeeze(0).permute(1,2,0)
            global_bev_gt        = torch.maximum(global_bev_gt, aligned_bev_gt[-t-1])

        # ------------------------------------------------------------------ #
        # (4) current frame – put straight into centre of canvas            #
        # ------------------------------------------------------------------ #
        current = self.gt_bev_cache[self.T+idx]                       # (1,1,200,200)
        bev_pad_cur = torch.zeros_like(current.new_zeros(1,1,pad_size,pad_size))
        bev_pad_cur[:, :, inset:inset+200, inset:inset+200] = current
        aligned_bev_gt[-1] = bev_pad_cur.squeeze(0).permute(1,2,0)
        global_bev_gt      = torch.maximum(global_bev_gt, aligned_bev_gt[-1])

        # optional visual check at a given index
        from matplotlib.colors import ListedColormap
        # cmap = ListedColormap(["black",   # 0   → empty
        #            "mediumspringgreen",   # 128 → target
        #            "dodgerblue"])         # 255 → obstacle

        output_folder = os.path.join(os.getcwd(), "gt_bevs_semantics_plots")
        os.makedirs(output_folder, exist_ok=True)
        plt.figure(figsize=(12, 12))
        plt.imshow(global_bev_gt.cpu().numpy().squeeze(), origin = "lower", cmap='viridis')
        plt.axis("off")
        plt.title("Global Padded BEV semantics")
        plot_path = os.path.join(output_folder, "Global Padded BEV semantics.png")
        plt.savefig(plot_path)
        plt.close()

    def plot_temporal_fused_bev_semantics(self, temporal_bevs_semantics, title_prefix="GT_BEV_Semantics"):
        '''
        Plots all time steps' 200x200 BEV features for all samples in the batch.
        
        Input:
        bev_features: torch.Tensor of shape (B, T, H, W, C)
        title_prefix: Prefix for the plot title.
        '''
        print("Plotting aligned GT BEV features")
        import os
        import matplotlib.pyplot as plt

        B, T, H, W, C = temporal_bevs_semantics.shape

        # Create a new folder under the current folder
        output_folder = os.path.join(os.getcwd(), "gt_bevs_semantics_plots")
        os.makedirs(output_folder, exist_ok=True)
        # sum the aligned BEV features along the time dimension
        temporal_fused_bevs_semantics = torch.mean(temporal_bevs_semantics, dim=1)  # Shape: (B, 200, 200, 1)
        # Create a figure with B rows:
        fig, axes = plt.subplots(B, 1, figsize=(1 * 3, B * 3), squeeze=False)
        for b in range(B):
            feature_map = temporal_fused_bevs_semantics[b, :, :, 0].cpu().detach().numpy()
            ax = axes[b, 0]
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"Step_idx:{self.cur_idx2-B+b}")
            ax.set_xlabel("W")
            ax.set_ylabel("H")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle(title_prefix)
        plot_path = os.path.join(output_folder, f"{title_prefix}_all_samples.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close()
        # Save the plot
        print(f"Saved GT BEV Segmentation plots for {B} samples to {plot_path}")

    def fuse_gt_bev_semantics(self, data):

        # Align the gt_bev_semantics to the current frame
        restart_idx = data['restart'] # Shape: (B, 1)
        B = len(restart_idx) # Batch size
        
        temporal_bevs_semantics = []  # List to store the aligned BEV features for each time step and each data sample in the batch
        not_aligned_temporal_bevs_semantics = []
        ############################################################
        if self.reset_flag:
            print("Resetting task counter, iterator")
            self.reset_state()
            print("task_offsets: ", data['task_offsets'][0,:,:].detach().tolist())
            self.reset_flag = False
            # import pdb; pdb.set_trace()
        ############################################################

        if self.delta_ego_pose_cache == None:
            # Initialize the task offsets
            self.task_offsets_iter2 = iter(data['task_offsets'][0,:,:].detach().tolist())
            # Initialize the delta_ego_pose_cache with T zeros tensors cat with data['delta_ego_pos']
            self.delta_ego_pose_cache = torch.cat((torch.zeros((self.T, 3)).to(data['ego_pos'].device), (data['ego_pos_next'] - data['ego_pos'])), dim = 0)  # Shape: (B+T, 3), e.g.: (30, 3)
            self.ego_pose_cache = torch.cat((torch.zeros((self.T, 3)).to(data['ego_pos'].device), data['ego_pos']), dim=0)  # Shape: (B+T, 3), e.g.: (30, 3)
            # import pdb; pdb.set_trace()
            # Initialize the BEV cache with T zeros tensors cat with fuse_feature
            self.gt_bev_cache = torch.cat((torch.zeros((self.T, 1, 200, 200)).to(data['segmentation'] .device), data['segmentation'] ), dim=0)  # Shape: (B+self.T, 200, 200, 1), e.g.: (30, 200, 200, 1)
        else:
            self.delta_ego_pose_cache = torch.cat((self.delta_ego_pose_cache.to(data['ego_pos'].device), (data['ego_pos_next'] - data['ego_pos'])), dim = 0)
            self.ego_pose_cahce = torch.cat((self.ego_pose_cache.to(data['ego_pos'].device), data['ego_pos']), dim=0)  # Shape: (B+T, 3), e.g.: (30, 3)
            self.gt_bev_cache = torch.cat((self.gt_bev_cache.to(data['segmentation'] .device)[-self.T:,:,:,:], data['segmentation'] ), dim=0)  # Shape: (B+self.T, 200, 200, 1), e.g.: (30, 200, 200, 1)
            self.yaw_cache
        # idx is the index of the current frame in the batch
        for idx in range(B):
            # print("idx: ", idx)
            # if this is the first frame of the task, reinitialize the BEV cache
            if restart_idx[idx]: # No need to iterate if the task just starts
                print("Reinitializing BEV cache, idx = : ", self.cur_idx2)
                # import pdb; pdb.set_trace()
                self.cur_start_idx2 = next(self.task_offsets_iter2)[0]
            # Update the temporal BEV cache with the current frame and the valid past frames
            temporal_bevs_semantics.append(self.align_gt_bev_semantics(idx)[0])
            not_aligned_temporal_bevs_semantics.append(self.align_gt_bev_semantics(idx)[1])
            if self.cur_idx2 == 70:
                self.padded_align_bev(idx)
            # import pdb; pdb.set_trace()
            # Print the tracked current index and the start index of the current task
            # print(self.cur_idx, self.cur_start_idx)
            self.cur_idx2 += 1
            

        # convert from bev space to token sapce
        temporal_bevs_semantics = torch.stack(temporal_bevs_semantics)  # Convert list to tensor
        not_aligned_temporal_bevs_semantics = torch.stack(not_aligned_temporal_bevs_semantics)  # Convert list to tensor
        # Plot the aligned BEV features for debugging
        if self.cur_idx2 == 180:
            # import pdb; pdb.set_trace()
            self.plot_fused_bev_semantics(temporal_bevs_semantics, title_prefix="Aligned GT BEV Semantics")
            # Plot the not aligned BEV features for debugging
            self.plot_fused_bev_semantics(not_aligned_temporal_bevs_semantics, title_prefix="Not Aligned GT BEV Semantics")

            self.plot_temporal_fused_bev_semantics(temporal_bevs_semantics, title_prefix="Temporal Fused BEV Semantics")

            self.plot_temporal_fused_bev_semantics(not_aligned_temporal_bevs_semantics, title_prefix="Not_aligned Temporal Fused BEV Semantics")

            # import pdb; pdb.set_trace()

        # temporal_bevs = temporal_bevs.reshape(B, self.T, 256, 264)  # Shape: (B, T=10, 16, 16, 264)
        # B, T, L, C = temporal_bevs.shape  # B: batch size, L: feature length, T: time steps, C: channels

        # # Update the delta_ego_pose_cache
        # self.delta_ego_pose_cache = data['delta_ego_pos'][-self.T:,:].detach()  # Update the delta_ego_motion for next step temporal fusion
        # # TODO: Accumulated Memory is sorted by detaching bev_cache
        # # TODO: however, is this the right way to do it?
        # # Update the BEV cache with the fuse_feature BEV
        # self.bev_cache = (fuse_feature[-self.T:,:,:]).detach()  # Shape: (B + T, 256, 264) 

        # # Flatten the BEV features for self-attention
        # flattened_bev = temporal_bevs.reshape(B, T, L * C)  # Shape: (B, T, L * C)
        # temporal_bevs = temporal_bevs.detach()
        # # Compute self-attention scores
        # attention_scores = torch.bmm(flattened_bev, flattened_bev.transpose(1, 2))  # Shape: (B, T, T)
        # attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores along the time dimension

        # # Apply attention weights to the BEV cache
        # fused_bev = torch.bmm(attention_scores, flattened_bev)  # Shape: (B, T, L * C)

        # # Reduce the temporal dimension to get a single fused feature
        # fused_bev = fused_bev.mean(dim=1, keepdim=True)  # Shape: (B, 1, L * C)

        # # Reshape back to (B, L, 1, C)
        # fused_bev = fused_bev.view(B, L, 1, C)  # Shape: (B, L, 1, C)

        # # Reshape to (B, 256, 264) for final output
        # final_features = fused_bev.squeeze(2)  # Shape: (B, L, C)

        # self.bev_cache = self.bev_cache.detach()
        # # self.last_delta_xy = self.delta_xy.detach()  # Detach the last_ego_xy for next step
        # return final_features