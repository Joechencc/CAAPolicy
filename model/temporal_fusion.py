import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TemporalFusion(nn.Module):  # Added nn.Module inheritance
    def __init__(self, cfg):
        super().__init__()
        # self.cfg = cfg
        self.bev_cache = None # bev_cache: torch.size(B, 256, T, 264), e.g.: T=10
        self.last_delta_xy = None # Initialize as empty tensor
        self.last_yaw = None  # Initialize last_yaw as None

        # Define the MLP layer to reduce feature dimensions
        self.mlp = nn.Sequential(
            nn.Linear(256 * 2, 256),  # Input: concatenated avg and max pool features
            nn.ReLU(),
            nn.Linear(256, 256)  # Output: reduced feature dimension
        )

    def forward(self, fuse_feature, delta_xy, delta_yaw, restart = True): # cur_yaw
        '''
        Input:
        fuse_feature: torch.size(B, 256, 264)
        delta_ego_motion: [delta_x, delta_y, delta_yaw] (unit in meters)
        
        Output:
        Updated BEV cache with all T steps projected to the current ego-centric BEV range.
        '''
        # TODO: calculate delta yaw by cur_yaw - self.last_yaw
        # if self.last_delta_xy is None:
        #     self.last_delta_xy = torch.zeros(fuse_feature.shape[0], 2, device=fuse_feature.device)
        # if self.last_yaw is None:
        #     delta_yaw = torch.zeros(fuse_feature.shape[0], 1, device=fuse_feature.device)
        # else:
        #     delta_yaw = cur_yaw - self.last_yaw
        # import pdb; pdb.set_trace()
        delta_ego_motion = torch.cat((delta_xy, delta_yaw), dim=1) # Shape: (B, 3)

        # delta_ego_motion = torch.cat((self.last_delta_xy, delta_yaw), dim=1) # Shape: (B, 3)
        # self.last_yaw = cur_yaw.detach()  # Update last_yaw for the next step

        # Initialize the BEV cache by repeat the first fuse_feature 10 times, if it's empty
        # Or reinitialize the BEV cache if it is the first frame of the task
        if self.bev_cache is None or restart:
            self.bev_cache = fuse_feature.unsqueeze(2).repeat(1, 1, 10, 1)  # Shape: (B, 256, T=10, 264)
        B, L, T, C = self.bev_cache.shape  # B: batch size, L: feature length, T: time steps, C: channels
        # Reshape the new fuse_feature to (B, 256, 1, 264)
        new_feature = fuse_feature.unsqueeze(2)  # Shape: (B, 256, 1, 264)

        # Remove the oldest time step (T=0)
        self.bev_cache = self.bev_cache[:, :, 1:, :]  # Shape: (B, 256, T-1, 264)

        # Add the new fuse_feature as the 10th time step
        self.bev_cache = torch.cat((self.bev_cache, new_feature), dim=2)  # Shape: (B, 256, T=10, 264)
        # Crop and project all T steps in bev_cache to the current ego-centric BEV range
        aligned_bev = self.align_bev(self.bev_cache, delta_ego_motion)
        # plot the BEV features for debugging
        # self.plot_bev_features(aligned_bev)

        self.delta_ego_motion = delta_ego_motion.detach()  # Update the delta_ego_motion for next step temporal fusion
        # TODO: Accumulated Memory is sorted by detaching bev_cache
        # TODO: however, is this the right way to do it?
        # Update the BEV cache with the aligned BEV
        self.bev_cache = aligned_bev.view(B, L, T, C) 

        # Calculate the self attention score between T bev caches
        flattened_bev = self.bev_cache.permute(0, 2, 1, 3).reshape(B, T, L * C)  # Shape: (B, T, L * C)
        
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

    def align_bev(self, raw_bev_cache, delta_ego_motion):
        '''
        Align the egocentric BEV image at all T steps by aligning the first T-1 channels to the Tth channel.
        Fill blank areas with 0s.
        
        Input:
        raw_bev_cache: torch.size(B, 16, 16, T, 264)
        delta_ego_motion: torch.size(B, 3) [delta_x, delta_y, delta_yaw] (unit in meters)
        
        Output:
        torch.size(B, 16, 16, T, 264) with the first T-1 channels aligned to the Tth channel.
        '''
        B, _, T, C = raw_bev_cache.shape
        raw_bev_cache = raw_bev_cache.view(B, 16, 16, T, C)  # Ensure the shape is (B, H, W, T, C)
        _, H, W, _, _ = raw_bev_cache.shape  # B: batch size, H: height, W: width, T: time steps, C: channels
        bev_resolution = 200 / 16  # Each grid represents 12.5 meters (200m downsampled to 16 grids)

        # Convert delta_ego_motion to grid units
        delta_x = delta_ego_motion[:, 0] / bev_resolution  # Convert delta_x to grid units
        delta_y = delta_ego_motion[:, 1] / bev_resolution  # Convert delta_y to grid units
        delta_yaw = delta_ego_motion[:, 2]  # Rotation in radians

        # Compute relative motion for all batches
        dx = delta_x.view(B, 1, 1)  # Shape: (B, 1, 1)
        dy = delta_y.view(B, 1, 1)  # Shape: (B, 1, 1)
        dyaw = delta_yaw.view(B, 1, 1)  # Shape: (B, 1, 1)

        # Create the affine transformation matrix for all T-1 channels
        theta = torch.zeros((B, 2, 3), device=raw_bev_cache.device)  # Shape: (B, 2, 3)
        cos_dyaw = torch.cos(delta_yaw.squeeze())
        sin_dyaw = torch.sin(delta_yaw.squeeze())

        # Rotation part (inverse rotation)
        theta[:, 0, 0] = cos_dyaw
        theta[:, 0, 1] = sin_dyaw # Note: sin( -dyaw) = -sin(dyaw)
        theta[:, 1, 0] = -sin_dyaw # Note: sin(-dyaw) = -sin(dyaw)
        theta[:, 1, 1] = cos_dyaw

        # Translation part (inverse translation in the rotated frame, mapped to [-1, 1] range)
        theta[:, 0, 2] = -(cos_dyaw * delta_x.squeeze() + sin_dyaw * delta_y.squeeze())
        theta[:, 1, 2] = (sin_dyaw * delta_x.squeeze() - cos_dyaw * delta_y.squeeze())

        # Reshape raw_bev_cache to (B * T, C, H, W)
        bev_cache = raw_bev_cache.permute(0, 3, 4, 1, 2).contiguous().view(B * T, C, H, W)

        # Generate sampling grid for affine transformation
        grid = F.affine_grid(theta.repeat(T - 1, 1, 1), size=(B * (T - 1), C, H, W), align_corners=False)  # Shape: (B * (T-1), H, W, 2)

        # Apply affine transformation to the first T-1 channels
        transformed_bev = F.grid_sample(bev_cache[:B * (T - 1)], grid, padding_mode='zeros', align_corners=False)

        # Reshape back to (B, H, W, T-1, C)
        transformed_bev = transformed_bev.view(B, T - 1, C, H, W).permute(0, 3, 4, 1, 2) # torch.Size([1, 16, 16, 9, 264])

        # Combine the transformed first T-1 channels with the unaltered Tth channel
        aligned_bev = torch.cat((transformed_bev, raw_bev_cache[:, :, :, -1:, :]), dim=3)
        return aligned_bev # torch.Size([1, 16, 16, 10, 264])

    def plot_bev_features(self, bev_features, title_prefix="BEV Feature 10 Time Step"):
        '''
        Plots all time steps' 16x16 BEV features in the same plot.
        
        Input:
        bev_features: torch.Tensor of shape (B, 16, 16, T, C)
        title_prefix: Prefix for the plot title.
        '''
        import os
        B, H, W, T, C = bev_features.shape

        # Create a new folder under the current folder
        output_folder = os.path.join(os.getcwd(), "bev_feature_plots")
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # Create a figure
        fig, axes = plt.subplots(1, T, figsize=(T * 4, 4))  # Create T subplots in a single row

        # Loop through each time step
        for t in range(T):
            # TODO: Visualize other channels if needed
            # Take the first batch (B=0) and the first channel (C=0) for visualization
            feature_map = bev_features[0, :, :, t, 0].cpu().detach().numpy()  # Shape: (16, 16)

            # Plot the feature map in the corresponding subplot
            ax = axes[t] if T > 1 else axes  # Handle case where T=1
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"{title_prefix} {t}")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            fig.colorbar(im, ax=ax)

        # Save the plot to the output folder
        plot_path = os.path.join(output_folder, f"{title_prefix}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory


# import torch
# import torch.nn.functional as F
# from torch import nn

# class TemporalFusion(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # Assuming cfg might contain parameters like bev_size, etc.
#         self.bev_height = 16  # Downsampled BEV height
#         self.bev_width = 16   # Downsampled BEV width

#     def forward(self, bev_cache, delta_ego_motion):
#         cropped_bev = self.crop_bev(bev_cache, delta_ego_motion)
#         # Further processing can be added here if needed
#         return cropped_bev

#     def crop_bev(self, raw_bev_cache, delta_ego_motion):
#         B, C, T, _ = raw_bev_cache.shape
#         H, W = self.bev_height, self.bev_width
#         # Reshape BEV cache to (B, C, T, H, W)
#         bev_cache = raw_bev_cache.view(B, C, T, H, W)
        
#         # Compute cumulative deltas for each timestep t
#         if T > 1:
#             reversed_cumsum = torch.flip(delta_ego_motion, dims=[1]).cumsum(dim=1)
#             reversed_cumsum = torch.flip(reversed_cumsum, dims=[1])
#             # Pad with zeros for the current frame (t=T-1)
#             cumulative_deltas = torch.cat([
#                 reversed_cumsum,
#                 torch.zeros((B, 1, 3), device=delta_ego_motion.device)
#             ], dim=1)
#         else:
#             # If T=1, no historical frames, return original
#             cumulative_deltas = torch.zeros((B, 1, 3), device=delta_ego_motion.device)
        
#         # Extract components
#         dx = cumulative_deltas[..., 0]  # (B, T)
#         dy = cumulative_deltas[..., 1]
#         dyaw = cumulative_deltas[..., 2]
        
#         # Compute cos and sin of yaw
#         cos_theta = torch.cos(dyaw)
#         sin_theta = torch.sin(dyaw)
        
#         # Compute translation components after rotation
#         tx = -dx * cos_theta - dy * sin_theta
#         ty = dx * sin_theta - dy * cos_theta
        
#         # Construct affine transformation matrices (B, T, 2, 3)
#         theta = torch.zeros(B, T, 2, 3, device=bev_cache.device)
#         theta[:, :, 0, 0] = cos_theta
#         theta[:, :, 0, 1] = sin_theta
#         theta[:, :, 0, 2] = tx
#         theta[:, :, 1, 0] = -sin_theta
#         theta[:, :, 1, 1] = cos_theta
#         theta[:, :, 1, 2] = ty
        
#         # Reshape theta for affine_grid (B*T, 2, 3)
#         theta = theta.view(B * T, 2, 3)
        
#         # Generate sampling grid
#         grid = F.affine_grid(
#             theta,
#             size=(B * T, C, H, W),
#             align_corners=False
#         )
        
#         # Prepare BEV features for grid_sample (B*T, C, H, W)
#         bev_reshaped = bev_cache.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
#         # Sample using grid
#         sampled = F.grid_sample(
#             bev_reshaped,
#             grid,
#             padding_mode='zeros',
#             align_corners=False
#         )
        
#         # Reshape back to (B, C, T, H, W)
#         sampled = sampled.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        
#         # Flatten spatial dimensions
#         cropped_bev = sampled.view(B, C, T, H * W)
        
#         return cropped_bev