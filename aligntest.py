import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import math # Import math for converting degrees to radians

# Assume the TemporalFusion class definition from your previous code is available
# (including the __init__ and plot_bev_features methods).
# We will only redefine the align_bev method here with the corrected logic for the test.
# In a real scenario, you'd use the full class.

class TemporalFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Minimal init for testing align_bev
        self.bev_cache = None
        self.delta_ego_motion = None
        self.last_yaw = None
        # Dummy MLP - not used in align_bev test
        self.mlp = nn.Identity() # Use Identity so it does nothing

    # Keep the plot_bev_features method from before
    def plot_bev_features(self, bev_features, title_prefix="BEV Feature 10 Time Step"):
        '''
        Plots all time steps' HxW BEV features in the same plot for the first batch.

        Input:
        bev_features: torch.Tensor of shape (B, H, W, T, C)
        title_prefix: Prefix for the plot title.
        '''
        if 'plt' not in globals():
             print("Matplotlib not available. Skipping plotting.")
             return

        B, H, W, T, C = bev_features.shape

        if B == 0:
             print("Batch size is 0, skipping plot.")
             return

        # Create a new folder under the current folder
        output_folder = os.path.join(os.getcwd(), "bev_feature_plots")
        os.makedirs(output_folder, exist_ok=True)

        # Create a figure
        fig, axes = plt.subplots(1, T, figsize=(T * 4, 4))
        if T == 1:
            axes = [axes]

        # Loop through each time step
        for t in range(T):
            # Take the first batch (B=0) and the first channel (C=0) for visualization
            feature_map = bev_features[0, :, :, t, 0].cpu().detach().numpy()

            ax = axes[t]
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"{title_prefix} {t}")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            fig.colorbar(im, ax=ax)

        plot_path = os.path.join(output_folder, f"{title_prefix}.png")
        plt.savefig(plot_path)
        plt.close(fig) # Close the figure

    # Redefine align_bev with corrected theta logic for this test
    def align_bev(self, raw_bev_cache, delta_ego_motion):
        '''
        Align the egocentric BEV image at all T steps by aligning the first T-1 channels to the Tth channel.
        Fill blank areas with 0s.

        Input:
        raw_bev_cache: torch.size(B, H, W, T, C)
        delta_ego_motion: torch.size(B, 3) [delta_x, delta_y, delta_yaw] (unit in meters)
                          This delta is the motion *from the previous frame to the current frame*.

        Output:
        torch.size(B, H, W, T, C) with the first T-1 channels aligned to the Tth channel.
        '''
        B, H, W, T, C = raw_bev_cache.shape

        # Define the spatial range and resolution based on your assumptions
        map_range = 200.0 # meters
        # delta_x/y in [-1, 1] grid units
        delta_x_norm = delta_ego_motion[:, 0] / (map_range / 2)
        delta_y_norm = delta_ego_motion[:, 1] / (map_range / 2)
        delta_yaw = delta_ego_motion[:, 2]  # Rotation in radians

        # Compute relative motion for all batches
        dx_norm = delta_x_norm.view(B, 1, 1)  # Shape: (B, 1, 1)
        dy_norm = delta_y_norm.view(B, 1, 1)  # Shape: (B, 1, 1)
        dyaw = delta_yaw.view(B, 1, 1)  # Shape: (B, 1, 1)

        # Create the affine transformation matrix (target -> source)
        # This matrix maps coordinates in the target grid [-1, 1] to coordinates
        # in the source image's [-1, 1] space.
        # We need the inverse of the ego-motion transformation.
        # Inverse Rotation matrix: [ cos(dyaw)  sin(dyaw) ]
        #                         [ -sin(dyaw) cos(dyaw) ]
        # Inverse Translation vector (in [-1, 1] grid space): -(R_inv @ T_norm)
        # T_norm = [delta_x_norm, delta_y_norm]^T

        theta = torch.zeros((B, 2, 3), device=raw_bev_cache.device)  # Shape: (B, 2, 3)

        cos_dyaw = torch.cos(dyaw.squeeze())
        sin_dyaw = torch.sin(dyaw.squeeze())

        # Rotation part (inverse rotation)
        theta[:, 0, 0] = cos_dyaw
        theta[:, 0, 1] = sin_dyaw
        theta[:, 1, 0] = -sin_dyaw
        theta[:, 1, 1] = cos_dyaw

        # Translation part (inverse translation in the rotated frame)
        # This maps target coords (x', y') to source coords (x, y)
        # x = cos(dyaw)*(x' - dx_norm) + sin(dyaw)*(y' - dy_norm)
        # y = -sin(dyaw)*(x' - dx_norm) + cos(dyaw)*(y' - dy_norm)
        # The translation component in the matrix is evaluated at the origin (x'=0, y'=0)
        # tx = -cos(dyaw)*dx_norm - sin(dyaw)*dy_norm
        # ty = sin(dyaw)*dx_norm - cos(dyaw)*dy_norm
        theta[:, 0, 2] = -(cos_dyaw * dx_norm.squeeze() + sin_dyaw * dy_norm.squeeze())
        theta[:, 1, 2] = (sin_dyaw * dx_norm.squeeze() - cos_dyaw * dy_norm.squeeze())

        # Reshape raw_bev_cache to (B * T, C, H, W) for grid_sample
        # We only transform the first T-1 frames
        bev_cache_to_transform = raw_bev_cache[:, :, :, :T-1, :].permute(0, 3, 4, 1, 2).contiguous().view(B * (T-1), C, H, W)

        # Generate sampling grid for affine transformation
        # Repeat the theta matrix for each of the B*(T-1) images to transform
        theta_repeated = theta.repeat(T - 1, 1, 1) # Repeats batch-wise

        # Grid shape: (N, H_out, W_out, 2). N = B * (T-1). Output size is (C, H, W).
        grid = F.affine_grid(theta_repeated, size=(B * (T - 1), C, H, W), align_corners=False)

        # Apply affine transformation to the first T-1 channels
        transformed_bev = F.grid_sample(bev_cache_to_transform, grid, padding_mode='zeros', align_corners=False) # Shape (B*(T-1), C, H, W)

        # Reshape back to (B, T-1, C, H, W) then permute to (B, H, W, T-1, C)
        transformed_bev = transformed_bev.view(B, T - 1, C, H, W).permute(0, 3, 4, 1, 2) # Shape: (B, H, W, T-1, C)

        # Reshape the current frame's feature (Tth channel) to (B, H, W, 1, C)
        current_bev = raw_bev_cache[:, :, :, T-1:T, :] # Already (B, H, W, 1, C)

        # Combine the transformed first T-1 channels with the unaltered Tth channel
        aligned_bev = torch.cat((transformed_bev, current_bev), dim=3) # Shape: (B, H, W, T, C)

        return aligned_bev # torch.Size([B, H, W, T, C])


def test_align_bev():
    print("Testing align_bev function with synthetic data...")

    # Test parameters
    B = 1         # Batch size
    H, W = 16, 16 # Spatial dimensions
    T = 2         # Number of time steps for the test (past, current)
    C = 4         # Feature channels (only visualize the first)
    map_range = 200.0 # BEV map range in meters

    # Create a dummy model instance to use the align_bev and plot methods
    class DummyConfig: pass
    model = TemporalFusion(DummyConfig())

    # --- 1. Create Synthetic BEV Feature Maps ---
    past_bev = torch.zeros(B, H, W, C)
    current_bev = torch.zeros(B, H, W, C)

    # Define feature blob properties (center pixel)
    blob_center_pixel_past = (int(H * 0.75), int(W * 0.25)) # Example: bottom-left quadrant
    blob_size = 3

    # Place a blob in the past BEV
    ph, pw = blob_center_pixel_past
    past_bev[:, ph-blob_size//2 : ph+blob_size//2 + 1,
             pw-blob_size//2 : pw+blob_size//2 + 1, :] = 1.0

    # Define Ego-motion from Past to Current frame (in meters and radians)
    delta_x_meters = 10.0 # Move 10 meters forward (assuming x is forward)
    delta_y_meters = -5.0 # Move 5 meters right (assuming y is left/right, positive is left)
    delta_yaw_degrees = 15.0 # Rotate 15 degrees counter-clockwise
    delta_yaw_radians = math.radians(delta_yaw_degrees)

    delta_ego_motion = torch.tensor([[delta_x_meters, delta_y_meters, delta_yaw_radians]], dtype=torch.float32) # Shape (B, 3)


    # --- Calculate the expected position of the blob in the current BEV ---
    # Assuming BEV grid center (pixel H/2-0.5, W/2-0.5) corresponds to world (0,0)
    # Pixel coordinates to world coordinates
    center_pixel = (H/2 - 0.5, W/2 - 0.5)
    bev_resolution = map_range / H # meters per pixel

    past_wx = (blob_center_pixel_past[1] - center_pixel[1]) * bev_resolution # Note: width is x, height is y in imshow
    past_wy = (blob_center_pixel_past[0] - center_pixel[0]) * bev_resolution

    # Apply ego-motion (rotation then translation) to the world point
    # The ego-motion moves the vehicle from the past frame's origin to the current frame's origin.
    # A static point in the world moves *relative* to the vehicle by the *inverse* ego-motion.
    # But the problem says delta_ego_motion is from previous to current.
    # Let's use the standard transformation:
    # A point (wx, wy) in the previous frame's coordinates ends up at (wx', wy') in the current frame's coordinates if it moves *with the vehicle*.
    # No, that's not right. A static point (wx, wy) in the world at t=0 (past frame) is at (wx, wy) in the current frame (t=1) *world* coords.
    # Its position *relative to the vehicle* changes.
    # If vehicle goes from (0,0) yaw 0 to (dx, dy) yaw dyaw, a world point (px, py) in frame 0 vehicle coords
    # is at R(dyaw)*(px, py) + (dx, dy) in frame 1 vehicle coords.
    # Let's stick to the definition: delta_ego_motion is the movement of the vehicle from past to current.
    # So a static point at (wx, wy) in the *past vehicle frame* should be at (wx', wy') in the *current vehicle frame* where:
    # (wx', wy') = Rotate(delta_yaw) * (wx, wy) + (delta_x_meters, delta_y_meters)
    # This is the transformation needed to map a point from the past frame *into* the current frame.
    # This is also the matrix needed for `affine_grid` to map target (current) to source (past).

    cos_dyaw = math.cos(delta_yaw_radians)
    sin_dyaw = math.sin(delta_yaw_radians)

    current_wx = past_wx * cos_dyaw - past_wy * sin_dyaw + delta_x_meters
    current_wy = past_wx * sin_dyaw + past_wy * cos_dyaw + delta_y_meters

    # World coordinates back to pixel coordinates
    current_px = int(current_wx / bev_resolution + center_pixel[1])
    current_py = int(current_wy / bev_resolution + center_pixel[0])

    blob_center_pixel_current = (current_py, current_px)
    print(f"Past blob center pixel: {blob_center_pixel_past}")
    print(f"Expected current blob center pixel: {blob_center_pixel_current}")


    # Place a blob in the current BEV at the calculated position
    # Need to handle boundary cases if the blob moves off the grid
    py_curr, px_curr = blob_center_pixel_current
    slice_y = slice(max(0, py_curr - blob_size//2), min(H, py_curr + blob_size//2 + 1))
    slice_x = slice(max(0, px_curr - blob_size//2), min(W, px_curr + blob_size//2 + 1))

    current_bev[:, slice_y, slice_x, :] = 1.0


    # Combine into the raw_bev_cache format (B, H, W, T, C)
    raw_bev_cache = torch.stack([past_bev, current_bev], dim=3) # Shape (B, H, W, 2, C)

    # Ensure data is on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    raw_bev_cache = raw_bev_cache.to(device)
    delta_ego_motion = delta_ego_motion.to(device)

    # --- 2. Call align_bev ---
    # We need T=2 in the input cache, the first time step (index 0) is the past, the last (index 1) is the current.
    # align_bev expects (B, H, W, T, C) where T >= 2
    aligned_bev = model.align_bev(raw_bev_cache, delta_ego_motion)

    # --- 3. Visualize Results ---
    # The output aligned_bev is (B, H, W, T, C)
    # We want to see:
    # 1. The original past BEV (raw_bev_cache[:, :, :, 0, :])
    # 2. The original current BEV (raw_bev_cache[:, :, :, 1, :])
    # 3. The aligned past BEV (aligned_bev[:, :, :, 0, :])

    # Reshape for plotting (add a T=1 dimension for consistency with plot_bev_features input)
    original_past_for_plot = raw_bev_cache[:, :, :, [0], :] # Shape (B, H, W, 1, C)
    original_current_for_plot = raw_bev_cache[:, :, :, [1], :] # Shape (B, H, W, 1, C)
    aligned_past_for_plot = aligned_bev[:, :, :, [0], :] # Shape (B, H, W, 1, C)
    aligned_current_for_plot = aligned_bev[:, :, :, [1], :] # Should be identical to original current

    # Check if aligned_current is indeed the same as original_current
    # print("Checking if aligned_current is same as original_current:",
    #       torch.allclose(aligned_current_for_plot, original_current_for_plot))


    # Plotting
    model.plot_bev_features(original_past_for_plot, title_prefix="Original_Past_BEV")
    model.plot_bev_features(original_current_for_plot, title_prefix="Original_Current_BEV")
    model.plot_bev_features(aligned_past_for_plot, title_prefix="Aligned_Past_BEV")
    # model.plot_bev_features(aligned_current_for_plot, title_prefix="Aligned_Current_BEV (Should match Original)")


    print("\nTest complete. Check the 'bev_feature_plots' folder for images.")
    print("The 'Aligned_Past_BEV' image should show the blob roughly in the same")
    print("location as the blob in the 'Original_Current_BEV' image.")

if __name__ == '__main__':
    test_align_bev()