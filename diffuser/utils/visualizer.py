import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_with_yaw(traj, arrow_length=0.5):
    """
    traj: torch.Tensor of shape (T, 3) where columns = (x, y, rad)
    arrow_length: length of yaw arrows in plot units
    """
    x = traj[:, 0].cpu().numpy()
    y = traj[:, 1].cpu().numpy()
    yaw = traj[:, 2].cpu().numpy()  # in radians

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', markersize=3, label="Trajectory")
    plt.scatter(x[0], y[0], color='green', label="Start", zorder=5)
    plt.scatter(x[-1], y[-1], color='red', label="End", zorder=5)

    # Compute arrow components
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    # Draw arrows
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)

    plt.axis('equal')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Trajectory with Yaw")
    plt.legend()
    plt.grid(True)
    plt.show()