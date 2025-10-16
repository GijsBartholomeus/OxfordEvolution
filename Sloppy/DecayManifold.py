import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define model: average of two decaying exponentials
def y_theta(t, theta1, theta2):
    return 0.5 * (np.exp(-theta1 * t) + np.exp(-theta2 * t))

# Parameter ranges (log spaced)
theta_vals = np.logspace(-2, 2, 100)
T = [1/3, 1, 3]

# Compute outputs for grid of theta1, theta2
Y1, Y2, Y3 = [], [], []
theta1_vals, theta2_vals = [], []
for theta1 in theta_vals:
    for theta2 in theta_vals:
        Y1.append(y_theta(T[0], theta1, theta2))
        Y2.append(y_theta(T[1], theta1, theta2))
        Y3.append(y_theta(T[2], theta1, theta2))
        theta1_vals.append(theta1)
        theta2_vals.append(theta2)

Y1, Y2, Y3 = np.array(Y1), np.array(Y2), np.array(Y3)
theta1_vals, theta2_vals = np.array(theta1_vals), np.array(theta2_vals)

# Identify boundary points (edges of the manifold)
# Use stricter boundary conditions to avoid overlap
theta1_low = (theta1_vals <= np.min(theta_vals) * 1.5) & (theta2_vals > np.min(theta_vals) * 1.5) & (theta2_vals < np.max(theta_vals) / 1.5)
theta1_high = (theta1_vals >= np.max(theta_vals) / 1.5) & (theta2_vals > np.min(theta_vals) * 1.5) & (theta2_vals < np.max(theta_vals) / 1.5)
theta2_low = (theta2_vals <= np.min(theta_vals) * 1.5) & (theta1_vals > np.min(theta_vals) * 1.5) & (theta1_vals < np.max(theta_vals) / 1.5)
theta2_high = (theta2_vals >= np.max(theta_vals) / 1.5) & (theta1_vals > np.min(theta_vals) * 1.5) & (theta1_vals < np.max(theta_vals) / 1.5)

# Create masks for different boundary regions
boundary_mask = theta1_low | theta1_high | theta2_low | theta2_high
interior_mask = ~boundary_mask

# 3D scatter plot of the model manifold projection
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot interior points in original color scheme
ax.scatter(Y1[interior_mask], Y2[interior_mask], Y3[interior_mask], 
          s=5, c=np.log10(Y2[interior_mask]), cmap='viridis', alpha=0.6, label='Interior')

# Plot boundary points in different colors
ax.scatter(Y1[theta1_low], Y2[theta1_low], Y3[theta1_low], 
          s=10, c='red', alpha=0.8, label='θ₁→0')
ax.scatter(Y1[theta1_high], Y2[theta1_high], Y3[theta1_high], 
          s=10, c='orange', alpha=0.8, label='θ₁→∞')
ax.scatter(Y1[theta2_low], Y2[theta2_low], Y3[theta2_low], 
          s=10, c='blue', alpha=0.8, label='θ₂→0')
ax.scatter(Y1[theta2_high], Y2[theta2_high], Y3[theta2_high], 
          s=10, c='purple', alpha=0.8, label='θ₂→∞')

# Add arrows pointing to the extreme boundary points
# Use actual boundary points for better visualization
if np.any(theta1_low):
    # theta1 -> 0: Find a representative point and show direction
    idx = np.where(theta1_low)[0][len(np.where(theta1_low)[0])//2]  # middle point
    arrow_start = [Y1[idx] - 0.02, Y2[idx] - 0.02, Y3[idx] - 0.02]
    arrow_end = [Y1[idx], Y2[idx], Y3[idx]]
    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
              color='red', arrow_length_ratio=0.3, linewidth=3, label='θ₁→0 direction')

if np.any(theta1_high):
    # theta1 -> inf: Find a representative point and show direction  
    idx = np.where(theta1_high)[0][len(np.where(theta1_high)[0])//2]
    arrow_start = [Y1[idx] + 0.02, Y2[idx] + 0.02, Y3[idx] + 0.02]
    arrow_end = [Y1[idx], Y2[idx], Y3[idx]]
    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
              color='orange', arrow_length_ratio=0.3, linewidth=3, label='θ₁→∞ direction')

if np.any(theta2_low):
    # theta2 -> 0: Find a representative point and show direction
    idx = np.where(theta2_low)[0][len(np.where(theta2_low)[0])//2]
    arrow_start = [Y1[idx] - 0.02, Y2[idx] - 0.02, Y3[idx] - 0.02]
    arrow_end = [Y1[idx], Y2[idx], Y3[idx]]
    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
              color='blue', arrow_length_ratio=0.3, linewidth=3, label='θ₂→0 direction')

if np.any(theta2_high):
    # theta2 -> inf: Find a representative point and show direction
    idx = np.where(theta2_high)[0][len(np.where(theta2_high)[0])//2]
    arrow_start = [Y1[idx] + 0.02, Y2[idx] + 0.02, Y3[idx] + 0.02]
    arrow_end = [Y1[idx], Y2[idx], Y3[idx]]
    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
              color='purple', arrow_length_ratio=0.3, linewidth=3, label='θ₂→∞ direction')

ax.set_xlabel('y(1/3)')
ax.set_ylabel('y(1)')
ax.set_zlabel('y(3)')
ax.set_title('Model manifold: N=2 exponentials with boundary highlighting\nProjection into (y(1/3), y(1), y(3))')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
