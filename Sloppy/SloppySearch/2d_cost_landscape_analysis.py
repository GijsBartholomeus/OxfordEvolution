# Compute 2D cost landscape over a grid of parameter combinations
# Focus on two key parameters: k1_aa_over_CT (index 0) and k4 (index 3)
# -------------------------------------------------------------------------

import time

# Choose two parameters to vary (indices into the 11-parameter vector)
param_idx_1 = 0  # k1_aa_over_CT
param_idx_2 = 3  # k4

# Get parameter names for labels
param_names = ['k1_aa_over_CT', 'k2', 'k3_CT', 'k4', 'k4prime', 'k5_minusP', 
               'k6', 'k7', 'k8_minusP', 'k9', 'CT']

# Define grid ranges as percentages around the optimized theta values
n_grid = 25  # Grid resolution (25x25 = 625 points)
grid_range = 0.5  # Vary ±50% around theta values

with torch.no_grad():
    # Extract the two parameters from optimized theta
    p1_center = theta_new[param_idx_1].item()
    p2_center = theta_new[param_idx_2].item()
    
    # Create grid ranges
    p1_min = max(0, p1_center * (1 - grid_range))
    p1_max = p1_center * (1 + grid_range)
    p2_min = max(0, p2_center * (1 - grid_range))
    p2_max = p2_center * (1 + grid_range)
    
    p1_vals = np.linspace(p1_min, p1_max, n_grid)
    p2_vals = np.linspace(p2_min, p2_max, n_grid)
    P1_mesh, P2_mesh = np.meshgrid(p1_vals, p2_vals)
    
    # Initialize cost grid
    Cost_mesh = np.zeros((n_grid, n_grid))
    
    print(f"Computing 2D cost landscape over {n_grid}x{n_grid} = {n_grid**2} points")
    print(f"Parameter 1: {param_names[param_idx_1]} in [{p1_min:.3e}, {p1_max:.3e}]")
    print(f"Parameter 2: {param_names[param_idx_2]} in [{p2_min:.3e}, {p2_max:.3e}]")
    print("=" * 70)
    
    start_time = time.time()
    
    # Compute cost at each grid point
    for i in range(n_grid):
        for j in range(n_grid):
            # Create parameter vector based on theta_new
            w_test = theta_new.clone()
            w_test[param_idx_1] = P1_mesh[i, j]
            w_test[param_idx_2] = P2_mesh[i, j]
            w_test = w_test.clamp(min=0.0)
            
            # Compute cost
            C_val, _ = trajectory_cost_and_grad_w(w_test, Xavg)
            Cost_mesh[i, j] = C_val.item()
        
        # Progress update
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / n_grid
            eta = elapsed / progress - elapsed
            print(f"  Progress: {i+1:2d}/{n_grid} rows | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Grid computation complete! Total time: {total_time:.1f}s")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Filled contour plot
contour = ax1.contourf(P1_mesh, P2_mesh, Cost_mesh, levels=25, cmap='viridis')
ax1.plot(p1_center, p2_center, 'r*', markersize=20, 
         markeredgecolor='white', markeredgewidth=2, label='Optimized θ')
ax1.plot(v1[param_idx_1].item(), v1[param_idx_2].item(), 'cs', 
         markersize=12, markeredgecolor='white', markeredgewidth=1.5, label='v1')
ax1.plot(v2[param_idx_1].item(), v2[param_idx_2].item(), 'ms',
         markersize=12, markeredgecolor='white', markeredgewidth=1.5, label='v2')
ax1.set_xlabel(param_names[param_idx_1], fontsize=12)
ax1.set_ylabel(param_names[param_idx_2], fontsize=12)
ax1.set_title('Cost Landscape (2-Parameter Slice)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
cbar1 = plt.colorbar(contour, ax=ax1)
cbar1.set_label('Cost C(w)', fontsize=11)

# Right: Contour lines with gradient field
levels = np.linspace(Cost_mesh.min(), Cost_mesh.max(), 15)
contour2 = ax2.contour(P1_mesh, P2_mesh, Cost_mesh, levels=levels, 
                       colors='black', linewidths=1, alpha=0.6)
ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.2e')
contourf2 = ax2.contourf(P1_mesh, P2_mesh, Cost_mesh, levels=levels, 
                         cmap='RdYlBu_r', alpha=0.7)
ax2.plot(p1_center, p2_center, 'r*', markersize=20,
         markeredgecolor='white', markeredgewidth=2, label='Optimized θ')
ax2.plot(v1[param_idx_1].item(), v1[param_idx_2].item(), 'cs',
         markersize=12, markeredgecolor='white', markeredgewidth=1.5, label='v1')
ax2.plot(v2[param_idx_1].item(), v2[param_idx_2].item(), 'ms',
         markersize=12, markeredgecolor='white', markeredgewidth=1.5, label='v2')
ax2.set_xlabel(param_names[param_idx_1], fontsize=12)
ax2.set_ylabel(param_names[param_idx_2], fontsize=12)
ax2.set_title('Cost Contours with Levels', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(contourf2, ax=ax2)
cbar2.set_label('Cost C(w)', fontsize=11)

plt.suptitle(f'2D Cost Landscape: {param_names[param_idx_1]} vs {param_names[param_idx_2]}',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nCost at optimized θ: {Cost_mesh[n_grid//2, n_grid//2]:.6e}")
print(f"Min cost on grid:    {Cost_mesh.min():.6e}")
print(f"Max cost on grid:    {Cost_mesh.max():.6e}")