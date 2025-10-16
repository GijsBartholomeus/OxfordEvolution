import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================
# Helper functions
# ============================================

def y_theta(t, thetas):
    """Model output y(t) = (1/N) Σ exp(-θ_i t)."""
    return np.mean(np.exp(-np.outer(thetas, t)), axis=0)

def sample_thetas(N, num_samples, theta_min=1e-2, theta_max=1e2):
    """Sample θ from ρ(θ) ∝ 1/θ, i.e., uniform in log-space."""
    log_thetas = np.random.uniform(np.log10(theta_min), np.log10(theta_max), size=(num_samples, N))
    return 10 ** log_thetas


# ============================================
# (a) Model manifold for N=2 on grid
# ============================================
t_values = np.array([1/3, 1, 3])
theta_vals = np.logspace(-2, 2, 80)
theta1, theta2 = np.meshgrid(theta_vals, theta_vals)

Y = np.array([y_theta(t_values, [t1, t2]) for t1, t2 in zip(theta1.ravel(), theta2.ravel())])
y1, y2, y3 = Y[:,0], Y[:,1], Y[:,2]

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(y1, y2, y3, cmap='viridis', alpha=0.8, linewidth=0)
ax.set_xlabel("y(1/3)")
ax.set_ylabel("y(1)")
ax.set_zlabel("y(3)")
ax.set_title("Model Manifold for N=2 (Grid Sampling)")
plt.tight_layout()
plt.show()

# The three cusp points correspond roughly to θ1, θ2 -> {0, ∞}, giving
# constant-like, single-exponential models (simpler, 1-parameter limits).


# ============================================
# (b) Random sampling (ρ(θ) ∝ 1/θ) for N=2 and N=7
# ============================================

num_samples = 5000
for N, color in [(2, "teal"), (7, "darkorange")]:
    theta_samples = sample_thetas(N, num_samples)
    Yrand = np.array([y_theta(t_values, thetas) for thetas in theta_samples])

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Yrand[:,0], Yrand[:,1], Yrand[:,2], s=2, alpha=0.4, c=color)
    ax.set_xlabel("y(1/3)")
    ax.set_ylabel("y(1)")
    ax.set_zlabel("y(3)")
    ax.set_title(f"Random Model Manifold for N={N} (ρ(θ)∝1/θ)")
    plt.tight_layout()
    plt.show()


# ============================================
# (d) PCA test for N=2, trajectories y(t) over t ∈ (0,10)
# ============================================

t_full = np.linspace(0, 10, 20)
N = 2
theta_samples = sample_thetas(N, num_samples)
Y_full = np.array([y_theta(t_full, thetas) for thetas in theta_samples])

# PCA projection
pca = PCA(n_components=3)
Y_pca = pca.fit_transform(Y_full)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_pca[:,0], Y_pca[:,1], Y_pca[:,2], s=2, alpha=0.4, c='slateblue')
ax.set_title("PCA of N=2 trajectories (first 3 PCs)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Set equal aspect ratio for 3D plot
max_range = np.array([Y_pca[:,0].max()-Y_pca[:,0].min(), 
                      Y_pca[:,1].max()-Y_pca[:,1].min(),
                      Y_pca[:,2].max()-Y_pca[:,2].min()]).max() / 2.0
mid_x = (Y_pca[:,0].max()+Y_pca[:,0].min()) * 0.5
mid_y = (Y_pca[:,1].max()+Y_pca[:,1].min()) * 0.5
mid_z = (Y_pca[:,2].max()+Y_pca[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()

print("Explained variance ratios (N=2):", pca.explained_variance_ratio_[:3])


# ============================================
# (e) PCA for N=7, see “thinning” hyperribbon
# ============================================

N = 7
theta_samples = sample_thetas(N, num_samples)
Y_full = np.array([y_theta(t_full, thetas) for thetas in theta_samples])

pca = PCA(n_components=6)
Y_pca = pca.fit_transform(Y_full)

# First 3 PCs
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Y_pca[:,0], Y_pca[:,1], Y_pca[:,2], s=2, alpha=0.4, c='crimson')
ax.set_title("PCA Projection (N=7, first 3 PCs)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Set equal aspect ratio for 3D plot
max_range = np.array([Y_pca[:,0].max()-Y_pca[:,0].min(), 
                      Y_pca[:,1].max()-Y_pca[:,1].min(),
                      Y_pca[:,2].max()-Y_pca[:,2].min()]).max() / 2.0
mid_x = (Y_pca[:,0].max()+Y_pca[:,0].min()) * 0.5
mid_y = (Y_pca[:,1].max()+Y_pca[:,1].min()) * 0.5
mid_z = (Y_pca[:,2].max()+Y_pca[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()

# Next 3 PCs
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Y_pca[:,3], Y_pca[:,4], Y_pca[:,5], s=2, alpha=0.4, c='darkgreen')
ax.set_title("PCA Projection (N=7, next 3 PCs)")
ax.set_xlabel("PC4")
ax.set_ylabel("PC5")
ax.set_zlabel("PC6")

# Set equal aspect ratio for 3D plot
max_range = np.array([Y_pca[:,3].max()-Y_pca[:,3].min(), 
                      Y_pca[:,4].max()-Y_pca[:,4].min(),
                      Y_pca[:,5].max()-Y_pca[:,5].min()]).max() / 2.0
mid_x = (Y_pca[:,3].max()+Y_pca[:,3].min()) * 0.5
mid_y = (Y_pca[:,4].max()+Y_pca[:,4].min()) * 0.5
mid_z = (Y_pca[:,5].max()+Y_pca[:,5].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()

print("Explained variance ratios (N=7):", pca.explained_variance_ratio_)
