#!/usr/bin/env python3
"""
gibbs_hellinger_viz.py

Visualize a 3-state Gibbs family:
 - Left: curve inside the probability simplex (triangle in 3D)
 - Right: Hellinger embedding psi = 2 * sqrt(p) on the positive octant of sphere (radius 2)

Optional: compute Hellinger distance and sphere geodesic distance between two selected betas.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Parameters ---------------------------------------------------------------
E = np.array([0.0, 1.0, 2.0])     # energies for the 3 states
betas = np.linspace(-5.0, 5.0, 400)  # parameter range

# choose two beta values to compare distances (optional diagnostics)
beta_a = -1.0
beta_b = 2.0

# --- Helper functions --------------------------------------------------------
def probs(beta, E=E):
    """Return Gibbs probabilities for given beta and energy vector E."""
    weights = np.exp(-beta * E - np.max(-beta * E))  # stability trick
    return weights / np.sum(weights)

def hellinger_distance(p, q):
    """Squared Hellinger distance H^2 = 2 - 2 * sum sqrt(p*q). Return H (not squared)."""
    s = np.sum(np.sqrt(p * q))
    H2 = 2.0 - 2.0 * s
    return np.sqrt(max(H2, 0.0))

def sphere_geodesic_distance(psi_a, psi_b, R=2.0):
    """
    Great-circle distance on sphere radius R between psi_a and psi_b.
    psi vectors are on the sphere of radius R.
    geodesic_length = R * arccos( dot(psi_a, psi_b) / (R^2) )
    Numerical clamping applied to the arccos argument.
    """
    denom = (R * R)
    dot = float(np.dot(psi_a, psi_b))
    arg = dot / denom
    arg = np.clip(arg, -1.0, 1.0)
    theta = np.arccos(arg)
    return R * theta

# --- Compute curves ----------------------------------------------------------
p = np.array([probs(b) for b in betas])     # shape (len(betas), 3)
psi = 2.0 * np.sqrt(p)                      # Hellinger embedding, on sphere radius 2

# Optional: distances between two chosen betas
p_a = probs(beta_a)
p_b = probs(beta_b)
psi_a = 2.0 * np.sqrt(p_a)
psi_b = 2.0 * np.sqrt(p_b)

H = hellinger_distance(p_a, p_b)
geo = sphere_geodesic_distance(psi_a, psi_b, R=2.0)

# --- Plotting ---------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# Left: Gibbs curve on simplex triangle
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tri = Poly3DCollection([vertices], alpha=0.15, facecolor="lightgray", edgecolor='k')
ax1.add_collection3d(tri)

# plot Gibbs curve and markers
ax1.plot(p[:, 0], p[:, 1], p[:, 2], lw=2, label='Gibbs curve (p)')
for b_val in [5, 2, 1, 0, -1, -5]:
    px = probs(b_val)
    ax1.scatter(px[0], px[1], px[2], s=50, depthshade=True, label=f'β={b_val}')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)
ax1.set_xlabel('p0')
ax1.set_ylabel('p1')
ax1.set_zlabel('p2')
ax1.set_title('Gibbs curve on probability simplex')
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize='small')

# Right: Hellinger embedding on positive octant of sphere (R=2)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
R = 2.0
# spherical patch (positive octant)
u = np.linspace(0, np.pi/2, 80)
v = np.linspace(0, np.pi/2, 80)
u, v = np.meshgrid(u, v)
Xs = R * np.sin(u) * np.cos(v)
Ys = R * np.sin(u) * np.sin(v)
Zs = R * np.cos(u)
ax2.plot_surface(Xs, Ys, Zs, alpha=0.12, rstride=1, cstride=1, linewidth=0)

# plot embedded Gibbs curve and markers
ax2.plot(psi[:, 0], psi[:, 1], psi[:, 2], lw=2, label='Hellinger-embedded Gibbs curve (ψ)')
for b_val in [5, 2, 1, 0, -1, -5]:
    px = probs(b_val)
    ps = 2.0 * np.sqrt(px)
    ax2.scatter(ps[0], ps[1], ps[2], s=50, depthshade=True, label=f'β={b_val}')

ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)
ax2.set_zlim(0, 2)
ax2.set_xlabel('ψ0 = 2√p0')
ax2.set_ylabel('ψ1 = 2√p1')
ax2.set_zlabel('ψ2 = 2√p2')
ax2.set_title('Hellinger embedding on positive octant of sphere (R=2)')
ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize='small')

plt.tight_layout()

# --- Annotate distances (optional) ------------------------------------------
txt = f"Hellinger(p(β={beta_a}), p(β={beta_b})) = {H:.6f}\n" \
      f"Sphere geodesic length (R=2) = {geo:.6f}"
# place annotation in the figure
fig.text(0.5, 0.02, txt, ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.3))

# --- Save or show -----------------------------------------------------------
# Uncomment one of the following depending on what you want:
plt.show()
# plt.savefig('gibbs_hellinger_viz.png', dpi=300, bbox_inches='tight')

# End of script
