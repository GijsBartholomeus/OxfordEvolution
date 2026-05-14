import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T = np.array([1/3, 1.0, 3.0])

def predict(th1, th2):
    th1 = np.atleast_1d(np.asarray(th1, float))
    th2 = np.atleast_1d(np.asarray(th2, float))
    return np.exp(-th1[..., None] * T) + np.exp(-th2[..., None] * T)

def jacobian(th1, th2):
    """J[i,alpha] = dy(ti)/d(theta_alpha).  Shape (3,2)."""
    J = np.zeros((3, 2))
    J[:, 0] = -T * np.exp(-th1 * T)
    J[:, 1] = -T * np.exp(-th2 * T)
    return J

def modes(th1, th2):
    """Eigenvalues and eigenvectors of H=J^TJ, ascending (sloppy first)."""
    J = jacobian(th1, th2)
    H = J.T @ J
    evals, evecs = np.linalg.eigh(H)   # evals[0]=sloppy, evals[1]=stiff
    return evals, evecs, J

# ── print analytical results for key points ────────────────────────────────
test_points = [
    (1.0,   1.0,   'diagonal:       θ₁=θ₂=1'),
    (1.0,   5.0,   'off-diagonal:   θ₁=1, θ₂=5'),
    (0.01,  1000., 'corner:         θ₁→0, θ₂→∞'),
]
print(f"{'Point':<30} {'λ_sloppy':>12} {'λ_stiff':>12} {'ratio':>8}  "
      f"{'sloppy dir (th1,th2)':>22}  {'stiff dir':>20}")
print("-" * 105)
for th1, th2, label in test_points:
    evals, evecs, _ = modes(th1, th2)
    ratio = evals[1] / (evals[0] + 1e-20)
    sv = evecs[:, 0]; tv = evecs[:, 1]
    print(f"{label:<30} {evals[0]:>12.5f} {evals[1]:>12.5f} {ratio:>8.1f}  "
          f"  ({sv[0]:+.3f}, {sv[1]:+.3f})        ({tv[0]:+.3f}, {tv[1]:+.3f})")

# ── figure: 3D manifold + 2D parameter-space eigenvectors ─────────────────
fig = plt.figure(figsize=(16, 7))

# ── left: 3D manifold with arrows ─────────────────────────────────────────
ax3 = fig.add_subplot(121, projection='3d')

N  = 80
th = np.exp(np.linspace(-5, 5, N))
TH1, TH2 = np.meshgrid(th, th)
Y  = predict(TH1, TH2)
Y[TH1 > TH2] = np.nan
ax3.plot_surface(Y[:,:,0], Y[:,:,1], Y[:,:,2],
                 color='steelblue', alpha=0.18, edgecolor='none')

th_f = np.exp(np.linspace(-5, 5, 600))
EPS, INF = 1e-6, 1e5
ax3.plot(*predict(th_f, th_f).T,                    color='firebrick',    lw=1.5)
ax3.plot(*predict(np.full(600,EPS), th_f).T,         color='seagreen',     lw=1.5)
ax3.plot(*predict(th_f, np.full(600,INF)).T,         color='mediumpurple', lw=1.5)

# Draw stiff (red) and sloppy (blue) tangent arrows at selected points
arrow_pts = [
    (1.0,  1.0,  'θ₁=θ₂=1'),
    (1.0,  5.0,  'θ₁=1, θ₂=5'),
    (0.01, 1000.,'θ₁→0, θ₂→∞'),
]
scale = 0.12
for th1, th2, lbl in arrow_pts:
    pt = predict(th1, th2)[0]
    evals, evecs, J = modes(th1, th2)

    # Project parameter-space eigenvectors into behavior space via J
    stiff_b  = J @ evecs[:, 1];  stiff_b  /= (np.linalg.norm(stiff_b)  + 1e-12)
    sloppy_b = J @ evecs[:, 0];  sloppy_b /= (np.linalg.norm(sloppy_b) + 1e-12)

    ax3.scatter(pt[0], pt[1], pt[2], color='black', s=60, zorder=10, depthshade=False)
    ax3.quiver(pt[0], pt[1], pt[2],
               stiff_b[0]*scale, stiff_b[1]*scale, stiff_b[2]*scale,
               color='red',  linewidth=2, arrow_length_ratio=0.3, label='stiff' if lbl==arrow_pts[0][2] else '')
    ax3.quiver(pt[0], pt[1], pt[2],
               sloppy_b[0]*scale, sloppy_b[1]*scale, sloppy_b[2]*scale,
               color='dodgerblue', linewidth=2, arrow_length_ratio=0.3, label='sloppy' if lbl==arrow_pts[0][2] else '')

ax3.set_xlabel('y(t=1/3)'); ax3.set_ylabel('y(t=1)'); ax3.set_zlabel('y(t=3)')
ax3.set_title('Stiff (red) and sloppy (blue)\ntangent vectors on manifold')
ax3.legend(fontsize=9)
ax3.view_init(elev=20, azim=40)

# ── right: 2D parameter space with eigenvector field ──────────────────────
ax2 = fig.add_subplot(122)

log_vals = np.linspace(-2.5, 2.5, 10)
th_grid  = np.exp(log_vals)
TH1g, TH2g = np.meshgrid(th_grid, th_grid)

qs = 0.18   # quiver scale

for i in range(len(th_grid)):
    for j in range(len(th_grid)):
        th1, th2 = TH1g[i,j], TH2g[i,j]
        if th1 > th2:
            continue                   # fundamental domain only
        x, y = np.log10(th1), np.log10(th2)
        evals, evecs, _ = modes(th1, th2)

        # Stiff eigenvector (bilateral arrow)
        dx, dy = evecs[:, 1] * qs
        ax2.annotate('', xy=(x+dx, y+dy), xytext=(x-dx, y-dy),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.8))
        # Sloppy eigenvector
        dx, dy = evecs[:, 0] * qs
        ax2.annotate('', xy=(x+dx, y+dy), xytext=(x-dx, y-dy),
                     arrowprops=dict(arrowstyle='<->', color='dodgerblue',
                                     lw=1.2, linestyle='dashed'))

# Shade the region where eigenvalue ratio > 10 (visibly sloppy)
log_fine = np.linspace(-2.5, 2.5, 200)
TH1f, TH2f = np.meshgrid(np.exp(log_fine), np.exp(log_fine))
ratio_map  = np.full_like(TH1f, np.nan)
for i in range(200):
    for j in range(200):
        if TH1f[i,j] > TH2f[i,j]:
            continue
        ev, _, _ = modes(TH1f[i,j], TH2f[i,j])
        ratio_map[i,j] = np.log10(ev[1] / (ev[0]+1e-20))
ax2.contourf(log_fine, log_fine, ratio_map,
             levels=15, cmap='RdYlBu_r', alpha=0.3)
plt.colorbar(ax2.contourf(log_fine, log_fine, ratio_map, levels=15,
                           cmap='RdYlBu_r', alpha=0.3),
             ax=ax2, label='log₁₀(λ_stiff / λ_sloppy)')

diag = np.linspace(-2.5, 2.5, 100)
ax2.plot(diag, diag, 'k--', lw=1.2, label='θ₁=θ₂ diagonal')
ax2.set_xlabel('log₁₀(θ₁)', fontsize=11)
ax2.set_ylabel('log₁₀(θ₂)', fontsize=11)
ax2.set_title('Parameter space: stiff (red) & sloppy (blue)\nColor = log condition number', fontsize=11)
ax2.set_aspect('equal')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()