import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T = np.array([1/3, 1.0, 3.0])

def predict(th1, th2):
    th1 = np.atleast_1d(np.asarray(th1, float))
    th2 = np.atleast_1d(np.asarray(th2, float))
    return np.exp(-th1[..., None] * T) + np.exp(-th2[..., None] * T)

# Extend range much further so the surface actually reaches the boundaries
N  = 100
th = np.exp(np.linspace(-5, 5, N))    # theta from ~0.007 to ~150
TH1, TH2 = np.meshgrid(th, th)
Y = predict(TH1, TH2)                 # shape (N, N, 3)
Y[TH1 > TH2] = np.nan                 # fundamental domain only

fig = plt.figure(figsize=(12, 9))
ax  = fig.add_subplot(111, projection='3d')

ax.plot_surface(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2],
                color='steelblue', alpha=0.35, edgecolor='none')

# Boundaries — also extend to match the surface range
th_fine = np.exp(np.linspace(-5, 5, 800))
EPS, INF = 1e-6, 1e5

crease     = predict(th_fine, th_fine)
bound_low  = predict(np.full(800, EPS), th_fine)    # theta1->0, theta2 varies
bound_high = predict(th_fine, np.full(800, INF))    # theta2->inf, theta1 varies
# (by symmetry these are also the theta2->0 and theta1->inf boundaries)

c1 = predict(EPS, EPS)[0]   # (2,2,2)
c2 = predict(EPS, INF)[0]   # (1,1,1)
c3 = predict(INF, INF)[0]   # (0,0,0)

ax.plot(crease[:,0],     crease[:,1],     crease[:,2],
        color='firebrick',    lw=2.5,
        label='θ₁ = θ₂  (also: mirror boundary by symmetry)')
ax.plot(bound_low[:,0],  bound_low[:,1],  bound_low[:,2],
        color='seagreen',     lw=2.5,
        label='θ₁→0  (= θ₂→0 by symmetry)')
ax.plot(bound_high[:,0], bound_high[:,1], bound_high[:,2],
        color='mediumpurple', lw=2.5,
        label='θ₂→∞  (= θ₁→∞ by symmetry)')

for pt, col, lab in [
    (c1, 'firebrick',    '(2,2,2)'),
    (c2, 'seagreen',     '(1,1,1) fold corner'),
    (c3, 'mediumpurple', '(0,0,0)'),
]:
    ax.scatter(pt[0], pt[1], pt[2], color=col, s=150, zorder=10, depthshade=False)
    ax.text(pt[0]+0.02, pt[1]+0.02, pt[2]+0.02, lab, fontsize=9)

ax.set_xlabel('y(t=1/3)', fontsize=11)
ax.set_ylabel('y(t=1)',   fontsize=11)
ax.set_zlabel('y(t=3)',   fontsize=11)
ax.set_title('Model manifold (fundamental domain θ₁ ≤ θ₂)', fontsize=13)
ax.legend(fontsize=9)
ax.view_init(elev=20, azim=40)

plt.tight_layout()
plt.show()