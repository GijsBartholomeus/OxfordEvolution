# Sloppiness analysis for the Tyson-like 6-state ODE example
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import eigh

# === Base parameters (user provided) ===
p = {
    "k1_aa_over_CT": 0.015,
    "k2": 0.0,
    "k3_CT": 200.0,
    "k4": 180.0,
    "k4prime": 0.018,
    "k5_minusP": 0.0,
    "k6": 1.0,
    "k7": 0.6,
    "k8_minusP": 100.0,
    "k9": 50.0,
    "CT": 1.0
}

# We'll analyze these parameters (drop k2 and k5_minusP; add k8_minusP and k9)
param_names = [
    "k1_aa_over_CT",
    "k3_CT",
    "k4",
    "k4prime",
    "k6",
    "k7",
    "k8_minusP",
    "k9"
]
pvec = np.array([p[name] for name in param_names])
p_index = {name:i for i,name in enumerate(param_names)}
P = len(param_names)


# === model right-hand side and helper functions ===
CT = p["CT"]

def F_M(M, p):
    return p["k4prime"] + p["k4"] * (M / p["CT"])**2

def dF_dM(M, p):
    # derivative of F_M wrt M
    return p["k4"] * 2.0 * (M / p["CT"]) * (1.0 / p["CT"])

def f_rhs(t, x, p):
    # x = [C2, CP, pM, M, Y, YP]
    C2, CP, pM, M, Y, YP = x
    k3 = p["k3_CT"] / p["CT"]
    k1 = p["k1_aa_over_CT"] * p['CT']
    dC2 = p["k6"] * M - p["k8_minusP"] * C2 + p["k9"] * CP
    dCP = -k3 * CP * Y + p["k8_minusP"] * C2 - p["k9"] * CP
    dpM = k3 * CP * Y - pM * F_M(M, p) + p["k5_minusP"] * M
    dM  = pM * F_M(M, p) - p["k5_minusP"] * M - p["k6"] * M
    dY  = k1 - p["k2"] * Y - k3 * CP * Y
    dYP = p["k6"] * M - p["k7"] * YP
    return np.array([dC2, dCP, dpM, dM, dY, dYP])

def jacobian_fx(x, p):
    """Compute df/dx (6x6) analytically at state x."""
    C2, CP, pM, M, Y, YP = x
    k3 = p["k3_CT"] / p["CT"]
    # dF/dM
    dF = dF_dM(M, p)
    J = np.zeros((6,6))
    # dC2
    J[0,0] = -p["k8_minusP"]    # d(dC2)/dC2
    J[0,1] = p["k9"]           # d(dC2)/dCP
    J[0,3] = p["k6"]           # d(dC2)/dM
    # dCP
    J[1,0] = p["k8_minusP"]
    J[1,1] = -k3*Y - p["k9"]
    J[1,4] = -k3*CP
    # dpM
    J[2,1] = k3*Y
    J[2,2] = -F_M(M, p)
    J[2,3] = -pM * dF + p["k5_minusP"]
    J[2,4] = k3*CP
    # dM
    J[3,2] = F_M(M, p)
    J[3,3] = pM * dF - p["k5_minusP"] - p["k6"]
    # dY
    k1 = p["k1_aa_over_CT"] * p['CT']
    J[4,1] = -k3 * Y
    J[4,4] = -p["k2"] - k3 * CP
    # dYP
    J[5,3] = p["k6"]
    J[5,5] = -p["k7"]
    return J

def df_dparam(x, p):
    """Compute df/dh for each parameter in param_names (returns 6 x P array).
       Now supports parameters: k1_aa_over_CT, k3_CT, k4, k4prime, k6, k7, k8_minusP, k9
    """
    C2, CP, pM, M, Y, YP = x
    k3 = p["k3_CT"] / p["CT"]
    # precompute pieces
    dfdh = np.zeros((6, P))
    for j,name in enumerate(param_names):
        if name == "k1_aa_over_CT":
            # k1 = k1_aa_over_CT * CT appears only in dY as +k1
            df = np.zeros(6); df[4] = p['CT']  # d(dY)/d k1_aa_over_CT = CT
            dfdh[:,j] = df
        elif name == "k3_CT":
            # k3 = k3_CT / CT
            coeff = 1.0 / p["CT"]
            df = np.zeros(6)
            df[1] = -coeff * CP * Y     # d(dCP)/d k3_CT
            df[2] =  coeff * CP * Y     # d(dpM)/d k3_CT
            df[4] = -coeff * CP * Y     # d(dY)/d k3_CT
            dfdh[:,j] = df
        elif name == "k4":
            df = np.zeros(6)
            dFdk4 = (M / p["CT"])**2
            df[2] = -pM * dFdk4   # dpM
            df[3] =  pM * dFdk4   # dM
            dfdh[:,j] = df
        elif name == "k4prime":
            df = np.zeros(6)
            dFdk4p = 1.0
            df[2] = -pM * dFdk4p
            df[3] =  pM * dFdk4p
            dfdh[:,j] = df
        elif name == "k6":
            df = np.zeros(6)
            df[0] = M          # d(dC2)/d k6 = M
            df[3] = -M         # d(dM)/d k6 = -M
            df[5] = M          # d(dYP)/d k6 = M
            dfdh[:,j] = df
        elif name == "k7":
            df = np.zeros(6)
            df[5] = -YP        # d(dYP)/d k7 = -YP
            dfdh[:,j] = df
        elif name == "k8_minusP":
            # k8 multiplies C2 with + in dCP and - in dC2
            df = np.zeros(6)
            df[0] = -C2        # d(dC2)/d k8 = -C2
            df[1] =  C2        # d(dCP)/d k8 = +C2
            dfdh[:,j] = df
        elif name == "k9":
            # k9 multiplies CP with + in dC2 and - in dCP
            df = np.zeros(6)
            df[0] =  CP        # d(dC2)/d k9 = +CP
            df[1] = -CP        # d(dCP)/d k9 = -CP
            dfdh[:,j] = df
        else:
            raise KeyError(name)
    return dfdh



# === Sensitivity ODE: augmented system ===
def augmented_rhs(t, z):
    """z contains [x(6), S_flat(6*P)] where S_j are sensitivities wrt log-params:
       S_j = d x / d log(h_j) = h_j * d x / d h_j.
    """
    x = z[:6]
    S_flat = z[6:]
    S = S_flat.reshape((6, P), order='F')  # columns are parameters
    # compute base dynamics
    xdot = f_rhs(t, x, p)
    # compute df/dx and df/dparam
    A = jacobian_fx(x, p)            # 6x6
    df_dh = df_dparam(x, p)         # 6xP (∂f/∂h)
    # convert df/dh to df/dlogh: df/dlogh = h * df/dh  (elementwise scalar multiply)
    df_dlogh = np.zeros_like(df_dh)
    for j,name in enumerate(param_names):
        hj = p[name]
        df_dlogh[:, j] = hj * df_dh[:, j]
    # sensitivity ODEs: dS_j/dt = A * S_j + df/dlogh_j
    Sdot = A.dot(S) + df_dlogh
    # pack
    return np.concatenate([xdot, Sdot.ravel(order='F')])

# === Integration ===
y0 = np.array([0.9, 0.05, 0.0, 0.005, 0.3, 0.0])
# initial sensitivities: ∂x(0)/∂log h_j ; if x0 independent of params -> zeros
S0 = np.zeros((6, P))
z0 = np.concatenate([y0, S0.ravel(order='F')])

t_span = (0.0, 100.0)
# choose times where we want to evaluate (optional)
t_eval = np.linspace(t_span[0], t_span[1], 1001)

sol_aug = solve_ivp(augmented_rhs, t_span, z0, method='BDF',
                    t_eval=t_eval, rtol=1e-6, atol=1e-8)

if not sol_aug.success:
    raise RuntimeError("Augmented integration failed: " + sol_aug.message)

t = sol_aug.t
X = sol_aug.y[:6, :]                       # 6 x m
S_flat = sol_aug.y[6:, :]                  # (6*P) x m
m = t.size

# reshape sensitivities: for each time i, S(:,:,i) is 6 x P
S_time = np.zeros((6, P, m))
for i in range(m):
    S_time[:,:,i] = S_flat[:,i].reshape((6,P), order='F')

# === Observables and output sensitivities ===
# Here g(x,h) = x (identity). So output sensitivity rows are simply S_time (state sensitivities).
# If you had a different g, you'd compute dg/dx @ S + dg/dlogh.
J_time = S_time.copy()   # shape (6, P, m) — observables x params x time

# === Build weighting matrix W and quadrature weights ===
Ns = 6           # number of species/observables
Nc = 1           # only one condition
Tc = t_span[1] - t_span[0]
sigma = 1.0      # per-species measurement noise (1 if unknown)
# diagonal weights per observable: each of the 6 species has same sigma here
W_diag = np.ones(Ns) * (1.0 / (Nc * Ns * Tc * sigma**2))
W = np.diag(W_diag)   # shape (6,6)

# quadrature weights (trapezoid)
dt = np.diff(t)
q = np.zeros(m)
q[0] = dt[0]/2.0
q[-1] = dt[-1]/2.0
q[1:-1] = 0.5*(dt[:-1] + dt[1:])

# === Accumulate H ===
H = np.zeros((P, P))
for i in range(m):
    J_i = J_time[:, :, i]        # (6 x P)
    A_i = J_i.T.dot(W).dot(J_i)  # P x P
    H += q[i] * A_i


# === Eigen-decompose H ===
eigvals, eigvecs = eigh(H)   # ascending
eigvals = eigvals[::-1]      # descending
eigvecs = eigvecs[:, ::-1]

# print eigenvalues and condition number
print("Eigenvalues (descending):")
for i, lam in enumerate(eigvals):
    print(f"{i+1:2d}: {lam:.3e}")


# === Plot spectrum ===
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, P+1), np.log10(eigvals), 'o-')
plt.xlabel("Eigenvalue index (descending)")
plt.ylabel("log10(eigenvalue)")
plt.title("Hessian / Fisher Information Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: inspect top eigenvector composition (which parameters dominate) ===
top_k = min(4, P)
for k in range(top_k):
    vec = eigvecs[:,k]
    print(f"\nEigenvector {k+1} (lambda={eigvals[k]:.3e}):")
    for j,name in enumerate(param_names):
        print(f"  {name:>15s}: {vec[j]:+.3f}")

import numpy as np

# 1) condition number and spread
eigvals_raw, _ = np.linalg.eig(H)
eigvals_real = np.real(eigvals_raw)  # numerical safety
eigvals_sorted = np.sort(eigvals_real)[::-1]
print("eigvals sorted (desc):", eigvals_sorted)
print("ratio largest/smallest (use smallest > 0):",
      eigvals_sorted[0] / max(eigvals_sorted[-1], 1e-300))
print("log10 span:", np.log10(eigvals_sorted[0]) - np.log10(max(eigvals_sorted[-1],1e-300)))

# 2) check H diagonal (which gives marginal info for each param)
print("H diagonal:", np.diag(H))

# 3) column norms of J (integrated contribution per parameter)
col_sq = np.sum([ (J_time[:,j,i]**2).sum() * q[i] for i in range(m) for j in range(P) ])
# simpler: compute L2 norm per parameter by integrating J(:,j,:)^2 over time
col_norms = np.zeros(P)
for j in range(P):
    col_norms[j] = sum(q[i] * np.sum(J_time[:, j, i]**2) for i in range(m))
print("per-parameter integrated squared-sensitivity (col_norms):", col_norms)

# 4) check for zero parameters in pvec
print("parameter vector:", pvec)


def simulate_at_params(p_local, t_eval):
    y0 = np.array([0.9, 0.05, 0.0, 0.005, 0.3, 0.0])
    sol = solve_ivp(lambda tt, xx: f_rhs(tt, xx, p_local), (t_eval[0], t_eval[-1]), y0,
                    method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError("Integrator failed: " + sol.message)
    return sol.t, sol.y  # t, (6 x m)

# choose index of smallest positive eigenvalue (most sloppy)
pos_idx = np.where(eigvals > 1e-30)[0]
if pos_idx.size == 0:
    raise RuntimeError("No positive eigenvalues found.")
sloppy_idx = pos_idx[-1]
lam = eigvals[sloppy_idx]
v = eigvecs[:, sloppy_idx]

print(f"Chosen sloppy eigen idx={sloppy_idx}, lambda={lam:.3e}")

# step size (quadratic rule)
delta = 0.1
alpha = np.sqrt(delta / max(lam, 1e-300))
max_logstep = 10
alpha = min(alpha, max_logstep)
dlogh = alpha * v
dlogh = np.clip(dlogh, -2.0, 2.0)  # safety clamp

# parameter names -> p vectors mapping (current)
p_current = dict(p)  # copy
P = len(param_names)

# build p_plus and p_minus
p_plus = dict(p_current)
p_minus = dict(p_current)

for j, name in enumerate(param_names):
    p_plus[name]  = p_current[name] * np.exp(+dlogh[j])
    p_minus[name] = p_current[name] * np.exp(-dlogh[j])


# choose time grid (reuse from your sensitivity run if available)
t_eval = np.linspace(0, 100, 501)

# simulate baseline and both candidates
t0, X0 = simulate_at_params(p_current, t_eval)
t_p, Xp = simulate_at_params(p_plus,  t_eval)
t_m, Xm = simulate_at_params(p_minus, t_eval)

# compute YT_rel and M_rel (use current CT)
CT_local = p_current['CT']
def compute_obs(X):
    C2, CP, pM, M, Y, YP = X
    YT = Y + YP + pM + M
    return YT / CT_local, M / CT_local

YT0, M0 = compute_obs(X0)
YTp, Mp = compute_obs(Xp)
YTm, Mm = compute_obs(Xm)

# approximate integrated squared difference (Euclidean with trapezoid weights)
dt = np.diff(t_eval)
q = np.zeros_like(t_eval)
q[0] = dt[0]/2.0; q[-1] = dt[-1]/2.0
q[1:-1] = 0.5*(dt[:-1] + dt[1:])

def integrated_sq_diff(YA, YB):
    return np.sum(q * ( (YA - YB)**2 ))

score_plus  = integrated_sq_diff(YT0, YTp) + integrated_sq_diff(M0, Mp)
score_minus = integrated_sq_diff(YT0, YTm) + integrated_sq_diff(M0, Mm)

# pick the better sign
if score_plus <= score_minus:
    score = score_plus
    chosen = 'plus'
    p_next = p_plus
    YT_next, M_next = YTp, Mp
else:
    chosen = 'minus'
    score = score_minus
    p_next = p_minus
    YT_next, M_next = YTm, Mm

print("---------Step result---------")
print("Picked direction:", chosen, "alpha:", alpha)
print('difference in parameters:', {name: (p_next[name] - p_current[name]) for name in param_names})
print('total  distance between P_current and P_next:',
      np.sqrt( sum( ((p_next[name]) - (p_current[name]))**2 for name in param_names ) ) )
print('new parameters:', {name: p_next[name] for name in param_names})
print('integrated squared-output-change at chosen step:', score)
# plot overlay
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_eval, YT0, label='base YT/CT', lw=1.5)
plt.plot(t_eval, YT_next, '--', label=f'after step ({chosen})', lw=1.5)
plt.xlabel('time')
plt.ylabel('YT / CT')
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(t_eval, M0, label='base M/CT', lw=1.5)
plt.plot(t_eval, M_next, '--', label=f'after step ({chosen})', lw=1.5)
plt.xlabel('time')
plt.ylabel('M / CT')
plt.legend(); plt.grid(True)

plt.suptitle(f"Step along sloppy eigen idx={sloppy_idx}, lambda={lam:.3e}")
plt.tight_layout()
plt.show()

