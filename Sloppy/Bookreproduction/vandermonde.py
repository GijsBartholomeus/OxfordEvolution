import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
T = 2  # can be any positive constant
N_values = np.arange(1, 101)
det_values = []

for N in N_values:
    if N == 1:
        det_values.append(1.0)
    else:
        Δ = 2 * T / (N - 1)
        log_det = (N * (N - 1) / 2) * np.log(Δ) + sum(math.lgamma(m + 1) for m in range(1, N))
        det_values.append(np.exp(log_det))

plt.figure(figsize=(8, 5))
plt.semilogy(N_values, det_values, 'o-', label=r'$|\det V|$')
plt.xlabel('N')
plt.ylabel(r'$|\det V|$ (log scale)')
plt.title('Magnitude of Vandermonde determinant for equally spaced points')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
