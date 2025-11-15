import numpy as np

# Define parameters (n decay rates)
theta = np.array([1.0, 2.0, 4.0, 8.0])   # example parameter values
n = len(theta)

# Define measurement times (m > n)
t = np.linspace(0.1, 10, 50)              # 50 time points between 0.1 and 10
m = len(t)

# Build the Hessian / Fisher matrix
H = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        H[i, j] = np.sum(t**2 * np.exp(-(theta[i] + theta[j]) * t))

# Symmetrize numerically (just to clean rounding)
H = 0.5 * (H + H.T)

# Compute eigenvalues (and optionally eigenvectors)
eigvals, eigvecs = np.linalg.eigh(H)

# Sort descending (largest first)
eigvals = np.flip(np.sort(eigvals))

print("Hessian matrix:\n", H)
print("\nEigenvalues (descending):\n", eigvals)
print("\nEigenvectors:\n", eigvecs)
