#!/usr/bin/env python3
"""
Linux-compatible Python analysis script for yeast cell cycle data
Clean version for copy-paste
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.io as sio

def load_and_analyze():
    """Load MATLAB results and create analysis plots"""
    
    # Load data
    if Path('matlab_results.mat').exists():
        data = sio.loadmat('matlab_results.mat')
        T = data['T'].flatten()
        allValues = data['allValues']
        allNames = [name[0] for name in data['allNames'].flatten()]
    else:
        print("MATLAB results not found. Run MATLAB script first.")
        return
    
    # Find key species indices
    try:
        cln2_idx = allNames.index('CLN2')
        clb2_idx = allNames.index('CLB2')
        ckit_idx = allNames.index('CKIT')
    except ValueError as e:
        print(f"Species not found: {e}")
        return
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time series
    axes[0,0].plot(T, allValues[:, cln2_idx], 'r-', linewidth=2, label='CLN2 (G1/S)')
    axes[0,0].plot(T, allValues[:, clb2_idx], 'b-', linewidth=2, label='CLB2 (G2/M)')
    axes[0,0].plot(T, allValues[:, ckit_idx], 'k--', linewidth=2, label='CKIT (Inhibitor)')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Concentration')
    axes[0,0].set_title('Cell Cycle Regulators')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Phase portrait
    axes[0,1].plot(allValues[:, cln2_idx], allValues[:, clb2_idx], 'b-', linewidth=2)
    axes[0,1].set_xlabel('CLN2 (G1/S Cyclin)')
    axes[0,1].set_ylabel('CLB2 (G2/M Cyclin)')
    axes[0,1].set_title('Phase Portrait')
    axes[0,1].grid(True)
    
    # Normalized comparison
    cln2_norm = allValues[:, cln2_idx] / np.max(allValues[:, cln2_idx])
    clb2_norm = allValues[:, clb2_idx] / np.max(allValues[:, clb2_idx])
    ckit_norm = allValues[:, ckit_idx] / np.max(allValues[:, ckit_idx])
    
    axes[1,0].plot(T, cln2_norm, 'r-', linewidth=2, label='CLN2')
    axes[1,0].plot(T, clb2_norm, 'b-', linewidth=2, label='CLB2')
    axes[1,0].plot(T, ckit_norm, 'k--', linewidth=2, label='CKIT')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Normalized Concentration')
    axes[1,0].set_title('Normalized Oscillations')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Inhibition dynamics
    axes[1,1].plot(allValues[:, ckit_idx], allValues[:, clb2_idx], 'k-', linewidth=2)
    axes[1,1].set_xlabel('CKIT (CDK Inhibitor)')
    axes[1,1].set_ylabel('CLB2 (G2/M Cyclin)')
    axes[1,1].set_title('Inhibition Dynamics')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('python_yeast_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print analysis summary
    period_estimate = estimate_period(T, allValues[:, clb2_idx])
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Simulation time: {T[-1]:.1f} time units")
    print(f"Estimated cell cycle period: {period_estimate:.1f} time units")
    print(f"Final CLN2: {allValues[-1, cln2_idx]:.3f}")
    print(f"Final CLB2: {allValues[-1, clb2_idx]:.3f}")
    print(f"Final CKIT: {allValues[-1, ckit_idx]:.3f}")

def estimate_period(time, signal):
    """Simple period estimation using zero crossings"""
    mean_val = np.mean(signal)
    crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
    if len(crossings) > 2:
        periods = np.diff(time[crossings[::2]])  # Every other crossing
        return np.mean(periods) * 2  # Full cycle
    return np.nan

if __name__ == "__main__":
    load_and_analyze()