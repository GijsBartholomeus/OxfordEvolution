# Budding Yeast Cell Cycle Model - Linux Setup Guide

## 🚀 Quick Setup for Linux (Hyprland)

### Prerequisites Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install MATLAB (if available) or GNU Octave as alternative
sudo apt install octave octave-signal  # For Octave alternative

# Install Python and conda/mamba
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

# Create bioevo environment
mamba create -n bioevo python=3.11 -y
mamba activate bioevo
mamba install tellurium matplotlib pandas numpy scipy jupyter -y
```

### Essential Files Setup
```bash
# Create project directory
mkdir -p ~/yeast_cell_cycle
cd ~/yeast_cell_cycle

# Download required files (you'll need to copy these from your current project):
# - BuddingYeastCellCycle_2015.m
# - run_yeast_model.m 
# - test_matlab.m
```

### Key MATLAB Commands for Linux
```bash
# Test MATLAB installation
matlab -nodisplay -nosplash -nodesktop -batch 'disp("MATLAB works!"); quit;'

# Run the yeast model
matlab -nodisplay -nosplash -nodesktop -batch 'run_yeast_model'

# Alternative with Octave (if MATLAB not available)
octave --no-gui --eval 'run_yeast_model'
```

### Essential MATLAB Script (run_yeast_model.m)
```matlab
% COPY-PASTABLE MATLAB SCRIPT FOR LINUX
% Run Budding Yeast Cell Cycle Model - Linux Optimized

clear all; clc;

% Set working directory (adjust path as needed)
cd('~/yeast_cell_cycle');  % Or your actual path

% Run model simulation
timespan = [0, 300];
[T, Y, yinit, param, allNames, allValues] = BuddingYeastCellCycle_2015(timespan);

fprintf('Simulation completed: %d time points, %d species\n', length(T), length(allNames));

% Find key species
cln2_idx = find(strcmp(allNames, 'CLN2'), 1);
clb2_idx = find(strcmp(allNames, 'CLB2'), 1);
ckit_idx = find(strcmp(allNames, 'CKIT'), 1);  % SIC1 substitute

% Save data
save('matlab_results.mat', 'T', 'Y', 'allNames', 'allValues');
writematrix(T, 'time_data.csv');
writematrix(allValues, 'species_data.csv');

% Create focused plot
figure('Visible', 'off');  % No display for Linux headless

subplot(2,2,1);
hold on;
if ~isempty(cln2_idx), plot(T, allValues(:, cln2_idx), 'r-', 'LineWidth', 2); end
if ~isempty(clb2_idx), plot(T, allValues(:, clb2_idx), 'b-', 'LineWidth', 2); end
if ~isempty(ckit_idx), plot(T, allValues(:, ckit_idx), 'k--', 'LineWidth', 2); end
xlabel('Time'); ylabel('Concentration');
title('Key Cell Cycle Regulators');
legend('CLN2 (G1/S)', 'CLB2 (G2/M)', 'CKIT (Inhibitor)');
grid on;

% Phase portrait
subplot(2,2,2);
if ~isempty(cln2_idx) && ~isempty(clb2_idx)
    plot(allValues(:, cln2_idx), allValues(:, clb2_idx), 'b-', 'LineWidth', 2);
    xlabel('CLN2'); ylabel('CLB2');
    title('Phase Portrait');
    grid on;
end

saveas(gcf, 'yeast_analysis.png');
close(gcf);

fprintf('Analysis complete! Files saved:\n');
fprintf('- matlab_results.mat\n- time_data.csv\n- species_data.csv\n- yeast_analysis.png\n');
```

### Python Analysis Script (analyze_results.py)
```python
#!/usr/bin/env python3
"""
Linux-compatible Python analysis script for yeast cell cycle data
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
```

### Complete Linux Workflow
```bash
# 1. Setup environment
mamba activate bioevo
cd ~/yeast_cell_cycle

# 2. Run MATLAB simulation
matlab -nodisplay -nosplash -nodesktop -batch 'run_yeast_model'

# 3. Analyze with Python
python analyze_results.py

# 4. View results
ls -la *.png *.csv *.mat
```

### For Hyprland Window Manager
```bash
# Install additional packages for graphics
sudo apt install libgl1-mesa-glx libqt5gui5

# Set display variables if needed
export DISPLAY=:0
export QT_QPA_PLATFORM=wayland  # For Wayland support
```

### Alternative: Octave Setup (if no MATLAB)
```bash
# Install Octave
sudo apt install octave octave-signal octave-control

# Modify run_yeast_model.m for Octave compatibility:
# Replace 'writematrix' with 'csvwrite'
# Replace 'contains' with 'strfind' or manual search
```

## 🎯 Key Files You Need

1. **BuddingYeastCellCycle_2015.m** - Main model file
2. **run_yeast_model.m** - Execution script (copy from above)
3. **analyze_results.py** - Python analysis (copy from above)

## 🔧 Troubleshooting
- If MATLAB hangs: Always use `-nodisplay -nosplash -nodesktop` flags
- If Octave fails: Install `octave-signal` package
- If Python plotting fails: Install `python3-tk` or use `matplotlib.use('Agg')`

## 📊 Expected Output
- **yeast_analysis.png** - MATLAB-generated plot
- **python_yeast_analysis.png** - Python-enhanced analysis
- **matlab_results.mat** - Full simulation data
- **time_data.csv** & **species_data.csv** - Exported data