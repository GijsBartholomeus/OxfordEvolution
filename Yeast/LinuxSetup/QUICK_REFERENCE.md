# 🚀 QUICK REFERENCE: Yeast Cell Cycle on Linux

## Key Command (Most Important!)
```bash
matlab -nodisplay -nosplash -nodesktop -batch 'run_yeast_model_linux'
```

## Full Workflow
```bash
# 1. Setup (one time)
mamba create -n bioevo python=3.11 tellurium matplotlib pandas scipy -y

# 2. Every time you run
mamba activate bioevo
cd ~/yeast_cell_cycle
matlab -nodisplay -nosplash -nodesktop -batch 'run_yeast_model_linux'
python analyze_results_linux.py
```

## Expected Output Files
- `matlab_results.mat` - Full simulation data
- `yeast_analysis_linux.png` - MATLAB plots
- `python_yeast_analysis.png` - Python plots
- `time_data.csv` & `species_data.csv` - Exported data

## Troubleshooting
- If MATLAB hangs: Make sure you use ALL three flags (-nodisplay -nosplash -nodesktop)
- If no MATLAB: Install `octave octave-signal` and modify script slightly
- If graphics fail: Install `python3-tk` or set `matplotlib.use('Agg')`

## Files to Transfer
1. `BuddingYeastCellCycle_2015.m` (main model)
2. `run_yeast_model_linux.m` (execution script)
3. `analyze_results_linux.py` (python analysis)
4. `Linux_Setup_Guide.md` (full documentation)