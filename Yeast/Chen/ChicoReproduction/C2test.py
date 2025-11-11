# file: chico_minimal.py
"""
Minimal tester for ChicoOscillation short-run debug:
- Loads Chen 2004 SBML
- Prints model variable ids
- Simulates robustly trying to obtain 'CLB2' and 'C2'
- Plots results (if present)
Requirements: tellurium, roadrunner, numpy, matplotlib
"""

import os
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
import tellurium as te
import roadrunner
import psutil

# Silence RoadRunner log messages completely
try:
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_CRITICAL)
except Exception:
    pass

# -------------------------
# Configuration (small)
# -------------------------
SIMULATION_TIME = 500
SIMULATION_POINTS = 501
DIVERGENCE_THRESHOLD = 250

# Candidate model paths (edit if needed)
LINUX_PATH = "/home/gijs/Documents/OxfordEvolution/Yeast/Chen/chen2004_biomd56.xml"
MAC_PATH = "/Users/gijsbartholomeus/Documents/STUDIE/OxfordEvolution/code/Yeast/Chen/chen2004_biomd56.xml"
RELATIVE_PATHS = [
    "chen2004_biomd56.xml",
    "Chen/chen2004_biomd56.xml",
    "../Chen/chen2004_biomd56.xml"
]

# -------------------------
# Helpers
# -------------------------
def detect_cpu_config():
    """Lightweight CPU detection (informational only)."""
    cpu_count_total = os.cpu_count() or 1
    cpu_info = platform.processor() or ""
    is_amd = any(k in cpu_info.lower() for k in ['amd', 'radeon', 'ryzen', 'epyc', 'athlon'])
    try:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        physical = None
        logical = cpu_count_total
    if is_amd:
        optimal_workers = max(1, int((physical or cpu_count_total) * 0.8))
        print(f"🔧 AMD-like CPU detected ({cpu_info}) — suggested workers: {optimal_workers}")
        return optimal_workers, True
    else:
        optimal_workers = max(1, cpu_count_total - 1)
        print(f"🔧 Non-AMD CPU (or unknown): using {optimal_workers} suggested workers")
        return optimal_workers, False

def get_model_path():
    """Find the chen2004_biomd56.xml file in expected locations."""
    if os.path.exists(LINUX_PATH):
        return LINUX_PATH
    if os.path.exists(MAC_PATH):
        return MAC_PATH
    for p in RELATIVE_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    # not found -> raise with helpful message
    raise FileNotFoundError(
        "Could not find 'chen2004_biomd56.xml'. Searched:\n"
        f"  {LINUX_PATH}\n  {MAC_PATH}\n  relative: {RELATIVE_PATHS}\n"
        f"Current working dir: {os.getcwd()}\n"
        "Place the SBML file in one of these locations or edit the paths in this script."
    )

# -------------------------
# Robust simulation
# -------------------------
def simulate_chico(rr, T=SIMULATION_TIME, npoints=SIMULATION_POINTS, debug=True):
    """
    Simulate and robustly extract 'time', 'CLB2', and 'C2' if present.
    Returns dict: {'time': ..., 'CLB2': ..., 'C2': ...}
    """
    selections = ["time", "CLB2", "C2"]
    # Try to pass selections into simulate if supported, else set rr.selections
    try:
        traj = rr.simulate(start=0, end=T, steps=npoints-1, selections=selections)
    except TypeError:
        rr.selections = selections
        traj = rr.simulate(start=0, end=T, steps=npoints-1)
    except Exception as e:
        # fallback: try simulate with default behaviour
        if debug:
            print("simulate() raised:", e)
            print("Retrying simulate without selections...")
        try:
            traj = rr.simulate(0, T, npoints-1)
        except Exception as e2:
            raise RuntimeError("Simulation failed: " + str(e2))

    if debug:
        # Print available model floating species & global parameters
        try:
            print("Available floating species:", rr.model.getFloatingSpeciesIds())
        except Exception:
            pass
        try:
            print("Available global parameters (sample):", rr.getGlobalParameterIds()[:40])
        except Exception:
            pass

    # Helper to probe column names / attributes on traj
    def get_colnames(obj):
        for attr in ('columnNames', 'colnames', 'getColumnNames', 'columns', 'headers'):
            if hasattr(obj, attr):
                try:
                    candidate = getattr(obj, attr)
                    return candidate() if callable(candidate) else list(candidate)
                except Exception:
                    continue
        # try to infer from rr.selections if array-like
        try:
            arr = np.asarray(obj)
            if arr.ndim == 2 and hasattr(rr, 'selections') and rr.selections and len(rr.selections) == arr.shape[1]:
                return list(rr.selections)
        except Exception:
            pass
        return None

    colnames = get_colnames(traj)
    if debug:
        print("Detected trajectory column names:", colnames)

    def extract_var(obj, name):
        # 1) direct accessor
        try:
            if hasattr(obj, 'getVariableTrajectory'):
                val = obj.getVariableTrajectory(name)
                return np.asarray(val)
        except Exception:
            pass
        # 2) dict-like
        try:
            val = obj[name]
            return np.asarray(val)
        except Exception:
            pass
        # 3) column-name indexing
        try:
            cols = get_colnames(obj)
            if cols and name in cols:
                idx = cols.index(name)
                arr = np.asarray(obj)
                return arr[:, idx]
        except Exception:
            pass
        # 4) rr.selections fallback
        try:
            if hasattr(rr, 'selections') and name in rr.selections:
                idx = list(rr.selections).index(name)
                arr = np.asarray(obj)
                return arr[:, idx]
        except Exception:
            pass
        return None

    out = {}
    out['time'] = extract_var(traj, 'time')
    if out['time'] is None:
        # try attribute .time
        try:
            out['time'] = np.asarray(getattr(traj, 'time'))
            if debug:
                print("Extracted time via traj.time attribute.")
        except Exception:
            # last resort: first column
            try:
                arr = np.asarray(traj)
                out['time'] = arr[:, 0]
                if debug:
                    print("Using first column as time axis.")
            except Exception:
                out['time'] = None

    out['CLB2'] = extract_var(traj, 'CLB2')
    out['C2']   = extract_var(traj, 'C2')

    if debug:
        for k in ('CLB2', 'C2'):
            v = out.get(k)
            if v is None:
                print(f"-> {k}: NOT FOUND in trajectory")
            else:
                print(f"-> {k}: found (length={len(v)}) min/max = ({np.nanmin(v):.4g}, {np.nanmax(v):.4g})")

    return out

# -------------------------
# Main execution
# -------------------------
def main():
    print("🧪 ChicoOscillation (minimal) — startup")
    detect_cpu_config()

    model_path = get_model_path()
    print("Loading model from:", model_path)

    rr = te.loadSBMLModel(model_path)

    # quick check: list floating species to help label debugging
    try:
        print("Model floating species ids:", rr.model.getFloatingSpeciesIds())
    except Exception:
        pass

    # Run a short simulation
    result = simulate_chico(rr, T=SIMULATION_TIME, npoints=SIMULATION_POINTS, debug=True)

    t = result.get('time')
    clb2 = result.get('CLB2')
    c2 = result.get('C2')

    # Basic plotting (only plot variables we have)
    if t is None:
        print("No time axis found in simulation output. Aborting plot.")
        return

    plt.figure(figsize=(8,4))
    plotted = False
    if clb2 is not None:
        plt.plot(t, clb2, label='CLB2')
        plotted = True
    if c2 is not None:
        plt.plot(t, c2, label='C2')
        plotted = True

    if not plotted:
        print("Neither CLB2 nor C2 were found in the trajectory. See detected column names above.")
        # print a slice of the numeric trajectory to inspect
        try:
            arr = np.asarray(rr.simulate(0, SIMULATION_TIME, SIMULATION_POINTS-1))
            print("Trajectory sample (first row):", arr[0, :min(12, arr.shape[1])])
            print("Trajectory shape:", arr.shape)
        except Exception:
            pass
        return

    plt.xlabel('time')
    plt.ylabel('concentration / arbitrary units')
    plt.title('Chico minimal: CLB2 and C2 (if present)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
