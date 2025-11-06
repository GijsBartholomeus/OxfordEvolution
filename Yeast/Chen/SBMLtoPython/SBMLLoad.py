#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Chen SBML → JSON conversion by running a CPU ODE integration.

Requires:
    pip install numpy scipy matplotlib
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import re
from pathlib import Path

# === CONFIG ===
BASE = Path("/home/gijs/Documents/OxfordEvolution/Yeast/Chen")
MODEL_JSON = BASE / "chen_model_export.json"
ARR_FILE   = BASE / "chen_arrays.npz"

# === LOAD EXPORTS ===
with open(MODEL_JSON) as f:
    model = json.load(f)
npz = np.load(ARR_FILE)

species_ids = model["species"]["ids"]
param_ids   = model["parameters"]["global_ids"]
odes_str    = model["odes"]
param_vals  = npz["param_values"].astype(float)
y0          = npz["species_init"].astype(float)
n_species   = len(species_ids)
param_map   = {pid: param_vals[i] for i, pid in enumerate(param_ids)}

print(f"Loaded model with {n_species} species, {len(param_ids)} parameters.")


# === COMPILE RHS ===
# We'll make a Python-evaluable version of the ODE system.
# Replace species IDs with y[index], parameters with p["name"]

def sanitize_expr(expr: str) -> str:
    """Simple cleanup: remove SBML function remnants."""
    expr = expr.replace("^", "**")
    expr = re.sub(r"\bpi\b", "np.pi", expr)
    return expr

import numpy as np

def Mass_Action_1_222(k1, S1):
    """Unimolecular mass-action rate: k1 * S1"""
    return k1 * S1

def Mass_Action_2_221(k1, S1, S2):
    """Bimolecular mass-action rate: k1 * S1 * S2"""
    return k1 * S1 * S2

def MichaelisMenten_220(M1, J1, k1, S1):
    """Michaelis–Menten rate: (k1 * S1 * M1) / (J1 + S1)"""
    return (k1 * S1 * M1) / (J1 + S1)

def BB_218(A1, A2, A3, A4):
    """Goldbeter–Koshland helper function: A3*A2 + A4*A1"""
    return A3 * A2 + A4 * A1

def GK_219(A1, A2, A3, A4):
    """
    Goldbeter–Koshland steady-state switch.
    SBML math:
      2*A4*A1 / ((A2 - A1) + A3*A2 + A4*A1
                  + sqrt(((A2 - A1) + A3*A2 + A4*A1)**2
                         - 4*(A2 - A1)*A4*A1))
    """
    term = (A2 - A1) + A3 * A2 + A4 * A1
    disc = term**2 - 4 * (A2 - A1) * A4 * A1
    # clamp to avoid negative sqrt due to rounding
    disc = np.maximum(disc, 0.0)
    return 2 * A4 * A1 / (term + np.sqrt(disc))

FUNC_ENV = {
    "Mass_Action_1_222": Mass_Action_1_222,
    "Mass_Action_2_221": Mass_Action_2_221,
    "MichaelisMenten_220": MichaelisMenten_220,
    "BB_218": BB_218,
    "GK_219": GK_219,
}


compiled_odes = []

# Pre-sort for longest-first replacement (avoids partial matches)
sorted_params  = sorted(param_ids, key=len, reverse=True)
sorted_species = sorted(species_ids, key=len, reverse=True)

for i, sid in enumerate(species_ids):
    rhs = sanitize_expr(odes_str[sid])

    # Replace parameters (e.g. kdbud → p["kdbud"])
    for pid in sorted_params:
        rhs = re.sub(rf"\b{re.escape(pid)}\b", f'p["{pid}"]', rhs)

    # Replace species (e.g. CLB5 → y[12])
    for j, sname in enumerate(sorted_species):
        rhs = re.sub(rf"\b{re.escape(sname)}\b", f"y[{species_ids.index(sname)}]", rhs)

    compiled_odes.append(compile(rhs, f"<ODE {sid}>", "eval"))




def f(t, y):
    dydt = np.empty_like(y)
    env = {"np": np, "y": y, "p": param_map, "t": t, **FUNC_ENV}
    for i, code in enumerate(compiled_odes):
        try:
            dydt[i] = eval(code, env)
        except Exception as e:
            raise RuntimeError(f"ODE[{i}] ({species_ids[i]}) eval failed: {e}")
    return dydt

for i, (sid, expr) in enumerate(model["odes"].items()):
    print(f"{sid}: {expr}")
    if i > 5:
        break

for i, (sid, expr) in enumerate(model["odes"].items()):
    print(f"{sid}: {expr}")
    if i > 5:
        break


# === INTEGRATE ===
t_end = 400
t_eval = np.linspace(0, t_end, 2000)

print("Running solve_ivp...")
sol = solve_ivp(f, (0, t_end), y0, method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-9)
print("Integration successful:", sol.success)

# === PLOT ===
to_plot = ["CLB2", "CLB5", "CDC20", "SIC1"]
idxs = [species_ids.index(s) for s in to_plot if s in species_ids]

plt.figure(figsize=(8,5))
for i, idx in enumerate(idxs):
    plt.plot(sol.t, sol.y[idx], label=species_ids[idx])
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Chen model (CPU ODE integration)")
plt.legend()
plt.tight_layout()
plt.show()
