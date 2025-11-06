#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBML → GPU export for Chen 2004 yeast cell cycle (or any SBML Level 2/3 model)

Outputs:
  - chen_model_export.json   (species, parameters, reactions, ODEs, index maps)
  - chen_arrays.npz          (species_init, stoich, reactant_matrix, product_matrix)
  - chen_indices.json        (flat indices & counts for GPU kernels)

Tested with Python 3.9. Requires: python-libsbml, numpy
    pip install python-libsbml numpy
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
try:
    import libsbml
except ImportError as e:
    raise SystemExit("Missing dependency: python-libsbml. Install with: pip install python-libsbml") from e


# === CONFIG ===
SBML_PATH = "/home/gijs/Documents/OxfordEvolution/Yeast/Chen/chen2004_biomd56.xml"
OUT_DIR = Path(SBML_PATH).with_suffix("").parent  # same folder as input
BASENAME = "chen"  # prefix for output files


# === UTIL ===
def fail(msg: str) -> None:
    raise SystemExit(f"❌ {msg}")

def formula_str(ast: libsbml.ASTNode) -> str:
    """Convert MathML AST → infix string."""
    return libsbml.formulaToString(ast)

def is_floating_species(s: libsbml.Species) -> bool:
    # RoadRunner-style "floating species": not boundary, participates dynamically
    return not s.getBoundaryCondition()

def species_initial_value(s: libsbml.Species) -> Tuple[float, str]:
    """
    Return initial numeric value and kind ('amount' or 'concentration').
    Prefer initialConcentration if set, else amount, else 0.0.
    """
    if s.isSetInitialConcentration():
        return s.getInitialConcentration(), "concentration"
    if s.isSetInitialAmount():
        return s.getInitialAmount(), "amount"
    return 0.0, "unspecified"


# === PARSE SBML ===
def load_sbml_model(path: str) -> libsbml.Model:
    if not os.path.exists(path):
        fail(f"SBML file not found: {path}")
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(path)
    if doc.getNumErrors() > 0 and doc.getErrorLog().getNumFailsWithSeverity(libsbml.LIBSBML_SEV_ERROR) > 0:
        fail(f"SBML read errors:\n{doc.getErrorLog().toString()}")
    model = doc.getModel()
    if model is None:
        fail("No <model> element in SBML.")
    return model

def collect_species(model: libsbml.Model) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """Return floating species ids, index map, and initial values (float32)."""
    ids: List[str] = []
    init_vals: List[float] = []
    for s in model.getListOfSpecies():
        if is_floating_species(s):
            ids.append(s.getId())
            v, _ = species_initial_value(s)
            init_vals.append(float(v))
    index = {sid: i for i, sid in enumerate(ids)}
    return ids, index, np.array(init_vals, dtype=np.float32)

def collect_global_parameters(model: libsbml.Model) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for p in model.getListOfParameters():
        pid = p.getId()
        val = float(p.getValue()) if p.isSetValue() else 0.0
        params[pid] = val
    return params

def reaction_stoich(r: libsbml.Reaction) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return reactants and products stoichiometries as dicts {species_id: stoich}."""
    react: Dict[str, float] = {}
    prod: Dict[str, float] = {}
    for sr in r.getListOfReactants():
        sid = sr.getSpecies()
        coeff = sr.getStoichiometry() if sr.isSetStoichiometry() else (sr.getStoichiometryMath().evaluate() if sr.isSetStoichiometryMath() else 1.0)
        react[sid] = react.get(sid, 0.0) + float(coeff)
    for sp in r.getListOfProducts():
        sid = sp.getSpecies()
        coeff = sp.getStoichiometry() if sp.isSetStoichiometry() else (sp.getStoichiometryMath().evaluate() if sp.isSetStoichiometryMath() else 1.0)
        prod[sid] = prod.get(sid, 0.0) + float(coeff)
    return react, prod

def kinetic_law_info(r: libsbml.Reaction) -> Tuple[str, Dict[str, float]]:
    """Return (rate_expr_infix, local_params_dict) for a reaction."""
    kl = r.getKineticLaw()
    if kl is None or kl.getMath() is None:
        return "0.0", {}
    expr = formula_str(kl.getMath())
    locals_dict: Dict[str, float] = {}
    for lp in kl.getListOfParameters():
        locals_dict[lp.getId()] = float(lp.getValue()) if lp.isSetValue() else 0.0
    return expr, locals_dict

def build_stoich_matrix(model: libsbml.Model, species_ids: List[str], reaction_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build stoichiometry matrix N (n_species x n_rxn),
    and also separate reactant and product incidence matrices (float32).
    """
    nS, nR = len(species_ids), len(reaction_ids)
    sid_to_idx = {sid: i for i, sid in enumerate(species_ids)}
    N = np.zeros((nS, nR), dtype=np.float32)
    Rmat = np.zeros((nS, nR), dtype=np.float32)
    Pmat = np.zeros((nS, nR), dtype=np.float32)

    for j, rid in enumerate(reaction_ids):
        r = model.getReaction(rid)
        react, prod = reaction_stoich(r)
        for sid, v in react.items():
            if sid in sid_to_idx:
                i = sid_to_idx[sid]
                N[i, j] -= v
                Rmat[i, j] += v
        for sid, v in prod.items():
            if sid in sid_to_idx:
                i = sid_to_idx[sid]
                N[i, j] += v
                Pmat[i, j] += v
    return N, Rmat, Pmat

def build_reaction_package(model: libsbml.Model) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return reaction IDs and a per-reaction dict with stoichiometry & kinetic law info."""
    rxn_ids: List[str] = [r.getId() if r.isSetId() else f"R{idx}" for idx, r in enumerate(model.getListOfReactions())]
    rxns: List[Dict[str, Any]] = []
    for rid in rxn_ids:
        r = model.getReaction(rid)
        rate_expr, local_params = kinetic_law_info(r)
        react, prod = reaction_stoich(r)
        rxns.append({
            "id": rid,
            "rate": rate_expr,         # infix string
            "local_params": local_params,  # {name: value}
            "reactants": react,        # {species_id: coeff}
            "products": prod,          # {species_id: coeff}
        })
    return rxn_ids, rxns

def substitute_locals_in_rate(rate: str, local_params: Dict[str, float]) -> str:
    """Inline local parameters as numeric literals in a naive, safe-ish way (identifier-boundaries)."""
    # very simple replacement respecting identifier boundaries with braces
    out = rate
    for name, val in local_params.items():
        # wrap with parentheses; avoid partial matches using a crude tokenization
        out = out.replace(name, f"({val})")
    return out

def extract_species_odes(model):
    """
    Construct species-level ODEs from SBML stoichiometry × kinetic laws.
    Returns a dict mapping each species ID -> Python expression for dy/dt.
    """
    import libsbml

    # Initialize every species with 0.0 so we can safely append terms
    odes = {s.getId(): "0.0" for s in model.getListOfSpecies()}

    for reaction in model.getListOfReactions():
        kinetic_law = reaction.getKineticLaw()
        if kinetic_law is None or kinetic_law.getMath() is None:
            continue

        # Convert MathML to infix formula string
        try:
            rate_expr = libsbml.formulaToL3String(kinetic_law.getMath())
        except Exception:
            rate_expr = libsbml.formulaToString(kinetic_law.getMath())

        # --- Subtract reactants (consumption) ---
        for reactant in reaction.getListOfReactants():
            sid = reactant.getSpecies()
            stoich = reactant.getStoichiometry()
            if stoich is None or stoich == 0:
                stoich = 1.0
            odes[sid] += f" + (-{stoich})*({rate_expr})"

        # --- Add products (formation) ---
        for product in reaction.getListOfProducts():
            sid = product.getSpecies()
            stoich = product.getStoichiometry()
            if stoich is None or stoich == 0:
                stoich = 1.0
            odes[sid] += f" + ({stoich})*({rate_expr})"

    # Clean up whitespace for readability
    for sid, expr in odes.items():
        odes[sid] = " ".join(expr.split())

    return odes



# === MAIN EXPORT ===
def main() -> None:
    model = load_sbml_model(SBML_PATH)

    # Species
    species_ids, species_index, species_init = collect_species(model)

    # Parameters (global)
    global_params = collect_global_parameters(model)

    # Reactions
    reaction_ids, reactions = build_reaction_package(model)

    # Stoichiometry (arrays for GPU)
    N, Rmat, Pmat = build_stoich_matrix(model, species_ids, reaction_ids)

    # ODE strings (with local params inlined)
    odes = extract_species_odes(model)

    # Build parameter index (only globals—locals are inlined)
    param_ids = sorted(global_params.keys())
    param_index = {pid: i for i, pid in enumerate(param_ids)}
    param_values = np.array([global_params[pid] for pid in param_ids], dtype=np.float32)

    # JSON export (human-readable + for codegen)
    export = {
        "meta": {
            "source": str(SBML_PATH),
            "species_count": len(species_ids),
            "reaction_count": len(reaction_ids),
            "global_param_count": len(param_ids),
            "notes": "Local (per-reaction) parameters have been inlined numerically into reaction rate expressions."
        },
        "species": {
            "ids": species_ids,
            "index": species_index,
            "initial": species_init.tolist(),   # float list
        },
        "parameters": {
            "global_ids": param_ids,
            "global_index": param_index,
            "global_values": param_values.tolist()
        },
        "reactions": reactions,   # includes reactants/products, rate (original), local_params (values)
        "reactions_inlined": [
            {
                "id": r["id"],
                "rate_inlined": substitute_locals_in_rate(r["rate"], r["local_params"]),
            } for r in reactions
        ],
        "odes": odes,             # d(species)/dt as infix strings
        "indices": {
            "n_species": len(species_ids),
            "n_reactions": len(reaction_ids),
            "n_params": len(param_ids)
        }
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{BASENAME}_model_export.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"✅ Wrote JSON model export: {json_path}")

    # NPZ arrays for GPU loaders
    npz_path = OUT_DIR / f"{BASENAME}_arrays.npz"
    # Note: integer stoichiometries are stored as float32 here for simplicity with OpenCL buffers
    np.savez_compressed(
        npz_path,
        species_init=species_init.astype(np.float32),
        param_values=param_values.astype(np.float32),
        stoich=N.astype(np.float32),
        reactants=Rmat.astype(np.float32),
        products=Pmat.astype(np.float32),
        # helpful counts
        n_species=np.array([len(species_ids)], dtype=np.int32),
        n_reactions=np.array([len(reaction_ids)], dtype=np.int32),
        n_params=np.array([len(param_ids)], dtype=np.int32),
    )
    print(f"✅ Wrote arrays NPZ:         {npz_path}")

    # Flat indices for kernel codegen
    indices_path = OUT_DIR / f"{BASENAME}_indices.json"
    with open(indices_path, "w") as f:
        json.dump({
            "species_index": species_index,
            "param_index": param_index,
            "n_species": len(species_ids),
            "n_reactions": len(reaction_ids),
            "n_params": len(param_ids),
        }, f, indent=2)
    print(f"✅ Wrote indices JSON:       {indices_path}")

    # Small console summary
    print("\n— SUMMARY —")
    print(f"Species ({len(species_ids)}):", ", ".join(species_ids[:8]) + (" ..." if len(species_ids) > 8 else ""))
    print(f"Parameters ({len(param_ids)}):", ", ".join(param_ids[:8]) + (" ..." if len(param_ids) > 8 else ""))
    print(f"Reactions ({len(reaction_ids)}):", ", ".join(reaction_ids[:6]) + (" ..." if len(reaction_ids) > 6 else ""))
    print("Example ODE:", next(iter(odes.items())))


if __name__ == "__main__":
    main()
