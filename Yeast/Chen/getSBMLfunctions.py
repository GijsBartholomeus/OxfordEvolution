#!/usr/bin/env python3
import libsbml
import re
from pathlib import Path

# --- CONFIG ---
SBML_PATH = Path("/home/gijs/Documents/OxfordEvolution/Yeast/Chen/chen2004_biomd56.xml")
OUT_PATH  = SBML_PATH.with_name("chen_functions.py")  # optional output

# --- LOAD MODEL ---
reader = libsbml.SBMLReader()
doc = reader.readSBML(str(SBML_PATH))
model = doc.getModel()
if model is None:
    raise SystemExit("❌ No <model> found in SBML.")

funcs = []

# --- EXTRACT FUNCTION DEFINITIONS ---
for fd in model.getListOfFunctionDefinitions():
    fid = fd.getId() or fd.getName()
    math = fd.getMath()
    if math is None:
        continue
    # libsbml gives us a lambda expression like "lambda(k, S) k * S"
    expr = libsbml.formulaToL3String(math)
    expr = expr.replace("^", "**")  # Python exponentiation

    funcs.append((fid, expr))

# --- PRINT AND OPTIONAL SAVE ---
print(f"Found {len(funcs)} functionDefinitions:\n")

py_lines = [
    "# Auto-generated from SBML <functionDefinition>",
    "import numpy as np",
    "",
]

for fid, expr in funcs:
    # Example expr: 'lambda(k, S) k * S'
    # Convert to proper Python def
    m = re.match(r"lambda\s*\(([^)]*)\)\s*(.*)", expr)
    if not m:
        print(f"⚠️  Could not parse {fid}: {expr}")
        continue
    args, body = m.groups()
    py_lines.append(f"def {fid}({args}):")
    py_lines.append(f"    return {body}")
    py_lines.append("")  # blank line
    print(f"{fid}({args}) = {body}")

# Optionally write to a .py file
OUT_PATH.write_text("\n".join(py_lines))
print(f"\n✅ Wrote Python function stubs to: {OUT_PATH}")
