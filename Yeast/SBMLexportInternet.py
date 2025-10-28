#!/usr/bin/env python3
"""
This file reads an SBML file using libSBML, expands all function definitions
and initial assignments, converts local parameters to global ones,
then writes out the resulting ODE system for use with scipy.integrate.

It emits a function called simulateModel(t0, tend, numPoints),
calls it, plots the result, and writes the generated code to 'generated.py'.
"""

from libsbml import *
import sys


def generateCodeForFile(filename, t0=0, tEnd=10, numPoints=1000):
    # Read the SBML from file
    doc = readSBMLFromFile(filename)
    if doc.getNumErrors(LIBSBML_SEV_FATAL):
        print("Encountered serious errors while reading file")
        print(doc.getErrorLog().toString())
        sys.exit(1)

    # Clear errors
    doc.getErrorLog().clearLog()

    # Perform conversions
    for opt in ["promoteLocalParameters", "expandInitialAssignments", "expandFunctionDefinitions"]:
        props = ConversionProperties()
        props.addOption(opt, True)
        if doc.convert(props) != LIBSBML_OPERATION_SUCCESS:
            print(f"The document could not be converted using {opt}")
            print(doc.getErrorLog().toString())

    # Determine variable species
    mod = doc.getModel()
    variables = {}
    for i in range(mod.getNumSpecies()):
        species = mod.getSpecies(i)
        if species.getBoundaryCondition() or species.getId() in variables:
            continue
        variables[species.getId()] = []

    # Start generating code
    code_lines = []
    code_lines.append("from numpy import *\n")
    code_lines.append("from matplotlib.pylab import *\n")
    code_lines.append("from matplotlib.pyplot import *\n")
    code_lines.append("from scipy.integrate import odeint\n\n")

    code_lines.append("def simulateModel(t0, tend, numPoints):\n")

    # Compartments
    code_lines.append("  # compartments\n")
    for i in range(mod.getNumCompartments()):
        c = mod.getCompartment(i)
        code_lines.append(f"  {c.getId()} = {c.getSize()}\n")

    # Global parameters
    code_lines.append("\n  # global parameters\n")
    for i in range(mod.getNumParameters()):
        p = mod.getParameter(i)
        code_lines.append(f"  {p.getId()} = {p.getValue()}\n")

    # Boundary species
    code_lines.append("\n  # boundary species\n")
    for i in range(mod.getNumSpecies()):
        s = mod.getSpecies(i)
        if not s.getBoundaryCondition():
            continue
        if s.isSetInitialConcentration():
            val = s.getInitialConcentration()
        else:
            val = s.getInitialAmount()
        code_lines.append(f"  {s.getId()} = {val}\n")

    # Define ODE function
    code_lines.append("\n  def ode_fun(__Y__, t):\n")
    keys = list(variables.keys())
    for i, key in enumerate(keys):
        code_lines.append(f"    {key} = __Y__[{i}]\n")
    code_lines.append("\n")

    # Reactions and kinetics
    for i in range(mod.getNumReactions()):
        r = mod.getReaction(i)
        kinetics = r.getKineticLaw()
        code_lines.append(f"    {r.getId()} = {kinetics.getFormula()}\n")

        for j in range(r.getNumReactants()):
            ref = r.getReactant(j)
            s = mod.getSpecies(ref.getSpecies())
            if s.getBoundaryCondition():
                continue
            stoich = ref.getStoichiometry()
            expr = f"-({stoich})*{r.getId()}" if stoich != 1.0 else f"-{r.getId()}"
            variables[s.getId()].append(expr)

        for j in range(r.getNumProducts()):
            ref = r.getProduct(j)
            s = mod.getSpecies(ref.getSpecies())
            if s.getBoundaryCondition():
                continue
            stoich = ref.getStoichiometry()
            expr = f"({stoich})*{r.getId()}" if stoich != 1.0 else f"{r.getId()}"
            variables[s.getId()].append(expr)

    code_lines.append("\n    return array([\n")
    for i, key in enumerate(keys):
        expr = " + ".join(variables[key]) if variables[key] else "0"
        sep = ",\n" if i < len(keys) - 1 else ""
        code_lines.append(f"      ({expr}){sep}\n")
    code_lines.append("    ])\n\n")

    # Time and initial values
    code_lines.append("  time = linspace(t0, tend, numPoints)\n")
    code_lines.append("  yinit = array([\n")
    for i, key in enumerate(keys):
        e = mod.getElementBySId(key)
        if e.getTypeCode() == SBML_PARAMETER:
            val = e.getValue()
        elif e.getTypeCode() == SBML_SPECIES:
            val = e.getInitialConcentration() if e.isSetInitialConcentration() else e.getInitialAmount()
        else:
            val = e.getSize()
        sep = ",\n" if i < len(keys) - 1 else ""
        code_lines.append(f"    {val}{sep}\n")
    code_lines.append("  ])\n\n")

    code_lines.append("  y = odeint(ode_fun, yinit, time)\n\n")
    code_lines.append("  return time, y\n\n\n")

    # Call simulation and plot
    code_lines.append(f"time, result = simulateModel({t0}, {tEnd}, {numPoints})\n\n")
    code_lines.append("fig = figure()\n")
    code_lines.append("ax = subplot(111)\n")
    for i, key in enumerate(keys):
        code_lines.append(f"plot(time, result[:, {i}], label='{key}', lw=1.5)\n")
    code_lines.append("box = ax.get_position()\n")
    code_lines.append("ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])\n")
    code_lines.append("xlabel('time')\n")
    code_lines.append("ylabel('concentration')\n")
    code_lines.append("legend(loc='center left', bbox_to_anchor=(1, 0.5))\n")
    code_lines.append("show()\n")

    # Return the generated Python code
    return "".join(code_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python SBMLexportInternet.py <model_file.xml>")
        sys.exit(1)

    result = generateCodeForFile(sys.argv[1])
    with open("generated.py", "w") as f:
        f.write(result)

    exec(result)
