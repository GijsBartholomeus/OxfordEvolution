import re
import libsbml

def parse_matlab_model(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    # --- Step 1: Extract species names ---
    # Looks like allNames = {'CLB2','CLN2','SIC1',...}
    species_match = re.search(r"allNames\s*=\s*\{([^\}]*)\}", text, re.S)
    species_raw = species_match.group(1)
    species_list = [s.strip(" '\n") for s in species_raw.split(",") if s.strip()]
    
    # --- Step 2: Extract parameters ---
    # Look for param(i) = value
    params = {}
    for m in re.finditer(r"param\((\d+)\)\s*=\s*([0-9eE\.\-\+]+)", text):
        idx, val = m.groups()
        params[f"p{idx}"] = float(val)
    
    # --- Step 3: Extract ODEs ---
    # Look for dydt(i) = expression
    odes = {}
    for m in re.finditer(r"dydt\((\d+)\)\s*=\s*(.+)", text):
        idx, expr = m.groups()
        expr = expr.strip().rstrip(";")
        expr = expr.replace(".*", "*").replace("./", "/").replace(".^", "^")
        if int(idx) <= len(species_list):
            odes[species_list[int(idx)-1]] = expr
    
    return species_list, params, odes

def build_sbml(species_list, params, odes, output_file="model.xml"):
    # --- Step 4: Create SBML doc ---
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.createCompartment().setId("cell")
    
    # Species (no initial concentrations here yet, set to 0 by default)
    for s in species_list:
        sp = model.createSpecies()
        sp.setId(s)
        sp.setCompartment("cell")
        sp.setInitialConcentration(0.0)
    
    # Parameters
    for p, val in params.items():
        param = model.createParameter()
        param.setId(p)
        param.setValue(val)
    
    # Rate rules
    for s, ode in odes.items():
        rule = model.createRateRule()
        rule.setVariable(s)
        rule.setFormula(ode)
    
    libsbml.writeSBMLToFile(doc, output_file)
    print(f"SBML model written to {output_file}")

# --- Example usage ---
species, params, odes = parse_matlab_model("BuddingYeastCellCycle_2015.m")
build_sbml(species, params, odes, "BuddingYeastCellCycle_2015.xml")
