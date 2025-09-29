import re
import libsbml
import numpy as np

def parse_matlab_model(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    # --- Step 1: Extract species names from allNames array ---
    species_match = re.search(r"allNames\s*=\s*\{([^\}]*)\}", text, re.S)
    if species_match:
        species_raw = species_match.group(1)
        species_list = [s.strip(" '\n;") for s in species_raw.split(";") if s.strip()]
    else:
        raise ValueError("Could not find allNames array")
    
    print(f"Found {len(species_list)} species")
    
    # --- Step 2: Extract initial conditions from yinit array ---
    initial_conditions = {}
    
    # Find the yinit array
    yinit_match = re.search(r"yinit\s*=\s*\[(.*?)\];", text, re.S)
    if yinit_match:
        yinit_raw = yinit_match.group(1)
        # Extract values and comments
        yinit_lines = [line.strip() for line in yinit_raw.split('\n') if line.strip()]
        
        for line in yinit_lines:
            if ';' in line and '%' in line:
                value_part = line.split(';')[0].strip()
                comment_part = line.split('%')[1].strip()
                
                # Extract value
                try:
                    value = float(value_part)
                except:
                    continue
                
                # Extract species name from comment
                species_match = re.search(r"'([^']*)'", comment_part)
                if species_match:
                    species_name = species_match.group(1)
                    initial_conditions[species_name] = value
    
    print(f"Found {len(initial_conditions)} initial conditions")
    
    # --- Step 3: Extract parameters from param array ---
    parameters = {}
    
    # Find the param array
    param_match = re.search(r"param\s*=\s*\[(.*?)\];", text, re.S)
    if param_match:
        param_raw = param_match.group(1)
        # Extract values and comments  
        param_lines = [line.strip() for line in param_raw.split('\n') if line.strip()]
        
        for line in param_lines:
            if ';' in line and '%' in line:
                value_part = line.split(';')[0].strip()
                comment_part = line.split('%')[1].strip()
                
                # Extract value
                try:
                    # Handle expressions like (1.0 ./ 602.0)
                    if '(' in value_part:
                        value = eval(value_part.replace('./', '/').replace('.*', '*'))
                    else:
                        value = float(value_part)
                except:
                    continue
                
                # Extract parameter name from comment
                param_match = re.search(r"'([^']*)'", comment_part)
                if param_match:
                    param_name = param_match.group(1)
                    parameters[param_name] = value
    
    print(f"Found {len(parameters)} parameters")
    
    # --- Step 4: Extract intermediate functions ---
    functions = {}
    
    # Find all functions in the getRow and f functions
    function_blocks = re.findall(r"% Functions\n(.*?)% (?:Rates|OutputFunctions)", text, re.S)
    
    if function_blocks:
        function_text = function_blocks[0]
        function_lines = [line.strip() for line in function_text.split('\n') if line.strip() and not line.strip().startswith('%')]
        
        for line in function_lines:
            if '=' in line and not line.startswith('%'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    func_name = parts[0].strip()
                    func_expr = parts[1].strip().rstrip(';')
                    # Convert MATLAB syntax to more standard form
                    func_expr = convert_matlab_syntax(func_expr)
                    functions[func_name] = func_expr
    
    print(f"Found {len(functions)} intermediate functions")
    
    # --- Step 5: Extract ODEs from dydt array ---
    odes = {}
    
    # Find the dydt array assignment
    dydt_match = re.search(r"dydt\s*=\s*\[(.*?)\];", text, re.S)
    if dydt_match:
        dydt_raw = dydt_match.group(1)
        dydt_lines = [line.strip() for line in dydt_raw.split('\n') if line.strip()]
        
        for i, line in enumerate(dydt_lines):
            if ';' in line and '%' in line:
                expr_part = line.split(';')[0].strip()
                comment_part = line.split('%')[1].strip()
                
                # Remove parentheses from expression
                if expr_part.startswith('(') and expr_part.endswith(')'):
                    expr_part = expr_part[1:-1]
                
                # Extract species name from comment
                if 'rate for' in comment_part:
                    species_name = comment_part.split('rate for')[1].strip().strip("'")
                    # Convert MATLAB syntax
                    expr_part = convert_matlab_syntax(expr_part)
                    odes[species_name] = expr_part
                elif i < len(species_list):
                    # Fallback: use position in species list
                    expr_part = convert_matlab_syntax(expr_part)  
                    odes[species_list[i]] = expr_part
    
    print(f"Found {len(odes)} ODEs")
    
    return species_list, initial_conditions, parameters, functions, odes

def convert_matlab_syntax(expr):
    """Convert MATLAB-specific syntax to more standard mathematical notation"""
    # Convert element-wise operations
    expr = expr.replace('.*', '*')
    expr = expr.replace('./', '/')
    expr = expr.replace('.^', '^')
    
    # Convert logical operations
    expr = expr.replace('!', 'not ')
    
    # Convert conditional expressions (MATLAB style)
    # Handle patterns like ((condition) .* expression1) + ((!condition) .* expression2)
    # This is a simplified conversion - full conversion would require more sophisticated parsing
    
    return expr

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
