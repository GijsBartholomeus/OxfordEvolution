import re
import numpy as np

# Try to import libsbml, fall back to simple XML writer if not available
try:
    import libsbml
    HAS_LIBSBML = True
    print("Using libsbml for SBML generation")
except ImportError:
    HAS_LIBSBML = False
    print("libsbml not found, using simple XML writer")
    from simple_sbml_writer import create_simple_sbml

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

def build_sbml(species_list, initial_conditions, parameters, functions, odes, output_file="model.xml"):
    # --- Step 4: Create SBML doc ---
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("BuddingYeastCellCycle_2015")
    model.setName("Budding Yeast Cell Cycle Model 2015")
    
    # Create a default compartment
    compartment = model.createCompartment()
    compartment.setId("cell")
    compartment.setName("Cell")
    compartment.setConstant(True)
    compartment.setSize(1.0)
    
    print(f"Creating SBML with {len(species_list)} species...")
    
    # Add species with initial concentrations
    for i, species_name in enumerate(species_list):
        if species_name in odes:  # Only add species that have ODEs
            species = model.createSpecies()
            species.setId(species_name)
            species.setName(species_name)
            species.setCompartment("cell")
            species.setHasOnlySubstanceUnits(False)
            species.setConstant(False)
            species.setBoundaryCondition(False)
            
            # Set initial concentration
            initial_value = initial_conditions.get(species_name, 0.0)
            species.setInitialConcentration(initial_value)
    
    print(f"Added {model.getNumSpecies()} species to SBML")
    
    # Add parameters
    for param_name, param_value in parameters.items():
        parameter = model.createParameter()
        parameter.setId(param_name)
        parameter.setName(param_name)
        parameter.setValue(param_value)
        parameter.setConstant(True)
    
    print(f"Added {model.getNumParameters()} parameters to SBML")
    
    # Add intermediate functions as assignment rules
    for func_name, func_expr in functions.items():
        if func_name not in odes:  # Don't add species as assignment rules
            assignment_rule = model.createAssignmentRule()
            assignment_rule.setVariable(func_name)
            assignment_rule.setFormula(func_expr)
    
    print(f"Added {model.getNumRules()} assignment rules to SBML")
    
    # Add rate rules for ODEs
    for species_name, ode_expr in odes.items():
        if species_name in [s.getId() for s in model.getListOfSpecies()]:
            rate_rule = model.createRateRule()
            rate_rule.setVariable(species_name)
            rate_rule.setFormula(ode_expr)
    
    print(f"Added rate rules for {len([r for r in model.getListOfRules() if r.isRate()])} species")
    
    # Write SBML to file
    writer = libsbml.SBMLWriter()
    success = writer.writeSBMLToFile(doc, output_file)
    
    if success:
        print(f"SBML model successfully written to {output_file}")
    else:
        print(f"Error writing SBML model to {output_file}")
        
    # Validate the model
    doc.checkConsistency()
    errors = doc.getNumErrors()
    if errors > 0:
        print(f"SBML validation found {errors} issues:")
        for i in range(errors):
            error = doc.getError(i)
            print(f"  {error.getSeverityAsString()}: {error.getMessage()}")
    else:
        print("SBML model passed validation")
    
    return doc

# --- Example usage ---
if __name__ == "__main__":
    try:
        print("Parsing Budding Yeast Cell Cycle MATLAB model...")
        species, initial_conditions, params, functions, odes = parse_matlab_model("BuddingYeastCellCycle_2015.m")
        
        print("\n=== Parsing Summary ===")
        print(f"Species found: {len(species)}")
        print(f"Initial conditions: {len(initial_conditions)}")
        print(f"Parameters: {len(params)}")
        print(f"Functions: {len(functions)}")
        print(f"ODEs: {len(odes)}")
        
        print("\nBuilding SBML model...")
        doc = build_sbml(species, initial_conditions, params, functions, odes, "BuddingYeastCellCycle_2015.xml")
        
        print("\n=== Conversion Complete ===")
        print("SBML file: BuddingYeastCellCycle_2015.xml")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
