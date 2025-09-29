"""
Simple SBML XML writer without libsbml dependency
This creates basic SBML XML structure for testing the parser
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_simple_sbml(species_list, initial_conditions, parameters, functions, odes, output_file="model.xml"):
    """
    Create a simplified SBML XML file without libsbml
    """
    
    # Create root SBML element
    sbml = ET.Element("sbml")
    sbml.set("xmlns", "http://www.sbml.org/sbml/level3/version2/core")
    sbml.set("level", "3")
    sbml.set("version", "2")
    
    # Create model
    model = ET.SubElement(sbml, "model")
    model.set("id", "BuddingYeastCellCycle_2015")
    model.set("name", "Budding Yeast Cell Cycle Model 2015")
    
    # Create compartments
    list_of_compartments = ET.SubElement(model, "listOfCompartments")
    compartment = ET.SubElement(list_of_compartments, "compartment")
    compartment.set("id", "cell")
    compartment.set("name", "Cell")
    compartment.set("spatialDimensions", "3")
    compartment.set("size", "1")
    compartment.set("constant", "true")
    
    # Create species
    list_of_species = ET.SubElement(model, "listOfSpecies")
    
    species_count = 0
    for species_name in species_list:
        if species_name in odes:  # Only add species that have ODEs
            species = ET.SubElement(list_of_species, "species")
            species.set("id", species_name)
            species.set("name", species_name)
            species.set("compartment", "cell")
            species.set("hasOnlySubstanceUnits", "false")
            species.set("boundaryCondition", "false")
            species.set("constant", "false")
            
            # Set initial concentration
            initial_value = initial_conditions.get(species_name, 0.0)
            species.set("initialConcentration", str(initial_value))
            species_count += 1
    
    print(f"Added {species_count} species to SBML")
    
    # Create parameters
    list_of_parameters = ET.SubElement(model, "listOfParameters")
    
    param_count = 0
    for param_name, param_value in parameters.items():
        parameter = ET.SubElement(list_of_parameters, "parameter")
        parameter.set("id", param_name)
        parameter.set("name", param_name)
        parameter.set("value", str(param_value))
        parameter.set("constant", "true")
        param_count += 1
    
    print(f"Added {param_count} parameters to SBML")
    
    # Create rules
    list_of_rules = ET.SubElement(model, "listOfRules")
    
    # Add assignment rules for functions
    assignment_count = 0
    for func_name, func_expr in functions.items():
        if func_name not in odes:  # Don't add species as assignment rules
            assignment_rule = ET.SubElement(list_of_rules, "assignmentRule")
            assignment_rule.set("variable", func_name)
            
            math = ET.SubElement(assignment_rule, "math")
            math.set("xmlns", "http://www.w3.org/1998/Math/MathML")
            
            # For now, add the expression as text (would need proper MathML conversion)
            apply = ET.SubElement(math, "apply")
            ci = ET.SubElement(apply, "ci")
            ci.text = func_expr
            assignment_count += 1
    
    # Add rate rules for ODEs
    rate_count = 0
    for species_name, ode_expr in odes.items():
        if any(s.get("id") == species_name for s in list_of_species.findall("species")):
            rate_rule = ET.SubElement(list_of_rules, "rateRule")
            rate_rule.set("variable", species_name)
            
            math = ET.SubElement(rate_rule, "math")
            math.set("xmlns", "http://www.w3.org/1998/Math/MathML")
            
            # For now, add the expression as text (would need proper MathML conversion)
            apply = ET.SubElement(math, "apply")
            ci = ET.SubElement(apply, "ci")
            ci.text = ode_expr
            rate_count += 1
    
    print(f"Added {assignment_count} assignment rules and {rate_count} rate rules to SBML")
    
    # Write to file with pretty formatting
    rough_string = ET.tostring(sbml, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    with open(output_file, 'w') as f:
        f.write(pretty_xml)
    
    print(f"Simple SBML model written to {output_file}")
    
    return True