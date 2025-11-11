#!/usr/bin/env python3
# list_sbml_ids.py
import xml.etree.ElementTree as ET
import sys, os

fn = sys.argv[1] if len(sys.argv) > 1 else 'BIOMD0000000056.xml'
if not os.path.exists(fn):
    print("File not found:", fn)
    sys.exit(1)

tree = ET.parse(fn)
root = tree.getroot()
ns = {'sbml': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

def find_tag(tag):
    if ns:
        return root.findall('.//{{{}}}{}'.format(ns['sbml'], tag))
    else:
        return root.findall('.//{}'.format(tag))

print("Parsing:", fn)
print()

# species
species = find_tag('species')
print("Total species:", len(species))
for i,s in enumerate(species[:200], start=1):
    sid = s.attrib.get('id') or s.attrib.get('name')
    name = s.attrib.get('name','')
    initial = s.attrib.get('initialAmount') or s.attrib.get('initialConcentration','')
    print(f"{i:3d} id={sid:30s} name={name:30s} initial={initial}")
print()

# parameters (global)
params = find_tag('parameter')
print("Total global parameters:", len(params))
for i,p in enumerate(params[:200], start=1):
    pid = p.attrib.get('id') or p.attrib.get('name')
    val = p.attrib.get('value') or p.attrib.get('initialValue','')
    print(f"{i:3d} id={pid:30s} value={val}")
print()

# optionally, list reactions summary
reactions = find_tag('reaction')
print("Total reactions:", len(reactions))
for i,r in enumerate(reactions[:100], start=1):
    rid = r.attrib.get('id') or r.attrib.get('name') or f"reaction_{i}"
    # try get kinetic law text
    kl = r.find('.//{{{}}}math'.format(ns['sbml'])) if ns else r.find('.//math')
    kl_text = ET.tostring(kl, encoding='unicode')[:160] if kl is not None else ''
    print(f"{i:3d} id={rid:30s} kinetics={kl_text}")
