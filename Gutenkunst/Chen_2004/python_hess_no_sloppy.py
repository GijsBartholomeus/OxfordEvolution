#!/usr/bin/env python3
# inspect_hess_no_sloppy.py
# Inspect hessian_keys.dat, hessian.dat and hess_dict.*.bp without needing SloppyCell.
# Usage:
#   python inspect_hess_no_sloppy.py                # runs in current dir
#   python inspect_hess_no_sloppy.py /path/to/Chen_2004

import os, sys, glob, pickle
import numpy as np

SEARCH_TERMS = ['CLB2', 'SIC1', 'CLB2T', 'SIC1T', 'CLB2_SIC1', 'CLB2:SIC1']

def read_keys(path):
    if not os.path.exists(path):
        print("No", path)
        return []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        keys = [ln.strip() for ln in f if ln.strip()]
    return keys

def load_hessian_dat(path):
    if not os.path.exists(path):
        print("No", path)
        return None
    # try text matrix first
    try:
        H = np.loadtxt(path)
        H = np.atleast_2d(H)
        return H
    except Exception:
        pass
    # fallback: try pickle (sometimes saved as array pickles)
    try:
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)
            arr = np.array(obj)
            return np.atleast_2d(arr)
    except Exception as e:
        print("Could not load hessian.dat:", e)
        return None

def try_unpickle(fn):
    try:
        with open(fn, 'rb') as fh:
            obj = pickle.load(fh)
        return obj
    except Exception as e:
        # try reading as ascii lines if it's text
        try:
            with open(fn, 'r', encoding='utf-8', errors='ignore') as fh:
                txt = fh.read()
            return txt
        except Exception:
            return ("UNREADABLE", str(e))

def inspect_keys_and_hessian(model_dir):
    os.chdir(model_dir)
    print("Inspecting:", os.getcwd())
    keys = read_keys('hessian_keys.dat')
    print("\nTotal keys in hessian_keys.dat:", len(keys))
    # Print first 40 keys as sample
    print("\nSample keys (first 40):")
    for i,k in enumerate(keys[:40]):
        print("{:3d} {}".format(i, k))
    # Search for matches
    matched = [(i,k) for i,k in enumerate(keys) if any(term in k for term in SEARCH_TERMS)]
    if matched:
        print("\nMatched keys (CLB2 / SIC1 related):")
        for idx,k in matched:
            print("  {:4d}  {}".format(idx, k))
    else:
        print("\nNo matches for CLB2/SIC1 in hessian_keys.dat (try other substrings).")

    # Load full hessian
    H = load_hessian_dat('hessian.dat')
    if H is None:
        print("\nCould not load hessian.dat.")
    else:
        print("\nLoaded hessian.dat shape:", H.shape)
        # If we found matches, print top couplings
        for idx,k in matched:
            if idx < H.shape[0]:
                row = H[idx,:]
                order = np.argsort(-np.abs(row))[:12]
                print("\nTop couplings for index {} ({})".format(idx, k))
                for j in order:
                    key_j = keys[j] if j < len(keys) else "<unknown>"
                    print("  {:4d} {:<50s} {: .5e}".format(int(j), key_j, row[j]))
            else:
                print("Index {} out of range for H ({}).".format(idx, H.shape[0]))

def inspect_hess_dict_files(model_dir):
    os.chdir(model_dir)
    bpfiles = sorted(glob.glob('hess_dict.*.bp'))
    if not bpfiles:
        print("\nNo hess_dict.*.bp files found in this directory.")
        return
    print("\nFound {} hess_dict.*.bp files:".format(len(bpfiles)))
    for fn in bpfiles:
        print("\n---", fn, "---")
        obj = try_unpickle(fn)
        if isinstance(obj, str):
            print("File read as text; first 400 chars:\n", obj[:400])
            continue
        if obj == ("UNREADABLE",):
            print("Could not unpickle", fn)
            continue
        # Many Salv saved structure: (h, h_d, keys)
        if isinstance(obj, (list, tuple)) and len(obj) >= 3:
            h = obj[0]
            h_d = obj[1]
            keys_local = obj[2]
            print(" Num keys in this condition:", len(keys_local))
            # show sample keys
            print(" Sample keys:", keys_local[:10])
            # find any local keys that match search terms
            matched_local = [k for k in keys_local if any(term in k for term in SEARCH_TERMS)]
            if matched_local:
                print(" Matched local keys:", matched_local)
            # show info about h_d if dict-like
            if isinstance(h_d, dict):
                print(" h_d is dict with {} entries. Sample keys:".format(len(h_d)))
                for count,k in enumerate(list(h_d.keys())[:20]):
                    val = h_d[k]
                    try:
                        arr = np.array(val)
                        norm = np.linalg.norm(arr)
                        print("   {:2d} {:<40s} shape:{:<12} norm:{: .4e}".format(count, str(k), arr.shape, norm))
                    except Exception:
                        print("   {:2d} {:<40s} type:{}".format(count, str(k), type(val)))
            else:
                print(" h_d is of type", type(h_d))
        else:
            print("Unexpected object type:", type(obj))
            # If it's a dict directly, print keys
            if isinstance(obj, dict):
                print("dict keys:", list(obj.keys())[:40])

def grep_nets_and_py(model_dir):
    print("\nSearching python files for CLB2/SIC1 mentions (Nets.py, reproduction.py, etc):")
    for root, dirs, files in os.walk(model_dir):
        for fn in files:
            if fn.endswith('.py'):
                full = os.path.join(root, fn)
                try:
                    with open(full, 'r', encoding='utf-8', errors='ignore') as fh:
                        for i,line in enumerate(fh, start=1):
                            if any(term in line for term in SEARCH_TERMS):
                                print(" {}:{}: {}".format(full, i, line.strip()))
                except Exception:
                    pass

if __name__ == '__main__':
    model_dir = sys.argv[1] if len(sys.argv)>1 else os.getcwd()
    inspect_keys_and_hessian(model_dir)
    inspect_hess_dict_files(model_dir)
    grep_nets_and_py(model_dir)
