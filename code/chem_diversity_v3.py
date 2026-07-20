#!/usr/bin/env python3
"""Cross-target chemical diversity analysis using Morgan fingerprints + Tanimoto."""
import json, os, glob
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

TARGETS = ["3V8D", "1ERE", "3EML", "1EVE", "4DFR", "3PJC", "4MNE"]
METHODS = ["random", "imgep_naive", "imgep", "curiosity", "bo", "ga"]
METHOD_LABELS = {
    "random": "Random", "imgep_naive": "IMGEP (naive)", "imgep": "IMGEP (adaptive)",
    "curiosity": "Curiosity-IMGEP", "bo": "Bayesian Opt.", "ga": "Genetic Alg."
}
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument('--results-dir', default='results',
                 help='directory containing per-run results.json (default: results)')
_ap.add_argument('--out', dest='output_dir', default='figures',
                 help='directory to write outputs (default: figures)')
_args, _ = _ap.parse_known_args()
RESULTS_DIR = _args.results_dir
OUTPUT_DIR = _args.output_dir

def compute_tanimoto_diversity(smiles_list):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    dists = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            dists.append(1.0 - TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(dists))

def get_murcko_scaffolds(smiles_list):
    scaffs = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            scaff = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
            scaffs.add(Chem.MolToSmiles(scaff))
    return len(scaffs)

def compute_qed(smiles_list):
    qeds = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                qeds.append(Descriptors.qed(mol))
            except:
                pass
    return float(np.mean(qeds)) if qeds else 0.0

def lipinski_pass_rate(smiles_list):
    ok = 0
    total = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            total += 1
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                ok += 1
    return ok / total if total > 0 else 0.0

# Collect per-method stats across all targets
method_tani = defaultdict(list)
method_scaff = defaultdict(list)
method_qed = defaultdict(list)
method_lip = defaultdict(list)
method_umols = defaultdict(list)

for target in TARGETS:
    for method in METHODS:
        for seed in range(10):
            rdir = f"{RESULTS_DIR}/{target}/{method}_seed{seed}"
            rfile = os.path.join(rdir, "results.json")
            if not os.path.exists(rfile):
                continue
            with open(rfile) as f:
                data = json.load(f)
            smiles = list(set(d["smiles"] for d in data["discoveries"] if d.get("smiles")))
            if len(smiles) < 5:
                continue
            tani = compute_tanimoto_diversity(smiles)
            scaff = get_murcko_scaffolds(smiles)
            qed = compute_qed(smiles)
            lip = lipinski_pass_rate(smiles)
            
            method_tani[method].append(tani)
            method_scaff[method].append(scaff)
            method_qed[method].append(qed)
            method_lip[method].append(lip)
            method_umols[method].append(len(smiles))

print("=" * 80)
print("CHEMICAL DIVERSITY ANALYSIS — Cross-target (4 targets × 10 seeds)")
print("=" * 80)
print(f"{'Method':<22} {'Tanimoto':>12} {'Scaffolds':>12} {'QED':>10} {'Lipinski%':>10} {'UniqMols':>10}")
print("-" * 80)
for method in METHODS:
    if not method_tani[method]:
        continue
    label = METHOD_LABELS[method]
    t = method_tani[method]
    s = method_scaff[method]
    q = method_qed[method]
    l = method_lip[method]
    u = method_umols[method]
    print(f"{label:<22} {np.mean(t):.3f}±{np.std(t):.3f} {np.mean(s):.0f}±{np.std(s):.0f}    {np.mean(q):.3f}±{np.std(q):.3f} {np.mean(l)*100:.1f}%±{np.std(l)*100:.1f}  {np.mean(u):.0f}±{np.std(u):.0f}")

# Print per-target breakdown for Curiosity vs Random
print("\n" + "=" * 80)
print("PER-TARGET: Curiosity-IMGEP vs Random (Tanimoto diversity)")
print("=" * 80)
for target in TARGETS:
    cur_tan = []
    ran_tan = []
    for seed in range(10):
        for method, lst in [("curiosity", cur_tan), ("random", ran_tan)]:
            rfile = f"{RESULTS_DIR}/{target}/{method}_seed{seed}/results.json"
            if not os.path.exists(rfile):
                continue
            with open(rfile) as f:
                data = json.load(f)
            smiles = list(set(d["smiles"] for d in data["discoveries"] if d.get("smiles")))
            lst.append(compute_tanimoto_diversity(smiles))
    if cur_tan and ran_tan:
        print(f"{target}: Curiosity {np.mean(cur_tan):.3f}±{np.std(cur_tan):.3f}  Random {np.mean(ran_tan):.3f}±{np.std(ran_tan):.3f}  Δ={np.mean(cur_tan)-np.mean(ran_tan):+.3f}")
