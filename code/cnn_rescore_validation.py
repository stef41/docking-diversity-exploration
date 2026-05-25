"""
CNN rescoring orthogonal validation.

Re-docks molecules discovered by each method using GNINA with CNN scoring
(instead of Vina scoring used during search), extracts interactions via PLIP,
and computes unique profile counts under the alternative scoring function.

This validates whether diversity gains are robust to scoring function choice.
"""
import json
import os
import sys
import subprocess
import tempfile
import time
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import logging
logging.getLogger('plip').setLevel(logging.CRITICAL)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rdkit import Chem
from rdkit.Chem import AllChem
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport
from Bio.PDB import PDBParser

# --- Configuration ---
GNINA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                          'examples', 'docking', 'systems', 'gnina')
TARGETS_DIR = os.path.join(os.path.dirname(__file__), 'targets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_v3')

TARGET = sys.argv[1] if len(sys.argv) > 1 else '3PJC'
PROTEIN = os.path.join(TARGETS_DIR, f'{TARGET}_A_protein.pdb')
BOX_FILE = os.path.join(TARGETS_DIR, f'{TARGET}_A_box.txt')

METHODS = ['random', 'curiosity', 'mapelites', 'ga', 'bo', 'novelty']
SEED = 0

d_MIN = 1.5
d_MAX = 4.0
INTERACTION_TYPES = ["hydrophobic", "hbond_donor", "hbond_acceptor"]


def read_box(box_file):
    with open(box_file) as f:
        lines = f.readlines()
    vals = {}
    for line in lines:
        key, val = line.strip().split('=')
        vals[key.strip()] = float(val.strip())
    return vals


def residues_in_bounding_box(pdb_file, center, size):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    cx, cy, cz = center
    sx, sy, sz = size
    xmin, xmax = cx - sx/2, cx + sx/2
    ymin, ymax = cy - sy/2, cy + sy/2
    zmin, zmax = cz - sz/2, cz + sz/2

    residue_ids = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.coord
                    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                        residue_ids.add(residue.get_id()[1])
                        break
    return sorted(residue_ids)


def generate_ligand_pdb(smiles, output_file):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    ret = AllChem.EmbedMolecule(mol, randomSeed=42)
    if ret != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000000)
    except Exception:
        pass
    Chem.MolToPDBFile(mol, output_file)
    return True


def dock_with_cnn(protein, ligand_pdb, out_pdb, box_params):
    cmd = [
        GNINA_PATH,
        "--center_x", str(box_params['center_x']),
        "--center_y", str(box_params['center_y']),
        "--center_z", str(box_params['center_z']),
        "--size_x", str(box_params['size_x']),
        "--size_y", str(box_params['size_y']),
        "--size_z", str(box_params['size_z']),
        "--num_mc_saved", "1",
        "--num_modes", "1",
        "--seed", "42",
        "--autobox_extend", "1",
        "--exhaustiveness", "16",
        "--cnn_scoring", "rescore",  # KEY: use CNN rescoring
        "--no_gpu",
        "--verbosity=0",
        "-r", protein,
        "-l", ligand_pdb,
        "--out", out_pdb
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=120)
    return result.returncode == 0


def extract_interactions(complex_pdb_str):
    protlig = PDBComplex()
    protlig.load_pdb(complex_pdb_str, as_string=True)
    for ligand in protlig.ligands:
        protlig.characterize_complex(ligand)

    key_site = sorted(protlig.interaction_sets.items())
    if not key_site:
        return {t: [] for t in INTERACTION_TYPES}
    key, site = key_site[0]
    bs = BindingSiteReport(site)

    interactions = {}
    hydrophobic_info = getattr(bs, "hydrophobic_info", [])
    interactions["hydrophobic"] = [(int(r[0]), float(r[6])) for r in hydrophobic_info]
    hbond_info = getattr(bs, "hbond_info", [])
    interactions["hbond_donor"] = [(int(r[0]), float(r[7])) for r in hbond_info if r[10] is True]
    interactions["hbond_acceptor"] = [(int(r[0]), float(r[7])) for r in hbond_info if r[10] is False]
    return interactions


def compute_fingerprint(interactions, residue_ids):
    stats = []
    for res_id in residue_ids:
        for int_type in INTERACTION_TYPES:
            matching = [d for (r, d) in interactions.get(int_type, []) if r == res_id]
            if matching:
                d_near = min(matching)
                val = (d_MAX - d_near) / (d_MAX - d_MIN)
                stats.append(max(0.0, min(1.0, val)))
            else:
                stats.append(0.0)
    stats.append(0.0)  # placeholder Vina (not used for profile counting)
    return tuple(round(x, 2) for x in stats)


# Module-level globals set by main() before Pool is created
_PROT_BLOCK = None
_RESIDUE_IDS = None
_BOX = None

def dock_one(smi):
    """Worker function: dock one SMILES with CNN rescoring, return (smi, fingerprint_or_None)."""
    with tempfile.TemporaryDirectory(prefix="cnn_") as tmp:
        lig_pdb = os.path.join(tmp, "lig.pdb")
        dock_pdb = os.path.join(tmp, "docked.pdb")

        if not generate_ligand_pdb(smi, lig_pdb):
            return (smi, None)

        try:
            ok = dock_with_cnn(PROTEIN, lig_pdb, dock_pdb, _BOX)
        except subprocess.TimeoutExpired:
            return (smi, None)

        if not ok or not os.path.exists(dock_pdb) or os.path.getsize(dock_pdb) == 0:
            return (smi, None)

        with open(dock_pdb) as f:
            lig_het = [l for l in f.readlines() if l.startswith('HETATM')]
        if not lig_het:
            return (smi, None)

        cplx = _PROT_BLOCK + ''.join(lig_het) + 'END\n'
        interactions = extract_interactions(cplx)
        fp = compute_fingerprint(interactions, _RESIDUE_IDS)
        return (smi, fp)


def main():
    t_start = time.time()
    box = read_box(BOX_FILE)
    residue_ids = residues_in_bounding_box(
        PROTEIN,
        (box['center_x'], box['center_y'], box['center_z']),
        (box['size_x'], box['size_y'], box['size_z'])
    )
    print(f"Target: {TARGET}, {len(residue_ids)} pocket residues")
    print(f"Fingerprint dims: {len(residue_ids)*3 + 1}")
    print()

    # --- Collect unique SMILES per method ---
    method_smiles = {}
    for method in METHODS:
        path = os.path.join(RESULTS_DIR, TARGET, f'{method}_seed{SEED}', 'results.json')
        with open(path) as f:
            data = json.load(f)
        seen = set()
        uniq = []
        for d in data['discoveries']:
            if d['smiles'] not in seen:
                seen.add(d['smiles'])
                uniq.append(d['smiles'])
        method_smiles[method] = uniq
        print(f"{method}: {len(uniq)} unique SMILES")

    # --- Deduplicate across all methods ---
    all_smiles = set()
    for sms in method_smiles.values():
        all_smiles.update(sms)
    all_smiles = sorted(all_smiles)
    print(f"\nTotal unique SMILES (deduped): {len(all_smiles)}")

    # --- Read protein lines once ---
    with open(PROTEIN) as f:
        prot_lines = [l for l in f.readlines() if l.startswith('ATOM')]

    # Set module-level globals for worker processes
    global _PROT_BLOCK, _RESIDUE_IDS, _BOX
    _PROT_BLOCK = ''.join(prot_lines)
    _RESIDUE_IDS = residue_ids
    _BOX = box

    # --- Re-dock all unique SMILES with CNN scoring (parallel) ---
    N_WORKERS = min(16, cpu_count())
    print(f"\nDocking with {N_WORKERS} parallel workers...")

    smi_to_fp = {}
    success, fail = 0, 0

    with Pool(N_WORKERS) as pool:
        for i, (smi, fp) in enumerate(pool.imap_unordered(dock_one, all_smiles)):
            if fp is not None:
                smi_to_fp[smi] = fp
                success += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(all_smiles) - i - 1) / rate / 3600
                print(f"  [{i+1}/{len(all_smiles)}] {success} ok, {fail} fail, "
                      f"{len(set(smi_to_fp.values()))} unique profiles, "
                      f"~{remaining:.1f}h remaining")

    elapsed = time.time() - t_start
    print(f"\nDone: {success} docked, {fail} failed in {elapsed/3600:.1f}h")
    print(f"Total unique CNN profiles: {len(set(smi_to_fp.values()))}")

    # --- Compute per-method stats ---
    print()
    print("=" * 70)
    print(f"{'Method':>12} {'SMILES':>7} {'Docked':>7} {'CNN profiles':>13} "
          f"{'Vina profiles':>14}")
    print("-" * 70)

    results = {}
    for method in METHODS:
        # CNN profiles for this method
        cnn_fps = set()
        docked = 0
        for smi in method_smiles[method]:
            if smi in smi_to_fp:
                cnn_fps.add(smi_to_fp[smi])
                docked += 1

        # Original Vina profiles from results
        path = os.path.join(RESULTS_DIR, TARGET, f'{method}_seed{SEED}', 'results.json')
        with open(path) as f:
            data = json.load(f)
        vina_fps = set()
        for d in data['discoveries']:
            vina_fps.add(tuple(round(x, 2) for x in d['behavior']))

        print(f"{method:>12} {len(method_smiles[method]):>7} {docked:>7} "
              f"{len(cnn_fps):>13} {len(vina_fps):>14}")

        results[method] = {
            'n_unique_smiles': len(method_smiles[method]),
            'n_docked': docked,
            'n_profiles_cnn': len(cnn_fps),
            'n_profiles_vina': len(vina_fps),
        }

    # Save
    out_path = os.path.join(os.path.dirname(__file__), f'cnn_rescore_results_{TARGET}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'target': TARGET, 'seed': SEED,
            'results': results,
            'smi_to_fp': {s: list(fp) for s, fp in smi_to_fp.items()}
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
