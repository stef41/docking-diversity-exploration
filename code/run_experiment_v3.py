#!/usr/bin/env python3
"""
run_experiment_v3.py — Nature-level experiment runner.

Multi-target support, competitive baselines (BO, GA), larger CReM DB, 
ablation-ready behavior space.

Usage:
  python code/run_experiment_v3.py \\
      --method curiosity --seed 0 --iterations 500 \\
      --target 3V8D \\
      --output_dir results/3V8D/curiosity_seed0
"""
import argparse
import json
import os
import sys
import random
import time
import re
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, DataStructs
from crem.crem import mutate_mol, grow_mol
from Bio.PDB import PDBParser
from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport

import logging
for logger_name in ['plip', 'rdkit', 'urllib3']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ─── Target configurations ───
# BASE_DIR = repo root (parent of code/). Env var REPO_ROOT overrides for out-of-tree installs.
BASE_DIR = os.environ.get("REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# GNINA binary: expect on PATH or via GNINA_BIN env; the value below is only a hint.
GNINA_BIN = os.environ.get("GNINA_BIN", "gnina")

TARGETS = {
    "3V8D": {
        "name": "CYP7A1",
        "family": "Oxidoreductase",
        "protein": os.path.join(BASE_DIR, "targets/3V8D_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/3V8D_A_box.txt"),
    },
    "1ERE": {
        "name": "Estrogen Receptor α",
        "family": "Nuclear receptor",
        "protein": os.path.join(BASE_DIR, "targets/1ERE_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/1ERE_A_box.txt"),
    },
    "3EML": {
        "name": "A2AR",
        "family": "GPCR",
        "protein": os.path.join(BASE_DIR, "targets/3EML_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/3EML_A_box.txt"),
    },
    "1EVE": {
        "name": "AChE",
        "family": "Hydrolase",
        "protein": os.path.join(BASE_DIR, "targets/1EVE_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/1EVE_A_box.txt"),
    },
    "4DFR": {
        "name": "DHFR",
        "family": "Oxidoreductase",
        "protein": os.path.join(BASE_DIR, "targets/4DFR_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/4DFR_A_box.txt"),
    },
    "3PJC": {
        "name": "JAK3",
        "family": "Kinase",
        "protein": os.path.join(BASE_DIR, "targets/3PJC_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/3PJC_A_box.txt"),
    },
    "4MNE": {
        "name": "BRAF",
        "family": "Kinase",
        "protein": os.path.join(BASE_DIR, "targets/4MNE_A_protein.pdb"),
        "box": os.path.join(BASE_DIR, "targets/4MNE_A_box.txt"),
    },
}

# CReM databases (try large first, then v2, then v1)
CREM_DBS = [
    os.path.join(BASE_DIR, "targets/crem_large.db"),
]
CREM_DBS = [db for db in CREM_DBS if os.path.exists(db)]

SEED_MOLECULES = [
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",  # benzylpenicillin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                     # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",                           # aspirin
    "CC(=O)Nc1ccc(O)cc1",                              # acetaminophen
    "OC(=O)c1ccccc1O",                                 # salicylic acid
]

INTERACTION_TYPES = ['hydrophobic', 'hbond_donor', 'hbond_acceptor']
N_INTERACTION_TYPES = len(INTERACTION_TYPES)
d_MIN = 1.5
d_MAX = 4.0


# ─── Docking functions ───

def read_box_file(path):
    with open(path) as f:
        lines = f.readlines()
    return [float(l.split('=')[1].strip()) for l in lines]


def generate_ligand_pdb(smiles, output_file):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    m = Chem.AddHs(m)
    ret = AllChem.EmbedMolecule(m, randomSeed=42)
    if ret != 0:
        raise ValueError(f"Failed to embed: {smiles}")
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    Chem.MolToPDBFile(m, output_file)


def run_gnina(protein, ligand, output, cx, cy, cz, sx, sy, sz):
    cmd = [
        GNINA_BIN,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--num_mc_saved", "1", "--num_modes", "1",
        "--seed", "42", "--autobox_extend", "1",
        "--exhaustiveness", "16",
        "--cnn_scoring", "none",
        "--verbosity=0",
        "-r", protein, "-l", ligand, "--out", output
    ]
    env = os.environ.copy()
    block_nvidia = "/tmp/block_nvidia.so"
    if os.path.exists(block_nvidia):
        env["LD_PRELOAD"] = block_nvidia
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, env=env)


def get_vina_score(docked_pdb_path):
    try:
        with open(docked_pdb_path) as f:
            for line in f:
                if 'REMARK' in line and 'minimizedAffinity' in line:
                    match = re.search(r'minimizedAffinity\s+([-\d.]+)', line)
                    if match:
                        return float(match.group(1))
                if 'REMARK VINA RESULT' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
    except Exception:
        pass
    return 0.0


def merge_pdb(ligand_file, protein_file, output_file):
    with open(ligand_file) as f:
        lig_lines = f.readlines()
    with open(protein_file) as f:
        prot_lines = f.readlines()
    prot_atoms = [l for l in prot_lines if l.startswith('ATOM')]
    lig_hetatms = [l for l in lig_lines if l.startswith('HETATM')]
    combined = prot_atoms + lig_hetatms + ['END\n']
    with open(output_file, 'w') as f:
        f.writelines(combined)


def dock_smiles(smiles, protein_pdb, box_vals, iteration, tmp_dir):
    lig_pdb = os.path.join(tmp_dir, f"ligand_{iteration}.pdb")
    docked_pdb = os.path.join(tmp_dir, f"docked_{iteration}.pdb")
    complex_pdb = os.path.join(tmp_dir, f"complex_{iteration}.pdb")

    generate_ligand_pdb(smiles, lig_pdb)
    run_gnina(protein_pdb, lig_pdb, docked_pdb, *box_vals)

    vina_score = get_vina_score(docked_pdb)

    merge_pdb(docked_pdb, protein_pdb, complex_pdb)
    with open(complex_pdb) as f:
        content = f.read()

    for fp in [lig_pdb, docked_pdb, complex_pdb]:
        if os.path.exists(fp):
            os.unlink(fp)

    return content, vina_score


# ─── Fingerprint computation ───

def get_pocket_residues(protein_pdb, box_vals):
    cx, cy, cz, sx, sy, sz = box_vals
    xmin, xmax = cx - sx/2, cx + sx/2
    ymin, ymax = cy - sy/2, cy + sy/2
    zmin, zmax = cz - sz/2, cz + sz/2

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)
    residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.coord
                    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                        residues.add(residue.id[1])
                        break
    return sorted(residues)


def get_all_interactions(pdb_string):
    protlig = PDBComplex()
    protlig.load_pdb(pdb_string, as_string=True)
    for ligand in protlig.ligands:
        protlig.characterize_complex(ligand)
    sites = sorted(protlig.interaction_sets.items())
    if not sites:
        return {}

    _, site = sites[0]
    report = BindingSiteReport(site)

    residue_interactions = {}

    for info in getattr(report, 'hydrophobic_info', []):
        try:
            resnr = int(info[0])
            dist = float(info[6])
            if resnr not in residue_interactions:
                residue_interactions[resnr] = {t: [] for t in INTERACTION_TYPES}
            residue_interactions[resnr]['hydrophobic'].append(dist)
        except (ValueError, IndexError):
            pass

    for info in getattr(report, 'hbond_info', []):
        try:
            resnr = int(info[0])
            dist = float(info[7])
            is_prot_donor = (info[6] is True or info[6] == 'True')
            itype = 'hbond_donor' if is_prot_donor else 'hbond_acceptor'
            if resnr not in residue_interactions:
                residue_interactions[resnr] = {t: [] for t in INTERACTION_TYPES}
            residue_interactions[resnr][itype].append(dist)
        except (ValueError, IndexError):
            pass

    return residue_interactions


def compute_fingerprint(pdb_string, pocket_residues, vina_score):
    n_res = len(pocket_residues)
    dim = n_res * N_INTERACTION_TYPES + 1

    interactions = get_all_interactions(pdb_string)
    fp = np.zeros(dim)
    fp[-1] = min(1.0, max(0.0, -vina_score / 12.0))

    if not interactions:
        return fp

    res_to_idx = {r: i for i, r in enumerate(pocket_residues)}

    for resnr, type_dists in interactions.items():
        if resnr not in res_to_idx:
            continue
        ridx = res_to_idx[resnr]
        for tidx, itype in enumerate(INTERACTION_TYPES):
            dists = type_dists[itype]
            if dists:
                d = min(dists)
                val = (d_MAX - d) / (d_MAX - d_MIN)
                val = max(0.0, min(1.0, val))
                fp[ridx * N_INTERACTION_TYPES + tidx] = val

    return fp


# ─── Chemical operators ───

def lipinski_ok(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if Lipinski.NumHDonors(mol) > 5:
        return False
    if Lipinski.NumHAcceptors(mol) > 10:
        return False
    if Descriptors.MolWt(mol) > 500:
        return False
    if Crippen.MolLogP(mol) >= 5:
        return False
    try:
        m2 = Chem.AddHs(mol)
        ret = AllChem.EmbedMolecule(m2, randomSeed=42)
        if ret != 0:
            return False
    except Exception:
        return False
    return True


def mutate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    mol_h = Chem.AddHs(mol)

    candidates = []
    for db in CREM_DBS:
        try:
            for smi in grow_mol(mol_h, db_name=db):
                if lipinski_ok(smi):
                    candidates.append(smi)
                if len(candidates) >= 15:
                    break
        except Exception:
            pass
        try:
            for smi in mutate_mol(mol_h, db_name=db):
                if lipinski_ok(smi):
                    candidates.append(smi)
                if len(candidates) >= 15:
                    break
        except Exception:
            pass
        if len(candidates) >= 10:
            break

    if not candidates:
        return smiles
    return Chem.MolToSmiles(Chem.MolFromSmiles(random.choice(candidates)))


def tanimoto_distance(smi1, smi2):
    """Tanimoto distance between Morgan fingerprints."""
    mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 1.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


# ─── Exploration Methods ───

def select_parent_random(history_smiles, history_behaviors, **kw):
    return random.choice(history_smiles)


def select_parent_imgep_naive(history_smiles, history_behaviors, dim=None, **kw):
    """IMGEP with uniform [0,1]^dim goals (broken baseline)."""
    goal = np.random.uniform(0, 1, dim)
    dists = [np.linalg.norm(b - goal) for b in history_behaviors]
    return history_smiles[np.argmin(dists)]


def select_parent_imgep(history_smiles, history_behaviors, dim=None, **kw):
    """IMGEP with adaptive goal sampling (perturbation of observed)."""
    bmat = np.array(history_behaviors)
    obs_range = np.maximum(bmat.max(axis=0) - bmat.min(axis=0), 0.01)
    base_idx = random.randint(0, len(history_behaviors) - 1)
    base = history_behaviors[base_idx].copy()
    noise = np.random.normal(0, 0.3, dim) * obs_range
    goal = np.clip(base + noise, 0, 1)
    dists = [np.linalg.norm(b - goal) for b in history_behaviors]
    return history_smiles[np.argmin(dists)]


def select_parent_curiosity(history_smiles, history_behaviors, dim=None, **kw):
    """Curiosity-IMGEP: sparsity-biased goal sampling."""
    from scipy.spatial import KDTree
    bmat = np.array(history_behaviors)
    tree = KDTree(bmat)
    k = min(5, len(history_behaviors))
    dists_knn, _ = tree.query(history_behaviors, k=k)
    if len(dists_knn.shape) == 1:
        dists_knn = dists_knn.reshape(-1, 1)
    sparsity = np.mean(dists_knn, axis=1)
    if np.sum(sparsity) > 0:
        probs = sparsity / np.sum(sparsity)
        base_idx = np.random.choice(len(probs), p=probs)
    else:
        base_idx = random.randint(0, len(history_behaviors) - 1)
    base = history_behaviors[base_idx].copy()
    obs_range = np.maximum(bmat.max(axis=0) - bmat.min(axis=0), 0.01)
    noise = np.random.normal(0, 0.3, dim) * obs_range
    goal = np.clip(base + noise, 0, 1)
    dists = [np.linalg.norm(b - goal) for b in history_behaviors]
    return history_smiles[np.argmin(dists)]


def select_parent_bo(history_smiles, history_behaviors, dim=None, **kw):
    """Bayesian Optimization baseline: maximize binding affinity (last dim = Vina score).

    Uses UCB acquisition on Vina scores, Tanimoto kernel for molecule similarity.
    Since we can't do GP on molecule graphs efficiently, we use a simple
    UCB-like heuristic: select molecule with best score among those most
    different from recent best-scoring molecules.
    """
    scores = np.array([b[-1] for b in history_behaviors])  # normalized Vina scores
    n = len(scores)
    # Exploit: top-k molecules by score
    k_top = max(1, n // 5)
    top_indices = np.argsort(scores)[-k_top:]
    # Explore: among top molecules, pick the one most different from the current best
    best_idx = np.argmax(scores)
    if n < 10:
        return history_smiles[best_idx]
    # UCB-like: score + beta * novelty
    beta = 2.0 * np.sqrt(np.log(n + 1))  # UCB exploration term
    ucb_scores = np.zeros(n)
    for i in range(n):
        # Novelty = mean Tanimoto distance to top-k
        novelties = []
        for j in top_indices:
            if i != j:
                novelties.append(tanimoto_distance(history_smiles[i], history_smiles[j]))
        novelty = np.mean(novelties) if novelties else 0.0
        ucb_scores[i] = scores[i] + beta * novelty
    return history_smiles[np.argmax(ucb_scores)]


def select_parent_ga(history_smiles, history_behaviors, dim=None, **kw):
    """Genetic Algorithm baseline: tournament selection on binding affinity.

    Selects parent via tournament (k=3) on Vina score,
    mimicking standard evolutionary docking approaches.
    """
    scores = np.array([b[-1] for b in history_behaviors])
    n = len(scores)
    tournament_size = min(3, n)
    candidates = random.sample(range(n), tournament_size)
    winner = max(candidates, key=lambda i: scores[i])
    return history_smiles[winner]


def select_parent_mapelites(history_smiles, history_behaviors, dim=None,
                            archive=None, **kw):
    """MAP-Elites: select a random occupied cell from the archive, return its molecule."""
    if archive is None or len(archive) == 0:
        return random.choice(history_smiles)
    cell_key = random.choice(list(archive.keys()))
    return archive[cell_key]['smiles']


def select_parent_novelty(history_smiles, history_behaviors, dim=None, **kw):
    """Novelty search: tournament selection on novelty score.

    Novelty = mean distance to k nearest neighbors in behavior space.
    Uses tournament selection (k=3) on novelty scores, mirroring how
    Lehman & Stanley (2011) use novelty as the fitness in an EA.
    """
    from scipy.spatial import KDTree
    bmat = np.array(history_behaviors)
    k = min(15, len(history_behaviors))
    tree = KDTree(bmat)
    dists_knn, _ = tree.query(bmat, k=k)
    if len(dists_knn.shape) == 1:
        dists_knn = dists_knn.reshape(-1, 1)
    novelty = np.mean(dists_knn, axis=1)
    # Tournament selection (k=3) on novelty, matching GA's tournament
    n = len(history_smiles)
    tournament_size = min(3, n)
    candidates = random.sample(range(n), tournament_size)
    winner = max(candidates, key=lambda i: novelty[i])
    return history_smiles[winner]


def mapelites_cell_key(behavior, n_res, n_types):
    """Discretize behavior into a MAP-Elites cell.

    For each residue, record a binary contact (did any interaction type fire?).
    This gives a natural discrete cell = tuple of contacted residue indices.
    """
    contacted = []
    for ri in range(n_res):
        if any(behavior[ri * n_types + ti] > 0 for ti in range(n_types)):
            contacted.append(ri)
    return tuple(contacted)


METHODS = {
    "random": select_parent_random,
    "imgep_naive": select_parent_imgep_naive,
    "imgep": select_parent_imgep,
    "curiosity": select_parent_curiosity,
    "bo": select_parent_bo,
    "ga": select_parent_ga,
    "mapelites": select_parent_mapelites,
    "novelty": select_parent_novelty,
}


# ─── Main exploration loop ───

def run_exploration(method, n_iterations, seed, output_dir, target_id):
    random.seed(seed)
    np.random.seed(seed)

    target = TARGETS[target_id]
    protein_pdb = target["protein"]
    box_file = target["box"]

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp_dock")
    os.makedirs(tmp_dir, exist_ok=True)

    box_vals = read_box_file(box_file)
    pocket_residues = get_pocket_residues(protein_pdb, box_vals)
    dim = len(pocket_residues) * N_INTERACTION_TYPES + 1
    print(f"Target: {target_id} ({target['name']}, {target['family']})")
    print(f"Pocket residues: {len(pocket_residues)}")
    print(f"Behavior dim: {dim}")

    select_parent = METHODS[method]

    history_smiles = []
    history_behaviors = []
    discoveries = []

    # MAP-Elites archive: cell_key -> {smiles, behavior, vina_score}
    me_archive = {} if method == "mapelites" else None

    n_seeds = len(SEED_MOLECULES)
    t_start = time.time()

    for t in range(n_iterations):
        iter_start = time.time()

        try:
            if t < n_seeds:
                smiles = SEED_MOLECULES[t]
            elif len(history_behaviors) < 3:
                parent = random.choice(history_smiles) if history_smiles else SEED_MOLECULES[0]
                smiles = mutate_smiles(parent)
            else:
                parent = select_parent(
                    history_smiles, history_behaviors, dim=dim,
                    archive=me_archive)
                smiles = mutate_smiles(parent)

            complex_pdb, vina_score = dock_smiles(
                smiles, protein_pdb, box_vals, t, tmp_dir)

            fp = compute_fingerprint(complex_pdb, pocket_residues, vina_score)

            history_smiles.append(smiles)
            history_behaviors.append(fp)

            # Update MAP-Elites archive if applicable
            if me_archive is not None:
                n_res = len(pocket_residues)
                cell = mapelites_cell_key(fp, n_res, N_INTERACTION_TYPES)
                if cell not in me_archive or vina_score < me_archive[cell]['vina_score']:
                    me_archive[cell] = {
                        'smiles': smiles,
                        'behavior': fp,
                        'vina_score': vina_score
                    }

            n_res = len(pocket_residues)
            active_residues = 0
            for ri in range(n_res):
                if any(fp[ri * N_INTERACTION_TYPES + ti] > 0 for ti in range(N_INTERACTION_TYPES)):
                    active_residues += 1
            n_interactions = int(np.sum(fp[:-1] > 0))

            discovery = {
                "iteration": t,
                "smiles": smiles,
                "behavior": fp.tolist(),
                "n_interactions": n_interactions,
                "active_residues": active_residues,
                "vina_score": vina_score,
                "time_s": time.time() - iter_start
            }
            discoveries.append(discovery)

            if (t + 1) % 25 == 0 or t < n_seeds:
                elapsed = time.time() - t_start
                print(f"  [{t+1}/{n_iterations}] {smiles[:40]}... "
                      f"int={n_interactions} res={active_residues} "
                      f"vina={vina_score:.1f} t={elapsed:.0f}s")

        except Exception as e:
            fallback_smi = smiles if 'smiles' in dir() else SEED_MOLECULES[0]
            history_smiles.append(fallback_smi)
            history_behaviors.append(np.zeros(dim))
            discoveries.append({
                "iteration": t,
                "smiles": fallback_smi,
                "behavior": np.zeros(dim).tolist(),
                "n_interactions": 0,
                "active_residues": 0,
                "vina_score": 0.0,
                "error": str(e),
                "time_s": time.time() - iter_start
            })

    total_time = time.time() - t_start

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    results = {
        "method": method,
        "seed": seed,
        "target": target_id,
        "target_name": target["name"],
        "target_family": target["family"],
        "n_iterations": n_iterations,
        "total_time_s": total_time,
        "pocket_residues": pocket_residues,
        "n_pocket_residues": len(pocket_residues),
        "interaction_types": INTERACTION_TYPES,
        "behavior_dim": dim,
        "discoveries": discoveries
    }

    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {results_file}")
    print(f"Total: {total_time:.0f}s ({total_time/n_iterations:.1f}s/iter)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--target", choices=list(TARGETS.keys()), default="3V8D")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Method: {args.method}")
    print(f"  Target: {args.target}")
    print(f"  Seed: {args.seed}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    run_exploration(args.method, args.iterations, args.seed, args.output_dir, args.target)
