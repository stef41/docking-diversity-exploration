#!/usr/bin/env python3
"""
charged_curiosity_experiment.py — Exp 3: Curiosity-IMGEP with 5-channel
descriptor (3 original + 2 salt-bridge directional channels).

Adds directional charged contacts:
  channel 4 = lig-positive/prot-negative (via PLIP salt_bridge_info; NEG on protein)
  channel 5 = lig-negative/prot-positive (via PLIP salt_bridge_info; POS on protein)

For each residue r, we set psi[r, t=3] > 0 iff any salt-bridge with LIG_POS
involves that residue, and psi[r, t=4] > 0 iff any salt-bridge with LIG_NEG
involves that residue. Distance normalization: same clip(d_max - d) / (d_max - d_min).

Runs on 3 targets pre-selected by ionic-fraction ranking: 1ERE, 3EML, 4DFR.
5 seeds each, 500 iterations = 15 runs.

Launch:
  python3 charged_curiosity_experiment.py --target 1ERE --seed 0 --output_dir OUT
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree

# Path patch (g107 environment)
if 'GNINA_BIN_OVERRIDE' not in os.environ:
    # local fallback
    sys.path.insert(0, str(Path(__file__).parent))

import env_patch  # noqa: E402  patches paths
from run_experiment_v3 import (  # noqa: E402
    TARGETS, SEED_MOLECULES, N_INTERACTION_TYPES,
    read_box_file, get_pocket_residues, dock_smiles, mutate_smiles,
    d_MIN, d_MAX,
)

N_CHANNELS = 5  # 3 original + 2 charged
INTERACTION_TYPES_ORIG = ['hydrophobic', 'hbond_donor', 'hbond_acceptor']


def compute_fingerprint_5ch(pdb_string, pocket_residues, vina_score):
    """Compute 5-channel fingerprint including salt-bridge directional contacts."""
    from plip.structure.preparation import PDBComplex
    from plip.exchange.report import BindingSiteReport

    n_res = len(pocket_residues)
    dim = n_res * N_CHANNELS + 1
    fp = np.zeros(dim)
    fp[-1] = min(1.0, max(0.0, -vina_score / 12.0))

    try:
        protlig = PDBComplex()
        protlig.load_pdb(pdb_string, as_string=True)
        for ligand in protlig.ligands:
            protlig.characterize_complex(ligand)
        sites = sorted(protlig.interaction_sets.items())
        if not sites:
            return fp
        _, site = sites[0]
        report = BindingSiteReport(site)
    except Exception:
        return fp

    res_to_idx = {r: i for i, r in enumerate(pocket_residues)}
    residue_dists = {r: {c: [] for c in range(N_CHANNELS)} for r in pocket_residues}

    # channel 0: hydrophobic
    for info in getattr(report, 'hydrophobic_info', []):
        try:
            resnr = int(info[0])
            dist = float(info[6])
            if resnr in residue_dists:
                residue_dists[resnr][0].append(dist)
        except (ValueError, IndexError):
            pass

    # channels 1,2: HBD, HBA
    for info in getattr(report, 'hbond_info', []):
        try:
            resnr = int(info[0])
            dist = float(info[7])
            is_prot_donor = (info[6] is True or info[6] == 'True')
            ch = 1 if is_prot_donor else 2
            if resnr in residue_dists:
                residue_dists[resnr][ch].append(dist)
        except (ValueError, IndexError):
            pass

    # channels 3,4: salt bridges. PLIP saltbridge_info fields (index 0 = resnr, index 5 = protispos)
    # For salt-bridge: if the protein residue is +charged, the ligand side is -charged (channel 4).
    # If protein residue is -charged, ligand is +charged (channel 3).
    for info in getattr(report, 'saltbridge_info', []):
        try:
            resnr = int(info[0])
            dist = float(info[5])  # distance
            protispos = info[4]     # bool: is the protein side positive?
            # NB: PLIP schema for saltbridge_info: (resnr, restype, resnr_lig, restype_lig,
            #     protispos, distance, ligcarboxylate, lystrue, argtrue, histrue)
            # In older PLIP (2.2.x) the columns may be shifted; guard with try/except.
            ch = 4 if protispos else 3  # protispos: lig NEG paired w/ prot POS → ch 4
            if resnr in residue_dists:
                residue_dists[resnr][ch].append(dist)
        except (ValueError, IndexError, AttributeError):
            pass

    for r, chdists in residue_dists.items():
        ridx = res_to_idx[r]
        for c, dists in chdists.items():
            if dists:
                d = min(dists)
                val = (d_MAX - d) / (d_MAX - d_MIN)
                val = max(0.0, min(1.0, val))
                fp[ridx * N_CHANNELS + c] = val
    return fp


def select_parent_curiosity_5ch(hs, hb, dim, k_nn=5):
    """Standard Curiosity-IMGEP in the 5-channel space (φ dim = 5*n_res + 1)."""
    bmat = np.array(hb)
    tree = KDTree(bmat)
    k = min(k_nn, len(hb))
    dists, _ = tree.query(hb, k=k)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
    sparsity = np.mean(dists, axis=1)
    if sparsity.sum() > 0:
        probs = sparsity / sparsity.sum()
        base_idx = int(np.random.choice(len(hb), p=probs))
    else:
        base_idx = random.randint(0, len(hb) - 1)
    base = hb[base_idx].copy()
    obs_range = np.maximum(bmat.max(0) - bmat.min(0), 0.01)
    noise = np.random.normal(0, 0.3, dim) * obs_range
    goal = np.clip(base + noise, 0, 1)
    dists2 = [float(np.linalg.norm(b - goal)) for b in hb]
    return hs[int(np.argmin(dists2))]


def run(target: str, seed: int, method: str, n_iterations: int, output_dir: str):
    random.seed(seed); np.random.seed(seed)
    tgt = TARGETS[target]
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp_dock"); os.makedirs(tmp_dir, exist_ok=True)
    box_vals = read_box_file(tgt["box"])
    pocket = get_pocket_residues(tgt["protein"], box_vals)
    dim = len(pocket) * N_CHANNELS + 1
    print(f"[{method}-5ch] target={target} N_res={len(pocket)} phi_dim={dim} budget={n_iterations}")

    hs, hb, discs = [], [], []
    t0 = time.time()
    for t in range(n_iterations):
        it0 = time.time()
        try:
            if t < len(SEED_MOLECULES):
                smi = SEED_MOLECULES[t]
            elif len(hb) < 3:
                parent = random.choice(hs)
                smi = mutate_smiles(parent)
            else:
                if method == "curiosity_5ch":
                    parent = select_parent_curiosity_5ch(hs, hb, dim=dim)
                elif method == "random_5ch":
                    parent = random.choice(hs)
                else:
                    raise ValueError(method)
                smi = mutate_smiles(parent)
            pdb, vina = dock_smiles(smi, tgt["protein"], box_vals, t, tmp_dir)
            phi = compute_fingerprint_5ch(pdb, pocket, vina)
            hs.append(smi); hb.append(phi)
            n_int = int(np.sum(phi[:-1] > 0))
            active_res = int(np.sum([any(phi[i*N_CHANNELS + c] > 0 for c in range(N_CHANNELS))
                                     for i in range(len(pocket))]))
            discs.append({"iteration": t, "smiles": smi, "behavior": phi.tolist(),
                          "n_interactions": n_int, "active_residues": active_res,
                          "vina_score": float(vina), "time_s": time.time() - it0})
            if (t+1) % 25 == 0:
                print(f"  [{method}-5ch {t+1}/{n_iterations}] int={n_int} res={active_res} vina={vina:.1f} t={time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  iter {t} failure: {e}")
            hs.append(hs[-1] if hs else SEED_MOLECULES[0])
            hb.append(np.zeros(dim))
            discs.append({"iteration": t, "smiles": hs[-1], "behavior": np.zeros(dim).tolist(),
                          "n_interactions": 0, "active_residues": 0, "vina_score": 0.0,
                          "error": str(e), "time_s": time.time() - it0})

    import shutil
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    result = {"method": f"{method}_5ch", "seed": seed, "target": target,
              "target_name": tgt["name"], "target_family": tgt["family"],
              "n_iterations": n_iterations, "total_time_s": time.time() - t0,
              "pocket_residues": pocket, "n_pocket_residues": len(pocket),
              "interaction_types": INTERACTION_TYPES_ORIG + ["saltbridge_lig_pos", "saltbridge_lig_neg"],
              "behavior_dim": dim, "discoveries": discs,
              "config": {"n_channels": N_CHANNELS, "note": "Exp 3 — R3-3a charged-channel ablation"}}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {output_dir}/results.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", default="curiosity_5ch", choices=["curiosity_5ch", "random_5ch"])
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(args.target, args.seed, args.method, args.iterations, args.output_dir)
