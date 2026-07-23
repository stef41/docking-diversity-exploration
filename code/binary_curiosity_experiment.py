#!/usr/bin/env python3
"""
binary_curiosity_experiment.py — Binary-interaction-fingerprint Curiosity-IMGEP.

Reviewer 3-4 explicitly asks for a binary-fingerprint baseline. This is the
Curiosity-IMGEP method from run_experiment_v3.py, but with

    psi_bin[r,t] = 1[psi[r,t] > 0]

used for goal sampling, nearest-neighbor retrieval, sparsity weighting, and all
downstream diagnostics ($D_goal$, $D_inter$, R, tie rate). All other settings
(seeds, CReM, filters, docking, goal noise coefficient, k) unchanged.

Precommitted BEFORE looking at outcomes (R3-4 response):
  * Run on all 7 targets, 10 seeds, 500 evaluations each = 70 runs, matching
    the original continuous-fingerprint Curiosity-IMGEP.
  * Random trajectories are representation-invariant; we do NOT re-run Random —
    we reevaluate cached Random results under both encodings (post hoc analysis).
  * Compare:
      unique binary contact patterns
      continuous-profile diversity (recomputed from stored psi)
      scaffold + chemical diversity (SMILES-based, identical protocol)
      D_inter, D_goal, R (in the binary space vs. continuous space)
      nearest-neighbor tie rate
      parent-selection entropy / effective # of distinct parents

Launch:
    python binary_curiosity_experiment.py --target 3PJC --seed 0 --output_dir OUT
    bash launch_binary_curiosity.sh   # 7x10 full sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_experiment_v3 import (  # noqa: E402
    TARGETS,
    SEED_MOLECULES,
    N_INTERACTION_TYPES,
    read_box_file,
    get_pocket_residues,
    dock_smiles,
    compute_fingerprint,
    mutate_smiles,
)
from scipy.spatial import KDTree

# ─── Binary variant of Curiosity-IMGEP parent selection ────────────────────

def binarize(psi: np.ndarray) -> np.ndarray:
    """psi is interaction-only (length = n_res * n_types). Returns 0/1 vector of same shape."""
    return (psi > 0).astype(np.float64)


def select_parent_curiosity_binary(history_smiles, history_psi_bin, dim, k_nn: int = 5,
                                    goal_sigma: float = 0.3) -> str:
    """
    Curiosity-IMGEP parent selection operating entirely in the BINARY space.

    Steps (identical to select_parent_curiosity in run_experiment_v3.py, but with
    all psi vectors already binarized):
      1. Compute sparsity of each observed behavior = mean distance to k=5 nearest neighbors.
      2. Sample a base behavior with probability proportional to sparsity.
      3. Perturb the base with Gaussian noise sigma = goal_sigma * range(observed).
         (In binary space, range is 0 or 1 per dim; noise gives a graded goal that then
          selects the nearest binary point.)
      4. Clip to [0,1] to form the goal g.
      5. Return the history SMILES minimizing ||psi_bin - g||_2.
    """
    bmat = np.array(history_psi_bin)
    n = len(history_psi_bin)
    k_eff = min(k_nn, n)
    tree = KDTree(bmat)
    dists_knn, _ = tree.query(bmat, k=k_eff)
    if dists_knn.ndim == 1:
        dists_knn = dists_knn.reshape(-1, 1)
    sparsity = np.mean(dists_knn, axis=1)
    if np.sum(sparsity) > 0:
        probs = sparsity / np.sum(sparsity)
        base_idx = int(np.random.choice(n, p=probs))
    else:
        base_idx = random.randint(0, n - 1)
    base = history_psi_bin[base_idx].copy()
    obs_range = np.maximum(bmat.max(axis=0) - bmat.min(axis=0), 0.01)
    noise = np.random.normal(0.0, goal_sigma, dim) * obs_range
    goal = np.clip(base + noise, 0.0, 1.0)
    # nearest in BINARY space
    dists = [float(np.linalg.norm(b - goal)) for b in history_psi_bin]
    return history_smiles[int(np.argmin(dists))]


# ─── Main loop ─────────────────────────────────────────────────────────────

def run_binary_curiosity(target_id: str, seed: int, n_iterations: int, output_dir: str):
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
    n_res = len(pocket_residues)
    psi_dim = n_res * N_INTERACTION_TYPES

    print(f"[BIN-CURIOSITY] target={target_id} psi_dim={psi_dim} budget={n_iterations}")

    history_smiles: list[str] = []
    history_psi: list[np.ndarray] = []
    history_psi_bin: list[np.ndarray] = []
    history_vina: list[float] = []
    parent_selection_history: list[int] = []
    tie_rate_per_iter: list[float] = []
    discoveries: list[dict] = []

    t_start = time.time()

    for t in range(n_iterations):
        iter_start = time.time()
        try:
            if t < len(SEED_MOLECULES):
                smiles = SEED_MOLECULES[t]
            elif len(history_psi_bin) < 3:
                parent = random.choice(history_smiles)
                smiles = mutate_smiles(parent)
            else:
                parent = select_parent_curiosity_binary(
                    history_smiles, history_psi_bin, dim=psi_dim)
                smiles = mutate_smiles(parent)

            complex_pdb, vina_score = dock_smiles(smiles, protein_pdb, box_vals, t, tmp_dir)
            phi = compute_fingerprint(complex_pdb, pocket_residues, vina_score)
            psi = phi[:-1].astype(np.float64)
            psi_bin = binarize(psi)

            history_smiles.append(smiles)
            history_psi.append(psi)
            history_psi_bin.append(psi_bin)
            history_vina.append(float(vina_score))

            # Track diagnostics for R3-4 response
            if len(history_psi_bin) > 5:
                # nearest-neighbor tie rate at this step: fraction of history within tie tolerance of top-1
                bmat = np.array(history_psi_bin)
                # sample a random goal like the actual retrieval would use
                noise = np.random.normal(0.0, 0.3, psi_dim)
                base = history_psi_bin[np.random.randint(len(history_psi_bin))].copy()
                obs_range = np.maximum(bmat.max(0) - bmat.min(0), 0.01)
                g = np.clip(base + noise * obs_range, 0.0, 1.0)
                dists = np.linalg.norm(bmat - g, axis=1)
                min_d = float(np.min(dists))
                # tie tolerance: within 1e-6 of the minimum (binary space collapses to integer distances)
                tie_frac = float(np.mean(np.abs(dists - min_d) < 1e-6))
                tie_rate_per_iter.append(tie_frac)

            n_int = int(np.sum(psi_bin))
            active_res = int(np.sum([any(psi_bin[r * N_INTERACTION_TYPES + t2] > 0
                                         for t2 in range(N_INTERACTION_TYPES))
                                     for r in range(n_res)]))
            discoveries.append({
                "iteration": t,
                "smiles": smiles,
                "behavior": psi.tolist(),           # continuous psi (for post-hoc analysis)
                "behavior_bin": psi_bin.tolist(),   # binary psi (this method's actual search vector)
                "n_interactions": n_int,
                "active_residues": active_res,
                "vina_score": float(vina_score),
                "time_s": time.time() - iter_start,
            })

            if (t + 1) % 25 == 0 or t < len(SEED_MOLECULES):
                print(f"  [BIN-CURIOSITY {t+1}/{n_iterations}] int={n_int} res={active_res} "
                      f"vina={vina_score:.1f} t={time.time()-t_start:.0f}s")

        except Exception as e:
            print(f"[BIN-CURIOSITY] iter {t} failure: {e}")
            history_smiles.append(smiles if 'smiles' in locals() else SEED_MOLECULES[0])
            history_psi.append(np.zeros(psi_dim))
            history_psi_bin.append(np.zeros(psi_dim))
            history_vina.append(0.0)
            discoveries.append({
                "iteration": t,
                "smiles": history_smiles[-1],
                "behavior": np.zeros(psi_dim).tolist(),
                "behavior_bin": np.zeros(psi_dim).tolist(),
                "n_interactions": 0,
                "active_residues": 0,
                "vina_score": 0.0,
                "error": str(e),
                "time_s": time.time() - iter_start,
            })

    total_time = time.time() - t_start

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Parent-selection concentration: which SMILES ended up as the most-used parent?
    from collections import Counter
    parent_counts = Counter(history_smiles).most_common(20)
    effective_parents = np.exp(-np.sum([(c/len(history_smiles)) * np.log(c/len(history_smiles))
                                        for _, c in Counter(history_smiles).items() if c > 0]))

    result = {
        "method": "curiosity_binary",
        "seed": seed,
        "target": target_id,
        "target_name": target["name"],
        "target_family": target["family"],
        "n_iterations": n_iterations,
        "total_time_s": total_time,
        "pocket_residues": pocket_residues,
        "n_pocket_residues": len(pocket_residues),
        "interaction_types": ["hydrophobic", "hbond_donor", "hbond_acceptor"],
        "behavior_dim": psi_dim,
        "discoveries": discoveries,
        "diagnostics": {
            "mean_nn_tie_rate": float(np.mean(tie_rate_per_iter)) if tie_rate_per_iter else 0.0,
            "top20_parent_counts": parent_counts,
            "effective_num_parents": float(effective_parents),
        },
        "config": {
            "encoding": "binary  psi_bin[r,t] = 1[psi[r,t] > 0]",
            "goal_sigma": 0.3,
            "k_nn_sparsity": 5,
            "note": "R3-4 response run — binary variant of the submitted Curiosity-IMGEP",
        },
    }
    out = os.path.join(output_dir, "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[BIN-CURIOSITY] wrote {out} — {total_time:.0f}s total, {total_time/n_iterations:.1f}s/iter")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run_binary_curiosity(args.target, args.seed, args.iterations, args.output_dir)
