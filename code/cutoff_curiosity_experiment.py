#!/usr/bin/env python3
"""
cutoff_curiosity_experiment.py — Exp 4: Curiosity-IMGEP with alternative
distance cutoff d_max and optional q(m)-exclusion.

Sensitivity axis 1: d_max ∈ {3.5, 4.0, 4.5} Å (paper uses 4.0)
Sensitivity axis 2: --exclude-q flag → trajectory-level ψ-only search (no q coord)

Runs on 2 pre-selected targets (1EVE = large pocket, 4DFR = small pocket).
5 seeds × 3 d_max values × {with q, without q} = 60 runs at 500 iters each,
but we pre-commit to only 30 additional runs (skip d_max=4.0 which is already in
the paper) = 2 targets × 5 seeds × (2 new d_max + 1 q-exclusion) = 30 runs.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
import env_patch  # noqa: E402
from run_experiment_v3 import (  # noqa: E402
    TARGETS, SEED_MOLECULES, N_INTERACTION_TYPES,
    read_box_file, get_pocket_residues, dock_smiles, mutate_smiles,
)
import run_experiment_v3 as R  # to override d_MIN/d_MAX


def select_parent_curiosity(hs, hb, dim, k_nn=5):
    bmat = np.array(hb)
    tree = KDTree(bmat)
    k = min(k_nn, len(hb))
    dists, _ = tree.query(hb, k=k)
    if dists.ndim == 1: dists = dists.reshape(-1, 1)
    sparsity = np.mean(dists, axis=1)
    if sparsity.sum() > 0:
        base_idx = int(np.random.choice(len(hb), p=sparsity / sparsity.sum()))
    else:
        base_idx = random.randint(0, len(hb) - 1)
    base = hb[base_idx].copy()
    obs_range = np.maximum(bmat.max(0) - bmat.min(0), 0.01)
    noise = np.random.normal(0, 0.3, dim) * obs_range
    goal = np.clip(base + noise, 0, 1)
    dists2 = [float(np.linalg.norm(b - goal)) for b in hb]
    return hs[int(np.argmin(dists2))]


def run(target: str, seed: int, d_max: float, exclude_q: bool, n_iterations: int, output_dir: str):
    R.d_MAX = d_max  # override the module-level constant used in compute_fingerprint
    random.seed(seed); np.random.seed(seed)
    tgt = TARGETS[target]
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp_dock"); os.makedirs(tmp_dir, exist_ok=True)
    box_vals = read_box_file(tgt["box"])
    pocket = get_pocket_residues(tgt["protein"], box_vals)
    n_res = len(pocket)
    # If we're excluding q, the "behavior" for retrieval is psi (3*n_res). Otherwise phi (3*n_res+1).
    dim = n_res * N_INTERACTION_TYPES + (0 if exclude_q else 1)
    print(f"[cutoff] target={target} d_max={d_max} exclude_q={exclude_q} dim={dim} budget={n_iterations}")

    from run_experiment_v3 import compute_fingerprint

    hs, hb, discs = [], [], []
    t0 = time.time()
    for t in range(n_iterations):
        it0 = time.time()
        try:
            if t < len(SEED_MOLECULES):
                smi = SEED_MOLECULES[t]
            elif len(hb) < 3:
                smi = mutate_smiles(random.choice(hs))
            else:
                parent = select_parent_curiosity(hs, hb, dim=dim)
                smi = mutate_smiles(parent)
            pdb, vina = dock_smiles(smi, tgt["protein"], box_vals, t, tmp_dir)
            phi_full = compute_fingerprint(pdb, pocket, vina)  # 3*n_res + 1 (uses updated d_MAX)
            b = phi_full[:-1] if exclude_q else phi_full  # ψ if excluding q, else φ
            hs.append(smi); hb.append(b.astype(np.float64))
            n_int = int(np.sum(phi_full[:-1] > 0))
            active_res = int(np.sum([any(phi_full[i*3+t2] > 0 for t2 in range(3)) for i in range(n_res)]))
            discs.append({"iteration": t, "smiles": smi, "behavior": b.tolist(),
                          "phi_full": phi_full.tolist(),  # keep full for post-hoc analysis
                          "n_interactions": n_int, "active_residues": active_res,
                          "vina_score": float(vina), "time_s": time.time() - it0})
            if (t+1) % 25 == 0:
                print(f"  [cutoff {t+1}/{n_iterations}] int={n_int} vina={vina:.1f} t={time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  iter {t} failure: {e}")
            hs.append(hs[-1] if hs else SEED_MOLECULES[0])
            hb.append(np.zeros(dim))
            discs.append({"iteration": t, "smiles": hs[-1], "behavior": np.zeros(dim).tolist(),
                          "phi_full": np.zeros(n_res*3+1).tolist(),
                          "n_interactions": 0, "active_residues": 0, "vina_score": 0.0,
                          "error": str(e), "time_s": time.time() - it0})

    import shutil
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    result = {"method": f"curiosity_dmax{d_max}{'_noq' if exclude_q else ''}", "seed": seed,
              "target": target, "target_name": tgt["name"], "target_family": tgt["family"],
              "n_iterations": n_iterations, "total_time_s": time.time() - t0,
              "pocket_residues": pocket, "n_pocket_residues": n_res,
              "behavior_dim": dim, "discoveries": discs,
              "config": {"d_max": d_max, "exclude_q_from_retrieval": exclude_q,
                         "note": "Exp 4 — R1-c/R1-d cutoff + q-exclusion sensitivity"}}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {output_dir}/results.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--d-max", type=float, default=4.0)
    p.add_argument("--exclude-q", action="store_true")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(args.target, args.seed, args.d_max, args.exclude_q, args.iterations, args.output_dir)
