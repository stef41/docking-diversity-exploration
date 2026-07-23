#!/usr/bin/env python3
"""
nsga2_experiment.py — Budget-matched NSGA-II baseline for JCIM revision.

Reviewer 2 asked explicitly for NSGA-II. Reviewer 3 (via editor's emphasis)
implicitly reinforces that a standard Pareto-optimization baseline is expected.

Design choices frozen BEFORE looking at results (R2-1 response commitment):
  * Population size = 50.
  * Initialization: 5 shared seed molecules + 45 one-edit CReM descendants of seeds.
  * Objectives (MINIMIZE both, so we negate reward-like quantities):
      -q(m)          : minimize the negative of normalized Vina quality
      -novelty(m)    : minimize the negative of interaction novelty
    where novelty(m) = mean Euclidean distance in psi to k=15 nearest previously
    evaluated behaviors (or all evaluated if <k available).
  * Parent selection: binary tournament on (rank, crowding distance).
  * Variation: single CReM edit via mutate_smiles() from run_experiment_v3.py.
  * Replacement: steady-state combine (parents + offspring), then NSGA-II
    nondominated sorting + crowding-distance truncation to population size.
  * Budget: EXACTLY 500 docking evaluations per run, including unchanged-parent
    outcomes. First 5 evaluations = seed molecules (mirroring the other methods).
  * Seeds: 10 (same as the other methods).
  * Targets: 7 (all).

Deps: numpy, scipy, rdkit, crem, plip, biopython + run_experiment_v3.py primitives.

Launch on g107 GPU cluster:
  # smoke test (1 target × 1 seed × 50 evals)
  python nsga2_experiment.py --target 3PJC --seed 0 --iterations 50 --smoke-test

  # full run: 7 targets × 10 seeds × 500 evals
  bash launch_nsga2.sh

Output directory structure mirrors the existing v3 results:
  results_v3/<TARGET>/nsga2_seed<N>/results.json
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

# Reuse all primitives from the production runner.
# (Assumes this file is placed alongside run_experiment_v3.py in code/.)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the primitives we need. If run_experiment_v3.py isn't on the cluster
# at this path, adjust PYTHONPATH before launching.
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


# ─── NSGA-II primitives ──────────────────────────────────────────────────

def dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """f1 dominates f2 (both to be MINIMIZED) if f1 <= f2 elementwise and f1 < f2 somewhere."""
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def nondominated_sort(F: np.ndarray) -> list[list[int]]:
    """Fast nondominated sort. Returns list of fronts (each = list of indices into F)."""
    n = len(F)
    S = [[] for _ in range(n)]        # solutions dominated by i
    n_dom = [0] * n                   # count of solutions dominating i
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(F[i], F[j]):
                S[i].append(j)
            elif dominates(F[j], F[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F: np.ndarray, indices: list[int]) -> np.ndarray:
    """Standard NSGA-II crowding distance for one front."""
    m = len(indices)
    dist = np.zeros(m)
    if m <= 2:
        dist[:] = np.inf
        return dist
    Fp = F[indices]
    for obj in range(Fp.shape[1]):
        order = np.argsort(Fp[:, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_min, f_max = Fp[order[0], obj], Fp[order[-1], obj]
        if f_max == f_min:
            continue
        for k in range(1, m - 1):
            dist[order[k]] += (Fp[order[k + 1], obj] - Fp[order[k - 1], obj]) / (f_max - f_min)
    return dist


def binary_tournament(rank: np.ndarray, crowd: np.ndarray, pop_size: int, n_picks: int, rng: random.Random) -> list[int]:
    """Pick n_picks parents via binary tournament on (rank, crowding)."""
    picks = []
    for _ in range(n_picks):
        a, b = rng.sample(range(pop_size), 2)
        if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] > crowd[b]):
            picks.append(a)
        else:
            picks.append(b)
    return picks


# ─── Novelty (k-NN mean distance in interaction fingerprint space) ────────

def novelty_score(psi: np.ndarray, history_psi: np.ndarray, k: int = 15) -> float:
    """Mean Euclidean distance in ψ to k non-self nearest previously observed behaviors.

    If `psi` appears in `history_psi` (e.g., because history includes the
    current population), the self-distance of 0 is skipped so `k` reflects
    non-self neighbors as documented.
    """
    if len(history_psi) == 0:
        return 0.0
    diffs = history_psi - psi
    dists = np.linalg.norm(diffs, axis=1)
    sorted_dists = np.sort(dists)
    # Drop leading zero self-match (if any). np.isclose handles float noise.
    if sorted_dists.size and np.isclose(sorted_dists[0], 0.0):
        sorted_dists = sorted_dists[1:]
    if sorted_dists.size == 0:
        return 0.0
    k_eff = min(k, len(sorted_dists))
    return float(np.mean(sorted_dists[:k_eff]))


# ─── Main NSGA-II loop ────────────────────────────────────────────────────

POP_SIZE = 50
N_OBJ = 2
KNN_NOVELTY = 15


def run_nsga2(target_id: str, seed: int, n_iterations: int, output_dir: str, smoke_test: bool = False):
    rng = random.Random(seed)
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
    psi_dim = n_res * N_INTERACTION_TYPES  # interaction-only dim (no vina coordinate)

    print(f"[NSGA-II] target={target_id} ({target['name']}, {target['family']})")
    print(f"[NSGA-II] N_res={n_res}, psi_dim={psi_dim}, budget={n_iterations}")

    # History across the whole run (for novelty computation + logging)
    history_smiles: list[str] = []
    history_psi: list[np.ndarray] = []       # interaction-only fingerprint (dim = psi_dim)
    history_q: list[float] = []              # normalized Vina quality
    history_vina: list[float] = []           # raw Vina score
    discoveries: list[dict] = []

    def evaluate(smiles: str, iter_idx: int) -> tuple[np.ndarray, float, float]:
        """Dock, compute psi + q. Returns (psi, q, vina_score)."""
        try:
            complex_pdb, vina_score = dock_smiles(smiles, protein_pdb, box_vals, iter_idx, tmp_dir)
            phi = compute_fingerprint(complex_pdb, pocket_residues, vina_score)  # length psi_dim+1
            psi = phi[:-1].astype(np.float64)
            q = float(np.clip(-vina_score / 12.0, 0.0, 1.0))
            return psi, q, float(vina_score)
        except Exception as e:
            print(f"[NSGA-II] eval failure at iter {iter_idx}: {e}")
            return np.zeros(psi_dim), 0.0, 0.0

    def record(iter_idx: int, smiles: str, psi: np.ndarray, q: float, vina: float, dt: float):
        history_smiles.append(smiles)
        history_psi.append(psi)
        history_q.append(q)
        history_vina.append(vina)
        n_int = int(np.sum(psi > 0))
        active_res = int(np.sum([any(psi[r * N_INTERACTION_TYPES + t] > 0 for t in range(N_INTERACTION_TYPES))
                                 for r in range(n_res)]))
        discoveries.append({
            "iteration": iter_idx,
            "smiles": smiles,
            "behavior": psi.tolist(),
            "n_interactions": n_int,
            "active_residues": active_res,
            "vina_score": vina,
            "time_s": dt,
        })

    t_start = time.time()

    # ── Phase 0: seed the first 5 evaluations with the shared seed molecules ──
    for t in range(min(5, n_iterations)):
        it_start = time.time()
        psi, q, vina = evaluate(SEED_MOLECULES[t], t)
        record(t, SEED_MOLECULES[t], psi, q, vina, time.time() - it_start)

    # ── Phase 1: fill the initial NSGA-II population to POP_SIZE by mutating seeds ──
    pop_smiles: list[str] = list(history_smiles[:5])
    pop_psi: list[np.ndarray] = list(history_psi[:5])
    pop_q: list[float] = list(history_q[:5])

    t = 5
    while len(pop_smiles) < POP_SIZE and t < n_iterations:
        parent = pop_smiles[rng.randrange(len(pop_smiles))]
        child = mutate_smiles(parent)
        it_start = time.time()
        psi, q, vina = evaluate(child, t)
        record(t, child, psi, q, vina, time.time() - it_start)
        pop_smiles.append(child)
        pop_psi.append(psi)
        pop_q.append(q)
        t += 1

    # If budget didn't allow filling the pop, we're done.
    if t >= n_iterations:
        _write_output(target_id, seed, n_iterations, pop_smiles, pop_psi, pop_q,
                      history_smiles, history_psi, history_q, history_vina,
                      pocket_residues, discoveries, t_start, output_dir)
        return

    # ── Phase 2: steady-state NSGA-II main loop ──
    while t < n_iterations:
        # Compute objectives + rank/crowding for the current population
        F = _compute_objectives(pop_psi, pop_q, history_psi[:t])  # (POP_SIZE, N_OBJ)
        fronts = nondominated_sort(F)
        rank = np.zeros(POP_SIZE, dtype=int)
        crowd = np.zeros(POP_SIZE)
        for fi, front in enumerate(fronts):
            rank[front] = fi
            crowd[front] = crowding_distance(F, front)

        # Binary tournament → pick 1 parent, apply 1 CReM edit → 1 offspring per generation step
        parent_idx = binary_tournament(rank, crowd, POP_SIZE, n_picks=1, rng=rng)[0]
        child = mutate_smiles(pop_smiles[parent_idx])

        it_start = time.time()
        child_psi, child_q, child_vina = evaluate(child, t)
        record(t, child, child_psi, child_q, child_vina, time.time() - it_start)
        t += 1

        # Combine (parents + child), then NSGA-II truncate to POP_SIZE
        combined_smiles = pop_smiles + [child]
        combined_psi = pop_psi + [child_psi]
        combined_q = pop_q + [child_q]
        F_all = _compute_objectives(combined_psi, combined_q, history_psi[:t])
        fronts_all = nondominated_sort(F_all)
        new_indices: list[int] = []
        for front in fronts_all:
            if len(new_indices) + len(front) <= POP_SIZE:
                new_indices.extend(front)
            else:
                cd = crowding_distance(F_all, front)
                order = np.argsort(-cd)  # highest crowding first
                remaining = POP_SIZE - len(new_indices)
                new_indices.extend([front[i] for i in order[:remaining]])
                break
        pop_smiles = [combined_smiles[i] for i in new_indices]
        pop_psi = [combined_psi[i] for i in new_indices]
        pop_q = [combined_q[i] for i in new_indices]

        if (t + 1) % 25 == 0:
            elapsed = time.time() - t_start
            best_q = max(pop_q) if pop_q else 0.0
            print(f"  [NSGA-II {t+1}/{n_iterations}] pop_size={len(pop_smiles)} "
                  f"best_q={best_q:.3f} t={elapsed:.0f}s")

    _write_output(target_id, seed, n_iterations, pop_smiles, pop_psi, pop_q,
                  history_smiles, history_psi, history_q, history_vina,
                  pocket_residues, discoveries, t_start, output_dir)


def _compute_objectives(pop_psi, pop_q, history_psi_before_this_step) -> np.ndarray:
    """(POP_SIZE, 2) — [ -q, -novelty ] both to be MINIMIZED."""
    P = len(pop_psi)
    F = np.zeros((P, N_OBJ))
    hist = np.array(history_psi_before_this_step) if history_psi_before_this_step else None
    for i in range(P):
        F[i, 0] = -pop_q[i]
        F[i, 1] = -novelty_score(pop_psi[i], hist if hist is not None else np.zeros((0, len(pop_psi[i]))),
                                 k=KNN_NOVELTY)
    return F


def _write_output(target_id, seed, n_iterations, pop_smiles, pop_psi, pop_q,
                  history_smiles, history_psi, history_q, history_vina,
                  pocket_residues, discoveries, t_start, output_dir):
    import shutil
    tmp_dir = os.path.join(output_dir, "tmp_dock")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    total_time = time.time() - t_start
    target = TARGETS[target_id]
    result = {
        "method": "nsga2",
        "seed": seed,
        "target": target_id,
        "target_name": target["name"],
        "target_family": target["family"],
        "n_iterations": n_iterations,
        "total_time_s": total_time,
        "pocket_residues": pocket_residues,
        "n_pocket_residues": len(pocket_residues),
        "interaction_types": ["hydrophobic", "hbond_donor", "hbond_acceptor"],
        "behavior_dim": len(pocket_residues) * N_INTERACTION_TYPES,
        "discoveries": discoveries,
        "final_population_smiles": pop_smiles,
        "final_population_q": pop_q,
        "nsga2_config": {
            "pop_size": POP_SIZE,
            "objectives": ["-q", "-novelty_knn15"],
            "selection": "binary tournament (rank, crowding)",
            "replacement": "steady-state, combine + NSGA-II truncate",
            "novelty_k": KNN_NOVELTY,
        },
    }
    out = os.path.join(output_dir, "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[NSGA-II] wrote {out} — {total_time:.0f}s total, {total_time/n_iterations:.1f}s/iter")


# ─── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smoke-test", action="store_true", help="Quick sanity run; use with --iterations 50")
    args = p.parse_args()
    run_nsga2(args.target, args.seed, args.iterations, args.output_dir, smoke_test=args.smoke_test)
