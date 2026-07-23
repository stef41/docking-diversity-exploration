#!/usr/bin/env python3
"""Compute all final numbers needed to fill remaining placeholders."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist

ROOT = Path("/tmp/acs-review/docking-diversity-exploration/results")
TARGETS = ["3V8D","1ERE","3EML","1EVE","4DFR","3PJC","4MNE"]

def load_all_of(method):
    out = defaultdict(list)  # target -> list of run dicts
    for T in TARGETS:
        for d in (ROOT / T).iterdir() if (ROOT / T).exists() else []:
            if d.name.startswith(f"{method}_seed"):
                with open(d / "results.json") as f:
                    r = json.load(f)
                out[T].append(r)
    return out

def compute_run_metrics(run, method_name=""):
    disc = run["discoveries"]
    behs = [np.array(d["behavior"]) for d in disc if d.get("behavior")]
    if not behs: return None
    smiles = list({d["smiles"] for d in disc if d.get("smiles")})
    n_res = run.get("n_pocket_residues", 0)
    # NSGA-II stores psi; others store phi. For phi analysis on nsga2, augment with q.
    is_nsga2 = method_name == "nsga2"
    if is_nsga2:
        phi = np.array([list(b) + [min(1.0, max(0.0, -disc[i]["vina_score"]/12.0))]
                        for i, b in enumerate(behs)])
    else:
        phi = np.array(behs)
    psi = phi[:, :-1]  # drop last (q) for interaction-only
    u_phi = len(set(map(tuple, np.round(phi, 2))))
    u_psi = len(set(map(tuple, np.round(psi, 2))))
    pw_phi = float(np.mean(pdist(phi, "euclidean"))) if len(phi) > 1 else 0
    # coverage
    contacted = set()
    for i, b in enumerate(psi):
        for r_idx in range(n_res):
            if any(b[r_idx*3+t] > 0 for t in range(3) if r_idx*3+t < len(b)):
                contacted.add(r_idx)
    coverage = len(contacted) / n_res if n_res else 0
    best_vina = min(d["vina_score"] for d in disc)
    zero_pct = 100 * sum(1 for b in psi if sum(b) == 0) / len(psi)
    return {"u_phi": u_phi, "u_psi": u_psi, "pw_phi": pw_phi,
            "coverage": coverage, "best_vina": best_vina, "zero_pct": zero_pct,
            "n_unique_smiles": len(smiles)}

def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2: return 0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0

# ═══ NSGA-II full 7-target stats ═══
print("="*70)
print("NSGA-II full 7-target stats")
print("="*70)
nsga = load_all_of("nsga2")
all_rows = []
for T in TARGETS:
    for r in nsga[T]:
        m = compute_run_metrics(r, "nsga2")
        if m: all_rows.append({"target": T, **m})
print(f"n={len(all_rows)}  targets: {sorted({r['target'] for r in all_rows})}")
for k in ["u_phi", "u_psi", "pw_phi", "coverage", "best_vina", "zero_pct"]:
    vals = [r[k] for r in all_rows]
    print(f"  {k:<15} mean={np.mean(vals):.3f}  std={np.std(vals, ddof=1):.3f}")
# vs Random comparison
random_runs = load_all_of("random")
random_rows = []
for T in TARGETS:
    for r in random_runs[T]:
        m = compute_run_metrics(r)
        if m: random_rows.append({"target": T, **m})
for k in ["u_phi", "u_psi", "pw_phi", "best_vina"]:
    a = [r[k] for r in all_rows]
    b = [r[k] for r in random_rows]
    d = cohen_d(a, b)
    try:
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        p = 1.0
    print(f"  {k:<15} d_vs_random={d:+.2f}  p={p:.3e}  ({np.mean(a):.2f} vs {np.mean(b):.2f})")

# Per-target Wilcoxon
print("\nPer-target Wilcoxon (mean per target, n=7 paired):")
for metric in ["u_phi", "best_vina"]:
    per_target = []
    for T in TARGETS:
        a = [r[metric] for r in all_rows if r["target"]==T]
        b = [r[metric] for r in random_rows if r["target"]==T]
        per_target.append((np.mean(a), np.mean(b)))
    diffs = [x[0]-x[1] for x in per_target]
    try:
        stat, p = stats.wilcoxon(diffs)
        print(f"  {metric:<12} n=7 mean_diff={np.mean(diffs):+.2f} W={stat:.1f} p={p:.4f}")
    except Exception as e:
        print(f"  {metric:<12} error: {e}")

# ═══ Binary Curiosity vs Continuous Curiosity ═══
print("\n"+"="*70)
print("BINARY CURIOSITY-IMGEP (ψ_bin) vs CONTINUOUS Curiosity-IMGEP")
print("="*70)
bin_curi = load_all_of("curiosity_binary")
cont_curi = load_all_of("curiosity")
bin_rows, cont_rows = [], []
for T in TARGETS:
    for r in bin_curi[T]:
        m = compute_run_metrics(r)
        if m: bin_rows.append({"target": T, **m})
    for r in cont_curi[T]:
        m = compute_run_metrics(r)
        if m: cont_rows.append({"target": T, **m})
print(f"binary n={len(bin_rows)}, continuous n={len(cont_rows)}")
print(f"{'metric':<20} {'binary':<15} {'continuous':<15} {'Δ pct':<10} {'d':<6} {'p':<10}")
for k, label in [("u_phi","unique φ"), ("u_psi","unique ψ"),
                 ("pw_phi","pw dist"), ("coverage","coverage"),
                 ("best_vina","best vina"), ("n_unique_smiles","unique SMILES")]:
    b = [r[k] for r in bin_rows]
    c = [r[k] for r in cont_rows]
    delta_pct = 100 * (np.mean(b) - np.mean(c)) / np.mean(c) if np.mean(c) else 0
    d = cohen_d(b, c)
    try:
        p = stats.mannwhitneyu(b, c, alternative="two-sided").pvalue
    except Exception:
        p = 1.0
    print(f"{label:<20} {np.mean(b):.2f}±{np.std(b, ddof=1):.2f}    {np.mean(c):.2f}±{np.std(c, ddof=1):.2f}    {delta_pct:+.2f}%    {d:+.2f}   {p:.2e}")

# ═══ Charged 5-channel vs standard 3-channel Curiosity ═══
print("\n"+"="*70)
print("CHARGED 5-CHANNEL vs 3-CHANNEL Curiosity-IMGEP (on 1ERE, 3EML, 4DFR)")
print("="*70)
charged_curi = load_all_of("curiosity_5ch")
charged_rand = load_all_of("random_5ch")
CHARGED_TARGETS = ["1ERE","3EML","4DFR"]
print(f"{'target':<8} {'method':<20} {'metric':<15} {'value':<20}")
for T in CHARGED_TARGETS:
    for label, runs in [("Curiosity 5ch", charged_curi[T]),
                        ("Random 5ch", charged_rand[T]),
                        ("Curiosity 3ch", cont_curi[T]),
                        ("Random 3ch", random_runs[T])]:
        vals_u = [compute_run_metrics(r)["u_phi"] for r in runs if compute_run_metrics(r)]
        vals_c = [compute_run_metrics(r)["coverage"] for r in runs if compute_run_metrics(r)]
        vals_v = [compute_run_metrics(r)["best_vina"] for r in runs if compute_run_metrics(r)]
        if vals_u:
            print(f"{T:<8} {label:<20} u_phi          {np.mean(vals_u):.1f}±{np.std(vals_u,ddof=1):.1f}"
                  f" | cov {np.mean(vals_c)*100:.1f}% | best_vina {np.mean(vals_v):.2f}")
    print()

# ═══ Cutoff sensitivity ═══
print("="*70)
print("CUTOFF SENSITIVITY (Curiosity-IMGEP at different d_max)")
print("="*70)
CUT_TARGETS = ["1EVE", "4DFR"]
for T in CUT_TARGETS:
    print(f"\n{T}:")
    for method, label in [("curiosity", "d_max=4.0 (original)"),
                          ("curiosity_dmax35", "d_max=3.5"),
                          ("curiosity_dmax45", "d_max=4.5"),
                          ("curiosity_dmax40_noq", "d_max=4.0, no q(m) in retrieval")]:
        runs_of = load_all_of(method)
        vals = [compute_run_metrics(r) for r in runs_of[T]]
        vals = [v for v in vals if v]
        if vals:
            print(f"  {label:<38} u_phi={np.mean([v['u_phi'] for v in vals]):.1f}±{np.std([v['u_phi'] for v in vals],ddof=1):.1f}"
                  f"  cov={np.mean([v['coverage'] for v in vals])*100:.1f}%"
                  f"  best_vina={np.mean([v['best_vina'] for v in vals]):.2f}")

# Save all as JSON
out = {
    "nsga2_full": {"per_run": all_rows,
                   "aggregate": {k: {"mean": float(np.mean([r[k] for r in all_rows])),
                                     "std": float(np.std([r[k] for r in all_rows], ddof=1))}
                                 for k in ["u_phi","u_psi","pw_phi","coverage","best_vina","zero_pct"]}},
    "binary_vs_continuous_curiosity": {"binary": bin_rows, "continuous": cont_rows},
}
with open("/tmp/acs-review/revision/analyses/final_numbers.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved: /tmp/acs-review/revision/analyses/final_numbers.json")
