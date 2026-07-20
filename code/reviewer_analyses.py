#!/usr/bin/env python3
"""
Comprehensive analysis script addressing reviewer feedback.

Computes:
1. Per-target paired comparisons (Wilcoxon signed-rank, effect sizes, CIs)
2. Target-level win counts
3. Wall-clock computational cost by method
4. Failed docking rates
5. Diversity per evaluation (efficiency)
6. External validation: MACCS+Morgan fingerprint diversity (independent of search FP)
7. Bootstrap CIs for all pooled metrics
8. Multiple-comparison corrected p-values
"""

import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats
from itertools import combinations

import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument('--results-dir', default='results',
                 help='directory containing per-run results.json (default: results)')
_args, _ = _ap.parse_known_args()
RESULTS_DIR = _args.results_dir
TARGETS = ["3V8D", "1ERE", "3EML", "1EVE", "4DFR", "3PJC", "4MNE"]
METHOD_MAP = {
    "random": "Random",
    "imgep_naive": "IMGEP (naive)",
    "imgep": "IMGEP (adaptive)",
    "curiosity": "Curiosity-IMGEP",
    "bo": "Aff-Div",
    "ga": "Genetic Alg.",
    "mapelites": "MAP-Elites",
    "novelty": "Novelty Search",
    "nsga2": "NSGA-II",
}
METHODS = list(METHOD_MAP.keys())
SEEDS = list(range(10))


def load_run(target, method, seed):
    p = os.path.join(RESULTS_DIR, target, f"{method}_seed{seed}", "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def compute_metrics(run):
    """Compute all metrics for a single run."""
    discoveries = run["discoveries"]
    behaviors = []
    smiles_list = []
    vina_scores = []
    times = []
    n_failed = 0

    for d in discoveries:
        b = d.get("behavior")
        if b is None or "error" in d:
            n_failed += 1
            continue
        behaviors.append(b)
        smiles_list.append(d["smiles"])
        vina_scores.append(d.get("vina_score", 0))
        times.append(d.get("time_s", 0))

    if len(behaviors) < 2:
        return None

    bmat = np.array(behaviors)

    # Unique profiles (2dp rounding)
    rounded = np.round(bmat, 2)
    unique_profiles = len(set(map(tuple, rounded)))

    # Pairwise distance
    from scipy.spatial.distance import pdist
    pw = pdist(bmat, "euclidean")
    mean_pw = float(np.mean(pw))

    # Coverage
    n_res = run.get("n_pocket_residues", (bmat.shape[1] - 1) // 3)
    active_any = set()
    for b in behaviors:
        for i in range(n_res):
            for j in range(3):
                if b[i * 3 + j] > 0:
                    active_any.add(i)
    coverage = len(active_any) / n_res if n_res > 0 else 0

    # Best Vina
    best_vina = min(vina_scores) if vina_scores else 0

    # Zero rate
    zero_count = sum(1 for b in behaviors if sum(b[:-1]) == 0)
    zero_rate = zero_count / len(behaviors)

    # Timing
    total_time = run.get("total_time_s", sum(times))
    mean_iter_time = np.mean(times) if times else 0

    # Unique SMILES
    unique_smiles = len(set(smiles_list))

    return {
        "unique_profiles": unique_profiles,
        "mean_pw": mean_pw,
        "coverage": coverage,
        "best_vina": best_vina,
        "zero_rate": zero_rate,
        "total_time": total_time,
        "mean_iter_time": mean_iter_time,
        "n_failed": n_failed,
        "n_valid": len(behaviors),
        "unique_smiles": unique_smiles,
        "smiles_list": smiles_list,
        "profiles_per_eval": unique_profiles / 500,
    }


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap CI for the mean."""
    data = np.array(data)
    n = len(data)
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(data, n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_d_ci(a, b, n_boot=10000, ci=0.95):
    rng = np.random.RandomState(42)
    a, b = np.array(a), np.array(b)
    ds = []
    for _ in range(n_boot):
        ai = rng.choice(a, len(a), replace=True)
        bi = rng.choice(b, len(b), replace=True)
        ds.append(cohens_d(ai, bi))
    lo = np.percentile(ds, (1 - ci) / 2 * 100)
    hi = np.percentile(ds, (1 + ci) / 2 * 100)
    return lo, hi


def compute_external_diversity(smiles_list):
    """Compute Morgan and MACCS fingerprint diversity (independent of search FP)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import DataStructs
    except ImportError:
        return None, None

    mols = []
    for s in set(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)

    if len(mols) < 2:
        return None, None

    # Morgan fingerprints
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    morgan_sims = []
    for i in range(len(morgan_fps)):
        for j in range(i + 1, len(morgan_fps)):
            morgan_sims.append(DataStructs.TanimotoSimilarity(morgan_fps[i], morgan_fps[j]))
    morgan_div = 1 - np.mean(morgan_sims) if morgan_sims else 0

    # MACCS keys
    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    maccs_sims = []
    for i in range(len(maccs_fps)):
        for j in range(i + 1, len(maccs_fps)):
            maccs_sims.append(DataStructs.TanimotoSimilarity(maccs_fps[i], maccs_fps[j]))
    maccs_div = 1 - np.mean(maccs_sims) if maccs_sims else 0

    return morgan_div, maccs_div


def compute_scaffold_count(smiles_list):
    """Count unique Bemis-Murcko scaffolds."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return None
    scaffolds = set()
    for s in set(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            try:
                sc = MurckoScaffold.GetScaffoldForMol(m)
                scaffolds.add(Chem.MolToSmiles(sc))
            except Exception:
                pass
    return len(scaffolds)


# ==================== MAIN ====================
print("=" * 80)
print("COMPREHENSIVE REVIEWER-RESPONSE ANALYSES")
print("=" * 80)

# Load all data
all_data = defaultdict(lambda: defaultdict(dict))  # method -> target -> seed -> metrics
all_smiles = defaultdict(lambda: defaultdict(dict))

print("\nLoading data...")
n_loaded = 0
for method in METHODS:
    for target in TARGETS:
        for seed in SEEDS:
            run = load_run(target, method, seed)
            if run is None:
                continue
            metrics = compute_metrics(run)
            if metrics is None:
                continue
            all_data[method][target][seed] = metrics
            all_smiles[method][target][seed] = metrics["smiles_list"]
            n_loaded += 1
print(f"Loaded {n_loaded} runs")

# ==================== 1. PER-TARGET PAIRED COMPARISONS ====================
print("\n" + "=" * 80)
print("1. PER-TARGET PAIRED COMPARISONS (vs Random)")
print("=" * 80)

metric_key = "unique_profiles"
print(f"\nMetric: {metric_key}")
print(f"{'Method':<20} {'Target':<8} {'Method mean':>12} {'Random mean':>12} {'Δ':>8} {'d':>8} {'p (MW)':>10}")
print("-" * 80)

for method in METHODS:
    if method == "random":
        continue
    for target in TARGETS:
        m_vals = [all_data[method][target][s][metric_key] for s in SEEDS if s in all_data[method][target]]
        r_vals = [all_data["random"][target][s][metric_key] for s in SEEDS if s in all_data["random"][target]]
        if len(m_vals) < 2 or len(r_vals) < 2:
            continue
        d = cohens_d(m_vals, r_vals)
        u_stat, p_val = stats.mannwhitneyu(m_vals, r_vals, alternative="two-sided")
        delta = np.mean(m_vals) - np.mean(r_vals)
        print(f"{METHOD_MAP[method]:<20} {target:<8} {np.mean(m_vals):>12.1f} {np.mean(r_vals):>12.1f} {delta:>+8.1f} {d:>8.2f} {p_val:>10.4f}")

# ==================== 2. TARGET-LEVEL WIN COUNTS ====================
print("\n" + "=" * 80)
print("2. TARGET-LEVEL WIN COUNTS (Unique Profiles)")
print("=" * 80)

print(f"{'Method':<20} {'Wins':>6} {'Ties':>6} {'Losses':>6} {'Win targets'}")
print("-" * 70)

for method in METHODS:
    if method == "random":
        continue
    wins, ties, losses = 0, 0, 0
    win_targets = []
    for target in TARGETS:
        m_vals = [all_data[method][target][s][metric_key] for s in SEEDS if s in all_data[method][target]]
        r_vals = [all_data["random"][target][s][metric_key] for s in SEEDS if s in all_data["random"][target]]
        if not m_vals or not r_vals:
            continue
        mm, rm = np.mean(m_vals), np.mean(r_vals)
        u_stat, p_val = stats.mannwhitneyu(m_vals, r_vals, alternative="two-sided")
        if p_val < 0.05:
            if mm > rm:
                wins += 1
                win_targets.append(target)
            else:
                losses += 1
        else:
            ties += 1
    print(f"{METHOD_MAP[method]:<20} {wins:>6} {ties:>6} {losses:>6} {', '.join(win_targets)}")


# ==================== 3. POOLED STATS WITH BOOTSTRAP CIs ====================
print("\n" + "=" * 80)
print("3. POOLED STATISTICS WITH BOOTSTRAP 95% CIs")
print("=" * 80)

for metric in ["unique_profiles", "mean_pw", "coverage", "best_vina"]:
    print(f"\n--- {metric} ---")
    print(f"{'Method':<20} {'Mean':>10} {'SD':>8} {'95% CI':>20} {'d vs Rand':>10} {'d 95% CI':>20} {'p (MW)':>10}")
    
    random_vals = []
    for target in TARGETS:
        for seed in SEEDS:
            if seed in all_data["random"][target]:
                random_vals.append(all_data["random"][target][seed][metric])
    
    for method in METHODS:
        vals = []
        for target in TARGETS:
            for seed in SEEDS:
                if seed in all_data[method][target]:
                    vals.append(all_data[method][target][seed][metric])
        if not vals:
            continue
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1)
        ci_lo, ci_hi = bootstrap_ci(vals)
        
        if method != "random":
            d = cohens_d(vals, random_vals)
            d_lo, d_hi = bootstrap_d_ci(vals, random_vals)
            u_stat, p = stats.mannwhitneyu(vals, random_vals, alternative="two-sided")
            d_str = f"{d:>10.2f}"
            d_ci_str = f"[{d_lo:.2f}, {d_hi:.2f}]"
            p_str = f"{p:>10.1e}" if p < 0.001 else f"{p:>10.4f}"
        else:
            d_str = "—"
            d_ci_str = "—"
            p_str = "—"
        
        print(f"{METHOD_MAP[method]:<20} {mean:>10.2f} {sd:>8.2f} [{ci_lo:.2f}, {ci_hi:.2f}]{' ':>3} {d_str} {d_ci_str:>20} {p_str}")


# ==================== 4. TARGET-LEVEL WILCOXON SIGNED-RANK ====================
print("\n" + "=" * 80)
print("4. TARGET-LEVEL WILCOXON SIGNED-RANK TEST (n=7 targets)")
print("=" * 80)

print(f"{'Method':<20} {'Metric':<18} {'W':>6} {'p':>10} {'Target means'}")
print("-" * 90)

for method in ["curiosity", "mapelites", "novelty", "ga", "bo"]:
    for metric in ["unique_profiles", "mean_pw"]:
        m_means = []
        r_means = []
        for target in TARGETS:
            m_vals = [all_data[method][target][s][metric] for s in SEEDS if s in all_data[method][target]]
            r_vals = [all_data["random"][target][s][metric] for s in SEEDS if s in all_data["random"][target]]
            if m_vals and r_vals:
                m_means.append(np.mean(m_vals))
                r_means.append(np.mean(r_vals))
        if len(m_means) >= 5:
            diffs = [m - r for m, r in zip(m_means, r_means)]
            try:
                w, p = stats.wilcoxon(diffs)
            except ValueError:
                w, p = 0, 1.0
            diffs_str = ", ".join([f"{d:+.1f}" for d in diffs])
            print(f"{METHOD_MAP[method]:<20} {metric:<18} {w:>6.0f} {p:>10.4f} [{diffs_str}]")


# ==================== 5. BONFERRONI CORRECTED P-VALUES ====================
print("\n" + "=" * 80)
print("5. BONFERRONI-CORRECTED P-VALUES (7 comparisons per metric)")
print("=" * 80)

metric = "unique_profiles"
n_comparisons = 7  # 7 methods vs random
print(f"Metric: {metric}, Bonferroni correction for {n_comparisons} comparisons")
print(f"{'Method':<20} {'p (raw)':>12} {'p (Bonf)':>12} {'Sig?':>6}")
print("-" * 55)

for method in METHODS:
    if method == "random":
        continue
    m_vals = []
    r_vals = []
    for target in TARGETS:
        for seed in SEEDS:
            if seed in all_data[method][target]:
                m_vals.append(all_data[method][target][seed][metric])
            if seed in all_data["random"][target]:
                r_vals.append(all_data["random"][target][seed][metric])
    if m_vals and r_vals:
        u, p_raw = stats.mannwhitneyu(m_vals, r_vals, alternative="two-sided")
        p_bonf = min(p_raw * n_comparisons, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns"))
        print(f"{METHOD_MAP[method]:<20} {p_raw:>12.2e} {p_bonf:>12.2e} {sig:>6}")


# ==================== 6. COMPUTATIONAL COST ====================
print("\n" + "=" * 80)
print("6. COMPUTATIONAL COST BY METHOD")
print("=" * 80)

print(f"{'Method':<20} {'Wall-clock (h)':>15} {'Per-iter (s)':>13} {'Failed (%)':>12} {'Prof/eval':>10}")
print("-" * 75)

for method in METHODS:
    total_times = []
    iter_times = []
    failed_rates = []
    prof_per_eval = []
    for target in TARGETS:
        for seed in SEEDS:
            if seed in all_data[method][target]:
                m = all_data[method][target][seed]
                total_times.append(m["total_time"])
                iter_times.append(m["mean_iter_time"])
                failed_rates.append(m["n_failed"] / 500 * 100)
                prof_per_eval.append(m["profiles_per_eval"])
    if total_times:
        print(f"{METHOD_MAP[method]:<20} {np.mean(total_times)/3600:>15.2f} {np.mean(iter_times):>13.1f} {np.mean(failed_rates):>12.1f} {np.mean(prof_per_eval):>10.3f}")


# ==================== 7. EXTERNAL VALIDATION (Morgan + MACCS diversity) ====================
print("\n" + "=" * 80)
print("7. EXTERNAL VALIDATION: MORGAN & MACCS FINGERPRINT DIVERSITY")
print("(Independent of search fingerprint)")
print("=" * 80)

morgan_data = defaultdict(list)
maccs_data = defaultdict(list)
scaffold_data = defaultdict(list)

for method in METHODS:
    print(f"  Computing for {METHOD_MAP[method]}...", end="", flush=True)
    for target in TARGETS:
        for seed in SEEDS:
            if seed not in all_data[method][target]:
                continue
            smiles = all_smiles[method][target][seed]
            morgan_div, maccs_div = compute_external_diversity(smiles)
            if morgan_div is not None:
                morgan_data[method].append(morgan_div)
                maccs_data[method].append(maccs_div)
            sc = compute_scaffold_count(smiles)
            if sc is not None:
                scaffold_data[method].append(sc)
    print(f" done ({len(morgan_data[method])} runs)")

print(f"\n{'Method':<20} {'Morgan div':>12} {'MACCS div':>12} {'Scaffolds':>12} {'d(Morg)':>10} {'d(MACCS)':>10} {'d(Scaff)':>10}")
print("-" * 90)

random_morgan = morgan_data.get("random", [])
random_maccs = maccs_data.get("random", [])
random_scaff = scaffold_data.get("random", [])

for method in METHODS:
    mg = morgan_data.get(method, [])
    mc = maccs_data.get(method, [])
    sc = scaffold_data.get(method, [])
    if not mg:
        continue
    mg_str = f"{np.mean(mg):.3f}±{np.std(mg, ddof=1):.3f}"
    mc_str = f"{np.mean(mc):.3f}±{np.std(mc, ddof=1):.3f}"
    sc_str = f"{np.mean(sc):.1f}±{np.std(sc, ddof=1):.1f}"
    
    if method != "random" and random_morgan:
        d_mg = cohens_d(mg, random_morgan)
        d_mc = cohens_d(mc, random_maccs)
        d_sc = cohens_d(sc, random_scaff) if random_scaff else 0
        print(f"{METHOD_MAP[method]:<20} {mg_str:>12} {mc_str:>12} {sc_str:>12} {d_mg:>+10.2f} {d_mc:>+10.2f} {d_sc:>+10.2f}")
    else:
        print(f"{METHOD_MAP[method]:<20} {mg_str:>12} {mc_str:>12} {sc_str:>12} {'—':>10} {'—':>10} {'—':>10}")


# ==================== 8. PER-TARGET TABLE ====================
print("\n" + "=" * 80)
print("8. PER-TARGET UNIQUE PROFILES (for supplementary/appendix)")
print("=" * 80)

print(f"{'Method':<20}", end="")
for t in TARGETS:
    print(f" {t:>12}", end="")
print()
print("-" * (20 + 13 * 7))

for method in METHODS:
    print(f"{METHOD_MAP[method]:<20}", end="")
    for target in TARGETS:
        vals = [all_data[method][target][s]["unique_profiles"] for s in SEEDS if s in all_data[method][target]]
        if vals:
            print(f" {np.mean(vals):>5.0f}±{np.std(vals, ddof=1):>4.0f}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()


# ==================== 9. HEAD-TO-HEAD PAIRWISE COMPARISONS ====================
print("\n" + "=" * 80)
print("9. HEAD-TO-HEAD METHOD COMPARISONS (Profiles, pooled)")
print("=" * 80)

method_list = METHODS
print(f"{'':>20}", end="")
for m2 in method_list:
    print(f" {METHOD_MAP[m2][:8]:>8}", end="")
print()

for m1 in method_list:
    print(f"{METHOD_MAP[m1]:<20}", end="")
    vals1 = []
    for target in TARGETS:
        for seed in SEEDS:
            if seed in all_data[m1][target]:
                vals1.append(all_data[m1][target][seed]["unique_profiles"])
    for m2 in method_list:
        if m1 == m2:
            print(f" {'—':>8}", end="")
            continue
        vals2 = []
        for target in TARGETS:
            for seed in SEEDS:
                if seed in all_data[m2][target]:
                    vals2.append(all_data[m2][target][seed]["unique_profiles"])
        if vals1 and vals2:
            d = cohens_d(vals1, vals2)
            print(f" {d:>+8.2f}", end="")
        else:
            print(f" {'N/A':>8}", end="")
    print()


print("\n" + "=" * 80)
print("DONE — All analyses complete")
print("=" * 80)
