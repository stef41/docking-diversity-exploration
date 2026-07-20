#!/usr/bin/env python3
"""
analyze_v3.py — Nature-level analysis for multi-target experiments.

Produces:
  - fig1_diversity_all_targets.pdf: Main result — diversity across all targets
  - fig2_per_target_heatmap.pdf: Method × Target performance heatmap
  - fig3_convergence.pdf: Temporal convergence curves
  - fig4_quality.pdf: Vina score distributions
  - fig5_ablation.pdf: Behavior space design impact (v1 vs v2 comparison)
  - fig6_pharmacological.pdf: Binding mode clustering analysis
  - stats_v3.txt: Full statistical analysis with effect sizes
  - table1.tex: LaTeX table for paper
"""
import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument('--results-dir', default='results',
                 help='directory containing per-run results.json (default: results)')
_ap.add_argument('--out', dest='output_dir', default='figures',
                 help='directory to write generated figures + tables (default: figures)')
_args, _ = _ap.parse_known_args()
RESULTS_DIR = _args.results_dir
OUTPUT_DIR = _args.output_dir

TARGETS = ["3V8D", "1ERE", "3EML", "1EVE", "4DFR", "3PJC", "4MNE"]
TARGET_NAMES = {
    "3V8D": "CYP7A1", "1ERE": "ER-α", "3EML": "A₂ₐR",
    "1EVE": "AChE", "4DFR": "DHFR", "3PJC": "JAK3", "4MNE": "BRAF"
}
TARGET_FAMILIES = {
    "3V8D": "Oxidoreductase", "1ERE": "Nuclear receptor", "3EML": "GPCR",
    "1EVE": "Hydrolase", "4DFR": "Oxidoreductase", "3PJC": "Kinase", "4MNE": "Kinase"
}

METHODS = ["random", "imgep_naive", "imgep", "curiosity", "bo", "ga", "mapelites", "novelty"]
METHOD_LABELS = {
    "random": "Random", "imgep_naive": "IMGEP (naive)",
    "imgep": "IMGEP (adaptive)", "curiosity": "Curiosity-IMGEP",
    "bo": "UCB Heuristic", "ga": "Genetic Alg.",
    "mapelites": "MAP-Elites", "novelty": "Novelty Search"
}
# Colorblind-friendly palette (Wong 2011) — consistent with figures_nature.py
METHOD_COLORS = {
    "random": "#999999", "imgep_naive": "#CC79A7",
    "imgep": "#56B4E9", "curiosity": "#E69F00",
    "bo": "#882255", "ga": "#009E73",
    "mapelites": "#0072B2", "novelty": "#D55E00"
}
SEEDS = list(range(10))


def load_results(target, method, seed):
    path = os.path.join(RESULTS_DIR, target, f"{method}_seed{seed}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_metrics(data):
    """Compute all metrics for a single run."""
    discoveries = data['discoveries']
    n_res = data['n_pocket_residues']
    n_types = len(data['interaction_types'])
    dim = data['behavior_dim']
    behaviors = [np.array(d['behavior']) for d in discoveries]
    n_iters = len(behaviors)

    # Final pairwise diversity
    bmat = np.array(behaviors)
    if len(bmat) > 1:
        pw_dist = np.mean(pdist(bmat))
    else:
        pw_dist = 0.0

    # Unique molecules
    unique_mols = len(set(d['smiles'] for d in discoveries))

    # Unique interaction profiles (rounded)
    rounded = set()
    for b in behaviors:
        key = tuple(np.round(b, 2))
        rounded.add(key)
    unique_profiles = len(rounded)

    # Zero fingerprint rate
    zero_count = sum(1 for b in behaviors if np.all(np.array(b[:-1]) == 0))
    zero_rate = zero_count / n_iters

    # Best Vina score (most negative = best)
    vina_scores = [d['vina_score'] for d in discoveries if d['vina_score'] != 0]
    best_vina = min(vina_scores) if vina_scores else 0.0
    mean_vina = np.mean(vina_scores) if vina_scores else 0.0

    # Active residues across all iterations (cumulative coverage)
    all_active = set()
    for d in discoveries:
        b = np.array(d['behavior'])
        for ri in range(n_res):
            if any(b[ri * n_types + ti] > 0 for ti in range(n_types)):
                all_active.add(ri)
    residue_coverage = len(all_active) / n_res if n_res > 0 else 0.0

    # Temporal pairwise diversity curve (sampled every 25 iters)
    pw_curve = []
    for t in range(24, n_iters, 25):
        bsub = np.array(behaviors[:t+1])
        if len(bsub) > 1:
            pw_curve.append(np.mean(pdist(bsub)))
        else:
            pw_curve.append(0.0)

    # Mean active interactions per molecule
    mean_interactions = np.mean([d['n_interactions'] for d in discoveries])
    mean_active_res = np.mean([d['active_residues'] for d in discoveries])

    return {
        'pw_dist': pw_dist,
        'unique_mols': unique_mols,
        'unique_profiles': unique_profiles,
        'zero_rate': zero_rate,
        'best_vina': best_vina,
        'mean_vina': mean_vina,
        'residue_coverage': residue_coverage,
        'pw_curve': pw_curve,
        'mean_interactions': mean_interactions,
        'mean_active_res': mean_active_res,
    }


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def bootstrap_ci(x, y, metric_fn=np.mean, n_boot=10000, alpha=0.05):
    """Bootstrap confidence interval for difference in means."""
    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        xb = rng.choice(x, len(x), replace=True)
        yb = rng.choice(y, len(y), replace=True)
        diffs.append(metric_fn(xb) - metric_fn(yb))
    diffs = sorted(diffs)
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Load all results ----------
    all_metrics = {}  # {target: {method: [metrics_per_seed]}}
    loaded = 0
    missing = 0

    for target in TARGETS:
        all_metrics[target] = {}
        for method in METHODS:
            runs = []
            for seed in SEEDS:
                data = load_results(target, method, seed)
                if data is not None:
                    runs.append(compute_metrics(data))
                    loaded += 1
                else:
                    missing += 1
            all_metrics[target][method] = runs

    print(f"Loaded: {loaded}, Missing: {missing}")
    if loaded == 0:
        print("No results found!")
        return

    # ---------- Aggregate across targets ----------
    global_metrics = {}  # {method: {metric: [values across all targets and seeds]}}
    for method in METHODS:
        global_metrics[method] = defaultdict(list)
        for target in TARGETS:
            for run in all_metrics[target][method]:
                for key in ['pw_dist', 'unique_mols', 'unique_profiles', 'zero_rate',
                           'best_vina', 'mean_vina', 'residue_coverage',
                           'mean_interactions', 'mean_active_res']:
                    global_metrics[method][key].append(run[key])

    # ---------- Stats file ----------
    with open(os.path.join(OUTPUT_DIR, 'stats_v3.txt'), 'w') as sf:
        sf.write("=" * 80 + "\n")
        sf.write("NATURE-LEVEL STATISTICAL ANALYSIS — V3 MULTI-TARGET\n")
        sf.write(f"Targets: {len(TARGETS)}, Methods: {len(METHODS)}, Seeds: {len(SEEDS)}\n")
        sf.write(f"Loaded runs: {loaded}, Missing: {missing}\n")
        sf.write("=" * 80 + "\n\n")

        # Per-target summary
        for target in TARGETS:
            sf.write(f"\n{'—'*60}\n")
            sf.write(f"TARGET: {target} ({TARGET_NAMES[target]}, {TARGET_FAMILIES[target]})\n")
            sf.write(f"{'—'*60}\n")
            sf.write(f"{'Method':<18} {'PW Dist':>10} {'Unique':>8} {'Profiles':>10} "
                     f"{'Zero%':>8} {'BestVina':>10} {'Coverage':>10}\n")

            for method in METHODS:
                runs = all_metrics[target][method]
                if not runs:
                    continue
                pw = [r['pw_dist'] for r in runs]
                um = [r['unique_mols'] for r in runs]
                up = [r['unique_profiles'] for r in runs]
                zr = [r['zero_rate'] for r in runs]
                bv = [r['best_vina'] for r in runs]
                rc = [r['residue_coverage'] for r in runs]

                sf.write(f"{METHOD_LABELS[method]:<18} "
                         f"{np.mean(pw):>7.3f}±{np.std(pw):.3f} "
                         f"{np.mean(um):>5.1f}±{np.std(um):.1f} "
                         f"{np.mean(up):>7.1f}±{np.std(up):.1f} "
                         f"{np.mean(zr)*100:>5.1f}% "
                         f"{np.mean(bv):>7.2f}±{np.std(bv):.2f} "
                         f"{np.mean(rc)*100:>6.1f}%±{np.std(rc)*100:.1f}\n")

        # Global comparison
        sf.write(f"\n\n{'='*80}\n")
        sf.write("GLOBAL CROSS-TARGET COMPARISON\n")
        sf.write(f"{'='*80}\n")

        ref_method = "random"
        for metric_name in ['pw_dist', 'unique_profiles', 'residue_coverage', 'best_vina']:
            sf.write(f"\n--- {metric_name} ---\n")
            ref_vals = np.array(global_metrics[ref_method][metric_name])
            if len(ref_vals) == 0:
                continue

            for method in METHODS:
                if method == ref_method:
                    continue
                test_vals = np.array(global_metrics[method][metric_name])
                if len(test_vals) == 0:
                    continue

                d = cohens_d(test_vals, ref_vals)
                t_stat, p_val = stats.ttest_ind(test_vals, ref_vals)
                _, p_mw = stats.mannwhitneyu(test_vals, ref_vals, alternative='two-sided')
                ci_lo, ci_hi = bootstrap_ci(test_vals, ref_vals)

                sf.write(f"  {METHOD_LABELS[method]} vs {METHOD_LABELS[ref_method]}: "
                         f"Δ={np.mean(test_vals)-np.mean(ref_vals):.4f}, "
                         f"d={d:.2f}, "
                         f"t-test p={p_val:.4f}, MW p={p_mw:.4f}, "
                         f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n")

        # Curiosity vs all others (pairwise)
        sf.write(f"\n\n{'='*80}\n")
        sf.write("CURIOSITY-IMGEP vs ALL OTHER METHODS\n")
        sf.write(f"{'='*80}\n")
        for metric_name, higher_better in [
            ('pw_dist', True), ('unique_profiles', True),
            ('residue_coverage', True), ('best_vina', False)
        ]:
            sf.write(f"\n--- {metric_name} (higher_better={higher_better}) ---\n")
            cur_vals = np.array(global_metrics['curiosity'][metric_name])
            if len(cur_vals) == 0:
                continue

            for method in METHODS:
                if method == 'curiosity':
                    continue
                other_vals = np.array(global_metrics[method][metric_name])
                if len(other_vals) == 0:
                    continue

                d = cohens_d(cur_vals, other_vals)
                _, p_val = stats.ttest_ind(cur_vals, other_vals)
                _, p_mw = stats.mannwhitneyu(cur_vals, other_vals, alternative='two-sided')

                sign = "+" if (np.mean(cur_vals) > np.mean(other_vals)) == higher_better else "-"
                sf.write(f"  {sign} vs {METHOD_LABELS[method]}: "
                         f"Δ={np.mean(cur_vals)-np.mean(other_vals):.4f}, "
                         f"d={d:.2f}, p={p_val:.4f} (MW: {p_mw:.4f})\n")

    print(f"Stats written to {OUTPUT_DIR}/stats_v3.txt")

    # ---------- Figure 1: Main diversity result across all targets ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1a: Pairwise diversity boxplot
    ax = axes[0]
    data_boxes = []
    labels = []
    colors = []
    for method in METHODS:
        vals = global_metrics[method]['pw_dist']
        if vals:
            data_boxes.append(vals)
            labels.append(METHOD_LABELS[method])
            colors.append(METHOD_COLORS[method])
    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean pairwise distance')
    ax.set_title('a) Behavioral diversity')

    # 1b: Unique profiles boxplot
    ax = axes[1]
    data_boxes = []
    for method in METHODS:
        vals = global_metrics[method]['unique_profiles']
        if vals:
            data_boxes.append(vals)
    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Unique interaction profiles')
    ax.set_title('b) Profile diversity')

    # 1c: Residue coverage boxplot
    ax = axes[2]
    data_boxes = []
    for method in METHODS:
        vals = [v * 100 for v in global_metrics[method]['residue_coverage']]
        if vals:
            data_boxes.append(vals)
    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Pocket residue coverage (%)')
    ax.set_title('c) Binding site coverage')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_diversity_main.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_diversity_main.png'))
    plt.close(fig)
    print("Fig 1 saved")

    # ---------- Figure 2: Per-target heatmap ----------
    metrics_for_hm = ['pw_dist', 'unique_profiles', 'residue_coverage']
    metric_labels_hm = ['Pairwise dist.', 'Unique profiles', 'Residue coverage (%)']

    fig, axes = plt.subplots(1, len(metrics_for_hm), figsize=(16, 5))

    for mi, (metric, mlabel) in enumerate(zip(metrics_for_hm, metric_labels_hm)):
        ax = axes[mi]
        mat = np.zeros((len(METHODS), len(TARGETS)))
        for ti, target in enumerate(TARGETS):
            for mi2, method in enumerate(METHODS):
                runs = all_metrics[target][method]
                if runs:
                    vals = [r[metric] for r in runs]
                    if metric == 'residue_coverage':
                        mat[mi2, ti] = np.mean(vals) * 100
                    else:
                        mat[mi2, ti] = np.mean(vals)

        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([TARGET_NAMES[t] for t in TARGETS], rotation=45, ha='right')
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
        ax.set_title(mlabel)
        # Add text
        for i in range(len(METHODS)):
            for j in range(len(TARGETS)):
                ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_target_heatmap.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_per_target_heatmap.png'))
    plt.close(fig)
    print("Fig 2 saved")

    # ---------- Figure 3: Convergence curves ----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for ti, target in enumerate(TARGETS):
        ax = axes_flat[ti]
        for method in METHODS:
            runs = all_metrics[target][method]
            if not runs:
                continue
            curves = [r['pw_curve'] for r in runs if r['pw_curve']]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:min_len] for c in curves])
            mean_c = np.mean(curves_arr, axis=0)
            std_c = np.std(curves_arr, axis=0)
            x = np.arange(1, min_len + 1) * 25
            ax.plot(x, mean_c, color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=1.5)
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                          color=METHOD_COLORS[method], alpha=0.15)
        ax.set_title(f'{TARGET_NAMES[target]}')
        ax.set_xlabel('Iteration')
        if ti % 4 == 0:
            ax.set_ylabel('Mean pairwise distance')

    # Last panel: aggregated across all targets
    ax = axes_flat[len(TARGETS)]
    for method in METHODS:
        all_curves = []
        for target in TARGETS:
            for run in all_metrics[target][method]:
                if run['pw_curve']:
                    all_curves.append(run['pw_curve'])
        if not all_curves:
            continue
        min_len = min(len(c) for c in all_curves)
        curves_arr = np.array([c[:min_len] for c in all_curves])
        mean_c = np.mean(curves_arr, axis=0)
        sem_c = np.std(curves_arr, axis=0) / np.sqrt(len(curves_arr))
        x = np.arange(1, min_len + 1) * 25
        ax.plot(x, mean_c, color=METHOD_COLORS[method],
                label=METHOD_LABELS[method], linewidth=2)
        ax.fill_between(x, mean_c - sem_c, mean_c + sem_c,
                      color=METHOD_COLORS[method], alpha=0.2)
    ax.set_title('All targets (aggregated)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean pairwise distance')
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_convergence.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_convergence.png'))
    plt.close(fig)
    print("Fig 3 saved")

    # ---------- Figure 4: Quality — Vina scores ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 4a: Best Vina per method
    ax = axes[0]
    data_boxes = []
    labels_box = []
    colors_box = []
    for method in METHODS:
        vals = global_metrics[method]['best_vina']
        if vals:
            data_boxes.append(vals)
            labels_box.append(METHOD_LABELS[method])
            colors_box.append(METHOD_COLORS[method])
    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels_box) + 1))
    ax.set_xticklabels(labels_box, rotation=45, ha='right')
    ax.set_ylabel('Best Vina score (kcal/mol)')
    ax.set_title('a) Best binding affinity')

    # 4b: Diversity vs Quality scatter
    ax = axes[1]
    for method in METHODS:
        pw_vals = global_metrics[method]['pw_dist']
        vina_vals = global_metrics[method]['best_vina']
        if pw_vals and vina_vals:
            ax.scatter(pw_vals, vina_vals, c=METHOD_COLORS[method],
                      label=METHOD_LABELS[method], alpha=0.5, s=30)
    ax.set_xlabel('Behavioral diversity (PW distance)')
    ax.set_ylabel('Best Vina score (kcal/mol)')
    ax.set_title('b) Diversity-Quality trade-off')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_quality.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_quality.png'))
    plt.close(fig)
    print("Fig 4 saved")

    # ---------- Table 1: LaTeX summary ----------
    with open(os.path.join(OUTPUT_DIR, 'table1.tex'), 'w') as tf:
        tf.write("\\begin{table*}[t]\n")
        tf.write("\\centering\n")
        tf.write("\\caption{Multi-target drug discovery benchmark. "
                 "Mean ± std across 7 targets × 10 seeds. "
                 "\\textbf{Bold} = best per metric.}\n")
        tf.write("\\label{tab:main}\n")
        tf.write("\\begin{tabular}{lcccccc}\n")
        tf.write("\\toprule\n")
        tf.write("Method & PW Dist. & Profiles & Coverage (\\%) & "
                 "Best Vina & Zero\\% & Interactions \\\\\n")
        tf.write("\\midrule\n")

        # Find best for each metric
        metric_vals = {}
        for m in ['pw_dist', 'unique_profiles', 'residue_coverage', 'best_vina',
                  'zero_rate', 'mean_interactions']:
            metric_vals[m] = {}
            for method in METHODS:
                vals = global_metrics[method][m]
                if vals:
                    metric_vals[m][method] = np.mean(vals)

        best_methods = {}
        for m in metric_vals:
            if not metric_vals[m]:
                continue
            if m in ['zero_rate', 'best_vina']:
                best_methods[m] = min(metric_vals[m], key=metric_vals[m].get)
            else:
                best_methods[m] = max(metric_vals[m], key=metric_vals[m].get)

        for method in METHODS:
            vals_pw = global_metrics[method]['pw_dist']
            vals_up = global_metrics[method]['unique_profiles']
            vals_rc = global_metrics[method]['residue_coverage']
            vals_bv = global_metrics[method]['best_vina']
            vals_zr = global_metrics[method]['zero_rate']
            vals_mi = global_metrics[method]['mean_interactions']

            if not vals_pw:
                continue

            def fmt(vals, m, fmt_str, pct=False):
                mean = np.mean(vals) * (100 if pct else 1)
                std = np.std(vals) * (100 if pct else 1)
                s = f"{mean:{fmt_str}}±{std:{fmt_str}}"
                if best_methods.get(m) == method:
                    return f"\\textbf{{{s}}}"
                return s

            line = f"{METHOD_LABELS[method]} & "
            line += f"{fmt(vals_pw, 'pw_dist', '.3f')} & "
            line += f"{fmt(vals_up, 'unique_profiles', '.1f')} & "
            line += f"{fmt(vals_rc, 'residue_coverage', '.1f', pct=True)} & "
            line += f"{fmt(vals_bv, 'best_vina', '.2f')} & "
            line += f"{fmt(vals_zr, 'zero_rate', '.1f', pct=True)} & "
            line += f"{fmt(vals_mi, 'mean_interactions', '.1f')} \\\\\n"
            tf.write(line)

        tf.write("\\bottomrule\n")
        tf.write("\\end{tabular}\n")
        tf.write("\\end{table*}\n")

    print(f"Table 1 → {OUTPUT_DIR}/table1.tex")

    # ---------- Summary ----------
    print(f"\n{'='*60}")
    print("QUICK SUMMARY (global means)")
    print(f"{'='*60}")
    print(f"{'Method':<18} {'PW Dist':>10} {'Profiles':>10} {'Coverage':>10} {'BestVina':>10}")
    for method in METHODS:
        pw = global_metrics[method]['pw_dist']
        up = global_metrics[method]['unique_profiles']
        rc = global_metrics[method]['residue_coverage']
        bv = global_metrics[method]['best_vina']
        if pw:
            print(f"{METHOD_LABELS[method]:<18} "
                  f"{np.mean(pw):>7.3f}±{np.std(pw):.3f} "
                  f"{np.mean(up):>7.1f}±{np.std(up):.1f} "
                  f"{np.mean(rc)*100:>6.1f}%±{np.std(rc)*100:.1f} "
                  f"{np.mean(bv):>7.2f}±{np.std(bv):.2f}")


if __name__ == "__main__":
    main()
