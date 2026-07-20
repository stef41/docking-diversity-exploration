#!/usr/bin/env python3
"""
Generate protein-ligand docking visualizations for the paper.
Produces 4 figures:
  1. 2D molecule grid: diverse ligands discovered by Curiosity-IMGEP
  2. Interaction fingerprint heatmap: comparing methods
  3. Binding pocket schematic with residue contacts
  4. Re-dock top molecules to show 3D poses overlaid in the pocket
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument('--results-dir', default='results',
                 help='directory containing per-run results.json (default: results)')
_ap.add_argument('--out', dest='output_dir', default='figures',
                 help='directory to write generated figures (default: figures)')
_ap.add_argument('--pdb-file', default='targets/3V8D_A_protein.pdb',
                 help='reference PDB file for docking-schematic figure (default: targets/3V8D_A_protein.pdb)')
_args, _ = _ap.parse_known_args()
RESULTS_DIR = _args.results_dir
OUTPUT_DIR = _args.output_dir
PDB_FILE = _args.pdb_file

METHODS = {
    "curiosity": "Curiosity-IMGEP",
    "mapelites": "MAP-Elites",
    "random": "Random",
    "bo": "UCB Heuristic",
    "ga": "Genetic Alg.",
}
# Colorblind-friendly palette (Wong 2011) — consistent with figures_nature.py
METHOD_COLORS = {
    "Curiosity-IMGEP": "#E69F00",
    "MAP-Elites": "#0072B2",
    "Random": "#999999",
    "UCB Heuristic": "#882255",
    "Genetic Alg.": "#009E73",
}

TARGET = "3V8D"


def load_results(target, method, seed=0):
    path = os.path.join(RESULTS_DIR, target, f"{method}_seed{seed}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# FIGURE A: 2D molecule grid — diverse ligands from Curiosity
# ═══════════════════════════════════════════════════════════════

def fig_molecule_grid():
    """Grid of diverse 2D molecules discovered by Curiosity-IMGEP on CYP7A1."""
    print("Generating molecule grid...")
    data = load_results(TARGET, "curiosity", seed=0)
    if data is None:
        print("  No curiosity data found, skipping.")
        return

    discoveries = data["discoveries"]

    # Pick diverse molecules: best affinity, most residues, and spread across iterations
    selected = []
    # Best affinity
    by_vina = sorted(discoveries, key=lambda x: x["vina_score"])
    selected.append(by_vina[0])
    # Most residue contacts
    by_res = sorted(discoveries, key=lambda x: x["active_residues"], reverse=True)
    for d in by_res:
        if d["smiles"] not in [s["smiles"] for s in selected]:
            selected.append(d)
            break
    # Spread across iterations
    for target_iter in [50, 100, 200, 300, 400, 480]:
        closest = min(discoveries, key=lambda x: abs(x["iteration"] - target_iter))
        if closest["smiles"] not in [s["smiles"] for s in selected]:
            selected.append(closest)
        if len(selected) >= 9:
            break

    # Fill remaining with diverse molecules
    smiles_seen = {s["smiles"] for s in selected}
    for d in discoveries:
        if d["smiles"] not in smiles_seen and d["n_interactions"] >= 5:
            selected.append(d)
            smiles_seen.add(d["smiles"])
        if len(selected) >= 9:
            break

    selected = selected[:9]

    mols = []
    legends = []
    for s in selected:
        mol = Chem.MolFromSmiles(s["smiles"])
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            legends.append(
                f"Iter {s['iteration']} | Vina: {s['vina_score']:.1f}\n"
                f"Residues: {s['active_residues']} | Int: {s['n_interactions']}"
            )

    if not mols:
        print("  No valid molecules, skipping.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(400, 350),
        legends=legends,
        useSVG=False,
    )

    outpath = os.path.join(OUTPUT_DIR, "fig_molecule_grid.png")
    img.save(outpath)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════
# FIGURE B: Interaction fingerprint heatmaps
# ═══════════════════════════════════════════════════════════════

def fig_fingerprint_heatmap():
    """Compare fingerprint patterns across methods."""
    print("Generating fingerprint heatmaps...")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    method_list = ["curiosity", "mapelites", "random", "ga"]
    method_names = ["Curiosity-IMGEP", "MAP-Elites", "Random", "Genetic Alg."]

    cmap = LinearSegmentedColormap.from_list("pocket", ["#F7F7F7", "#2166AC", "#B2182B"], N=256)

    for idx, (method, name) in enumerate(zip(method_list, method_names)):
        data = load_results(TARGET, method, seed=0)
        if data is None:
            axes[idx].set_title(name)
            axes[idx].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[idx].transAxes)
            continue

        discoveries = data["discoveries"]
        n_res = data["n_pocket_residues"]
        n_types = len(data["interaction_types"])

        # Collect fingerprints — sample 50 evenly spaced
        indices = np.linspace(0, len(discoveries) - 1, min(50, len(discoveries)), dtype=int)
        behaviors = []
        for i in indices:
            b = discoveries[i]["behavior"]
            behaviors.append(b)

        mat = np.array(behaviors)
        # Show only residue dimensions (drop last Vina column)
        mat = mat[:, :-1]

        # Reshape to (n_mols, n_res, 3) and average across interaction types
        # for a cleaner display
        if mat.shape[1] == n_res * n_types:
            mat_reshaped = mat.reshape(mat.shape[0], n_res, n_types)
            mat_display = np.max(mat_reshaped, axis=2)  # max interaction per residue
        else:
            mat_display = mat

        ax = axes[idx]
        im = ax.imshow(mat_display.T, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(name, fontsize=13, fontweight='bold',
                     color=METHOD_COLORS.get(name, "black"))
        ax.set_xlabel("Molecule index", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Pocket residue", fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Interaction proximity")

    outpath = os.path.join(OUTPUT_DIR, "fig_fingerprint_heatmap.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════
# FIGURE C: Binding pocket contact map
# ═══════════════════════════════════════════════════════════════

def fig_pocket_contact_map():
    """Circular pocket residue contact map showing which residues are
    contacted by which methods."""
    print("Generating pocket contact map...")

    data_curiosity = load_results(TARGET, "curiosity", seed=0)
    data_random = load_results(TARGET, "random", seed=0)
    data_bo = load_results(TARGET, "bo", seed=0)

    if data_curiosity is None:
        print("  No data, skipping.")
        return

    n_res = data_curiosity["n_pocket_residues"]
    n_types = len(data_curiosity["interaction_types"])
    residues = data_curiosity["pocket_residues"]

    def get_residue_counts(data):
        counts = np.zeros(n_res)
        for d in data["discoveries"]:
            b = np.array(d["behavior"][:-1])  # drop Vina
            if len(b) == n_res * n_types:
                b_reshaped = b.reshape(n_res, n_types)
                contact = np.any(b_reshaped > 0, axis=1)
                counts += contact.astype(float)
        return counts / len(data["discoveries"])

    counts_cur = get_residue_counts(data_curiosity)
    counts_rnd = get_residue_counts(data_random) if data_random else np.zeros(n_res)
    counts_bo = get_residue_counts(data_bo) if data_bo else np.zeros(n_res)

    # Sort by curiosity count for clarity
    order = np.argsort(-counts_cur)

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(n_res)
    width = 0.28
    ax.bar(x - width, counts_cur[order], width, label="Curiosity-IMGEP",
           color=METHOD_COLORS["Curiosity-IMGEP"], alpha=0.85)
    ax.bar(x, counts_rnd[order], width, label="Random",
           color=METHOD_COLORS["Random"], alpha=0.85)
    ax.bar(x + width, counts_bo[order], width, label="Bayesian Opt.",
           color=METHOD_COLORS["Bayesian Opt."], alpha=0.85)

    # Label top residues
    res_labels = [str(residues[i]) if i < len(residues) else str(i) for i in order]
    tick_indices = np.arange(0, n_res, max(1, n_res // 20))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([res_labels[i] for i in tick_indices], rotation=45, fontsize=7)

    ax.set_xlabel("Pocket residue (sorted by Curiosity contact frequency)", fontsize=11)
    ax.set_ylabel("Contact frequency", fontsize=11)
    ax.set_title("CYP7A1 (3V8D) — Per-residue contact frequency by method", fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, n_res - 0.5)

    outpath = os.path.join(OUTPUT_DIR, "fig_pocket_contacts.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════
# FIGURE D: Conceptual schematic — exploration pipeline
# ═══════════════════════════════════════════════════════════════

def fig_pipeline_schematic():
    """Conceptual diagram showing the IMGEP docking exploration pipeline.
    Uses graphviz for automatic layout — no manual coordinate placement."""
    import graphviz
    print("Generating pipeline schematic...")

    # Professional palette: light gray boxes, dark text, one accent for feedback
    node_fill = '#F0F0F0'
    node_border = '#333333'
    text_color = '#222222'
    feedback_color = '#555555'

    g = graphviz.Digraph('pipeline', format='pdf',
                         graph_attr={
                             'rankdir': 'LR',
                             'bgcolor': 'white',
                             'fontname': 'Helvetica',
                             'label': '',
                             'pad': '0.3',
                             'nodesep': '0.5',
                             'ranksep': '1.0',
                             'dpi': '300',
                             'splines': 'polyline',
                         },
                         node_attr={
                             'shape': 'box',
                             'style': 'rounded,filled',
                             'fontname': 'Helvetica',
                             'fontsize': '11',
                             'fontcolor': text_color,
                             'fillcolor': node_fill,
                             'color': node_border,
                             'penwidth': '1.5',
                             'width': '1.6',
                             'height': '0.8',
                             'margin': '0.15,0.1',
                         },
                         edge_attr={
                             'fontname': 'Helvetica',
                             'fontsize': '9',
                             'color': node_border,
                             'fontcolor': '#555555',
                             'penwidth': '1.5',
                             'arrowsize': '0.8',
                         })

    # Nodes in pipeline order (archive is part of the loop)
    g.node('archive', 'Behavior\nArchive',
           fillcolor='#D9D9D9', style='rounded,filled,bold')
    g.node('goal', 'Goal\nSampling')
    g.node('nn', 'Nearest-Neighbor\nRetrieval')
    g.node('crem', 'CReM\nMutation')
    g.node('dock', 'GNINA\nDocking')
    g.node('plip', 'PLIP\nAnalysis')

    # Forward edges (single horizontal chain)
    g.edge('archive', 'goal', label=' goal selection ')
    g.edge('goal', 'nn', label=' goal g ')
    g.edge('nn', 'crem', label=' parent mol ')
    g.edge('crem', 'dock', label=" mol m' ")
    g.edge('dock', 'plip', label=' pose ')

    # Feedback: PLIP → Archive (closes the loop)
    g.edge('plip', 'archive', label=" store φ(m') ",
           style='dashed', constraint='false')

    outpath = os.path.join(OUTPUT_DIR, "fig_pipeline_schematic")
    g.render(outpath, cleanup=True)
    g.format = 'png'
    g.render(outpath, cleanup=True)
    print(f"  Saved: {outpath}.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE E: Behavior space visualization (t-SNE)
# ═══════════════════════════════════════════════════════════════

def fig_behavior_tsne():
    """t-SNE of behavior fingerprints color-coded by method."""
    print("Generating behavior space t-SNE...")

    from sklearn.manifold import TSNE

    all_behaviors = []
    all_labels = []
    all_vina = []

    method_list = ["curiosity", "mapelites", "random", "bo", "ga"]

    for method in method_list:
        data = load_results(TARGET, method, seed=0)
        if data is None:
            continue
        discoveries = data["discoveries"]
        # Sample up to 100 molecules per method
        indices = np.linspace(0, len(discoveries) - 1, min(100, len(discoveries)), dtype=int)
        for i in indices:
            all_behaviors.append(discoveries[i]["behavior"])
            all_labels.append(METHODS[method])
            all_vina.append(discoveries[i]["vina_score"])

    if not all_behaviors:
        print("  No data for t-SNE, skipping.")
        return

    X = np.array(all_behaviors)
    labels = np.array(all_labels)
    vina = np.array(all_vina)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    # Plot 1: by method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method_name in METHODS.values():
        mask = labels == method_name
        if mask.sum() == 0:
            continue
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=METHOD_COLORS[method_name], label=method_name,
                    alpha=0.6, s=25, edgecolors='white', linewidth=0.3)

    ax1.set_title("Behavior Space (t-SNE) — by Method", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, markerscale=1.5)
    ax1.set_xlabel("t-SNE 1", fontsize=11)
    ax1.set_ylabel("t-SNE 2", fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 2: by Vina score
    sc = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=vina, cmap='RdYlGn_r',
                     alpha=0.6, s=25, edgecolors='white', linewidth=0.3,
                     vmin=-12, vmax=-7)
    ax2.set_title("Behavior Space (t-SNE) — by Vina Score", fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax2, label="Vina score (kcal/mol)", shrink=0.8)
    ax2.set_xlabel("t-SNE 1", fontsize=11)
    ax2.set_ylabel("t-SNE 2", fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "fig_behavior_tsne.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════
# FIGURE F: Diversity comparison — discovered molecules
# ═══════════════════════════════════════════════════════════════

def fig_molecule_comparison():
    """Side-by-side 2D molecule comparison between methods."""
    print("Generating method molecule comparison...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.2)

    methods_to_show = [
        ("curiosity", "Curiosity-IMGEP"),
        ("mapelites", "MAP-Elites"),
        ("random", "Random"),
        ("ga", "Genetic Alg."),
        ("bo", "Bayesian Opt."),
    ]

    for idx, (method, name) in enumerate(methods_to_show):
        data = load_results(TARGET, method, seed=0)
        if data is None:
            continue

        discoveries = data["discoveries"]
        # Get 6 diverse molecules: spread across iterations
        n = len(discoveries)
        sample_iters = np.linspace(0, n - 1, 6, dtype=int)
        smiles_used = set()
        mols = []
        legends = []
        for i in sample_iters:
            d = discoveries[i]
            if d["smiles"] in smiles_used:
                continue
            mol = Chem.MolFromSmiles(d["smiles"])
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                legends.append(f"V:{d['vina_score']:.1f} R:{d['active_residues']}")
                smiles_used.add(d["smiles"])
            if len(mols) >= 6:
                break

        if not mols:
            continue

        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs[row, col])

        img = Draw.MolsToGridImage(mols[:6], molsPerRow=3, subImgSize=(250, 200),
                                   legends=legends[:6], useSVG=False)
        ax.imshow(img)
        ax.set_title(name, fontsize=14, fontweight='bold',
                     color=METHOD_COLORS.get(name, "black"))
        ax.axis('off')

    outpath = os.path.join(OUTPUT_DIR, "fig_molecule_comparison.png")
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_molecule_grid()
    fig_fingerprint_heatmap()
    fig_pocket_contact_map()
    fig_pipeline_schematic()
    fig_behavior_tsne()
    fig_molecule_comparison()

    print("\nAll figures generated!")
