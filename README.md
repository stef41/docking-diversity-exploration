# Diversity-Seeking Exploration for Molecular Docking

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docking: GNINA](https://img.shields.io/badge/docking-GNINA%20v1.0%20%2F%20v1.3.3-orange)](https://github.com/gnina/gnina)
[![Framework: adtool](https://img.shields.io/badge/framework-adtool-brightgreen)](https://github.com/flowersteam/adtool)
[![Runs: 760](https://img.shields.io/badge/runs-760-lightgrey)]()
[![Dockings: 380k](https://img.shields.io/badge/docking%20evaluations-380%2C000-lightgrey)]()

> **TL;DR** &nbsp; Treat docking as a *diversity-seeking* problem instead of an
> affinity-maximization one. **MAP-Elites** and **Curiosity-IMGEP** discover
> **+13.6%** more unique interaction profiles and **+21%** more unique
> Bemis&ndash;Murcko scaffolds than random search across **7 targets / 5 protein
> families**, at no cost to binding quality. A reviewer-requested **NSGA-II**
> Pareto-optimization baseline matches the Genetic Algorithm for the strongest
> predicted binding affinity (&minus;11.18 &plusmn; 1.08 kcal/mol) and produces
> the highest Bemis&ndash;Murcko scaffold count of any method
> (24.6 &plusmn; 12.3), while underperforming the QD methods on
> interaction-profile counts.

<p align="center">
  <img src="figures/fig_docking_3d.png" width="78%" alt="Three structurally distinct ligands discovered by Curiosity-IMGEP, all docking into the JAK3 ATP pocket via distinct residue contact patterns.">
</p>

<p align="center">
  <em>Three structurally distinct ligands discovered by Curiosity-IMGEP, all
  docking into the JAK3 ATP pocket (PDB&nbsp;3PJC) via distinct residue contact
  patterns &mdash; the kind of binding-mode diversity this work targets.</em>
</p>

---

## Why this exists

Virtual screening rewards a single number &mdash; binding affinity &mdash; and
converges on narrow chemical series exploring a single binding mode per pocket.
Yet a single pocket typically accommodates **multiple distinct binding modes**
with different hydrogen-bond networks, hydrophobic packing, and selectivity
implications. We show that off-the-shelf **quality-diversity** and
**curiosity-driven goal-exploration** algorithms, paired with a
**residue-level multi-interaction fingerprint**, systematically uncover this
diversity *in silico* under a fixed docking budget.

This repository contains everything needed to **reproduce every figure and
table in the paper from cached results**, and everything needed to
**re-run the docking campaign from scratch**.

## Headline results

| Method                 | Unique profiles | vs. Random | Unique scaffolds | Best Vina (kcal/mol) |
|------------------------|:---------------:|:----------:|:----------------:|:--------------------:|
| Random                 | 308 &plusmn; 29 | &mdash;    | 16.3 &plusmn; 3.2 | &minus;10.10 &plusmn; 0.89 |
| IMGEP (naive)          | 231 &plusmn; 64 | &minus;25% | 21.9 &plusmn; 15.7 | &minus;10.64 &plusmn; 1.10 |
| IMGEP (adaptive)       | 328 &plusmn; 30 | +6.5%      | 17.9 &plusmn; 4.7 | &minus;10.27 &plusmn; 0.89 |
| **Curiosity-IMGEP**    | 331 &plusmn; 29 | **+7.5%**  | **19.8 &plusmn; 5.8 (+21%)** | &minus;10.40 &plusmn; 0.99 |
| Aff&ndash;Div          | 105 &plusmn; 48 | &minus;66% | 11.5 &plusmn; 7.2 | &minus;9.63 &plusmn; 0.78  |
| Genetic Algorithm      | 219 &plusmn; 60 | &minus;29% | 19.3 &plusmn; 10.5 | **&minus;11.22 &plusmn; 1.20** |
| **MAP-Elites**         | **350 &plusmn; 23** | **+13.6%** | 21.4 &plusmn; 8.1 | &minus;10.36 &plusmn; 0.92 |
| Novelty Search         | 291 &plusmn; 37 | &minus;5.5% | 18.6 &plusmn; 7.9 | &minus;10.52 &plusmn; 0.99 |
| **NSGA-II**            | 282 &plusmn; 30 | &minus;8%   | **24.6 &plusmn; 12.3 (+51%)** | **&minus;11.18 &plusmn; 1.08** |

*Pooled across 7 targets &times; 10 seeds (n = 70). Diversity-seeking methods
(MAP-Elites, Curiosity-IMGEP) retain binding quality while uncovering interaction
modes that an affinity-focused GA misses by design. NSGA-II concentrates the
population near the affinity&ndash;novelty frontier: it ties the Genetic Algorithm
for the strongest predicted binding affinity and produces the highest
Bemis&ndash;Murcko scaffold count, but underperforms the quality-diversity methods
on interaction-profile counts.*

<p align="center">
  <img src="figures/fig3_convergence.png" width="62%" alt="Per-target pairwise behavioral distance over 500 iterations">
</p>

## Quickstart &mdash; reproduce the figures (no docking needed)

The 760 per-run `results.json` files in `results/` are sufficient to
regenerate every figure and table in the paper.

```bash
git clone https://github.com/stef41/docking-diversity-exploration
cd docking-diversity-exploration
pip install rdkit numpy pandas matplotlib seaborn scipy

python code/analyze_v3.py          --results-dir results --out figures/
python code/chem_diversity_v3.py   --results-dir results --out figures/
python code/generate_docking_figures.py
```

## Re-run the docking campaign from scratch

> Wall-clock cost: ~800 GPU-hours on a single A100 for the 630-run main
> benchmark; ~200 GPU-hours for the 130 reviewer-requested ablation runs.

```bash
# 1. Install the exploration framework
git clone https://github.com/flowersteam/adtool
pip install -e adtool

# 2. Install GNINA (docking) and CReM (fragment-based mutation)
#    See https://github.com/gnina/gnina and https://github.com/DrrDom/crem

# 3. Launch main benchmark: 630 runs (7 targets x 9 methods x 10 seeds)
bash code/run_all_v3.sh   # covers the original 6 methods; edit the METHODS list
                          # to add mapelites, novelty, and nsga2 as needed
```

The repository additionally contains 130 reviewer-requested ablation runs
(binary-fingerprint Curiosity-IMGEP, charged-channel 5-descriptor, cutoff
sensitivity, and inclusion of docking quality in the search vector); their
result folders follow the naming pattern `curiosity_binary_seed*`,
`curiosity_5ch_seed*`, `random_5ch_seed*`, `curiosity_dmax35_seed*`,
`curiosity_dmax45_seed*`, and `curiosity_dmax40_noq_seed*`.

## Repository layout

```
code/        Experiment runner (run_experiment_v3.py), analysis pipelines,
             chemical-diversity scripts, CNN-rescore validation, and
             figure-generation code.
configs/     JSON configurations for every (method, target, seed) run.
targets/     Prepared receptor PDB files + binding-box definitions for
             3V8D, 1ERE, 3EML, 1EVE, 4DFR, 3PJC, 4MNE (5 protein families).
results/     760 per-run results.json files containing SMILES, GNINA/Vina
             scores, residue-level interaction fingerprints, and full
             per-iteration trajectories. Breakdown: 630 nine-method main
             benchmark + 130 reviewer-requested ablation runs.
figures/     Final PDF/PNG figures used in the manuscript + table1.tex.
paper/       Manuscript LaTeX source (jcim_submission.tex, jcim.bib).
```

## The seven targets

| PDB  | Protein        | Family           | Indication            | Pocket residues |
|------|----------------|------------------|-----------------------|:---------------:|
| 3V8D | CYP7A1         | Oxidoreductase   | Cholesterol metab.    | 95 |
| 1ERE | ER-&alpha;     | Nuclear receptor | Breast cancer         | 71 |
| 3EML | A<sub>2A</sub>R| GPCR             | Neurological          | 65 |
| 1EVE | AChE           | Hydrolase        | Alzheimer's           | 76 |
| 4DFR | DHFR           | Oxidoreductase   | Infection             | 55 |
| 3PJC | JAK3           | Kinase           | Immune disorders      | 60 |
| 4MNE | BRAF           | Kinase           | Melanoma              | 68 |

## The reachability ratio

A central practical contribution: goal-directed exploration only outperforms
random search when nearest-neighbor retrieval is **discriminative**. We define

$$R \;=\; \frac{\mathbb{E}_{g \sim P_\text{goal}}\,\min_{b \in \mathcal{B}} \lVert g - b \rVert_2}{\mathbb{E}_{b, b' \sim \mathcal{B}}\,\lVert b - b' \rVert_2}$$

Empirically, **R &raquo; 10** &rarr; goal-directed retrieval degenerates
toward random parent selection. Our residue-level fingerprint reduces *R* from
**~128** (atom-level baseline) to **~8**, which is when IMGEP variants begin
to beat random search.

## Software versions

| Tool   | Version  | Purpose                                                          |
|--------|----------|------------------------------------------------------------------|
| GNINA  | v1.0 (+ v1.3.3 on a subset of ablation runs, verified compatible) | Molecular docking (Vina scoring, `--cnn_scoring none`, exh. 16) |
| CReM   | v0.2.16  | Fragment-based mutation (5,955-entry fragment DB; SHA-256 `40262010822204fdb7576ba84208512e6203992cfb1fcbe157648fde6e6ad664`) |
| RDKit  | latest   | Morgan / MACCS / Atom Pair fingerprints, Bemis&ndash;Murcko scaffolds |
| PLIP   | v2.2.2   | Protein&ndash;ligand interaction profiling                       |
| adtool | main     | Exploration framework (Random, IMGEP naive/adaptive, Curiosity-IMGEP, Aff&ndash;Div, GA, MAP-Elites, Novelty Search, NSGA-II) |

## License

[MIT](LICENSE).

## Citation

```bibtex
@article{bugaud2026diversity,
  title   = {Diversity-Seeking Docking-Guided Molecular Design Discovers
             Broader Predicted Interaction Profiles Across Protein Families},
  author  = {Bugaud, Z.},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026}
}
```
