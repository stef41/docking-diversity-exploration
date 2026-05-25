# Diversity-Seeking Exploration for Docking-Predicted Interaction Profiles

Code, configurations, prepared targets, per-run results, and figure outputs
for the manuscript *"Diversity-Seeking Exploration Discovers Broader
Docking-Predicted Interaction Profiles Across Protein Families"* (Bugaud, 2026),
submitted to the Journal of Chemical Information and Modeling.

This repository contains the **paper-specific** artifacts. The general-purpose
exploration framework used to run the experiments is the separate `adtool`
project:

- `adtool` framework: https://github.com/flowersteam/adtool

## Layout

```
code/         Experiment runner, analysis, chemical-diversity, CNN rescore,
              figure-generation, and config-generation scripts.
configs/      JSON configurations for every (method, target, seed) run.
targets/      Prepared receptor PDB files and binding-box definitions for the
              seven protein targets (3V8D, 1ERE, 3EML, 1EVE, 4DFR, 3PJC, 4MNE).
results/      Per-run `results.json` files: one per (target, method, seed),
              560 runs total (7 targets x 8 methods x 10 seeds).
figures/      Final PDF/PNG figures used in the manuscript plus `table1.tex`.
paper/        Manuscript LaTeX source (`jcim_submission.tex`, `jcim.bib`).
```

## Reproducing the figures

The processed results in `results/` are sufficient to regenerate every figure
and table in the paper:

```bash
python code/analyze_v3.py --results-dir results --out figures/
python code/chem_diversity_v3.py --results-dir results --out figures/
python code/generate_docking_figures.py
```

## Re-running the docking experiments

Re-running the full 280,000 docking evaluations requires the `adtool`
framework, GNINA (https://github.com/gnina/gnina), and CReM
(https://github.com/DrrDom/crem). Wall-clock cost is approximately 800 GPU
hours on an A100.

```bash
git clone https://github.com/flowersteam/adtool
pip install -e adtool
bash code/run_all_v3.sh
```

## Software versions

- GNINA v1.0 (Vina scoring, `--cnn_scoring none`, exhaustiveness 16)
- CReM v0.2.16 (fragment database of 5,955 entries)
- RDKit (Morgan, MACCS, Atom Pair fingerprints; Bemis-Murcko scaffolds)
- PLIP v2.2 (protein-ligand interaction profiling)

## License

Released under the same license as the parent `adtool` project (MIT).

## Citation

```
@article{bugaud2026diversity,
  title  = {Diversity-Seeking Exploration Discovers Broader Docking-Predicted
            Interaction Profiles Across Protein Families},
  author = {Bugaud, Zacharie},
  journal= {Journal of Chemical Information and Modeling},
  year   = {2026}
}
```
