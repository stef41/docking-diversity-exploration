#!/bin/bash
# rerun_after_fixes.sh — full-benchmark re-run script for post-audit corrections.
#
# Purpose: regenerate every archived result with the corrected code base so the
# hbond-donor / hbond-acceptor channels reflect PROTISDON (index 10) rather
# than SIDECHAIN (index 6), and so the six other code fixes (M-06, M-09,
# M-10, B-04, B-05) are baked into the archive.
#
# Cost estimate (g107, 8× H100, ≈ 40 min/CPU-run under 20-way parallelism):
#   9 methods × 7 targets × 10 seeds =  630 main-benchmark runs   ≈ 21 h
#   Extra: 5-channel × 3 targets × 5 seeds =  15 runs             ≈ 30 min
#   Extra: binary Curiosity × 7 × 10       =  70 runs             ≈ 2.3 h
#   Extra: cutoff sensitivity 2 × 5 × 3    =  30 runs             ≈ 1 h
#   Total: 745 runs, ≈ 24-30 wall-clock hours.
#
# Usage:
#   ssh g107
#   cd ~/docking-diversity-exploration
#   git pull origin master  # picks up B-02, B-04, B-05, M-06, M-09, M-10 fixes
#   bash rerun_after_fixes.sh --parallel 20 2>&1 | tee rerun.log
#
# Environment (verified working, 2026-07-22):
#   - GNINA v1.0 (HEAD 6381355, 2021-03) — canonical for all runs
#   - CReM v0.2.16 (PyPI)
#   - PLIP v2.2 (or 2.2.2; hbond_features tuple layout unchanged)
#   - RDKit 2024.09.6, Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Move the existing (pre-fix) archive out of the way so results.json won't be
# skipped by run_all_v3.sh.
if [ -d results ] && [ ! -d results_prefix_prefix_audit ]; then
    echo "[$(date +%F\ %T)] Archiving pre-fix results to results_prefix_audit/"
    mv results results_prefix_audit
fi
mkdir -p results

# 1. Main benchmark (9 methods × 7 targets × 10 seeds = 630 runs)
echo "============================================================"
echo "  1. Main benchmark rerun with corrected code"
echo "============================================================"
bash code/run_all_v3.sh --parallel "${PARALLEL:-20}"

# 2. Charged-channel ablation (5-channel Curiosity + Random on 3 ionic targets)
echo "============================================================"
echo "  2. Charged-channel ablation (Experiment 3)"
echo "============================================================"
python3 code/charged_curiosity_experiment.py

# 3. Binary-fingerprint Curiosity-IMGEP (Experiment 2)
echo "============================================================"
echo "  3. Binary-fingerprint Curiosity (Experiment 2)"
echo "============================================================"
python3 code/binary_curiosity_experiment.py

# 4. Cutoff sensitivity + q-exclusion (Experiment 4)
echo "============================================================"
echo "  4. Cutoff sensitivity + q-exclusion (Experiment 4)"
echo "============================================================"
python3 code/cutoff_curiosity_experiment.py

# 5. Regenerate all summary tables + figures
echo "============================================================"
echo "  5. Regenerate all analysis outputs"
echo "============================================================"
python3 code/analyze_v3.py
python3 code/chem_diversity_v3.py
python3 code/reviewer_analyses.py
python3 code/generate_docking_figures.py

echo "============================================================"
echo "  DONE. Regenerated results now live at results/"
echo "  Pre-fix archive preserved at results_prefix_audit/"
echo "  Diff summary tables: compare with results_prefix_audit/*/summary.json"
echo "============================================================"
