#!/usr/bin/env python3
"""
master_analysis.py — Runs all cached analyses for the JCIM revision.

Reads the 560 cached results.json files and outputs everything needed to fill
the placeholders in the manuscript, minus experiments that require new docking.

Outputs (under /tmp/acs-review/revision/analyses/):
  - failure_stats.json           (Group A4)
  - psi_vs_phi.json              (Group B1: unique-profile counts recomputed w/o q)
  - chem_diversity.json          (Group B2: Bemis-Murcko + Morgan + MACCS + AtomPair)
  - coverage.json                (Group B3: extended vs co-crystal-local)
  - salt_bridge_prevalence.json  (Group E1: pocket residue composition proxy)
  - summary.json                 (one-stop table of key numbers)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Paths (local clone)
REPO_ROOT = Path("/tmp/acs-review/docking-diversity-exploration")
RESULTS_DIR = REPO_ROOT / "results"
TARGETS_DIR = REPO_ROOT / "targets"
OUT_DIR = Path("/tmp/acs-review/revision/analyses")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["3V8D", "1ERE", "3EML", "1EVE", "4DFR", "3PJC", "4MNE"]
SEEDS = list(range(10))
# Method labels: source-code name -> display name
METHOD_MAP = {
    "random": "Random",
    "imgep_naive": "IMGEP (naive)",
    "imgep": "IMGEP (adaptive)",
    "curiosity": "Curiosity-IMGEP",
    "bo": "Aff-Div",              # aka UCB Heuristic in figures
    "ga": "Genetic Alg.",
    "mapelites": "MAP-Elites",
    "novelty": "Novelty Search",
    "nsga2":   "NSGA-II",
}
METHODS = list(METHOD_MAP)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


def _augment_nsga2_phi(run):
    """For nsga2 runs (which store ψ), append q(m)=clip(-vina/12,0,1) to reconstruct φ."""
    for d in run.get("discoveries", []):
        b = d.get("behavior")
        if b is None: continue
        v = d.get("vina_score", 0.0)
        q = min(1.0, max(0.0, -v / 12.0))
        # Only augment if length matches ψ (3*n_res, no +1)
        n_res = run.get("n_pocket_residues", 0)
        if n_res and len(b) == n_res * 3:
            d["behavior"] = list(b) + [q]
    return run

def load_run(target: str, method: str, seed: int) -> dict | None:
    p = RESULTS_DIR / target / f"{method}_seed{seed}" / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        run = json.load(f)
    if method == "nsga2":
        run = _augment_nsga2_phi(run)
    return run


def get_all_runs():
    runs = {}  # (target, method, seed) -> run dict
    for T in TARGETS:
        for M in METHODS:
            for S in SEEDS:
                r = load_run(T, M, S)
                if r is not None:
                    runs[(T, M, S)] = r
    print(f"[load] {len(runs)}/{7*8*10} runs loaded")
    return runs


# ═══════════════════════════════════════════════════════════════════════
# GROUP A4 — Failure statistics
# ═══════════════════════════════════════════════════════════════════════

def failure_stats(runs) -> dict:
    """
    Per-method and per-(method,target) failure statistics:
      - runtime_error_rate: fraction of discoveries with 'error' field
      - no_op_mutation_rate: fraction where SMILES == parent SMILES (proxy for
        no-valid-candidate; not perfectly measurable from cached data but the
        rate of *consecutive* duplicate SMILES is a reasonable proxy)
      - zero_interaction_rate: fraction with sum(behavior[:-1]) == 0
    """
    out = {"per_method": {}, "per_method_target": defaultdict(dict)}

    for M in METHODS:
        n_total = 0
        n_error = 0
        n_zero = 0
        n_consec_dup = 0
        for T in TARGETS:
            m_total = 0
            m_error = 0
            m_zero = 0
            m_dup = 0
            for S in SEEDS:
                r = runs.get((T, M, S))
                if not r:
                    continue
                discs = r["discoveries"]
                prev = None
                for d in discs:
                    m_total += 1
                    if "error" in d:
                        m_error += 1
                    b = d.get("behavior", [])
                    if b and sum(b[:-1]) == 0:
                        m_zero += 1
                    smi = d.get("smiles", "")
                    if prev is not None and smi == prev:
                        m_dup += 1
                    prev = smi
            if m_total > 0:
                out["per_method_target"][M][T] = {
                    "n_evals": m_total,
                    "runtime_error_rate": m_error / m_total,
                    "zero_interaction_rate": m_zero / m_total,
                    "consecutive_duplicate_smiles_rate": m_dup / m_total,
                }
            n_total += m_total
            n_error += m_error
            n_zero += m_zero
            n_consec_dup += m_dup
        out["per_method"][M] = {
            "n_evals": n_total,
            "runtime_error_rate": n_error / n_total if n_total else 0,
            "zero_interaction_rate": n_zero / n_total if n_total else 0,
            "consecutive_duplicate_smiles_rate": n_consec_dup / n_total if n_total else 0,
        }
    return out


# ═══════════════════════════════════════════════════════════════════════
# GROUP B1 — ψ vs φ (interaction-only vs augmented) unique-profile counts
# ═══════════════════════════════════════════════════════════════════════

def psi_vs_phi(runs) -> dict:
    """
    For each run, recompute unique-profile count using:
      - φ = full behavior vector (last coord = q(m))  ← original submission
      - ψ = interaction-only vector (drop last coord)
    Both after 2-decimal-place rounding.
    Report ratios and per-method aggregates.
    """
    def unique(behaviors, drop_last=False):
        arr = np.array([b[:-1] if drop_last else b for b in behaviors])
        rounded = np.round(arr, 2)
        return len(set(map(tuple, rounded)))

    per_run = []
    for (T, M, S), r in runs.items():
        behaviors = [d["behavior"] for d in r["discoveries"] if d.get("behavior")]
        if not behaviors:
            continue
        u_phi = unique(behaviors, drop_last=False)
        u_psi = unique(behaviors, drop_last=True)
        per_run.append({
            "target": T, "method": M, "seed": S,
            "unique_phi": u_phi, "unique_psi": u_psi,
            "delta_pct": 100 * (u_psi - u_phi) / u_phi if u_phi else 0.0,
        })

    # Per-method aggregate
    per_method = {}
    for M in METHODS:
        rows = [r for r in per_run if r["method"] == M]
        if not rows: continue
        u_phi = [r["unique_phi"] for r in rows]
        u_psi = [r["unique_psi"] for r in rows]
        per_method[M] = {
            "mean_phi": float(np.mean(u_phi)),
            "std_phi": float(np.std(u_phi, ddof=1)),
            "mean_psi": float(np.mean(u_psi)),
            "std_psi": float(np.std(u_psi, ddof=1)),
            "mean_delta_pct": float(np.mean([r["delta_pct"] for r in rows])),
            "max_delta_pct": float(np.max(np.abs([r["delta_pct"] for r in rows]))),
        }

    # Ranking stability check
    rank_phi = sorted(per_method.keys(), key=lambda M: -per_method[M]["mean_phi"])
    rank_psi = sorted(per_method.keys(), key=lambda M: -per_method[M]["mean_psi"])
    ranking_preserved = rank_phi == rank_psi

    return {
        "per_method": per_method,
        "ranking_phi": [METHOD_MAP[m] for m in rank_phi],
        "ranking_psi": [METHOD_MAP[m] for m in rank_psi],
        "ranking_preserved": ranking_preserved,
        "max_delta_pct_any_run": max((abs(r["delta_pct"]) for r in per_run), default=0),
    }


# ═══════════════════════════════════════════════════════════════════════
# GROUP B2 — Bemis-Murcko + Morgan + MACCS + AtomPair diversity
# ═══════════════════════════════════════════════════════════════════════

def chem_diversity(runs) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

    def mol_from(smi: str):
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    def scaffold_smiles(mol):
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:
            return None

    def pairwise_tanimoto(fps):
        if len(fps) < 2:
            return 0.0
        sims = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        return 1.0 - float(np.mean(sims))

    per_run = []
    for k, (T, M, S) in enumerate(sorted(runs.keys())):
        r = runs[(T, M, S)]
        smiles_list = list({d["smiles"] for d in r["discoveries"] if d.get("smiles")})
        mols = [m for m in (mol_from(s) for s in smiles_list) if m is not None]
        if not mols:
            continue
        scaffolds = {scaffold_smiles(m) for m in mols}
        scaffolds.discard(None)

        morgan = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
        maccs  = [MACCSkeys.GenMACCSKeys(m) for m in mols]
        atompair = [_atompair_gen.GetFingerprint(m) for m in mols]

        per_run.append({
            "target": T, "method": M, "seed": S,
            "n_unique_smiles": len(mols),
            "n_scaffolds": len(scaffolds),
            "morgan_div": pairwise_tanimoto(morgan),
            "maccs_div": pairwise_tanimoto(maccs),
            "atompair_div": pairwise_tanimoto(atompair),
        })
        if (k + 1) % 40 == 0:
            print(f"  [chem_div] processed {k+1}/{len(runs)} runs")

    # Per-method aggregate + Cohen's d vs Random
    per_method = {}
    random_rows = [r for r in per_run if r["method"] == "random"]
    for M in METHODS:
        rows = [r for r in per_run if r["method"] == M]
        if not rows: continue
        entry = {
            "n_runs": len(rows),
            "mean_scaffolds": float(np.mean([r["n_scaffolds"] for r in rows])),
            "std_scaffolds":  float(np.std ([r["n_scaffolds"] for r in rows], ddof=1)),
            "mean_morgan":    float(np.mean([r["morgan_div"] for r in rows])),
            "std_morgan":     float(np.std ([r["morgan_div"] for r in rows], ddof=1)),
            "mean_maccs":     float(np.mean([r["maccs_div"]  for r in rows])),
            "std_maccs":      float(np.std ([r["maccs_div"]  for r in rows], ddof=1)),
            "mean_atompair":  float(np.mean([r["atompair_div"] for r in rows])),
            "std_atompair":   float(np.std ([r["atompair_div"] for r in rows], ddof=1)),
        }
        if M != "random" and random_rows:
            for k, key in [("scaffolds","n_scaffolds"), ("morgan","morgan_div"),
                           ("maccs","maccs_div"), ("atompair","atompair_div")]:
                a = np.array([r[key] for r in rows])
                b = np.array([r[key] for r in random_rows])
                pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
                d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
                try:
                    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                except Exception:
                    p = 1.0
                entry[f"d_{k}_vs_random"] = float(d)
                entry[f"p_{k}_vs_random"] = float(p)
        per_method[M] = entry
    return {"per_method": per_method, "per_run": per_run}


# ═══════════════════════════════════════════════════════════════════════
# GROUP B3 — Extended vs co-crystal-local pocket coverage
# ═══════════════════════════════════════════════════════════════════════

def compute_coverage(runs) -> dict:
    """
    Two denominators:
      (i)  Extended pocket = residues with ≥1 atom inside the docking box
           (this is what the paper's runs already used; already in results.json)
      (ii) Co-crystal-local pocket = residues with ≥1 heavy atom within 4.0 Å
           of any co-crystallized-ligand heavy atom. We approximate the
           co-crystal ligand as HETATM records in the ORIGINAL PDB (the paper
           extracted chain A, but co-crystal ligand was retained in the
           preparation step for box definition — we may not have it in the
           prepared PDB, in which case we fall back to the box center as a
           surrogate).
    Since the prepared PDBs (targets/1ERE_A_protein.pdb etc.) may not contain
    the co-crystal ligand, we FETCH the original PDB from RCSB for each target
    and extract the co-crystallized ligand heavy atoms.
    """
    import urllib.request
    from Bio.PDB import PDBParser, MMCIFParser
    parser = PDBParser(QUIET=True)

    def fetch_original_pdb(pdb_id: str) -> Path:
        cache = OUT_DIR.parent / "pdb_cache"
        cache.mkdir(exist_ok=True)
        f = cache / f"{pdb_id}.pdb"
        if not f.exists():
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                urllib.request.urlretrieve(url, f)
            except Exception as e:
                print(f"  [pdb-fetch] {pdb_id} failed: {e}")
                return None
        return f

    def read_box(target: str):
        # "size_x = 22.5\nsize_y = 22.5\n..."
        with open(TARGETS_DIR / f"{target}_A_box.txt") as f:
            lines = f.readlines()
        vals = {}
        for l in lines:
            k, _, v = l.partition("=")
            vals[k.strip()] = float(v.strip())
        return vals

    def get_cocrystal_ligand_atoms(pdb_path: Path):
        """Return list of (x, y, z) for co-crystal ligand HETATM heavy atoms."""
        # Skip standard cofactors that aren't the co-crystal drug
        SKIP = {"HOH", "WAT", "SO4", "PO4", "NA", "K", "MG", "CA", "CL",
                "ZN", "FE", "MN", "GOL", "EDO", "PEG", "PGE", "DMS",
                "ACT", "IMD", "NAG", "MAN", "BMA", "FUC", "GAL", "BGC",
                "SEP", "TPO", "PTR"}
        atoms = []
        try:
            structure = parser.get_structure("x", str(pdb_path))
            # Find the largest HETATM residue that isn't a skip target
            hetatm_residues = defaultdict(list)
            for model in structure:
                for chain in model:
                    for res in chain:
                        if res.id[0] != " " and res.get_resname().strip() not in SKIP:
                            key = (chain.id, res.id[0], res.id[1], res.get_resname())
                            for atom in res:
                                if atom.element != "H":
                                    hetatm_residues[key].append(tuple(atom.coord))
                    break  # only first chain
                break  # only first model
            if not hetatm_residues:
                return []
            # Pick the LARGEST hetatm residue (the drug, not the cofactor)
            best_key = max(hetatm_residues, key=lambda k: len(hetatm_residues[k]))
            return hetatm_residues[best_key]
        except Exception as e:
            print(f"  [het-parse] {pdb_path.name}: {e}")
            return []

    def get_pocket_residues_local(pdb_path: Path, box_center, box_size):
        cx, cy, cz = box_center
        sx, sy, sz = box_size
        residues_in_box = set()
        structure = parser.get_structure("x", str(pdb_path))
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ":  # skip HETATM
                        continue
                    for atom in res:
                        x, y, z = atom.coord
                        if (abs(x - cx) <= sx/2 and abs(y - cy) <= sy/2 and abs(z - cz) <= sz/2):
                            residues_in_box.add(res.id[1])
                            break
            break
        return residues_in_box

    def get_pocket_residues_local_cocrystal(prepared_pdb: Path, cocrystal_atoms, cutoff=4.0):
        """Residues within `cutoff` Å of any co-crystal atom."""
        near = set()
        if not cocrystal_atoms:
            return near
        cx = np.array(cocrystal_atoms)  # (M, 3)
        structure = parser.get_structure("x", str(prepared_pdb))
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ":
                        continue
                    for atom in res:
                        if atom.element == "H":
                            continue
                        d = np.linalg.norm(cx - np.array(atom.coord), axis=1).min()
                        if d <= cutoff:
                            near.add(res.id[1])
                            break
            break
        return near

    out = {"per_target": {}, "per_method_target": defaultdict(dict)}
    for T in TARGETS:
        box = read_box(T)
        prepared = TARGETS_DIR / f"{T}_A_protein.pdb"
        original = fetch_original_pdb(T)
        if original is None:
            continue

        extended = get_pocket_residues_local(
            prepared,
            (box["center_x"], box["center_y"], box["center_z"]),
            (box["size_x"], box["size_y"], box["size_z"]))
        cocrystal = get_cocrystal_ligand_atoms(original)
        local = get_pocket_residues_local_cocrystal(prepared, cocrystal, 4.0)
        out["per_target"][T] = {
            "extended_pocket_size": len(extended),
            "extended_pocket_residues": sorted(extended),
            "cocrystal_ligand_atoms": len(cocrystal),
            "local_pocket_size": len(local),
            "local_pocket_residues": sorted(local),
        }

        # For each method+seed on this target, compute coverage using both denominators
        for M in METHODS:
            covs_ext = []
            covs_loc = []
            for S in SEEDS:
                r = runs.get((T, M, S))
                if not r: continue
                # extended coverage = union of active_residues across discoveries
                # (using n_res in the results.json which is len(pocket_residues))
                # We recompute against the residue indices we just computed
                pocket = r["pocket_residues"]  # list of PDB residue numbers
                # find which pocket residues had ANY interaction
                # behavior is (3*N + 1); a residue r (idx i) is contacted iff any of behavior[i*3:i*3+3] > 0
                contacted = set()
                for d in r["discoveries"]:
                    b = d.get("behavior", [])
                    for i, resnum in enumerate(pocket):
                        if any(b[i*3 + t] > 0 for t in range(3) if i*3 + t < len(b) - 1):
                            contacted.add(resnum)
                if extended:
                    covs_ext.append(len(contacted & extended) / len(extended))
                if local:
                    covs_loc.append(len(contacted & local) / len(local))
            if covs_ext:
                out["per_method_target"][M][T] = {
                    "extended_coverage_mean": float(np.mean(covs_ext)),
                    "extended_coverage_std":  float(np.std(covs_ext, ddof=1)),
                    "local_coverage_mean":    float(np.mean(covs_loc)) if covs_loc else None,
                    "local_coverage_std":     float(np.std(covs_loc, ddof=1)) if covs_loc else None,
                }
        print(f"  [coverage] {T}: extended={len(extended)}, local={len(local)}, cocrystal_atoms={len(cocrystal)}")

    # Aggregate per-method (average over targets)
    per_method_avg = {}
    for M in METHODS:
        ext_covs, loc_covs = [], []
        for T in TARGETS:
            entry = out["per_method_target"].get(M, {}).get(T, {})
            if "extended_coverage_mean" in entry:
                ext_covs.append(entry["extended_coverage_mean"])
            if entry.get("local_coverage_mean") is not None:
                loc_covs.append(entry["local_coverage_mean"])
        per_method_avg[M] = {
            "mean_extended_coverage": float(np.mean(ext_covs)) if ext_covs else None,
            "mean_local_coverage":    float(np.mean(loc_covs)) if loc_covs else None,
        }
    out["per_method_avg"] = per_method_avg
    return dict(out)


# ═══════════════════════════════════════════════════════════════════════
# GROUP E1 — Salt-bridge target prioritization (from PDB residue composition
#           within the co-crystal-local pocket)
# ═══════════════════════════════════════════════════════════════════════

def salt_bridge_prevalence(coverage_result) -> dict:
    """
    Proxy for salt-bridge potential: fraction of co-crystal-local pocket
    residues that are Asp/Glu (acidic) or Lys/Arg/His (basic).
    Also count backbone charged residues in the extended box.
    Rank targets by ionic-residue fraction to pick the 3 with highest prevalence.
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)

    ACIDIC = {"ASP", "GLU"}
    BASIC = {"LYS", "ARG", "HIS"}

    out = {}
    for T in TARGETS:
        prepared = TARGETS_DIR / f"{T}_A_protein.pdb"
        local_residues = set(coverage_result["per_target"].get(T, {}).get("local_pocket_residues", []))
        # Map resnum -> resname
        resmap = {}
        structure = parser.get_structure("x", str(prepared))
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ":
                        continue
                    resmap[res.id[1]] = res.get_resname().strip()
            break

        acidic_local = sum(1 for r in local_residues if resmap.get(r) in ACIDIC)
        basic_local = sum(1 for r in local_residues if resmap.get(r) in BASIC)
        total_local = len(local_residues)
        ionic_frac = (acidic_local + basic_local) / total_local if total_local else 0.0
        out[T] = {
            "local_pocket_size": total_local,
            "acidic_local": acidic_local,
            "basic_local": basic_local,
            "ionic_fraction_local": ionic_frac,
            "acidic_residues": sorted([r for r in local_residues if resmap.get(r) in ACIDIC]),
            "basic_residues":  sorted([r for r in local_residues if resmap.get(r) in BASIC]),
        }
    ranked = sorted(TARGETS, key=lambda t: -out[t]["ionic_fraction_local"])
    return {"per_target": out, "ranked_by_ionic_fraction": ranked, "top3": ranked[:3]}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Loading 560 cached runs...")
    runs = get_all_runs()

    print("\n" + "=" * 70)
    print("A4 — Failure statistics")
    fs = failure_stats(runs)
    with open(OUT_DIR / "failure_stats.json", "w") as f:
        json.dump(fs, f, indent=2, default=str)
    for M, entry in fs["per_method"].items():
        print(f"  {METHOD_MAP[M]:<18} n={entry['n_evals']:5d}  "
              f"runtime_err={entry['runtime_error_rate']*100:.2f}%  "
              f"zero_int={entry['zero_interaction_rate']*100:.2f}%  "
              f"consec_dup={entry['consecutive_duplicate_smiles_rate']*100:.2f}%")

    print("\n" + "=" * 70)
    print("B1 — ψ vs φ unique-profile recomputation")
    pp = psi_vs_phi(runs)
    with open(OUT_DIR / "psi_vs_phi.json", "w") as f:
        json.dump(pp, f, indent=2)
    for M, entry in pp["per_method"].items():
        print(f"  {METHOD_MAP[M]:<18} φ={entry['mean_phi']:.1f}±{entry['std_phi']:.1f}  "
              f"ψ={entry['mean_psi']:.1f}±{entry['std_psi']:.1f}  "
              f"Δmean={entry['mean_delta_pct']:+.2f}%  |Δ|max={entry['max_delta_pct']:.2f}%")
    print(f"  ranking preserved: {pp['ranking_preserved']}")
    print(f"  max |Δ| any run: {pp['max_delta_pct_any_run']:.2f}%")

    print("\n" + "=" * 70)
    print("B2 — Chemical diversity (Morgan/MACCS/AtomPair + Bemis-Murcko)")
    cd = chem_diversity(runs)
    with open(OUT_DIR / "chem_diversity.json", "w") as f:
        json.dump(cd, f, indent=2)
    for M, entry in cd["per_method"].items():
        s = f"  {METHOD_MAP[M]:<18} scaffolds={entry['mean_scaffolds']:.1f}±{entry['std_scaffolds']:.1f}  "
        s += f"morgan={entry['mean_morgan']:.3f}  maccs={entry['mean_maccs']:.3f}  atompair={entry['mean_atompair']:.3f}"
        if "d_scaffolds_vs_random" in entry:
            s += f"  d_scaff={entry['d_scaffolds_vs_random']:+.2f}"
        print(s)

    print("\n" + "=" * 70)
    print("B3 — Extended vs co-crystal-local coverage")
    cov = compute_coverage(runs)
    with open(OUT_DIR / "coverage.json", "w") as f:
        json.dump(cov, f, indent=2, default=str)
    for T, entry in cov["per_target"].items():
        print(f"  {T}: extended={entry['extended_pocket_size']}, local={entry['local_pocket_size']}, cocrystal_atoms={entry['cocrystal_ligand_atoms']}")
    print("  Per-method mean coverage across targets:")
    for M, entry in cov["per_method_avg"].items():
        ext = entry["mean_extended_coverage"]
        loc = entry["mean_local_coverage"]
        print(f"    {METHOD_MAP[M]:<18} extended={ext*100:.1f}%  local={loc*100:.1f}%" if ext and loc else f"    {METHOD_MAP[M]:<18} (incomplete)")

    print("\n" + "=" * 70)
    print("E1 — Salt-bridge target prioritization")
    sb = salt_bridge_prevalence(cov)
    with open(OUT_DIR / "salt_bridge_prevalence.json", "w") as f:
        json.dump(sb, f, indent=2)
    for T in sb["ranked_by_ionic_fraction"]:
        e = sb["per_target"][T]
        print(f"  {T}: ionic_frac={e['ionic_fraction_local']*100:.1f}%  "
              f"(acidic={e['acidic_local']}, basic={e['basic_local']}, total_local={e['local_pocket_size']})")
    print(f"  → TOP 3 for charged-ablation: {sb['top3']}")

    # Summary
    print("\n" + "=" * 70)
    print("Writing summary.json")
    summary = {
        "failure_stats": fs["per_method"],
        "psi_vs_phi": pp,
        "chem_diversity": {M: cd["per_method"][M] for M in cd["per_method"]},
        "coverage_per_target": cov["per_target"],
        "coverage_per_method_avg": cov["per_method_avg"],
        "salt_bridge_top3": sb["top3"],
        "salt_bridge_per_target": sb["per_target"],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
