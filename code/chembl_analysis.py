#!/usr/bin/env python3
"""
chembl_analysis.py — Group G: target-specific bioactivity-neighborhood analysis.

For each of the 7 PDB targets:
  1. Map PDB → UniProt accession via RCSB
  2. Query ChEMBL for the target
  3. Fetch binding-type activities with pChEMBL >= 6, direct binding, target_confidence>=8
  4. Standardize SMILES, compute Morgan fingerprints
  5. Compare generated compounds (from cached 560 runs) to actives:
     - exact InChIKey rediscovery
     - max Morgan similarity per compound
     - fraction with similarity >= {0.4, 0.5, 0.6}
     - distinct active Bemis-Murcko scaffold neighborhoods reached
  6. Aggregate per method across all 7 targets

Output: /tmp/acs-review/revision/analyses/chembl.json + summary in stdout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path("/tmp/acs-review/docking-diversity-exploration")
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = Path("/tmp/acs-review/revision/analyses")
CACHE_DIR = Path("/tmp/acs-review/revision/chembl_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["3V8D", "1ERE", "3EML", "1EVE", "4DFR", "3PJC", "4MNE"]
METHOD_MAP = {
    "random": "Random", "imgep_naive": "IMGEP (naive)", "imgep": "IMGEP (adaptive)",
    "curiosity": "Curiosity-IMGEP", "bo": "Aff-Div", "ga": "Genetic Alg.",
    "mapelites": "MAP-Elites", "novelty": "Novelty Search",
}
METHODS = list(METHOD_MAP)
SEEDS = list(range(10))

# Manual PDB → UniProt (avoids extra API calls; verified from RCSB deposition data)
PDB_TO_UNIPROT = {
    "3V8D": "P22680",  # CYP7A1 (Human cholesterol 7-alpha-monooxygenase)
    "1ERE": "P03372",  # Estrogen receptor alpha (Human)
    "3EML": "P29274",  # Adenosine A2A receptor (Human)
    "1EVE": "P22303",  # AChE (Human)
    "4DFR": "P00378",  # DHFR (E. coli — DIFFERENT from human!)
    "3PJC": "P52333",  # JAK3 (Human)
    "4MNE": "P15056",  # BRAF (Human)
}


# ═══════════════════════════════════════════════════════════════════════
# ChEMBL query via REST API (chembl_webresource_client has been flaky)
# ═══════════════════════════════════════════════════════════════════════

def chembl_get(endpoint: str, params: dict, cache_key: str = None) -> dict:
    """Cached ChEMBL REST API GET."""
    if cache_key:
        p = CACHE_DIR / f"{cache_key}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    url = f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}"
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if cache_key:
        with open(p, "w") as f:
            json.dump(data, f)
    return data


def uniprot_to_chembl_target(uniprot: str) -> str | None:
    """Get ChEMBL target_chembl_id from UniProt accession."""
    d = chembl_get("target.json",
                    {"target_components__accession": uniprot, "format": "json", "limit": 5},
                    cache_key=f"target_{uniprot}")
    targets = d.get("targets", [])
    if not targets:
        return None
    # Prefer SINGLE PROTEIN
    for t in targets:
        if t.get("target_type") == "SINGLE PROTEIN":
            return t.get("target_chembl_id")
    return targets[0].get("target_chembl_id")


def fetch_activities(chembl_target: str, uniprot: str) -> list[dict]:
    """
    Fetch binding-type activities with pChEMBL >= 6.
    Uses paginated fetch (max 1000/page).
    """
    activities = []
    offset = 0
    limit = 1000
    while True:
        cache_key = f"activities_{chembl_target}_o{offset}"
        d = chembl_get("activity.json", {
            "target_chembl_id": chembl_target,
            "assay_type": "B",
            "standard_relation": "=",
            "pchembl_value__gte": 6,
            "limit": limit,
            "offset": offset,
            "format": "json",
        }, cache_key=cache_key)
        batch = d.get("activities", [])
        activities.extend(batch)
        total = d.get("page_meta", {}).get("total_count", 0)
        if offset + limit >= total or not batch:
            break
        offset += limit
        time.sleep(0.5)
    print(f"  [chembl] {chembl_target} ({uniprot}): {len(activities)} activities")
    return activities


def activities_to_smiles_set(activities: list[dict]) -> tuple[set[str], list[dict]]:
    """
    Extract unique canonical SMILES + keep metadata.
    Filter: needs canonical_smiles.
    """
    canonical = {}
    for a in activities:
        smi = a.get("canonical_smiles")
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can not in canonical:
                canonical[can] = {
                    "chembl_id": a.get("molecule_chembl_id"),
                    "pchembl": a.get("pchembl_value"),
                }
        except Exception:
            continue
    return set(canonical), [{"smiles": s, **m} for s, m in canonical.items()]


# ═══════════════════════════════════════════════════════════════════════
# Similarity + scaffold utilities
# ═══════════════════════════════════════════════════════════════════════

def get_morgan_fp(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def get_inchi_key(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol) if mol else None
    except Exception:
        return None


def get_scaffold(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Load generated compounds
# ═══════════════════════════════════════════════════════════════════════

def load_generated_by_target_and_method() -> dict:
    """
    Returns {target: {method: set_of_unique_smiles}}
    """
    out = defaultdict(lambda: defaultdict(set))
    for T in TARGETS:
        for M in METHODS:
            for S in SEEDS:
                p = RESULTS_DIR / T / f"{M}_seed{S}" / "results.json"
                if not p.exists():
                    continue
                with open(p) as f:
                    d = json.load(f)
                for disc in d["discoveries"]:
                    smi = disc.get("smiles")
                    if smi:
                        out[T][M].add(smi)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_target(target: str, generated_by_method: dict, actives_smiles: list[dict]) -> dict:
    """For one target, compare each method's generated compounds to ChEMBL actives."""
    active_smiles = [a["smiles"] for a in actives_smiles]
    active_inchi = {get_inchi_key(s) for s in active_smiles} - {None}
    active_scaffolds = {get_scaffold(s) for s in active_smiles} - {None}
    active_fps = [(a["smiles"], get_morgan_fp(a["smiles"])) for a in actives_smiles]
    active_fps = [(s, fp) for s, fp in active_fps if fp is not None]
    active_fp_list = [fp for _, fp in active_fps]
    print(f"  [{target}] {len(active_smiles)} active SMILES, {len(active_inchi)} unique InChI, "
          f"{len(active_scaffolds)} unique scaffolds, {len(active_fp_list)} valid fingerprints")

    out = {}
    for M, gen_smiles in generated_by_method.items():
        gen_mols = [(s, get_morgan_fp(s)) for s in gen_smiles]
        gen_mols = [(s, fp) for s, fp in gen_mols if fp is not None]
        gen_inchi = {get_inchi_key(s) for s, _ in gen_mols} - {None}
        gen_scaffolds_reached = set()
        max_sims = []
        for gen_smi, gen_fp in gen_mols:
            if not active_fp_list:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(gen_fp, active_fp_list)
            max_s = max(sims) if sims else 0.0
            max_sims.append(max_s)
            # Find scaffold of best-matching active
            if max_s >= 0.4:
                best_idx = int(np.argmax(sims))
                best_active_smi = active_fps[best_idx][0]
                sc = get_scaffold(best_active_smi)
                if sc:
                    gen_scaffolds_reached.add(sc)
        exact_rediscovery = len(gen_inchi & active_inchi)
        out[M] = {
            "n_generated_unique_smiles": len(gen_smiles),
            "n_valid_mols": len(gen_mols),
            "exact_inchi_rediscovery": exact_rediscovery,
            "mean_max_similarity": float(np.mean(max_sims)) if max_sims else 0.0,
            "median_max_similarity": float(np.median(max_sims)) if max_sims else 0.0,
            "top10_max_similarity": float(np.mean(sorted(max_sims, reverse=True)[:10])) if max_sims else 0.0,
            "frac_sim_ge_0.4": float(np.mean([s >= 0.4 for s in max_sims])) if max_sims else 0.0,
            "frac_sim_ge_0.5": float(np.mean([s >= 0.5 for s in max_sims])) if max_sims else 0.0,
            "frac_sim_ge_0.6": float(np.mean([s >= 0.6 for s in max_sims])) if max_sims else 0.0,
            "active_scaffold_neighborhoods_reached": len(gen_scaffolds_reached),
        }
    return out


def main():
    print("=" * 70)
    print("ChEMBL bioactivity-neighborhood analysis")
    print("=" * 70)

    generated = load_generated_by_target_and_method()
    print(f"Loaded generated compounds for {len(generated)} targets")
    for T in TARGETS:
        totals = {M: len(smiles) for M, smiles in generated[T].items()}
        print(f"  {T}: {sum(totals.values())} generated compounds across {len(totals)} methods")

    all_results = {}
    target_summary = {}
    for T in TARGETS:
        print(f"\n--- {T} ---")
        uniprot = PDB_TO_UNIPROT[T]
        try:
            chembl_id = uniprot_to_chembl_target(uniprot)
        except Exception as e:
            print(f"  ChEMBL target lookup failed for {uniprot}: {e}")
            chembl_id = None
        print(f"  {T} → UniProt {uniprot} → ChEMBL {chembl_id}")
        if not chembl_id:
            all_results[T] = {"error": f"no ChEMBL target for UniProt {uniprot}"}
            continue

        try:
            activities = fetch_activities(chembl_id, uniprot)
        except Exception as e:
            print(f"  activity fetch failed: {e}")
            all_results[T] = {"error": f"activity fetch failed: {e}"}
            continue

        _, active_meta = activities_to_smiles_set(activities)
        target_summary[T] = {
            "uniprot": uniprot,
            "chembl_target": chembl_id,
            "n_activities": len(activities),
            "n_unique_actives": len(active_meta),
        }
        if len(active_meta) < 10:
            print(f"  WARNING: only {len(active_meta)} actives; results will be weak")
        result = analyze_target(T, generated[T], active_meta)
        all_results[T] = result

    # Aggregate per method across targets
    per_method_agg = {}
    for M in METHODS:
        vals = defaultdict(list)
        for T in TARGETS:
            r = all_results.get(T, {})
            if isinstance(r, dict) and M in r:
                for k, v in r[M].items():
                    if isinstance(v, (int, float)):
                        vals[k].append(v)
        per_method_agg[M] = {k: float(np.mean(v)) for k, v in vals.items()}

    out_final = {
        "target_summary": target_summary,
        "per_target_per_method": all_results,
        "per_method_avg": per_method_agg,
    }
    with open(OUT_DIR / "chembl.json", "w") as f:
        json.dump(out_final, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("PER-METHOD AGGREGATE (averaged across targets):")
    print("=" * 70)
    print(f"{'Method':<20} {'InChI redisc':>12} {'mean maxSim':>12} {'top10 mSim':>12} "
          f"{'≥0.4 %':>8} {'≥0.5 %':>8} {'≥0.6 %':>8} {'#scaff neigh':>14}")
    for M in METHODS:
        e = per_method_agg.get(M, {})
        if e:
            print(f"{METHOD_MAP[M]:<20} {e.get('exact_inchi_rediscovery',0):>12.2f} "
                  f"{e.get('mean_max_similarity',0):>12.3f} {e.get('top10_max_similarity',0):>12.3f} "
                  f"{e.get('frac_sim_ge_0.4',0)*100:>8.2f} {e.get('frac_sim_ge_0.5',0)*100:>8.2f} "
                  f"{e.get('frac_sim_ge_0.6',0)*100:>8.2f} {e.get('active_scaffold_neighborhoods_reached',0):>14.1f}")
    print(f"\nOutput → {OUT_DIR / 'chembl.json'}")


if __name__ == "__main__":
    main()
