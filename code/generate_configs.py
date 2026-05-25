#!/usr/bin/env python3
"""
Generate experiment configurations for all ablation studies.

Produces JSON config files for:
  - Main experiments (3 methods × 5 seeds)
  - Ablation: equilibration time sweep
  - Ablation: novelty weight sweep
  - Ablation: scaling (different iteration counts)
"""

import json
import os
from itertools import product

OUTPUT_DIR = "paper/experiments/configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Base templates ───

SYSTEM_3V8D = {
    "path": "examples.docking.systems.Docking.Docking",
    "config": {
        "biomolecule": "examples/docking/systems/P22680_3V8D_A.pdb",
        "bbox": "examples/docking/systems/P22680_3V8D_A_box_2.txt"
    }
}

SEED_SMILES = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"

SAVE_CALLBACK = {
    "path": "adtool.callbacks.on_discovery_callbacks.save_discovery_on_disk.SaveDiscoveryOnDisk",
    "config": {}
}


def make_config(save_dir, explorer_path, explorer_config, system=None):
    if system is None:
        system = SYSTEM_3V8D
    return {
        "experiment": {
            "config": {
                "save_location": save_dir,
                "save_frequency": 1,
                "bootstrap_size": 1
            }
        },
        "system": system,
        "explorer": {
            "path": explorer_path,
            "config": explorer_config
        },
        "input_wrappers": [],
        "output_representations": [],
        "logger_handlers": [],
        "callbacks": {
            "on_discovery": [SAVE_CALLBACK]
        }
    }


def imgep_config(equil_time=1):
    return {
        "mutator": "specific",
        "equil_time": equil_time,
        "behavior_map": "examples.docking.maps.DockingStatistics.DockingStatistics",
        "parameter_map": "examples.docking.maps.DockingParameterMap.DockingParameterMap",
        "mutator_config": {},
        "behavior_map_config": {},
        "parameter_map_config": {"seed_smiles": SEED_SMILES}
    }


def curiosity_config(equil_time=1, novelty_weight=0.5):
    cfg = imgep_config(equil_time)
    cfg["novelty_weight"] = novelty_weight
    return cfg


def random_config():
    return imgep_config(equil_time=10000)  # equil > N_iter → always random


def write_config(name, config):
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  {path}")


# ─── Main experiments ───

print("Main experiments:")
write_config("main_random_3v8d", make_config(
    "./paper/experiments/results/main_random_3v8d",
    "adtool.explorers.IMGEPExplorer.IMGEPExplorer",
    random_config()))

write_config("main_imgep_3v8d", make_config(
    "./paper/experiments/results/main_imgep_3v8d",
    "adtool.explorers.IMGEPExplorer.IMGEPExplorer",
    imgep_config()))

write_config("main_curiosity_3v8d", make_config(
    "./paper/experiments/results/main_curiosity_3v8d",
    "adtool.explorers.CuriosityIMGEPExplorer.IMGEPExplorer",
    curiosity_config()))


# ─── Ablation: equilibration time ───

print("\nAblation: equilibration time:")
for T in [1, 5, 10, 25]:
    name = f"ablation_equil{T}_imgep_3v8d"
    write_config(name, make_config(
        f"./paper/experiments/results/{name}",
        "adtool.explorers.IMGEPExplorer.IMGEPExplorer",
        imgep_config(equil_time=T)))


# ─── Ablation: novelty weight ───

print("\nAblation: novelty weight:")
for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    name = f"ablation_novelty{w:.2f}_curiosity_3v8d"
    write_config(name, make_config(
        f"./paper/experiments/results/{name}",
        "adtool.explorers.CuriosityIMGEPExplorer.IMGEPExplorer",
        curiosity_config(novelty_weight=w)))


print("\nAll configs generated.")
print(f"Total: {len(os.listdir(OUTPUT_DIR))} config files in {OUTPUT_DIR}/")
