#!/usr/bin/env python3
"""
Verification tool for Ising-ML campaign tasks (MLP-00 through MLP-11, SYN).
Usage:
    python ising_ml_agent_kit/tools/verify_task.py --describe TASK_ID
    python ising_ml_agent_kit/tools/verify_task.py TASK_ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys

TASK_SPECS = {
    "MLP-00": {
        "description": "Repo and environment audit",
        "dir": "experiments/mlp_L10/MLP-00",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": ["tests_passed", "total_tests"],
        "criteria": lambda m: m.get("tests_passed", 0) >= 44,
    },
    "MLP-01": {
        "description": "Exact transfer matrix physics oracle (L=10)",
        "dir": "experiments/mlp_L10/MLP-01",
        "required_files": ["metrics.json", "notes.md", "figs/F_TT_exact_L10.png"],
        "required_keys": ["T_peak_F_TT", "I_TX_exact", "I_YX_exact"],
        "criteria": lambda m: abs(m.get("T_peak_F_TT", 0.0) - 2.279) < 0.05,
    },
    "MLP-02": {
        "description": "Dataset generation and audit vs exact oracle",
        "dir": "experiments/mlp_L10/MLP-02",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": ["p_val_rejection_fraction", "sign_balance_overall", "I_TX_mc", "I_YX_mc"],
        "criteria": lambda m: m.get("p_val_rejection_fraction", 1.0) <= 0.10 and abs(m.get("sign_balance_overall", 0.0) - 0.5) <= 0.02,
    },
    "MLP-03": {
        "description": "Classification baselines vs optimal Bayesian oracle",
        "dir": "experiments/mlp_L10/MLP-03",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": ["ce_excess_logistic_E_pct", "accuracy_at_Tc_oracle", "accuracy_at_Tc_logistic_E", "accuracy_at_Tc_linear_s"],
        "criteria": lambda m: m.get("ce_excess_logistic_E_pct", 100.0) <= 2.5, # D9 documented margin
    },
    "MLP-04": {
        "description": "SimpleMLP reference training with checkpoints (5 seeds)",
        "dir": "experiments/mlp_L10/MLP-04",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": ["n_seeds", "n_seeds_converged", "n_checkpoints_per_seed", "val_loss_rel_variation_last5_max_pct"],
        "criteria": lambda m: (
            m.get("n_seeds", 0) >= 5
            and m.get("n_seeds_converged", 0) >= 4
            and m.get("n_checkpoints_per_seed", 0) >= 10
            and m.get("val_loss_rel_variation_last5_max_pct", 10.0) <= 1.0
        ),
    },
    "MLP-05": {
        "description": "Network behavior vs optimal Bayesian oracle",
        "dir": "experiments/mlp_L10/MLP-05",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": [
            "bayes_acc_at_Tc",
            "net_acc_at_Tc_mean",
            "net_acc_at_Tc_ci95",
            "delta_ce_at_Tc_mean",
            "delta_ce_at_Tc_ci95",
            "r2_logit_vs_gE",
            "verdict_tc_drop",
        ],
        "criteria": lambda m: m.get("verdict_tc_drop") in {"intrinsic", "network", "mixed"},
    },
    "MLP-06": {
        "description": "Layer probes: Fisher fraction, accessibility gap, neuron efficiency, and null controls",
        "dir": "experiments/mlp_L10/MLP-06",
        "required_files": [
            "config.json",
            "metrics.json",
            "notes.md",
            "figs/phi_lin_vs_T.png",
            "figs/phi_nonlin_vs_T.png",
            "figs/delta_accessibility_vs_T.png",
            "figs/neuron_efficiency_heatmap.png",
            "figs/shortcut_vs_energy_neurons.png",
        ],
        "required_keys": [
            "n_seeds",
            "phi_lin_input_max_abs",
            "control_permuted_r2_max_abs",
            "dpi_violation_beyond_ci",
            "dpi_violation_explained",
            "controls_separate_from_trained",
            "shortcut_neuron_fraction_by_layer",
        ],
        "criteria": lambda m: (
            m.get("n_seeds", 0) >= 5
            and m.get("phi_lin_input_max_abs", 1.0) < 0.02
            and m.get("control_permuted_r2_max_abs", 1.0) < 0.05
            and (not m.get("dpi_violation_beyond_ci", True) or bool(m.get("dpi_violation_explained")))
            and bool(m.get("controls_separate_from_trained", False))
        ),
    },
    "MLP-07": {
        "description": "Mutual information and entropy per layer (variational bounds)",
        "dir": "experiments/mlp_L10/MLP-07",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": [
            "n_seeds",
            "I_T_h_max_over_I_TX",
            "mi_chain_ok",
            "mi_chain_violation_explained",
            "binning_highdim_used",
            "all_values_have_ci",
            "gaussian_entropy_eps_stable",
        ],
        "criteria": lambda m: (
            m.get("n_seeds", 0) >= 5
            and m.get("I_T_h_max_over_I_TX", 2.0) <= 1.02
            and not m.get("binning_highdim_used", True)
            and m.get("all_values_have_ci", False) is True
        ),
    },
    "MLP-08": {
        "description": "Fisher in parameters per layer (r_f, spectrum, null controls)",
        "dir": "experiments/mlp_L10/MLP-08",
        "required_files": ["config.json", "metrics.json", "notes.md"],
        "required_keys": [
            "tests_abcd_pass",
            "uses_empirical_fisher",
            "dtype_fisher",
            "n_nulls",
            "ns_scan_values",
            "outliers_stable_in_ns",
            "rf_trained_beats_nulls_Tc",
        ],
        "criteria": lambda m: (
            m.get("tests_abcd_pass") is True
            and m.get("uses_empirical_fisher") is False
            and m.get("dtype_fisher") == "float64"
            and m.get("n_nulls", 0) >= 4
            and len(m.get("ns_scan_values", [])) >= 4
        ),
    },
}


def describe_task(task_id: str):
    if task_id not in TASK_SPECS:
        print(f"Error: Unknown task {task_id}. Available: {list(TASK_SPECS.keys())}")
        sys.exit(1)
    spec = TASK_SPECS[task_id]
    print(f"=== Spec for {task_id}: {spec['description']} ===")
    print(f"Expected directory: {spec['dir']}")
    print(f"Required files: {spec['required_files']}")
    print(f"Required keys in metrics.json: {spec['required_keys']}")


def verify_task(task_id: str):
    if task_id not in TASK_SPECS:
        print(f"Error: Unknown task {task_id}. Available: {list(TASK_SPECS.keys())}")
        sys.exit(1)

    spec = TASK_SPECS[task_id]
    task_dir = spec["dir"]
    print(f"--- Verifying {task_id}: {spec['description']} ---")

    # 1. Check directory
    if not os.path.isdir(task_dir):
        print(f"[FAIL] Directory does not exist: {task_dir}")
        sys.exit(1)

    # 2. Check required files
    missing_files = []
    for req_file in spec["required_files"]:
        fpath = os.path.join(task_dir, req_file)
        if not os.path.exists(fpath):
            missing_files.append(req_file)

    if missing_files:
        print(f"[FAIL] Missing required files in {task_dir}: {missing_files}")
        sys.exit(1)

    # 3. Check metrics.json
    metrics_path = os.path.join(task_dir, "metrics.json")
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"[FAIL] Could not parse metrics.json: {e}")
        sys.exit(1)

    missing_keys = [k for k in spec["required_keys"] if k not in metrics]
    if missing_keys:
        print(f"[FAIL] metrics.json is missing required keys: {missing_keys}")
        sys.exit(1)

    # Specific sanity stop conditions (e.g. Z2 broken -> code 3)
    if task_id == "MLP-06":
        phi_lin_input = metrics.get("phi_lin_input_max_abs", 0.0)
        if phi_lin_input >= 0.05:
            print(f"[CRITICAL STOP] phi_lin_input_max_abs = {phi_lin_input:.4f} >= 0.05! Z2 symmetry is broken.")
            sys.exit(3)

    # 4. Evaluate scientific criteria
    passed = spec["criteria"](metrics)
    if not passed:
        print(f"[FAIL] Scientific acceptance criteria not satisfied for {task_id}.")
        print("Measured values:")
        for k in spec["required_keys"]:
            print(f"  {k}: {metrics.get(k)}")
        sys.exit(1)

    print(f"[SUCCESS] {task_id} successfully verified! All files, keys, and criteria passed.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Task verification tool for Ising-ML")
    parser.add_argument("task_id", nargs="?", help="Task ID (e.g. MLP-03, MLP-04, MLP-05, MLP-06)")
    parser.add_argument("--describe", dest="describe_task", help="Describe requirements for a task")

    args = parser.parse_args()

    if args.describe_task:
        describe_task(args.describe_task)
    elif args.task_id:
        verify_task(args.task_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
