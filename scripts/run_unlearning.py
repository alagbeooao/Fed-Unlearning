"""
Run the three unlearning methods (gradient ascent, federated KD, FedEraser).

Prerequisites:
    - base federation trained (scripts/run_base_federation.py)
    - oracle trained (scripts/run_oracle.py)

After this completes you have five models to compare:
    pre_unlearning, oracle, grad_ascent, kd, fed_eraser
then run scripts/run_evaluation.py.

The manifest written here is MERGED with any existing manifest for the same
forgotten client, so you can run methods in separate invocations (e.g. only
--methods kd) without losing previously recorded paths.

Usage:
    python scripts/run_unlearning.py \
        --config configs/base.yaml \
        --forgotten_client 0 \
        --base_run_dir ./outputs/base_federation
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlearning.fed_eraser import run_fed_eraser_unlearning
from unlearning.gradient_ascent import run_gradient_ascent_unlearning
from unlearning.kd_unlearn import run_kd_unlearning


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--base_run_dir", required=True,
                   help="Path to the base-federation output directory")
    p.add_argument("--methods", nargs="*",
                   default=["grad_ascent", "kd", "fed_eraser"],
                   choices=["grad_ascent", "kd", "fed_eraser",
                            "kd_fraction", "kd_teacher_signal",
                            "kd_teacher_gap", "kd_oracle_divergence"],
                   help="kd_* variants run KD with the corresponding "
                        "push_scaling mode (see unlearning/push_scaling.py)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    output_root = config["experiment"]["output_dir"]
    base_run_dir = Path(args.base_run_dir)

    n_rounds = config["training"]["num_rounds"]
    base_model_path = base_run_dir / f"round_{n_rounds:04d}" / "global_model.pt"
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            f"Run scripts/run_base_federation.py first."
        )

    method_paths = {}

    if "grad_ascent" in args.methods:
        print("\n" + "=" * 60)
        print("UNLEARNING 1/3: Gradient Ascent (Halimi et al. 2022)")
        print("=" * 60)
        method_paths["grad_ascent"] = str(run_gradient_ascent_unlearning(
            config, processed_dir, output_root,
            args.forgotten_client, base_model_path,
        ))

    if "kd" in args.methods:
        print("\n" + "=" * 60)
        print("UNLEARNING 2/3: Federated Knowledge Distillation (Wu et al. 2022)")
        print("=" * 60)
        method_paths["kd"] = str(run_kd_unlearning(
            config, processed_dir, output_root,
            args.forgotten_client, base_model_path,
        ))

    # KD variants using the push-scaling modes
    oracle_model_path = (
        Path(output_root) / f"oracle_forgot_{args.forgotten_client}"
        / f"round_{n_rounds:04d}" / "global_model.pt")
    for variant, mode in [("kd_fraction", "fraction"),
                          ("kd_teacher_signal", "teacher_signal"),
                          ("kd_teacher_gap", "teacher_gap"),
                          ("kd_oracle_divergence", "oracle_divergence")]:
        if variant not in args.methods:
            continue
        print("\n" + "=" * 60)
        print(f"UNLEARNING: KD with push_scaling='{mode}'")
        print("=" * 60)
        if mode == "oracle_divergence" and not oracle_model_path.exists():
            raise FileNotFoundError(
                f"kd_oracle_divergence needs the oracle at {oracle_model_path}. "
                f"Run scripts/run_oracle.py for client {args.forgotten_client} first."
            )
        vcfg = copy.deepcopy(config)
        vcfg["unlearning"]["kd_unlearn"]["push_scaling"] = mode
        method_paths[variant] = str(run_kd_unlearning(
            vcfg, processed_dir, output_root,
            args.forgotten_client, base_model_path,
            oracle_model_path=(oracle_model_path
                               if mode == "oracle_divergence" else None),
        ))

    if "fed_eraser" in args.methods:
        print("\n" + "=" * 60)
        print("UNLEARNING 3/3: FedEraser (Liu et al. 2021)")
        print("=" * 60)
        method_paths["fed_eraser"] = str(run_fed_eraser_unlearning(
            config, processed_dir, output_root,
            args.forgotten_client, base_run_dir,
        ))

    # Merge with any existing manifest so separate invocations accumulate.
    manifest_path = (
        Path(output_root) / f"manifest_forgot_{args.forgotten_client}.json"
    )
    oracle_model = str(
        Path(output_root) / f"oracle_forgot_{args.forgotten_client}" /
        f"round_{n_rounds:04d}" / "global_model.pt"
    )
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault("unlearned_models", {})
        manifest["unlearned_models"].update(method_paths)
    else:
        manifest = {
            "forgotten_client": args.forgotten_client,
            "base_model": str(base_model_path),
            "oracle_model": oracle_model,
            "unlearned_models": method_paths,
        }
    # Keep base/oracle paths current.
    manifest["base_model"] = str(base_model_path)
    manifest["oracle_model"] = oracle_model

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n\nManifest saved: {manifest_path}")
    print(f"  methods recorded: {sorted(manifest['unlearned_models'])}")
    print("Run scripts/run_evaluation.py next.")


if __name__ == "__main__":
    main()
