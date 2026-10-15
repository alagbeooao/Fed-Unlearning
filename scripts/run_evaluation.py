"""
Run the full evaluation suite across all methods.

Computes for each model (pre-unlearning, oracle, grad_ascent, kd, fed_eraser):
1. Anomaly detection: pixel AUROC and Dice on tumour test slices
   (separately for forgotten client and remaining clients)
2. Unlearning efficacy: Wasserstein and KS distance to oracle's
   reconstruction-error distribution on forgotten client's data
3. Membership inference attack AUROC on forgotten client's data

Outputs a single results JSON that the notebook can read for tables/figures.

Usage:
    python scripts/run_evaluation.py \\
        --config configs/base.yaml \\
        --forgotten_client 2 \\
        --manifest ./outputs/manifest_forgot_2.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.anomaly_detection import evaluate_anomaly_detection
from evaluation.unlearning_efficacy import compare_all_methods, compute_efficacy
from evaluation.membership_inference import run_mia_all_methods


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--manifest", required=True,
                   help="Path to manifest JSON from run_unlearning.py")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.manifest) as f:
        manifest = json.load(f)

    processed_dir = config["data"]["processed_dir"]
    n_clients = config["data"]["n_clients"]
    forgotten = manifest["forgotten_client"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the full set of models to evaluate
    all_models = {
        "pre_unlearning": manifest["base_model"],
        "oracle": manifest["oracle_model"],
        **manifest["unlearned_models"],
    }

    results = {
        "config": args.config,
        "forgotten_client": forgotten,
        "models": list(all_models.keys()),
        "metrics": {},
    }

    # 1. anomaly detection
    print("\n" + "=" * 60)
    print("EVALUATION 1/3: Anomaly Detection (pixel AUROC, Dice)")
    print("=" * 60)
    for name, path in all_models.items():
        print(f"\n  {name}: {path}")

        # On remaining clients (utility check)
        ad_remaining = evaluate_anomaly_detection(
            model_path=path,
            processed_dir=processed_dir,
            n_clients=n_clients,
            config=config,
            splits=("test",),
            excluded_clients=[forgotten],
            device=device,
        )

        # On forgotten client (does the unlearned model still detect their tumours?)
        ad_forgotten = evaluate_anomaly_detection(
            model_path=path,
            processed_dir=processed_dir,
            n_clients=n_clients,
            config=config,
            splits=("test",),
            excluded_clients=[i for i in range(n_clients) if i != forgotten],
            device=device,
        )

        results["metrics"].setdefault(name, {})["anomaly_detection"] = {
            "remaining": ad_remaining,
            "forgotten": ad_forgotten,
        }
        print(f"    remaining: AUROC={ad_remaining['test']['auroc']:.4f}, "
              f"Dice={ad_remaining['test']['dice']:.4f}")
        print(f"    forgotten: AUROC={ad_forgotten['test']['auroc']:.4f}, "
              f"Dice={ad_forgotten['test']['dice']:.4f}")

    # 2. efficacy against the oracle
    print("\n" + "=" * 60)
    print("EVALUATION 2/3: Unlearning Efficacy (vs Oracle)")
    print("=" * 60)

    # Methods to compare against oracle: pre-unlearning + the three unlearning methods
    methods_for_efficacy = {
        k: v for k, v in all_models.items() if k != "oracle"
    }
    efficacy = compare_all_methods(
        method_paths=methods_for_efficacy,
        oracle_path=manifest["oracle_model"],
        processed_dir=processed_dir,
        forgotten_client=forgotten,
        config=config,
        device=device,
    )
    for name, eff in efficacy.items():
        results["metrics"].setdefault(name, {})["efficacy"] = {
            k: v for k, v in eff.items()
            if k not in ("errs_unlearned", "errs_oracle")  # drop raw arrays
        }
        # Save raw arrays separately for figure generation
        results["metrics"][name]["efficacy_raw"] = {
            "errs_unlearned": eff.get("errs_unlearned", []),
            "errs_oracle": eff.get("errs_oracle", []),
        }
        print(f"  {name}: W={eff.get('wasserstein', 'NA'):.4f}, "
              f"KS={eff.get('ks_statistic', 'NA'):.4f}")

    # 3. membership inference
    print("\n" + "=" * 60)
    print("EVALUATION 3/3: Membership Inference Attack")
    print("=" * 60)
    mia_results = run_mia_all_methods(
        method_paths=all_models,
        processed_dir=processed_dir,
        forgotten_client=forgotten,
        config=config,
        device=device,
    )
    for name, mia in mia_results.items():
        results["metrics"].setdefault(name, {})["mia"] = mia
        auroc = mia.get("attack_auroc", "NA")
        if isinstance(auroc, float):
            print(f"  {name}: attack AUROC = {auroc:.4f}")
        else:
            print(f"  {name}: attack AUROC = {auroc}")

    # save
    if args.out is None:
        args.out = (Path(config["experiment"]["output_dir"]) /
                    f"evaluation_results_forgot_{forgotten}.json")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved: {args.out}")
    print(f"Generate tables and figures with notebooks/03_results_analysis.ipynb")

    # Quick textual summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'AUROC_rem':>10} {'Dice_rem':>10} {'W_dist':>10} {'MIA_AUROC':>10}")
    for name in all_models.keys():
        m = results["metrics"][name]
        auroc = m.get("anomaly_detection", {}).get("remaining", {}).get("test", {}).get("auroc", float("nan"))
        dice = m.get("anomaly_detection", {}).get("remaining", {}).get("test", {}).get("dice", float("nan"))
        w = m.get("efficacy", {}).get("wasserstein", float("nan")) if name != "oracle" else 0.0
        mia = m.get("mia", {}).get("attack_auroc", float("nan"))
        print(f"{name:<20} {auroc:>10.4f} {dice:>10.4f} {w:>10.4f} {mia:>10.4f}")


if __name__ == "__main__":
    main()
