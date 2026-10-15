"""
Train the oracle baseline (retrain from scratch without the forgotten client).

The oracle is the reference every unlearning method is compared against.

Sanity checks after this completes, before running the unlearning methods:
    1. KS test on the forgotten client's healthy validation reconstruction
       errors, pre-unlearning model vs oracle, should be significant
       (p < 0.01). If not, the federation did not learn client-specific
       information and unlearning has nothing to remove.
    2. The oracle's anomaly-detection AUROC on remaining clients should be
       within roughly 0.02 of the pre-unlearning model's on those clients.
    3. Reconstruction-error histograms for pre-unlearning vs oracle on the
       forgotten client's data should look visibly different.

Usage:
    python scripts/run_oracle.py --config configs/base.yaml --forgotten_client 0
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unlearning.retrain_oracle import run_oracle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    output_root = config["experiment"]["output_dir"]
    active = [i for i in range(config["data"]["n_clients"])
              if i != args.forgotten_client]

    print("=" * 60)
    print("Oracle training")
    print(f"  Forgotten client: {args.forgotten_client}")
    print(f"  Active clients: {active}")
    print("=" * 60)

    final_path = run_oracle(
        config, processed_dir, output_root, args.forgotten_client
    )
    print(f"\nOracle complete: {final_path}")


if __name__ == "__main__":
    main()
