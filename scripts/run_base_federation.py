"""
Train the base federated AE (the pre-unlearning model).

This is the model every unlearning method starts from. After it runs you
will have, under {output_dir}/base_federation/:
    round_0001/ ... round_0100/   (global_model.pt + client_*_update.pt)
    training_log.json

Sanity checks before proceeding:
    - training_log.json shows decreasing loss over rounds
    - the round-100 global model reconstructs healthy validation slices well
    - pixel AUROC on tumour test slices is roughly 0.80-0.88

Usage:
    python scripts/run_base_federation.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.simulation import run_federation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_subdir", default="base_federation")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    output_dir = Path(config["experiment"]["output_dir"]) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Base federation training")
    print(f"  Config: {args.config}")
    print(f"  Processed data: {processed_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Clients: {config['data']['n_clients']}")
    print(f"  Rounds: {config['training']['num_rounds']}")
    print(f"  Local epochs/round: {config['training']['local_epochs']}")
    print("=" * 60)

    final_path, _ = run_federation(
        config=config,
        processed_dir=processed_dir,
        output_dir=output_dir,
        excluded_clients=None,
    )

    print(f"\nDone. Final model: {final_path}")
    print("\nNext: train the oracle with")
    print(f"  python scripts/run_oracle.py --config {args.config} "
          f"--forgotten_client 0")


if __name__ == "__main__":
    main()
