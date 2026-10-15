"""
Oracle baseline: retrain the federated AE from scratch with the forgotten
client excluded.

This is the reference point for unlearning. Every unlearning method is
measured by how closely its outputs match this oracle's: on the
reconstruction-error distribution over the forgotten client's data
(efficacy), on downstream anomaly-detection quality on remaining clients
(utility), and on membership-inference resistance (privacy).

The oracle is a federation over the remaining clients only, using the
same hyperparameters as the base federation. It therefore trains on
strictly less data, which matters when reading the utility numbers: a
method warm-started from the all-client model inherits a richer
representation than the oracle can learn from scratch.
"""
from pathlib import Path

import yaml

from federated.simulation import run_federation


def run_oracle(config, processed_dir, output_root, forgotten_client):
    """Train an oracle with the forgotten client excluded.

    Args:
        config: full config dict.
        processed_dir: path to processed/ root.
        output_root: e.g. ./outputs/.
        forgotten_client: int, the client to exclude.

    Returns:
        path to the oracle's final global model.
    """
    output_dir = Path(output_root) / f"oracle_forgot_{forgotten_client}"
    output_dir.mkdir(parents=True, exist_ok=True)

    active = [i for i in range(config["data"]["n_clients"])
              if i != forgotten_client]
    print(f"Training oracle with client {forgotten_client} excluded.")
    print(f"Active clients: {active}")

    final_path, _ = run_federation(
        config=config,
        processed_dir=processed_dir,
        output_dir=output_dir,
        excluded_clients={forgotten_client},
    )
    return final_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_oracle(
        config,
        config["data"]["processed_dir"],
        config["experiment"]["output_dir"],
        args.forgotten_client,
    )
