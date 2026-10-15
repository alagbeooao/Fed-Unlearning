"""
Federated simulation entry point (Flower).

Runs all active clients in a single-GPU Ray simulation. Each round the
server samples clients (here, all of them: fraction_fit=1.0), each client
trains locally for `local_epochs` epochs, the server FedAvg-aggregates and
checkpoints, and optionally evaluates on validation data.

Usage from a script:
    from federated.simulation import run_federation
    run_federation(config, processed_dir, output_dir, excluded_clients=None)
"""
from pathlib import Path

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters

from federated.client import AEClient, get_parameters, set_parameters
from federated.seeding import enable_determinism, set_global_seed
from federated.server import CheckpointingFedAvg
from models.ae_2d import AE2D


def fit_metrics_aggregation_fn(metrics):
    """Sample-weighted mean of per-client fit metrics."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    out = {}
    for n, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0.0) + v * n / total
    return out


def evaluate_metrics_aggregation_fn(metrics):
    return fit_metrics_aggregation_fn(metrics)


def run_federation(config, processed_dir, output_dir, excluded_clients=None,
                   num_rounds=None, init_model_path=None):
    """Run a federated training simulation.

    Args:
        config: full config dict.
        processed_dir: path to the root of processed client folders.
        output_dir: where to write checkpoints.
        excluded_clients: optional set of client IDs to exclude (the oracle
            and the unlearning fine-tunes exclude the forgotten client).
        num_rounds: override config['training']['num_rounds'].
        init_model_path: optional .pt checkpoint to initialise from (used by
            the unlearning fine-tune phases). Must contain a "parameters"
            list of tensors.

    Returns:
        (final_model_path, history)
        final_model_path = {output_dir}/round_{num_rounds:04d}/global_model.pt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(config["experiment"]["seed"])
    if config["experiment"].get("deterministic", True):
        enable_determinism(True)
    n_clients_total = config["data"]["n_clients"]
    excluded = set(excluded_clients or [])
    active_clients = [i for i in range(n_clients_total) if i not in excluded]
    n_active = len(active_clients)
    if n_active == 0:
        raise ValueError("No active clients after exclusion.")

    if num_rounds is None:
        num_rounds = config["training"]["num_rounds"]

    # Initial parameters (optionally loaded from a checkpoint).
    init_model = AE2D(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_channels=config["model"]["latent_channels"],
    )
    if init_model_path is not None:
        ckpt = torch.load(init_model_path, weights_only=False)
        set_parameters(init_model, [p.numpy() for p in ckpt["parameters"]])
    init_parameters = ndarrays_to_parameters(get_parameters(init_model))

    strategy = CheckpointingFedAvg(
        output_dir=output_dir,
        save_client_updates=True,
        fraction_fit=config["training"]["fraction_fit"],
        fraction_evaluate=config["training"]["fraction_evaluate"],
        min_fit_clients=n_active,
        min_evaluate_clients=n_active,
        min_available_clients=n_active,
        initial_parameters=init_parameters,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Flower passes string CIDs "0".."n_active-1"; map them onto the actual
    # active client IDs.
    cid_to_client = {str(i): cid for i, cid in enumerate(active_clients)}

    def client_fn(cid: str):
        actual_id = cid_to_client[cid]
        client_dir = Path(processed_dir) / f"client_{actual_id}"
        return AEClient(actual_id, client_dir, config, device).to_client()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_active,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 1.0 / n_active if device == "cuda" else 0,
        },
    )

    strategy.save_log()
    final_path = Path(output_dir) / f"round_{num_rounds:04d}" / "global_model.pt"
    print(f"\nFederation complete. Final model: {final_path}")
    return final_path, history
