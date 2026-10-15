"""
Flower server strategy with per-round checkpoint saving.

The per-round checkpoints are what make FedEraser (unlearning/fed_eraser.py)
possible: it reconstructs a counterfactual training trajectory from the
cached round-by-round client updates, so it needs both each round's
aggregated global model and each client's contributed update. Without
these checkpoints, FedEraser cannot run and the federation would have to
be repeated.

Saved per round under {output_dir}/round_{t:04d}/:
    global_model.pt        # aggregated weights for round t
    client_{i}_update.pt   # each participating client's contributed weights

Saved at the end:
    {output_dir}/training_log.json
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from flwr.common import (FitRes, Parameters, Scalar,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CheckpointingFedAvg(FedAvg):
    """FedAvg that persists per-round client updates and the global model.

    A thin wrapper around Flower's FedAvg with one extra responsibility:
    save enough state for FedEraser's calibrated replay to run later.
    """

    def __init__(self, output_dir, save_client_updates=True, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_client_updates = save_client_updates
        self.training_log = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # DETERMINISM: Ray returns client results in completion order, which
        # varies run to run. FedAvg sums them in that order, and floating-point
        # addition is not associative, so the aggregate differs slightly between
        # otherwise identical runs, a difference that compounds over rounds.
        # Sorting by client id fixes the summation order.
        results = sorted(
            results,
            key=lambda t: int(t[1].metrics.get("client_id", t[0].cid)),
        )
        round_dir = self.output_dir / f"round_{server_round:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Save individual client updates before aggregation (FedEraser
        # consumes these).
        if self.save_client_updates:
            for proxy, fit_res in results:
                client_id = fit_res.metrics.get("client_id", proxy.cid)
                arrays = parameters_to_ndarrays(fit_res.parameters)
                torch.save(
                    {
                        "client_id": client_id,
                        "num_examples": fit_res.num_examples,
                        "parameters": [torch.tensor(a) for a in arrays],
                        "metrics": fit_res.metrics,
                    },
                    round_dir / f"client_{client_id}_update.pt",
                )

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            arrays = parameters_to_ndarrays(aggregated_params)
            torch.save(
                {
                    "round": server_round,
                    "parameters": [torch.tensor(a) for a in arrays],
                    "metrics": aggregated_metrics,
                },
                round_dir / "global_model.pt",
            )

        self.training_log.append({
            "round": server_round,
            "phase": "fit",
            "metrics": dict(aggregated_metrics) if aggregated_metrics else {},
            "client_metrics": [dict(fr.metrics) for _, fr in results],
        })
        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        results = sorted(
            results,
            key=lambda t: int(t[1].metrics.get("client_id", t[0].cid)),
        )
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        if loss_aggregated is not None:
            self.training_log.append({
                "round": server_round,
                "phase": "evaluate",
                "loss": loss_aggregated,
                "metrics": dict(metrics_aggregated) if metrics_aggregated else {},
                "client_metrics": [dict(er.metrics) for _, er in results],
            })
        return loss_aggregated, metrics_aggregated

    def save_log(self):
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)


def load_global_model_at_round(output_dir, round_num):
    """Load the aggregated global model parameters for a given round.

    Returns the list of parameter tensors, ready for set_parameters().
    """
    ckpt_path = Path(output_dir) / f"round_{round_num:04d}" / "global_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, weights_only=False)["parameters"]


def load_client_update_at_round(output_dir, round_num, client_id):
    """Load a specific client's cached update for a given round.

    Used by FedEraser's calibrated replay.
    """
    ckpt_path = (
        Path(output_dir) / f"round_{round_num:04d}" /
        f"client_{client_id}_update.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, weights_only=False)
