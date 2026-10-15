"""
Knowledge-distillation unlearning, federated.

Wu et al. 2022, arXiv:2201.09441.

The pre-unlearning model is a frozen teacher. A student, initialised from it, is
trained to match the teacher on the remaining clients' data and to diverge from
it on the forgotten client's, the idea being to keep everything the teacher knew
except what came from the client that left.

This is the federated version. Each remaining client runs the combined objective
on its own data and the server FedAvg-aggregates the student each round, so data
stays local and the comparison against the federated oracle is like for like. A
centralised student would see all the remaining data at once, which the oracle
never does.

The forgotten client's data is needed for the push term during unlearning. That
is the standard assumption in this line of work: the client has left, but its
data is still available for the removal operation.

The push term maximises student-teacher distance on forgotten data, and left
alone it grows without bound and the student diverges everywhere. It is capped
at tau_push, which at this error scale is an active constraint rather than a
formality.
"""
import itertools
import json
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from flwr.common import ndarrays_to_parameters

from data.datasets import get_client_loader
from federated.client import get_parameters, set_parameters
from federated.seeding import derive_seed, enable_determinism, set_global_seed
from federated.server import CheckpointingFedAvg
from models.ae_2d import AE2D, reconstruction_loss
from unlearning.push_scaling import compute_push_scale


class KDStudentClient(fl.client.NumPyClient):
    """A remaining-client that trains the student with the KD objective.

    The teacher is loaded once per client from frozen weights. Each round
    the client receives the current student weights, runs local SGD on the
    combined MATCH (on its own data) + PUSH (on forgotten data) loss, and
    returns updated student weights.

    Args:
        client_id: int, this remaining client's id.
        client_dir: path to this client's processed folder.
        forgotten_client_dir: path to the forgotten client's processed
            folder. Each worker builds its own loader over this client's
            training data for the PUSH term (we pass a path, not a
            DataLoader, so Ray serialises only a string to each worker).
        teacher_params_np: frozen teacher weights (list of numpy arrays).
        config: full config dict.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, client_id, client_dir, forgotten_client_dir,
                 teacher_params_np, config, device="cuda",
                 alpha_push_effective=None):
        if config["experiment"].get("deterministic", True):
            enable_determinism(True)
        self.client_id = client_id
        self.config = config
        self.device = device
        self.loss_mode = config["model"]["loss"]

        kd = config["unlearning"]["kd_unlearn"]
        self.alpha_match = kd["alpha_match"]
        # either the configured value or a scaled one from push_scaling.py
        self.alpha_push = (kd["alpha_push"] if alpha_push_effective is None
                           else alpha_push_effective)
        self.tau_push = kd.get("tau_push", 0.05)
        self.distill_lr = kd["distill_lr"]
        self.local_epochs = config["training"]["local_epochs"]

        m = config["model"]
        # Teacher (frozen).
        self.teacher = AE2D(m["in_channels"], m["base_channels"],
                            m["latent_channels"]).to(device)
        set_parameters(self.teacher, teacher_params_np)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student (weights set each round from the server).
        self.student = AE2D(m["in_channels"], m["base_channels"],
                            m["latent_channels"]).to(device)

        # Local remaining-client data for the MATCH term.
        self.base_seed = config["experiment"]["seed"]
        self.match_loader = get_client_loader(
            client_dir, split="train", mode="healthy",
            batch_size=config["training"]["local_batch_size"],
            shuffle=True,
            seed=derive_seed(self.base_seed, "kd_match", client_id),
        )
        self.n_train = len(self.match_loader.dataset)

        # Shared forgotten-client data for the PUSH term. Built here (inside
        # the worker) from a path, so the loader is not serialised across
        # Ray workers — matching the base federation's per-client pattern.
        self.forgotten_loader = get_client_loader(
            forgotten_client_dir, split="train", mode="healthy",
            batch_size=config["training"]["local_batch_size"],
            shuffle=True,
            seed=derive_seed(self.base_seed, "kd_push", client_id),
        )

    def get_parameters(self, config):
        return get_parameters(self.student)

    def fit(self, parameters, config):
        set_parameters(self.student, parameters)
        # Reproducible per-round shuffling for both streams.
        rnd = int(config.get("server_round", 0))
        for tag, ld in (("kd_match", self.match_loader),
                        ("kd_push", self.forgotten_loader)):
            if ld.generator is not None:
                ld.generator.manual_seed(
                    derive_seed(self.base_seed, tag, self.client_id, rnd))
        self.student.train()
        opt = optim.Adam(self.student.parameters(), lr=self.distill_lr)

        forgotten_iter = itertools.cycle(self.forgotten_loader)
        run = {"match": 0.0, "push": 0.0}
        n_batches = 0

        for _ in range(self.local_epochs):
            for x_rem, _ in self.match_loader:
                x_rem = x_rem.to(self.device)
                x_for, _ = next(forgotten_iter)
                x_for = x_for.to(self.device)

                opt.zero_grad()

                # MATCH: student tracks teacher on remaining data, and also
                # reconstructs it (the second term keeps the student a valid
                # AE rather than only a teacher-mimic).
                with torch.no_grad():
                    t_rem = self.teacher(x_rem)
                s_rem = self.student(x_rem)
                loss_match = (F.l1_loss(s_rem, t_rem)
                              + reconstruction_loss(s_rem, x_rem,
                                                    mode=self.loss_mode))

                # PUSH: student diverges from teacher on forgotten data,
                # rewarded only up to tau_push.
                with torch.no_grad():
                    t_for = self.teacher(x_for)
                s_for = self.student(x_for)
                push_dist = F.l1_loss(s_for, t_for)
                loss_push = -torch.clamp(push_dist, max=self.tau_push)

                total = (self.alpha_match * loss_match
                         + self.alpha_push * loss_push)
                total.backward()
                opt.step()

                run["match"] += loss_match.item()
                run["push"] += push_dist.item()
                n_batches += 1

        metrics = {
            "client_id": self.client_id,
            "match": run["match"] / max(n_batches, 1),
            "push_dist": run["push"] / max(n_batches, 1),
        }
        return get_parameters(self.student), self.n_train, metrics

    def evaluate(self, parameters, config):
        # Evaluation handled centrally after distillation; nothing per-round.
        return 0.0, self.n_train, {"client_id": self.client_id}


def run_kd_unlearning(config, processed_dir, output_root,
                      forgotten_client, base_model_path,
                      oracle_model_path=None):
    """Federated KD unlearning.

    The student is trained as a federated model over the remaining
    clients with the MATCH + PUSH objective; the server FedAvg-aggregates
    the student each round. This is a fair comparison to the federated
    oracle: both keep client data local.

    Args:
        config: full config dict.
        processed_dir: path to processed/ root.
        output_root: e.g. ./outputs/.
        forgotten_client: int.
        base_model_path: path to the pre-unlearning model (the teacher).

    Returns:
        path to the unlearned student (.pt with {"parameters": [...]}).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(config["experiment"]["seed"])
    if config["experiment"].get("deterministic", True):
        enable_determinism(True)
    kd = config["unlearning"]["kd_unlearn"]
    m = config["model"]

    # Per-scaling-mode output dir so ablation variants do not overwrite each
    # other. 'fixed' keeps the original path for backward compatibility.
    _mode = kd.get("push_scaling", "fixed")
    _suffix = "" if _mode == "fixed" else f"_{_mode}"
    output_dir = (Path(output_root)
                  / f"unlearn_kd{_suffix}_forgot_{forgotten_client}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clients = config["data"]["n_clients"]
    remaining = [i for i in range(n_clients) if i != forgotten_client]
    n_active = len(remaining)
    n_rounds = kd["distill_rounds"]

    print(f"\n[kd] Federated KD unlearning, forgetting client {forgotten_client}")
    print(f"  Remaining clients: {remaining}")
    print(f"  Distill rounds: {n_rounds}, local epochs/round: "
          f"{config['training']['local_epochs']}")
    print(f"  alpha_match={kd['alpha_match']}, alpha_push={kd['alpha_push']}, "
          f"tau_push={kd.get('tau_push', 0.05)}")

    # Load teacher weights once (shared by all clients).
    ck = torch.load(base_model_path, weights_only=False)
    teacher_params_np = [p.numpy() for p in ck["parameters"]]

    # push weight scaling
    scaling_mode = kd.get("push_scaling", "fixed")
    oracle_params_np = None
    if scaling_mode == "oracle_divergence":
        if oracle_model_path is None:
            raise ValueError(
                "push_scaling='oracle_divergence' requires oracle_model_path")
        ock = torch.load(oracle_model_path, weights_only=False)
        oracle_params_np = [p.numpy() for p in ock["parameters"]]

    # Compute the scale on CPU by default. It is a one-off measurement of a
    # few forward passes, and running it on the GPU claims memory in the
    # driver process that Ray then cannot allocate to the client actors.
    scale_device = kd.get("scale_device", "cpu")
    push_scale, scale_diag = compute_push_scale(
        scaling_mode, config, processed_dir, forgotten_client,
        teacher_params_np, oracle_params_np, scale_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    alpha_push_effective = kd["alpha_push"] * push_scale
    print(f"  push_scaling='{scaling_mode}' -> scale={push_scale:.4f}, "
          f"alpha_push {kd['alpha_push']} -> {alpha_push_effective:.5f}")
    if scaling_mode == "teacher_signal":
        print(f"    teacher error: forgotten={scale_diag['e_forgotten']:.5f}, "
              f"remaining={scale_diag['e_remaining']:.5f}, "
              f"signal s={scale_diag['memorisation_signal']:+.4f}")
    with open(output_dir / "push_scaling.json", "w") as f:
        json.dump({k: v for k, v in scale_diag.items()}, f, indent=2,
                  default=float)

    # Initialise the student from the teacher (Wu 2022: faster convergence).
    init_model = AE2D(m["in_channels"], m["base_channels"],
                      m["latent_channels"])
    set_parameters(init_model, teacher_params_np)
    init_parameters = ndarrays_to_parameters(get_parameters(init_model))

    # Forgotten client's processed folder (path only). Each worker builds
    # its own loader from this path for the PUSH term, so no DataLoader is
    # serialised across Ray workers.
    forgotten_client_dir = Path(processed_dir) / f"client_{forgotten_client}"

    distill_dir = output_dir / "distill_rounds"
    strategy = CheckpointingFedAvg(
        output_dir=distill_dir,
        save_client_updates=False,  # not needed for KD
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=n_active,
        min_evaluate_clients=0,
        min_available_clients=n_active,
        initial_parameters=init_parameters,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        fit_metrics_aggregation_fn=_kd_metrics_agg,
    )

    cid_to_client = {str(i): cid for i, cid in enumerate(remaining)}

    def client_fn(cid: str):
        actual = cid_to_client[cid]
        client_dir = Path(processed_dir) / f"client_{actual}"
        return KDStudentClient(
            actual, client_dir, forgotten_client_dir,
            teacher_params_np, config, device,
            alpha_push_effective=alpha_push_effective,
        ).to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_active,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 1.0 / n_active if device == "cuda" else 0,
        },
    )
    strategy.save_log()

    # The final aggregated student is the last round's global model.
    final_global = distill_dir / f"round_{n_rounds:04d}" / "global_model.pt"
    final_ckpt = torch.load(final_global, weights_only=False)

    out_path = output_dir / "final_unlearned_model.pt"
    torch.save({"parameters": final_ckpt["parameters"]}, out_path)

    # Persist the per-round distillation metrics for diagnostics.
    with open(output_dir / "distillation_log.json", "w") as f:
        json.dump(strategy.training_log, f, indent=2)

    print(f"\n[kd] Final unlearned model saved to {out_path}")
    return out_path


def _kd_metrics_agg(metrics):
    """Aggregate per-client KD metrics (sample-weighted mean) for logging."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    out = {}
    for n, md in metrics:
        for k, v in md.items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0.0) + v * n / total
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--base_model_path", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_kd_unlearning(
        config,
        config["data"]["processed_dir"],
        config["experiment"]["output_dir"],
        args.forgotten_client,
        args.base_model_path,
    )
