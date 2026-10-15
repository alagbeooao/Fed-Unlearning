"""
Flower client for federated AE training.

Each client owns a local AE and its local healthy slices. Per round the
server sends global weights, the client trains for `local_epochs` epochs
on its local healthy data, and returns the updated weights plus its
training-sample count. The sample count drives FedAvg's weighted
aggregation.
"""
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch
import torch.optim as optim

from data.datasets import get_client_loader
from federated.seeding import derive_seed, enable_determinism
from models.ae_2d import AE2D, reconstruction_loss


def get_parameters(model):
    """Model parameters as a list of numpy arrays (Flower convention)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Load a list of numpy arrays into the model (state_dict order)."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class AEClient(fl.client.NumPyClient):
    """Federated AE client.

    Args:
        client_id: int, used to locate data and tag updates.
        client_dir: path to processed/client_{client_id}/.
        config: full config dict.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, client_id, client_dir, config, device="cuda"):
        # Ray runs each client in its own process, so determinism must be
        # enabled here as well as in the driver. Do it BEFORE any tensor is
        # moved to the GPU, since CUBLAS_WORKSPACE_CONFIG has to be set prior
        # to CUDA initialisation.
        if config["experiment"].get("deterministic", True):
            enable_determinism(True)
        self.client_id = client_id
        self.client_dir = client_dir
        self.config = config
        self.device = device

        m = config["model"]
        self.model = AE2D(
            in_channels=m["in_channels"],
            base_channels=m["base_channels"],
            latent_channels=m["latent_channels"],
        ).to(device)

        t = config["training"]
        # Reproducible shuffling: seed derived from the run seed and client id,
        # so each client shuffles differently but identically across runs.
        self.base_seed = config["experiment"]["seed"]
        self.train_loader = get_client_loader(
            client_dir, split="train", mode="healthy",
            batch_size=t["local_batch_size"], shuffle=True,
            seed=derive_seed(self.base_seed, "train", client_id),
        )
        self.val_loader = get_client_loader(
            client_dir, split="val", mode="healthy",
            batch_size=t["local_batch_size"], shuffle=False,
        )
        self.n_train = len(self.train_loader.dataset)
        self.loss_mode = m["loss"]

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # Re-seed the shuffle for this round: different order each round (so
        # training keeps its stochasticity) but identical across reruns.
        rnd = int(config.get("server_round", 0))
        if self.train_loader.generator is not None:
            self.train_loader.generator.manual_seed(
                derive_seed(self.base_seed, "train", self.client_id, rnd))
        t = self.config["training"]
        opt = optim.Adam(self.model.parameters(),
                         lr=t["local_lr"], weight_decay=t["weight_decay"])

        self.model.train()
        running_loss = 0.0
        n_batches = 0
        for _ in range(t["local_epochs"]):
            for x, _ in self.train_loader:
                x = x.to(self.device)
                opt.zero_grad()
                recon = self.model(x)
                loss = reconstruction_loss(recon, x, mode=self.loss_mode)
                loss.backward()
                opt.step()
                running_loss += loss.item()
                n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        metrics = {"train_loss": float(avg_loss), "client_id": self.client_id}
        return get_parameters(self.model), self.n_train, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        running_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                recon = self.model(x)
                loss = reconstruction_loss(recon, x, mode=self.loss_mode)
                running_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
        avg_loss = running_loss / max(n_samples, 1)
        metrics = {"val_loss": float(avg_loss), "client_id": self.client_id}
        return float(avg_loss), n_samples, metrics


def make_client_fn(processed_dir, config, device, excluded_clients=None):
    """Factory for Flower's simulation API.

    Args:
        processed_dir: path to root containing client_0/, client_1/, ...
        config: full config dict.
        device: 'cuda' or 'cpu'.
        excluded_clients: optional set of client IDs to exclude.
    """
    excluded = set(excluded_clients or [])

    def client_fn(cid: str):
        client_id = int(cid)
        if client_id in excluded:
            raise ValueError(
                f"Client {client_id} is excluded but was instantiated."
            )
        client_dir = Path(processed_dir) / f"client_{client_id}"
        return AEClient(client_id, client_dir, config, device).to_client()

    return client_fn
