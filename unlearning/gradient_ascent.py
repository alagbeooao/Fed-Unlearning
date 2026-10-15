"""
Gradient-ascent based federated unlearning.

Reference:
    Halimi et al. 2022. Federated unlearning: How to efficiently erase a
    client in FL? arXiv:2207.05521.

Procedure:
    1. Start from the trained federated model (the pre-unlearning model).
    2. Run K steps of gradient ASCENT on the forgotten client's data,
       i.e. maximise reconstruction loss there, pushing the model away
       from the forgotten distribution.
    3. Run R rounds of standard FedAvg on the REMAINING clients to recover
       utility, since the ascent also degrades the model on retained data.

Notes:
    Gradient ascent on a generative model is volatile, so the recovery
    fine-tune is essential. The published classifier-domain settings
    (K=10 steps, lr=1e-4) drove ~16x reconstruction-error divergence on
    this AE; the rescaled settings in configs/base.yaml (K=2, lr=1e-5,
    R=1) match the AE's error scale.
"""
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from data.datasets import get_client_loader
from federated.client import get_parameters, set_parameters
from federated.seeding import set_global_seed
from federated.simulation import run_federation
from models.ae_2d import AE2D, reconstruction_loss


def gradient_ascent_phase(model, forgotten_loader, n_steps, lr, loss_mode,
                          device):
    """Run K gradient-ASCENT steps on the forgotten client's data.

    Standard loss.backward() + step() performs descent; we ascend by
    backpropagating the negated loss, which flips the gradient sign.
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    step = 0
    while step < n_steps:
        for x, _ in forgotten_loader:
            if step >= n_steps:
                break
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = reconstruction_loss(recon, x, mode=loss_mode)
            (-loss).backward()
            opt.step()
            step += 1
            if step % 5 == 0:
                print(f"  ascent step {step}/{n_steps}, "
                      f"recon_loss={loss.item():.4f}")
    return model


def run_gradient_ascent_unlearning(config, processed_dir, output_root,
                                   forgotten_client, base_model_path):
    """Full gradient-ascent unlearning.

    Args:
        config: full config dict.
        processed_dir: path to processed/ root.
        output_root: e.g. ./outputs/.
        forgotten_client: int.
        base_model_path: path to the pre-unlearning model (.pt).

    Returns:
        path to the unlearned model (.pt with {"parameters": [...]}).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(config["experiment"]["seed"])
    ga = config["unlearning"]["gradient_ascent"]
    m = config["model"]

    output_dir = Path(output_root) / f"unlearn_grad_ascent_forgot_{forgotten_client}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the pre-unlearning model.
    print(f"\n[grad_ascent] Loading base model from {base_model_path}")
    ck = torch.load(base_model_path, weights_only=False)
    model = AE2D(m["in_channels"], m["base_channels"],
                 m["latent_channels"]).to(device)
    set_parameters(model, [p.numpy() for p in ck["parameters"]])

    # 2. Forgotten client's training data (healthy slices).
    forgotten_loader = get_client_loader(
        Path(processed_dir) / f"client_{forgotten_client}",
        split="train", mode="healthy",
        batch_size=config["training"]["local_batch_size"],
        shuffle=True,
    )

    # 3. Ascent phase.
    print(f"\n[grad_ascent] Phase 1: gradient ascent on client {forgotten_client}")
    print(f"  steps={ga['ascent_steps']}, lr={ga['ascent_lr']}")
    model = gradient_ascent_phase(
        model, forgotten_loader,
        n_steps=ga["ascent_steps"], lr=ga["ascent_lr"],
        loss_mode=m["loss"], device=device,
    )

    # Persist the post-ascent model for diagnostics, and as the init for
    # the recovery fine-tune.
    post_ascent = output_dir / "post_ascent_model.pt"
    torch.save({"parameters": [torch.tensor(a) for a in get_parameters(model)]},
               post_ascent)

    # 4. Recovery fine-tune on remaining clients via FedAvg.
    print(f"\n[grad_ascent] Phase 2: fine-tune on remaining clients "
          f"({ga['finetune_rounds']} rounds)")
    final_path, _ = run_federation(
        config=config,
        processed_dir=processed_dir,
        output_dir=output_dir / "finetune",
        excluded_clients={forgotten_client},
        num_rounds=ga["finetune_rounds"],
        init_model_path=post_ascent,
    )

    final_ckpt = torch.load(final_path, weights_only=False)
    out_path = output_dir / "final_unlearned_model.pt"
    torch.save(final_ckpt, out_path)
    print(f"\n[grad_ascent] Final unlearned model saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--base_model_path", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_gradient_ascent_unlearning(
        config,
        config["data"]["processed_dir"],
        config["experiment"]["output_dir"],
        args.forgotten_client,
        args.base_model_path,
    )
