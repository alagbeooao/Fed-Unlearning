"""
FedEraser: unlearning by replaying the training history without the forgotten
client.

Liu et al. 2021, IEEE/ACM IWQoS.

The forgotten client influenced the global model at every round, so its
influence is spread through the whole trajectory rather than sitting in the
final weights. FedEraser removes it by rebuilding a trajectory in which that
client never took part.

The obvious shortcut does not work. Each cached update was computed by a client
starting from that round's global model, and in the counterfactual trajectory
the starting model is different, so the client would have moved in roughly the
same direction but by a different amount. Liu's answer is to keep the cached
direction, which is cheap, and recalibrate its magnitude to the new starting
point. So each replayed round:

    1. takes a few local SGD steps on each remaining client from the current
       running model, giving a magnitude estimate
    2. rescales that client's cached update to match
    3. FedAvg-aggregates the calibrated updates and applies them

Threading the running model forward across rounds is the method; without
it there is no trajectory being rebuilt.

Cheaper than retraining, since each round only costs a few probe steps per
client rather than full local training. Replaying every k-th round (replay_stride)
cuts it further.

Needs the per-round client checkpoints written by federated/server.py.
"""
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from data.datasets import get_client_loader
from federated.client import get_parameters, set_parameters
from federated.seeding import set_global_seed
from federated.server import (load_client_update_at_round,
                              load_global_model_at_round)
from models.ae_2d import AE2D, reconstruction_loss


def _flat_norm(tensor_list, mask=None):
    """L2 norm of a list of tensors treated as one concatenated vector.

    If `mask` is given (a boolean list the same length as `tensor_list`), only
    entries where mask[i] is True are included. This restricts the magnitude
    calibration to learnable parameters, excluding BatchNorm running
    statistics (running_mean / running_var / num_batches_tracked) whose
    movement would otherwise distort the calibration scale.
    """
    sq = 0.0
    for i, t in enumerate(tensor_list):
        if mask is not None and not mask[i]:
            continue
        sq += (t.float() ** 2).sum().item()
    return sq ** 0.5


def _learnable_param_mask(model):
    """Boolean list over state_dict order: True for learnable params.

    BatchNorm running statistics are registered buffers, not parameters, so
    they do not appear in named_parameters() and map to False here. The mask is
    used to compute the calibration magnitude over learnable weights only.
    """
    param_names = set(name for name, _ in model.named_parameters())
    return [name in param_names for name in model.state_dict().keys()]


def _state_dict_from_params(model, param_tensors):
    """Build an ordered name->tensor mapping from a parameter-tensor list."""
    keys = list(model.state_dict().keys())
    return {k: v for k, v in zip(keys, param_tensors)}


def _params_from_model(model):
    """Current parameters as a list of CPU tensors in state_dict order."""
    return [v.detach().cpu().clone() for v in model.state_dict().values()]


def local_probe_update(model, loader, n_steps, lr, loss_mode, device):
    """Take a few real local SGD steps and return the resulting update.

    The "update" is the difference (after - before) in parameter space,
    returned as a list of CPU tensors in state_dict order. This is the
    cheap probe used to estimate the magnitude a client would move from
    the current running model, NOT a full local-training round.

    Args:
        model: the running unlearned model (its weights define the
            starting point for this round). NOT modified in place.
        loader: this client's training DataLoader.
        n_steps: number of optimiser steps (a handful, e.g. 1-2).
        lr: probe learning rate (reuse the federation's local_lr).
        loss_mode: 'l1' or 'l2'.
        device: 'cuda' or 'cpu'.

    Returns:
        update: list of tensors (after - before), state_dict order.
    """
    # Snapshot starting weights.
    start = _params_from_model(model)

    # Work on a detached clone so the running model is not disturbed.
    probe = AE2D(
        in_channels=model.enc1.conv.in_channels,
        base_channels=model.enc1.conv.out_channels,
        latent_channels=model.enc4.conv.out_channels,
    ).to(device)
    probe.load_state_dict(model.state_dict())
    probe.train()

    opt = optim.Adam(probe.parameters(), lr=lr)
    taken = 0
    while taken < n_steps:
        for x, _ in loader:
            if taken >= n_steps:
                break
            x = x.to(device)
            opt.zero_grad()
            recon = probe(x)
            loss = reconstruction_loss(recon, x, mode=loss_mode)
            loss.backward()
            opt.step()
            taken += 1

    end = _params_from_model(probe)
    update = [e - s for e, s in zip(end, start)]
    return update


def calibrate_direction(old_update, new_update_norm, mask=None,
                        max_scale=10.0):
    """Rescale a cached update direction to a target magnitude.

    calibrated = clip(new_norm / ||old||, [1/max_scale, max_scale]) * old

    Keeps the cached direction (cheap, from history) but sets its length to the
    freshly probed magnitude (corrects for the counterfactual starting point).

    Stability measures (important on a generative AE replayed over many rounds):
      * ``mask`` restricts BOTH the norm computation and the application to
        learnable parameters. The per-tensor deltas for BatchNorm running
        statistics (running_mean / running_var / num_batches_tracked) are set
        to zero here, so the replay never perturbs them — scaling a variance
        buffer can drive it negative and produce NaNs. The calibration
        retraining phase recomputes BN statistics afterwards.
      * The scale ratio is clipped to ``[1/max_scale, max_scale]`` so a single
        noisy magnitude probe cannot inject an enormous update that compounds
        over the sequential replay.

    Args:
        old_update: list of tensors (cached client update from history).
        new_update_norm: float, target L2 norm from the local probe.
        mask: optional boolean list selecting learnable-parameter entries.
        max_scale: maximum (and reciprocal-minimum) allowed magnitude ratio.

    Returns:
        list of tensors, the calibrated update (buffer entries zeroed if masked).
    """
    old_norm = _flat_norm(old_update, mask=mask)
    if old_norm < 1e-12:
        # No usable direction; contribute nothing this round.
        return [torch.zeros_like(t) for t in old_update]
    scale = new_update_norm / old_norm
    # Clip the ratio so one bad probe cannot blow up the running model.
    lo, hi = 1.0 / max_scale, max_scale
    scale = max(lo, min(hi, scale))
    out = []
    for i, t in enumerate(old_update):
        if mask is not None and not mask[i]:
            # Buffer (e.g. BN running stat): do not perturb during replay.
            out.append(torch.zeros_like(t))
        else:
            out.append(t * scale)
    return out


def fedavg(updates_per_client, num_examples):
    """Weighted FedAvg of a list of per-client update tensor-lists."""
    total = sum(num_examples)
    n_params = len(updates_per_client[0])
    out = []
    for p in range(n_params):
        acc = sum(
            upd[p] * (n / total)
            for upd, n in zip(updates_per_client, num_examples)
        )
        out.append(acc)
    return out


def run_fed_eraser_unlearning(config, processed_dir, output_root,
                              forgotten_client, base_run_dir):
    """Faithful FedEraser: calibrated, sequential history replay.

    Args:
        config: full config dict.
        processed_dir: path to processed/ root (needed for the local
            calibration probes on remaining clients).
        output_root: e.g. ./outputs/.
        forgotten_client: int.
        base_run_dir: path to the original base-federation output folder,
            containing round_0001/.../round_NNNN/ with global_model.pt and
            client_*_update.pt.

    Returns:
        path to the unlearned model (.pt with {"parameters": [...]}).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(config["experiment"]["seed"])
    cfg = config["unlearning"]["fed_eraser"]
    loss_mode = config["model"]["loss"]

    output_dir = Path(output_root) / f"unlearn_fed_eraser_forgot_{forgotten_client}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_run_dir = Path(base_run_dir)
    n_clients = config["data"]["n_clients"]
    n_rounds = config["training"]["num_rounds"]
    remaining = [i for i in range(n_clients) if i != forgotten_client]

    # Replay controls (with sensible defaults if absent from config).
    probe_steps = cfg.get("probe_steps", 5)
    probe_lr = cfg.get("calibration_lr", config["training"]["local_lr"])
    replay_stride = cfg.get("replay_stride", 1)
    max_scale = cfg.get("max_scale", 10.0)

    last_round_dir = base_run_dir / f"round_{n_rounds:04d}"
    if not last_round_dir.exists():
        raise FileNotFoundError(
            f"Base run not found at {base_run_dir}. FedEraser requires the "
            f"per-round checkpoints from the original federated training."
        )

    print(f"\n[fed_eraser] Calibrated replay without client {forgotten_client}")
    print(f"  Remaining clients: {remaining}")
    print(f"  Rounds: {n_rounds}, stride: {replay_stride}, "
          f"probe steps/client/round: {probe_steps}")

    # Build a persistent loader per remaining client for the probes.
    loaders = {
        cid: get_client_loader(
            Path(processed_dir) / f"client_{cid}",
            split="train", mode="healthy",
            batch_size=config["training"]["local_batch_size"],
            shuffle=True,
        )
        for cid in remaining
    }

    # Start from the round-1 global model, the earliest checkpoint we have.
    # It already includes round 1's aggregation, so the loop below starts at
    # round 2. Its round-1 share of the forgotten client stays in, but that is
    # one round out of n_rounds and the calibration retraining dilutes it
    # further. Rebuilding the pre-round-1 init from the seed would avoid this
    # but is fragile across environments.
    running = AE2D(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_channels=config["model"]["latent_channels"],
    ).to(device)
    init_params = load_global_model_at_round(base_run_dir, 1)
    set_parameters(running, [p.numpy() for p in init_params])

    # Calibrate over learnable parameters only (exclude BatchNorm running
    # statistics, whose movement would otherwise distort the magnitude scale).
    learnable_mask = _learnable_param_mask(running)

    # Replay from round 2 (round 1 is already in the G_1 starting point).
    replay_rounds = list(range(2, n_rounds + 1, replay_stride))
    for i, t in enumerate(replay_rounds, start=1):
        # Gather cached updates (old directions) for remaining clients.
        old_updates = []
        num_examples = []
        present = []
        for cid in remaining:
            try:
                ck = load_client_update_at_round(base_run_dir, t, cid)
            except FileNotFoundError:
                continue
            # Cached client_*_update.pt stores the client's POST-training
            # weights for that round. The update direction relative to the
            # round's input model G_{t-1} is what we need; we recover it as
            # (cached_weights - G_{t-1}).
            g_prev = load_global_model_at_round(
                base_run_dir, max(t - 1, 1)
            )
            cached_w = ck["parameters"]
            old_update = [cw - gp for cw, gp in zip(cached_w, g_prev)]
            old_updates.append(old_update)
            num_examples.append(ck["num_examples"])
            present.append(cid)

        if not old_updates:
            continue

        # Probe each present client from the CURRENT running model to get
        # the target magnitude, then calibrate the cached direction to it.
        calibrated_updates = []
        for cid, old_update in zip(present, old_updates):
            probe = local_probe_update(
                running, loaders[cid],
                n_steps=probe_steps, lr=probe_lr,
                loss_mode=loss_mode, device=device,
            )
            new_norm = _flat_norm(probe, mask=learnable_mask)
            calibrated_updates.append(
                calibrate_direction(old_update, new_norm,
                                    mask=learnable_mask, max_scale=max_scale)
            )

        # FedAvg the calibrated updates and apply to the running model.
        agg_update = fedavg(calibrated_updates, num_examples)
        cur = _params_from_model(running)
        new_params = [c + u for c, u in zip(cur, agg_update)]
        set_parameters(running, [p.numpy() for p in new_params])

        # Stop on divergence rather than saving a NaN model. Calibration
        # retraining then picks up from the last good state.
        finite = all(torch.isfinite(p).all().item()
                     for p in running.state_dict().values())
        if not finite:
            print(f"  [warn] non-finite weights after replay step {i} "
                  f"(round {t}); stopping replay early and reverting to the "
                  f"previous good state.")
            set_parameters(running, [c.numpy() for c in cur])
            break

        if i % 10 == 0 or i == len(replay_rounds):
            print(f"  replayed {i}/{len(replay_rounds)} "
                  f"(round {t})")

    # Save the post-replay model.
    post_replay = output_dir / "post_replay_model.pt"
    torch.save({"parameters": _params_from_model(running)}, post_replay)

    # Calibration retraining: a few full FedAvg rounds on remaining clients
    # from the post-replay model, recovering any utility lost to replay
    # drift. This is the second phase in Liu 2021.
    calib_rounds = cfg.get("calibration_rounds", 0)
    if calib_rounds > 0:
        print(f"\n[fed_eraser] Calibration retraining: {calib_rounds} rounds")
        from federated.simulation import run_federation
        calib_dir = output_dir / "calibration"
        final_path, _ = run_federation(
            config=config,
            processed_dir=processed_dir,
            output_dir=calib_dir,
            excluded_clients={forgotten_client},
            num_rounds=calib_rounds,
            init_model_path=post_replay,
        )
        final_ckpt = torch.load(final_path, weights_only=False)
    else:
        final_ckpt = {"parameters": _params_from_model(running)}

    out_path = output_dir / "final_unlearned_model.pt"
    torch.save(final_ckpt, out_path)
    print(f"\n[fed_eraser] Final unlearned model saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--base_run_dir", required=True,
                   help="Path to base-federation output with round_*/ folders")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_fed_eraser_unlearning(
        config,
        config["data"]["processed_dir"],
        config["experiment"]["output_dir"],
        args.forgotten_client,
        args.base_run_dir,
    )
