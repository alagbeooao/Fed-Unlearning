"""
Scaling the KD push weight by how much the teacher memorised the forgotten
client.

Fixed-weight KD over-pushes on small clients. The push term rewards divergence
from the teacher whether or not the teacher was a reasonable answer to begin
with, so when the federation barely fitted a client the student gets dragged
away from a model it was already close to.

The idea here was to scale the weight by a memorisation estimate:

    alpha_push_effective = alpha_push_base * scale,   scale in [0, 1]

Modes, set via unlearning.kd_unlearn.push_scaling in the config:

    fixed              scale = 1, the published behaviour
    fraction           scale by the client's share of the training data
    teacher_signal     teacher's error on this client vs the others
    teacher_gap        teacher's error on this client's train vs held-out data
    oracle_divergence  actual teacher-to-oracle distance; needs the oracle, so
                       not deployable, included only as a ceiling

The signal exploits the fact that an autoencoder reconstructs well what it has
learned. Unusually low error on one client suggests the model fitted that
client specifically, so there is something to remove. Non-positive signal means
there is not, and the scale floors at zero.

This did not survive validation and is not used in the paper. Estimating a
generalisation gap needs a reasonable held-out set, and most clients in this
partition hold about 32 cases. Changing the held-out split reversed the verdict
for four of six clients. min_heldout_slices guards against trusting those
estimates, but the underlying limitation is the data, not the code. See the
development notes.
"""
from pathlib import Path

import numpy as np
import torch

from data.datasets import get_client_loader
from federated.client import set_parameters
from models.ae_2d import AE2D, reconstruction_loss


def memorisation_signal(e_forgotten, e_remaining):
    """Relative memorisation advantage of the teacher on the forgotten client.

        s = (e_remaining - e_forgotten) / e_remaining

    s > 0 means the teacher reconstructs the forgotten client's data
    *better* than it reconstructs comparable retained data, i.e. it has
    fitted that client specifically. s <= 0 means it has not.
    """
    return (e_remaining - e_forgotten) / max(e_remaining, 1e-12)


def signal_to_scale(s, s_ref):
    """Map a memorisation signal to a push scale in [0, 1].

    Linear up to s_ref, saturating at 1, floored at 0 so a teacher that
    fits the forgotten client *worse* than average never pushes at all.
    """
    if s_ref <= 0:
        raise ValueError("s_ref must be positive")
    return float(np.clip(s / s_ref, 0.0, 1.0))


@torch.no_grad()
def _mean_recon_error(model, client_dir, config, split="train",
                      max_batches=None, device="cuda", seed=0):
    """Mean per-slice reconstruction error of `model` on a client's data.

    Returns (mean_error, n_slices).

    Evaluates the whole split by default. Capping batches with shuffle=False
    would take the first N in case order, which is a biased subset, and since
    the training split is much larger than the held-out one the two ends of the
    gap end up measured on non-comparable data. If a cap is given, batches are
    drawn with a seeded shuffle instead.
    """
    shuffle = max_batches is not None
    loader = get_client_loader(
        client_dir, split=split, mode="healthy",
        batch_size=config["training"]["local_batch_size"],
        shuffle=shuffle,
        seed=(seed if shuffle else None),
    )
    mode = config["model"]["loss"]
    model.eval()
    total, n_slices = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        # weight by slice count so partial final batches do not skew the mean
        total += reconstruction_loss(model(x), x, mode=mode).item() * x.size(0)
        n_slices += x.size(0)
    return (total / max(n_slices, 1)), n_slices


@torch.no_grad()
def _per_slice_errors(model, client_dir, config, split="val",
                      max_batches=None, device="cuda", seed=0):
    """Per-slice reconstruction errors (1-D array) for distributional comparison.

    As with _mean_recon_error, the full split is used by default; a cap draws a
    seeded random subsample rather than the first N batches in file order.
    """
    shuffle = max_batches is not None
    loader = get_client_loader(
        client_dir, split=split, mode="healthy",
        batch_size=config["training"]["local_batch_size"],
        shuffle=shuffle, seed=(seed if shuffle else None),
    )
    mode = config["model"]["loss"]
    model.eval()
    errs = []
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        recon = model(x)
        diff = (recon - x).abs() if mode == "l1" else (recon - x) ** 2
        errs.extend(diff.mean(dim=(1, 2, 3)).cpu().numpy().tolist())
    return np.asarray(errs)


def _free(*models):
    """Release scaling-time models before the Flower simulation claims the GPU."""
    for m in models:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build(params_np, config, device):
    m = config["model"]
    net = AE2D(m["in_channels"], m["base_channels"], m["latent_channels"]).to(device)
    set_parameters(net, params_np)
    net.eval()
    return net


def _client_train_sizes(processed_dir, n_clients, config):
    """Number of healthy training slices per client."""
    sizes = {}
    for cid in range(n_clients):
        loader = get_client_loader(
            Path(processed_dir) / f"client_{cid}", split="train", mode="healthy",
            batch_size=config["training"]["local_batch_size"], shuffle=False,
        )
        sizes[cid] = len(loader.dataset)
    return sizes


def compute_push_scale(mode, config, processed_dir, forgotten_client,
                       teacher_params_np, oracle_params_np=None,
                       device="cuda"):
    """Compute the multiplicative scale applied to alpha_push.

    Args:
        mode: 'fixed' | 'fraction' | 'teacher_signal' | 'oracle_divergence'.
        config: full config dict.
        processed_dir: root of processed client folders.
        forgotten_client: int.
        teacher_params_np: pre-unlearning weights (list of numpy arrays).
        oracle_params_np: oracle weights; required for 'oracle_divergence'.
        device: 'cuda' or 'cpu'.

    Returns:
        (scale, diagnostics_dict). scale is a float in [0, 1].
    """
    kd = config["unlearning"]["kd_unlearn"]
    n_clients = config["data"]["n_clients"]
    remaining = [c for c in range(n_clients) if c != forgotten_client]
    diag = {"mode": mode, "forgotten_client": forgotten_client}

    if mode == "fixed":
        diag["scale"] = 1.0
        return 1.0, diag

    if mode == "fraction":
        frac_ref = kd.get("scale_fraction_ref", 0.5)
        sizes = _client_train_sizes(processed_dir, n_clients, config)
        total = sum(sizes.values())
        frac = sizes[forgotten_client] / max(total, 1)
        scale = float(np.clip(frac / frac_ref, 0.0, 1.0))
        diag.update({"fraction": frac, "frac_ref": frac_ref, "scale": scale})
        return scale, diag

    if mode == "teacher_signal":
        s_ref = kd.get("scale_signal_ref", 0.10)
        teacher = _build(teacher_params_np, config, device)

        e_forgot, n_forgot = _mean_recon_error(
            teacher, Path(processed_dir) / f"client_{forgotten_client}",
            config, device=device)

        # Sample-weighted mean error across the remaining clients, so a tiny
        # remaining client does not dominate the reference.
        sizes = _client_train_sizes(processed_dir, n_clients, config)
        num, den = 0.0, 0
        per_client = {}
        for cid in remaining:
            e, _n = _mean_recon_error(
                teacher, Path(processed_dir) / f"client_{cid}",
                config, device=device)
            per_client[cid] = e
            num += e * sizes[cid]
            den += sizes[cid]
        e_remain = num / max(den, 1)

        # Relative memorisation advantage. Positive => the teacher fits the
        # forgotten client better than it fits comparable retained data.
        s = memorisation_signal(e_forgot, e_remain)
        scale = signal_to_scale(s, s_ref)
        diag.update({
            "e_forgotten": e_forgot, "e_remaining": e_remain,
            "memorisation_signal": s, "s_ref": s_ref,
            "per_client_error": per_client, "scale": scale,
        })
        _free(teacher)
        return scale, diag

    if mode == "teacher_gap":
        # Within-client generalisation gap. Controls for intrinsic client
        # difficulty, which the cross-client 'teacher_signal' does not: a
        # client whose images are simply easier to reconstruct would show a
        # low cross-client error without having been memorised. Here we
        # compare the teacher on the forgotten client's TRAIN data against
        # its own HELD-OUT data, so intrinsic difficulty cancels.
        #
        #   s = (e_heldout - e_train) / e_heldout
        #
        # A teacher that fits a client's training slices much better than
        # that same client's unseen slices has memorised them.
        s_ref = kd.get("scale_gap_ref", 0.10)
        teacher = _build(teacher_params_np, config, device)
        fdir = Path(processed_dir) / f"client_{forgotten_client}"
        e_train, n_train = _mean_recon_error(teacher, fdir, config,
                                             split="train", device=device)
        held = kd.get("scale_heldout_split", "test")
        e_held, n_held = _mean_recon_error(teacher, fdir, config, split=held,
                                           device=device)
        s = memorisation_signal(e_train, e_held)
        scale = signal_to_scale(s, s_ref)

        # The gap is a difference of two means, so the smaller held-out sample
        # dominates its error. With only a handful of held-out cases the number
        # reflects which cases landed in the split rather than memorisation: on
        # this dataset four of six clients flipped when the split changed from
        # test to val. Below the threshold, don't push at all. An unreliable
        # signal is worse than none, since pushing with nothing to remove is
        # what wrecks fidelity in the first place.
        min_slices = kd.get("min_heldout_slices", 200)
        reliable = n_held >= min_slices
        if not reliable:
            print(f"    [push_scaling] WARNING: held-out set has only {n_held} "
                  f"slices (< {min_slices}); gap estimate is unreliable. "
                  f"Falling back to scale=0 (no push).")
            scale = 0.0
        diag.update({"e_train": e_train, "e_heldout": e_held,
                     "n_train": n_train, "n_heldout": n_held,
                     "heldout_split": held, "memorisation_signal": s,
                     "s_ref": s_ref, "reliable": reliable,
                     "min_heldout_slices": min_slices,
                     "scale_before_guard": signal_to_scale(s, s_ref),
                     "scale": scale})
        _free(teacher)
        return scale, diag

    if mode == "oracle_divergence":
        if oracle_params_np is None:
            raise ValueError("oracle_divergence mode requires oracle_params_np")
        from scipy.stats import wasserstein_distance

        d_ref = kd.get("scale_divergence_ref", 0.004)
        teacher = _build(teacher_params_np, config, device)
        oracle = _build(oracle_params_np, config, device)
        fdir = Path(processed_dir) / f"client_{forgotten_client}"
        held = kd.get("scale_heldout_split", "test")
        et = _per_slice_errors(teacher, fdir, config, split=held, device=device)
        eo = _per_slice_errors(oracle, fdir, config, split=held, device=device)
        d = float(wasserstein_distance(et, eo))
        scale = float(np.clip(d / d_ref, 0.0, 1.0))
        diag.update({"teacher_oracle_w1": d, "d_ref": d_ref,
                     "heldout_split": held, "scale": scale})
        _free(teacher, oracle)
        return scale, diag

    raise ValueError(f"Unknown push_scaling mode: {mode}")
