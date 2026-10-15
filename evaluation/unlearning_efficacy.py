"""
Unlearning efficacy evaluation: how close is an unlearned model to the
oracle (retrain-from-scratch-without-forgotten-client) model?

The metric is computed on RECONSTRUCTION ERROR DISTRIBUTIONS over the
forgotten client's healthy data:

1. Run the unlearned model on forgotten client's held-out healthy slices.
   Compute mean reconstruction error per slice. This gives a 1D distribution.
2. Do the same with the oracle model.
3. Compute distance between the two distributions:
   - Wasserstein-1 (earth mover's): scale-aware, interpretable in loss units
   - KS statistic: scale-free, max distance between empirical CDFs

A SUCCESSFUL unlearning method produces a distribution close to the
oracle's. Smaller distance = better unlearning.

Sanity checks:
- Distance(oracle, oracle) = 0 (definition)
- Distance(do_nothing, oracle) is large (since pre-unlearning model
  was trained ON forgotten client and has lower reconstruction error
  on its data than oracle)
"""
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance, ks_2samp

from data.datasets import BratsSliceDataset
from federated.client import set_parameters
from models.ae_2d import AE2D, reconstruction_error_map


def per_slice_recon_error(model, dataset, device, batch_size=32):
    """Mean reconstruction error per slice. Returns 1D numpy array."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    per_slice = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            err_map = reconstruction_error_map(recon, x, mode="l1")  # (B, H, W)
            per_slice.append(err_map.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(per_slice, axis=0) if per_slice else np.array([])


def load_model(model_path, config, device):
    model = AE2D(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_channels=config["model"]["latent_channels"],
    ).to(device)
    ckpt = torch.load(model_path, weights_only=False)
    params_np = [p.numpy() for p in ckpt["parameters"]]
    set_parameters(model, params_np)
    return model


def compute_efficacy(unlearned_model_path, oracle_model_path,
                     processed_dir, forgotten_client, config,
                     split="val", device="cuda"):
    """Compute Wasserstein and KS distance between unlearned and oracle
    reconstruction-error distributions on forgotten client's data.

    Args:
        unlearned_model_path: path to .pt for the model under evaluation
        oracle_model_path: path to .pt for oracle (retrain w/o F)
        processed_dir: path to processed/
        forgotten_client: int
        config: full config
        split: which split of forgotten client's data to evaluate on.
               'val' is recommended — these are slices the model never
               saw at any point.
        device: 'cuda' or 'cpu'

    Returns:
        dict with wasserstein, ks_statistic, ks_pvalue, and the underlying
        distribution arrays.
    """
    unlearned = load_model(unlearned_model_path, config, device)
    oracle = load_model(oracle_model_path, config, device)

    forgotten_dir = Path(processed_dir) / f"client_{forgotten_client}"
    forgotten_ds = BratsSliceDataset(forgotten_dir, split=split, mode="healthy")
    if len(forgotten_ds) == 0:
        return {"error": f"No healthy slices for client {forgotten_client}/{split}"}

    errs_unlearned = per_slice_recon_error(unlearned, forgotten_ds, device)
    errs_oracle = per_slice_recon_error(oracle, forgotten_ds, device)

    w = float(wasserstein_distance(errs_unlearned, errs_oracle))
    ks_stat, ks_p = ks_2samp(errs_unlearned, errs_oracle)

    return {
        "wasserstein": w,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "n_slices": len(errs_unlearned),
        "unlearned_mean": float(np.mean(errs_unlearned)),
        "unlearned_std": float(np.std(errs_unlearned)),
        "oracle_mean": float(np.mean(errs_oracle)),
        "oracle_std": float(np.std(errs_oracle)),
        # Raw arrays for plotting (caller can save or discard)
        "errs_unlearned": errs_unlearned.tolist(),
        "errs_oracle": errs_oracle.tolist(),
    }


def compare_all_methods(method_paths, oracle_path, processed_dir,
                         forgotten_client, config, device="cuda"):
    """Compute efficacy for multiple unlearning methods at once.

    Args:
        method_paths: dict {method_name: path_to_unlearned_model.pt}
        oracle_path: path to oracle .pt
        ...

    Returns:
        dict {method_name: efficacy_dict}
    """
    results = {}
    for name, path in method_paths.items():
        print(f"  Computing efficacy for {name}...")
        results[name] = compute_efficacy(
            unlearned_model_path=path,
            oracle_model_path=oracle_path,
            processed_dir=processed_dir,
            forgotten_client=forgotten_client,
            config=config,
            device=device,
        )
    return results


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--unlearned_path", required=True)
    p.add_argument("--oracle_path", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = compute_efficacy(
        unlearned_model_path=args.unlearned_path,
        oracle_model_path=args.oracle_path,
        processed_dir=config["data"]["processed_dir"],
        forgotten_client=args.forgotten_client,
        config=config,
        device=device,
    )
    # Drop raw arrays from print
    summary = {k: v for k, v in res.items()
               if k not in ("errs_unlearned", "errs_oracle")}
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
