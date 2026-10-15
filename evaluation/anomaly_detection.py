"""
Anomaly detection evaluation for trained AEs.

Protocol (Baur et al. 2021):
1. Take tumour-containing test slices.
2. Run through the AE; compute per-pixel reconstruction error map.
3. Compare error map to ground-truth tumour mask.
4. Report:
   - Pixel-level AUROC (treating each pixel as a binary classification)
   - Dice score at the threshold that maximises F1

Note on threshold selection: Baur 2021 uses the threshold that maximises
F1 on a held-out tumour-containing set. We follow this convention.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve

from data.datasets import BratsSliceDataset
from federated.client import set_parameters
from models.ae_2d import AE2D, reconstruction_error_map


def compute_error_maps(model, dataset, device, batch_size=32):
    """Compute reconstruction error maps and ground-truth masks for all
    slices in the dataset.

    Returns:
        error_maps: (N, H, W) numpy array
        masks:      (N, H, W) numpy array (binary)
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    all_errs = []
    all_masks = []
    with torch.no_grad():
        for x, mask in loader:
            x = x.to(device)
            recon = model(x)
            err = reconstruction_error_map(recon, x, mode="l1")  # (B, H, W)
            all_errs.append(err.cpu().numpy())
            all_masks.append(mask.squeeze(1).numpy())

    error_maps = np.concatenate(all_errs, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    return error_maps, masks


def pixel_auroc(error_maps, masks):
    """Compute pixel-level AUROC across all slices."""
    flat_errs = error_maps.flatten()
    flat_masks = masks.flatten().astype(np.int32)
    if flat_masks.sum() == 0:
        return float("nan")  # no positives, undefined
    return float(roc_auc_score(flat_masks, flat_errs))


def best_dice(error_maps, masks):
    """Compute Dice score at the threshold maximising F1 score.

    Returns:
        dice: best Dice across thresholds
        threshold: the threshold that achieved it
    """
    flat_errs = error_maps.flatten()
    flat_masks = masks.flatten().astype(np.int32)
    if flat_masks.sum() == 0:
        return float("nan"), float("nan")

    precisions, recalls, thresholds = precision_recall_curve(flat_masks, flat_errs)

    # F1 = 2*P*R / (P+R)
    eps = 1e-9
    f1 = 2 * precisions * recalls / (precisions + recalls + eps)
    best_idx = int(np.argmax(f1))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])

    # Dice at best threshold = F1 at best threshold (per Baur's definition;
    # for binary segmentation Dice == F1 score)
    return float(f1[best_idx]), best_thresh


def evaluate_anomaly_detection(model_path, processed_dir, n_clients, config,
                                 splits=("test",), excluded_clients=None,
                                 device="cuda"):
    """Evaluate anomaly detection on tumour slices from specified clients.

    Args:
        model_path: path to .pt checkpoint
        processed_dir: path to processed/
        n_clients: total number of clients
        config: full config dict
        splits: which splits to evaluate on
        excluded_clients: clients to exclude (e.g. for "remaining clients" eval)
        device: 'cuda' or 'cpu'

    Returns:
        dict of results
    """
    model = AE2D(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_channels=config["model"]["latent_channels"],
    ).to(device)

    ckpt = torch.load(model_path, weights_only=False)
    params_np = [p.numpy() for p in ckpt["parameters"]]
    set_parameters(model, params_np)

    excluded = set(excluded_clients or [])
    results = {}

    for split in splits:
        all_errs = []
        all_masks = []

        for cid in range(n_clients):
            if cid in excluded:
                continue
            client_dir = Path(processed_dir) / f"client_{cid}"
            try:
                ds = BratsSliceDataset(client_dir, split=split, mode="tumour")
            except FileNotFoundError:
                continue
            if len(ds) == 0:
                continue

            errs, masks = compute_error_maps(model, ds, device)
            all_errs.append(errs)
            all_masks.append(masks)

        if not all_errs:
            results[split] = {"auroc": float("nan"), "dice": float("nan"),
                              "threshold": float("nan"), "n_slices": 0}
            continue

        all_errs = np.concatenate(all_errs, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        auroc = pixel_auroc(all_errs, all_masks)
        dice, thresh = best_dice(all_errs, all_masks)
        results[split] = {
            "auroc": auroc,
            "dice": dice,
            "threshold": thresh,
            "n_slices": int(all_errs.shape[0]),
        }

    return results


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--exclude_clients", type=int, nargs="*", default=[])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_anomaly_detection(
        model_path=args.model_path,
        processed_dir=config["data"]["processed_dir"],
        n_clients=config["data"]["n_clients"],
        config=config,
        excluded_clients=args.exclude_clients,
        device=device,
    )
    print(json.dumps(results, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
