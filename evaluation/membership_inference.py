"""
Membership-inference attack on the forgotten client.

The attacker sees per-slice reconstruction errors and tries to tell training
slices (members) from held-out ones (non-members). Attack AUROC near 0.5 means
they cannot.

This is a loss-threshold attack, the simplest kind. A shadow-model or
likelihood-ratio attack would be stronger. Worth reading the result with that in
mind, though in this setting even the pre-unlearning model sits at chance, which
suggests there is no membership signal to find rather than that the attack is
too weak.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.datasets import BratsSliceDataset
from evaluation.unlearning_efficacy import per_slice_recon_error, load_model


def loss_threshold_mia(member_losses, nonmember_losses):
    """Compute attack AUROC for a loss-threshold MIA.

    Members (in training set) tend to have LOWER loss => use -loss as
    the score for "is member". AUROC of 1.0 means perfect attack;
    AUROC of 0.5 means random.

    Args:
        member_losses: 1D array of per-sample losses for known members
        nonmember_losses: 1D array of per-sample losses for known non-members

    Returns:
        attack_auroc, summary stats
    """
    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        return float("nan"), {"error": "empty member or non-member set"}

    y_true = np.concatenate([
        np.ones(len(member_losses)),       # 1 = member
        np.zeros(len(nonmember_losses)),   # 0 = non-member
    ])
    # Lower loss = more likely member, so use NEGATIVE loss as score
    scores = np.concatenate([-member_losses, -nonmember_losses])

    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        return float("nan"), {"error": "AUROC undefined (need both classes)"}

    return float(auroc), {
        "n_members": len(member_losses),
        "n_nonmembers": len(nonmember_losses),
        "member_loss_mean": float(np.mean(member_losses)),
        "member_loss_std": float(np.std(member_losses)),
        "nonmember_loss_mean": float(np.mean(nonmember_losses)),
        "nonmember_loss_std": float(np.std(nonmember_losses)),
    }


def run_mia(model_path, processed_dir, forgotten_client, config,
            device="cuda"):
    """Run loss-threshold MIA on a model.

    Args:
        model_path: path to model .pt
        processed_dir: path to processed/
        forgotten_client: int — the client whose training data we
                          attempt to identify
        config: full config dict
        device: 'cuda' or 'cpu'

    Returns:
        dict with attack_auroc, plus underlying stats
    """
    model = load_model(model_path, config, device)
    forgotten_dir = Path(processed_dir) / f"client_{forgotten_client}"

    # Members: forgotten client's training slices (model trained on them
    # IF the model is the pre-unlearning model. For unlearned models, the
    # MEMBERSHIP relationship is what we're testing — was their influence
    # successfully removed?)
    members_ds = BratsSliceDataset(forgotten_dir, split="train", mode="healthy")
    # Non-members: forgotten client's test slices (model never saw these)
    nonmembers_ds = BratsSliceDataset(forgotten_dir, split="test", mode="healthy")

    if len(members_ds) == 0:
        return {"error": f"No training slices for client {forgotten_client}"}
    if len(nonmembers_ds) == 0:
        return {"error": f"No test slices for client {forgotten_client}"}

    member_losses = per_slice_recon_error(model, members_ds, device)
    nonmember_losses = per_slice_recon_error(model, nonmembers_ds, device)

    auroc, stats = loss_threshold_mia(member_losses, nonmember_losses)

    return {
        "attack_auroc": auroc,
        **stats,
    }


def run_mia_all_methods(method_paths, processed_dir, forgotten_client,
                          config, device="cuda"):
    """Run MIA across multiple models for direct comparison.

    Args:
        method_paths: dict {method_name: model_path.pt}, including
                      'pre_unlearning', 'oracle', and the unlearning
                      methods being evaluated

    Returns:
        dict {method_name: mia_result}
    """
    results = {}
    for name, path in method_paths.items():
        print(f"  MIA on {name}...")
        results[name] = run_mia(
            model_path=path,
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
    p.add_argument("--model_path", required=True)
    p.add_argument("--forgotten_client", type=int, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_mia(
        model_path=args.model_path,
        processed_dir=config["data"]["processed_dir"],
        forgotten_client=args.forgotten_client,
        config=config,
        device=device,
    )
    print(json.dumps(res, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
