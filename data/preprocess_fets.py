"""
FeTS 2022 preprocessing for site-aware federated learning.

This is the preprocessing pipeline for the MICCAI FeTS 2022 dataset
(NOT BraTS 2023/2024). FeTS ships explicit institution partitioning,
which gives us a real, defensible federated split — exactly what the
research question requires.

Procedure:
1. Read partitioning_1.csv to map each case to a partition (institution)
2. Select the top N partitions by case count -> these become federated clients
3. Drop cases not in the selected top-N partitions
4. Per case: load FLAIR (or other modality), z-score normalize, extract
   2D axial slices, save with healthy/tumour tag
5. Write per-client splits with case-level (not slice-level) train/val/test

Usage:
    python data/preprocess_fets.py \\
        --raw_dir /path/to/MICCAI_FeTS2022_TrainingData \\
        --partition_csv /path/to/MICCAI_FeTS2022_TrainingData/partitioning_1.csv \\
        --out_dir /path/to/processed \\
        --modality flair \\
        --slice_size 128 \\
        --n_clients 6
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True,
                   help="Path to MICCAI_FeTS2022_TrainingData root")
    p.add_argument("--partition_csv", required=True,
                   help="Path to partitioning_1.csv (institutional split used in the paper)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--modality", default="flair",
                   choices=["t1", "t1ce", "t2", "flair"])
    p.add_argument("--slice_size", type=int, default=128)
    p.add_argument("--n_clients", type=int, default=6,
                   help="Use top-N partitions by case count")
    p.add_argument("--min_brain_fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_partitioning(csv_path):
    """Read FeTS partitioning CSV. Returns dict {subject_id: partition_id}.

    The CSV format is:
        Subject_ID,Partition_ID
        FeTS2022_01341,1
        FeTS2022_01333,1
        ...
    """
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["Subject_ID"].strip()
            pid = int(row["Partition_ID"])
            mapping[sid] = pid
    return mapping


def select_top_n_partitions(case_to_partition, n):
    """Select the top-N partitions by case count.

    Returns:
        selected_partitions: list of partition IDs in size order (largest first)
        partition_to_client: dict mapping selected partition ID -> client index 0..N-1
    """
    counts = defaultdict(int)
    for pid in case_to_partition.values():
        counts[pid] += 1
    sorted_pids = sorted(counts.keys(), key=lambda p: -counts[p])
    selected = sorted_pids[:n]
    partition_to_client = {pid: idx for idx, pid in enumerate(selected)}

    print(f"Selected top {n} partitions by case count:")
    for client_idx, pid in enumerate(selected):
        print(f"  Client {client_idx} <- Partition {pid}: {counts[pid]} cases")

    return selected, partition_to_client


def load_fets_case(case_dir, modality):
    """Load a FeTS case. Files are named:
        FeTS2022_NNNNN_{modality}.nii  (uncompressed)
        FeTS2022_NNNNN_seg.nii

    Returns (volume, segmentation) numpy arrays, or (None, None) if missing.
    """
    case_id = case_dir.name
    # Try uncompressed first
    vol_path = case_dir / f"{case_id}_{modality}.nii"
    seg_path = case_dir / f"{case_id}_seg.nii"
    # Fall back to compressed if needed
    if not vol_path.exists():
        vol_path = case_dir / f"{case_id}_{modality}.nii.gz"
        seg_path = case_dir / f"{case_id}_seg.nii.gz"

    if not vol_path.exists():
        return None, None

    vol = nib.load(str(vol_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata() if seg_path.exists() else np.zeros_like(vol)
    return vol, seg


def normalize_volume(vol, brain_mask):
    """Z-score normalize within the brain region only."""
    brain_voxels = vol[brain_mask > 0]
    if brain_voxels.size == 0:
        return vol
    mean = brain_voxels.mean()
    std = brain_voxels.std() + 1e-8
    out = (vol - mean) / std
    out[brain_mask == 0] = 0
    return out


def extract_slices(vol, seg, slice_size, min_brain_fraction):
    """Extract 2D axial slices, resize, and tag healthy vs tumour.

    Returns list of (slice_2d, mask_2d, is_healthy) tuples.
    """
    brain_mask = (vol != 0).astype(np.float32)
    vol_norm = normalize_volume(vol, brain_mask)

    slices = []
    n_axial = vol.shape[2]
    total_pixels = vol.shape[0] * vol.shape[1]

    for z in range(n_axial):
        sl = vol_norm[:, :, z]
        sl_seg = seg[:, :, z]
        sl_brain = brain_mask[:, :, z]

        if sl_brain.sum() / total_pixels < min_brain_fraction:
            continue

        sl_img = Image.fromarray(sl.astype(np.float32), mode="F").resize(
            (slice_size, slice_size), Image.BILINEAR
        )
        sl_seg_img = Image.fromarray(sl_seg.astype(np.float32), mode="F").resize(
            (slice_size, slice_size), Image.NEAREST
        )

        sl_arr = np.array(sl_img, dtype=np.float32)
        seg_arr = np.array(sl_seg_img, dtype=np.float32)
        is_healthy = (seg_arr > 0).sum() == 0
        mask_2d = (seg_arr > 0).astype(np.uint8)
        slices.append((sl_arr, mask_2d, is_healthy))

    return slices


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: read partitioning
    print(f"Reading partitioning from {args.partition_csv}")
    case_to_partition = read_partitioning(args.partition_csv)
    print(f"  {len(case_to_partition)} cases in partitioning CSV")

    # Step 2: select top-N partitions -> clients
    selected_pids, pid_to_client = select_top_n_partitions(
        case_to_partition, args.n_clients
    )

    # Step 3: build {client_idx: [case_dir, ...]}
    client_cases = defaultdict(list)
    skipped_no_folder = 0
    skipped_wrong_partition = 0

    for case_id, pid in case_to_partition.items():
        if pid not in pid_to_client:
            skipped_wrong_partition += 1
            continue
        case_dir = raw_dir / case_id
        if not case_dir.exists():
            skipped_no_folder += 1
            continue
        client_idx = pid_to_client[pid]
        client_cases[client_idx].append(case_dir)

    print(f"\nSkipped {skipped_wrong_partition} cases (not in selected partitions)")
    print(f"Skipped {skipped_no_folder} cases (folder not found in {raw_dir})")
    total_used = sum(len(v) for v in client_cases.values())
    print(f"Total cases retained: {total_used}")

    # Step 4: per-client preprocessing
    metadata = {
        "source_dataset": "MICCAI_FeTS2022",
        "partitioning_csv": str(args.partition_csv),
        "selected_partition_ids": selected_pids,
        "modality": args.modality,
        "slice_size": args.slice_size,
        "n_clients": args.n_clients,
        "clients": {}
    }
    splits = {}

    for client_idx in sorted(client_cases.keys()):
        cases = client_cases[client_idx]
        rng.shuffle(cases)
        n_total = len(cases)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        train_cases = cases[:n_train]
        val_cases = cases[n_train:n_train + n_val]
        test_cases = cases[n_train + n_val:]

        client_dir = out_dir / f"client_{client_idx}"

        for split_name, split_cases in [("train", train_cases),
                                          ("val", val_cases),
                                          ("test", test_cases)]:
            split_dir = client_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            n_healthy = 0
            n_tumour = 0
            for case_dir in tqdm(split_cases,
                                  desc=f"Client {client_idx} / {split_name}"):
                vol, seg = load_fets_case(case_dir, args.modality)
                if vol is None:
                    continue
                slices = extract_slices(
                    vol, seg, args.slice_size, args.min_brain_fraction
                )
                for i, (sl, mask, healthy) in enumerate(slices):
                    label = "healthy" if healthy else "tumour"
                    out_path = (split_dir /
                                f"{case_dir.name}_z{i:03d}_{label}.npz")
                    np.savez_compressed(out_path, slice=sl, mask=mask,
                                         is_healthy=healthy)
                    if healthy:
                        n_healthy += 1
                    else:
                        n_tumour += 1

            metadata["clients"].setdefault(str(client_idx), {})[split_name] = {
                "n_cases": len(split_cases),
                "n_healthy_slices": n_healthy,
                "n_tumour_slices": n_tumour,
                "partition_id": selected_pids[client_idx],
            }

        splits[str(client_idx)] = {
            "partition_id": selected_pids[client_idx],
            "train": [c.name for c in train_cases],
            "val": [c.name for c in val_cases],
            "test": [c.name for c in test_cases]
        }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. Metadata at {out_dir / 'metadata.json'}")
    print("\nVerify with notebooks/01_data_exploration.ipynb before proceeding.")


if __name__ == "__main__":
    main()
