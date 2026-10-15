"""
PyTorch datasets over the preprocessed BraTS slices.

Each preprocessed slice is a .npz with keys: slice, mask, is_healthy.
This module provides:

- BratsSliceDataset: load slices for a given client/split, optionally
  filtered to healthy-only (for training the AE) or tumour-only (for
  anomaly detection eval)
- get_client_loader: convenience to build a DataLoader for a client
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BratsSliceDataset(Dataset):
    """Loads preprocessed 2D slices.

    Args:
        client_dir: path to processed/client_{i}/
        split: 'train', 'val', or 'test'
        mode: 'healthy' (only healthy slices, for AE training),
              'tumour' (only tumour slices, for anomaly eval),
              'all' (everything)
    """

    def __init__(self, client_dir, split="train", mode="healthy"):
        self.split_dir = Path(client_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        all_files = sorted(self.split_dir.glob("*.npz"))
        if mode == "healthy":
            self.files = [f for f in all_files if "_healthy.npz" in f.name]
        elif mode == "tumour":
            self.files = [f for f in all_files if "_tumour.npz" in f.name]
        elif mode == "all":
            self.files = all_files
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if len(self.files) == 0:
            print(f"WARNING: No files found for {client_dir}/{split}/{mode}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        sl = data["slice"].astype(np.float32)  # (H, W)
        mask = data["mask"].astype(np.float32)  # (H, W) binary
        # Add channel dim: (1, H, W)
        sl = sl[np.newaxis, :, :]
        mask = mask[np.newaxis, :, :]
        return torch.from_numpy(sl), torch.from_numpy(mask)


def get_client_loader(client_dir, split, mode, batch_size, shuffle=True,
                      num_workers=2, seed=None):
    """Build a DataLoader for one client/split/mode.

    Args:
        seed: if given, shuffling becomes reproducible. A torch.Generator is
            seeded with it and passed to the DataLoader, which is what drives
            RandomSampler's permutation. Workers are seeded from it too. Leave
            as None for the previous (nondeterministic) behaviour.

    The returned loader exposes `.generator`, so a caller can re-seed between
    rounds (see federated/client.py) to get per-round variation that is still
    reproducible run to run.
    """
    ds = BratsSliceDataset(client_dir, split=split, mode=mode)
    gen = None
    winit = None
    if seed is not None:
        from federated.seeding import loader_worker_init
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        winit = loader_worker_init
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=(shuffle and len(ds) > batch_size),
                      generator=gen, worker_init_fn=winit)
