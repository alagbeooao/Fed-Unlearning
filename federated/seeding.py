"""
Global seeding utility for reproducible experiments.

Every stochastic entry point (federated training, the unlearning methods)
calls set_global_seed() with the seed from config["experiment"]["seed"] at
the start, so that a given seed produces a repeatable run: model
initialisation, data shuffling, and optimiser noise all derive from the
same seeded RNGs.

Note on federated simulation: model initialisation and the server-side RNG
are seeded in the driver process. Client-side DataLoader shuffling runs in
Ray worker processes; for fully deterministic client shuffling one would
also seed per-worker, but the dominant sources of run-to-run variation
(initialisation and aggregation order) are controlled here, which is what
the multi-seed comparison in the paper relies on.
"""
import os
import random
import zlib

# Must be set before the first CUDA call anywhere in the process, so we do it
# at import time rather than inside a function. Ray workers inherit it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def derive_seed(*parts) -> int:
    """Deterministically combine integers/strings into a 32-bit seed.

    Python's built-in hash() is randomised per process (PYTHONHASHSEED), so it
    cannot be used for cross-process reproducibility. zlib.crc32 is stable
    across processes and runs, which is what Ray workers need.
    """
    payload = "|".join(str(x) for x in parts).encode()
    return zlib.crc32(payload) & 0x7FFFFFFF


def loader_worker_init(worker_id: int):
    """Seed RNGs inside a DataLoader worker process.

    PyTorch gives each worker a distinct base_seed derived from the DataLoader's
    generator; we propagate it to numpy and python so any worker-side randomness
    is reproducible too.
    """
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))
    random.seed((base + worker_id) % (2 ** 32))


def enable_determinism(strict: bool = True):
    """Force deterministic kernels wherever PyTorch supports it.

    Three separate switches are needed:
      * cuDNN algorithm selection (conv/convtranspose autotuning).
      * cuBLAS workspace configuration, without CUBLAS_WORKSPACE_CONFIG set
        before the first CUDA call, some matmul reductions are nondeterministic.
      * torch.use_deterministic_algorithms, which routes remaining ops to
        deterministic implementations. warn_only=True so an op lacking a
        deterministic kernel degrades to a warning rather than crashing a run.

    Note this is necessary but not sufficient on its own: aggregation order
    across federated clients must also be fixed (see federated/server.py).
    """
    if not strict:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:  # older torch versions
        print(f"[seeding] use_deterministic_algorithms unavailable: {e}")


def set_global_seed(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
