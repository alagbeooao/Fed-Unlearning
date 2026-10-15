# Utility Is Not Compliance

Code for the paper *"Utility Is Not Compliance: Verifying Federated Unlearning for
Autoencoder-Based Anomaly Detection"* (O. Alagbe), ACISM 2026.

This repository evaluates three classifier-derived federated unlearning (FU) methods on federated
2D convolutional autoencoders for unsupervised anomaly detection in multi-site brain
MRI, across two forgotten-client regimes and three random seeds. The paper's argument
is that FU evaluation should separate **utility** (downstream detection performance)
from **fidelity** (distributional closeness to an oracle retrained without the
forgotten client), because only the second speaks to regulatory compliance, and
because the two can be anti-correlated.

Headline results: gradient ascent is the only method whose fidelity improves
reliably; knowledge distillation beats the oracle on utility while its fidelity
collapses as the forgotten client shrinks; FedEraser does not transfer at all; and
reconstruction-loss membership inference is uninformative here, since it reports
chance-level attack success on the *pre-unlearning* model.

---

## Repository layout

```
configs/base.yaml                 all hyperparameters (the values used in the paper)
data/preprocess_fets.py           FeTS 2022 -> 2D slices (Partitioning 1)
data/datasets.py                  PyTorch datasets over the preprocessed slices
models/ae_2d.py                   2D convolutional autoencoder (Baur spatial variant)
federated/client.py               Flower client (local AE training)
federated/server.py               FedAvg with per-round checkpointing
federated/simulation.py           run_federation(...) entry point
federated/seeding.py              seeding and determinism utilities
unlearning/retrain_oracle.py      oracle: retrain without the forgotten client
unlearning/gradient_ascent.py     gradient-ascent unlearning (Halimi et al. 2022)
unlearning/kd_unlearn.py          federated knowledge distillation (Wu et al. 2022)
unlearning/fed_eraser.py          FedEraser calibrated replay (Liu et al. 2021)
unlearning/push_scaling.py        exploratory; NOT used for the paper (see note below)
evaluation/anomaly_detection.py   pixel AUROC / Dice (Baur 2021 protocol)
evaluation/unlearning_efficacy.py Wasserstein and KS distance to the oracle
evaluation/membership_inference.py loss-threshold membership-inference attack
scripts/                          runnable pipeline stages
tests/test_methods.py             13 CPU-only correctness tests
results/                          aggregated three-seed results (per-run JSONs are written here by a run)
analysis/comparative_analysis.py  rebuilds the paper's tables
analysis/make_paper_figures.py    rebuilds Figures 1-3 exactly as published
notebooks/                        the runs as executed, and a reproduction notebook
```

---

## Data

The experiments use the **MICCAI FeTS 2022** training set under **Partitioning 1** of
BraTS 2021. The data is not redistributed here; obtain it from the official challenge,
which requires registration. Retaining the six largest institutional partitions gives
clients of 357, 267, 32, 24, 24 and 23 cases.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is strongly recommended. The experiments were run on a single A100/L4; a base
federation of 100 rounds takes roughly 90 minutes.

---

## Reproducing the paper

### 0. Verify the implementations (no GPU, no data, seconds)

```bash
python -m tests.test_methods
```

Thirteen tests covering FedAvg weighting, the FedEraser magnitude calibration
(rescaling, direction preservation, clipping, exclusion of normalisation buffers),
the KD push clamp, and seed determinism. All should pass.

### 1. Preprocess

```bash
python data/preprocess_fets.py \
  --raw_dir /path/to/MICCAI_FeTS2022_TrainingData \
  --partition_csv /path/to/partitioning_1.csv \
  --out_dir ./data/processed \
  --modality flair --slice_size 128 --n_clients 6 \
  --min_brain_fraction 0.20 --seed 42
```

The preprocessing seed fixes the case-wise 70/15/15 split and stays at 42 for every
run; only the *experiment* seed varies. Expected healthy/tumour training slice counts
are 7057/18249, 5250/14160, 365/1669, 388/1254, 345/1419 and 378/1249 for clients 0-5.

### 2. One full experiment

Set `experiment.seed` in `configs/base.yaml`, then:

```bash
# base federation (the pre-unlearning model)
python scripts/run_base_federation.py --config configs/base.yaml

# oracle: retrain without the forgotten client
python scripts/run_oracle.py --config configs/base.yaml --forgotten_client 0

# the three unlearning methods
python scripts/run_unlearning.py \
  --config configs/base.yaml --forgotten_client 0 \
  --base_run_dir ./outputs/base_federation

# evaluation (utility, fidelity, membership inference)
python scripts/run_evaluation.py \
  --config configs/base.yaml --forgotten_client 0 \
  --manifest ./outputs/manifest_forgot_0.json
```

This writes `outputs/evaluation_results_forgot_0.json`.

### 3. Which command produces which result

| Paper | How to reproduce |
|---|---|
| **Table 1** (both regimes, 3 seeds) | Step 2 for `--forgotten_client 0` and `2`, seeds 42/123/2024 |
| **Table 2** (push-weight ablation) | Step 2 twice with `alpha_push: 0.05` then `alpha_push: 0.0` in `configs/base.yaml`, writing to different `output_dir`s |
| **Table 3** (per-seed appendix) | the individual JSONs from Table 1's runs |
| **Fig. 1** (utility vs fidelity) | `python analysis/make_paper_figures.py` |
| **Fig. 2** (AUROC by regime) | `python analysis/make_paper_figures.py` |
| **Fig. 3** (fidelity divergence) | `python analysis/make_paper_figures.py` |
| Post-replay diagnostic (FedEraser) | `outputs/unlearn_fed_eraser_forgot_*/post_replay_model.pt` |

Both analysis scripts run on CPU in seconds and need no data, so the tables and
figures can be checked without reproducing any training.

The three-seed results are currently held as literals at the top of
`analysis/comparative_analysis.py`, copied from the per-run
`evaluation_results_forgot_*.json` files the runs produced and summarised in
`results/aggregated_results.json`. Those per-run JSONs are the primary record; the
literals exist so the analysis runs without them present.

---

## Notebooks

Two notebooks, with different jobs.

| File | Role | Run it? |
|---|---|---|
| `notebooks/01_experiments_as_run.ipynb` | the working notebook from the experiments, with its original outputs | no |
| `notebooks/02_reproduction.ipynb` | rebuilds the tables and figures from the archived outputs | yes |

### 01_experiments_as_run.ipynb

The record of the runs reported in the paper, executed 19 June 2026 and
extended on 3 to 4 August 2026. It is published with its outputs intact,
because those outputs are the evidence of what each run produced. It is not
meant to be executed: re-running it would overwrite that evidence, and one
cell's archive no longer exists so it could not be reproduced.

Every number in Tables 1 and 3 comes from this notebook. Each of the six
(seed, forgotten client) runs prints its own results table, so any value in the
paper can be traced to a specific execution.

Four things in it need explanation, and all four are labelled in place.

**A superseded run.** One cell is labelled `SEED = 123` but reused the base
federation already on disk from the seed 42 run, because the guard checked only
whether a base model existed and not which seed produced it. Its output
repeats the seed 42 values, pre-unlearning 0.8951 and oracle 0.8463. The run
was discarded, the guard was replaced with per-seed output directories, and
the seed 123 numbers in the paper come from the seed-isolated run further down,
which reports pre-unlearning 0.8608. The superseded cell is kept because the
cell after it is the check that caught the problem.

**Two base federations for seed 42.** The seed 42 client 0 run trained its base
into a shared directory; the later seed-isolated runs wrote to per-seed
directories. Seed 42 therefore has two base federations, and because per-worker
shuffling was unseeded they are different models. The client 0 regime was
evaluated against the first, the client 2 regime against the second. Each cell
is internally consistent, and every comparison the paper makes is within a
regime. Re-evaluating client 0 against the second base gives a pre-unlearning
AUROC of 0.858 rather than 0.895, which `02_reproduction.ipynb` reports and
explains.

**A lost archive.** `outputs_seed42_forget0.zip` was written at the end of that
run and later found corrupt, so it was deleted. The oracle was recovered
separately as `oracle_seed42_forgot0.zip`. The run itself survives only as the
printed output in this notebook.

**One cell missing from the ablation aggregate.** The push-weight ablation was
run by a batch whose job list covers five of the six cells. Seed 123, client 2
is absent from it, so no `results_seed123_forget2.zip` was written, and the
aggregation cell that reads those archives reports five cells and labels its
client 2 summary as two seeds. That cell's ablation was run separately and its
archive was not kept. Table 2 of the paper is a three-seed mean in both
regimes. Tables 1 and 3 are unaffected: they cover all six cells and three
seeds in both regimes, and are harvested from the full
`outputs_seed*_forget*.zip` archives rather than from the slim ones that cell
reads.

Cells that were empty, duplicated, never executed, or pure housekeeping have
been removed. No code cell was edited and no output was altered, so every cell
still matches the output printed beneath it.

### 02_reproduction.ipynb

Reads the archived run outputs, rebuilds Tables 1 and 3, redraws Figures 1 to
3, and checks every value against the paper. On the archives as they stand it
reports 103 of 120 values matching, 13 differing in the seed 42 client 0 cell
for the reason above, 4 absent because the August batch did not run FedEraser
for that cell, and none unexplained. It also restates the four findings as
tests; all four hold.

Sections 1 to 7 need no data and no GPU. Section 7 runs the thirteen unit
tests in a few seconds, which is the cheapest way to check the implementations
against the paper's description.

---

## Notes on the implementations

These are the points a reader comparing the code against the published methods will
want, and where this implementation departs from a naive reading of the original
papers.

**Knowledge distillation is federated.** Each remaining client computes the
match + push objective on its own local data and the server FedAvg-aggregates the
student every round. A centralised student pooling the remaining clients' data would
have an information advantage the oracle does not, which would confound distillation
quality with centralisation.

**FedEraser's replay is sequential.** A running model is threaded through the
retained rounds, and at each round every remaining client's cached update *direction*
is rescaled to a magnitude probed from the *current* running model. Reusing cached
updates as absolute weights does not implement Liu et al.'s method, since each was
computed relative to a different input model. The magnitude ratio is clipped to
[1/10, 10] and normalisation buffers are excluded from rescaling; both are required to
prevent divergence.

**Hyperparameters are rescaled to the reconstruction-error scale.** The published
classifier-domain settings diverge on an autoencoder (gradient ascent by ~16x, KD by
~10x). `configs/base.yaml` ships the values used in the paper, with the reason for
each rescaling documented inline.

**The autoencoder is the spatial variant**, fully convolutional with a 256x8x8
bottleneck, not the dense variant with a fully-connected bottleneck.

**On `unlearning/push_scaling.py`.** This module explores scaling the KD push weight
by a measured memorisation signal. It is retained for completeness but is **not part
of the paper's method**: the estimate proved unusable for clients with few held-out
cases (changing the held-out split reversed its sign for four of six clients), and the
paper makes no adaptivity claim. Table 2's `alpha_p = 0` arm should be reproduced by
setting `alpha_push: 0.0` directly, which is what the paper describes. In the original
runs that value was reached through this module; the `min_heldout_slices` guard and
the unbiased-sampling fix were added afterwards.

---

## Reproducibility

Seeds 42, 123 and 2024 are used throughout. Per-worker dataloader shuffling was
unseeded in the runs reported in the paper, so those runs are statistically but not
bitwise reproducible. Rather than assume that variation negligible, it was measured directly:
two mathematically identical configurations differ by 0.0007 in Wasserstein distance,
and the paper reports effect sizes against that floor and makes no claim at or below
it.

`federated/seeding.py` now supports full determinism (`experiment.deterministic:
true`) via seeded dataloader generators, per-worker seeding, client-id-sorted FedAvg
aggregation, and deterministic cuDNN. Results produced with determinism enabled will
not match the paper's numbers exactly, since the paper's runs predate it.

`CHANGELOG_corrections.md` documents defects found and fixed during development,
including three in an earlier version of this codebase and several found by the
experiments themselves. It is written as a development record; a reader interested
only in the released implementation can skip it.

---

## Citation

```bibtex
@inproceedings{alagbe2026utility,
  author    = {Alagbe, Oluwapelumi},
  title     = {Utility Is Not Compliance: Verifying Federated Unlearning
               for Autoencoder-Based Anomaly Detection},
  booktitle = {Proceedings of the International Conference on Artificial
               Intelligence, Computer Science and Management (ACISM 2026)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026}
}
```

## Licence

MIT (see `LICENSE`). The FeTS 2022 data is subject to its own terms and is not
distributed here.
