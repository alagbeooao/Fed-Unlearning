# Project guide: architecture and design decisions

Operational companion to `README.md`. Documents the pipeline, the data
contracts between stages, and the design decisions behind the
implementation. For how to *run* the pipeline, see the README.

## Pipeline overview

```
preprocess_fets.py
        │   writes processed/client_{i}/{train,val,test}/*.npz
        ▼
run_base_federation.py ── federated/simulation.run_federation
        │   writes outputs/base_federation/round_*/{global_model,client_*_update}.pt
        ▼
run_oracle.py ── unlearning/retrain_oracle
        │   writes outputs/oracle_forgot_{c}/round_*/global_model.pt
        ▼
run_unlearning.py ── unlearning/{gradient_ascent,kd_unlearn,fed_eraser}
        │   writes outputs/unlearn_{method}_forgot_{c}/final_unlearned_model.pt
        │   merges  outputs/manifest_forgot_{c}.json
        ▼
run_evaluation.py ── evaluation/{anomaly_detection,unlearning_efficacy,membership_inference}
            writes outputs/evaluation_results_forgot_{c}.json
```

## Data contracts

These are the formats stages rely on; keep them stable when modifying code.

**Preprocessed slice** — one `.npz` per slice with keys `slice` (H×W
float), `mask` (H×W binary), `is_healthy` (bool). Filenames end in
`_healthy.npz` or `_tumour.npz` so datasets can filter by mode.

**Checkpoint** — `torch.save({"parameters": [tensor, ...], "round": int,
"metrics": {...}}, path)`. `parameters` is the model's `state_dict()`
values in order (the Flower convention). `set_parameters` zips these back
onto `state_dict().keys()`.

**Per-round artifacts** — under `round_{t:04d}/`: `global_model.pt`
(aggregated weights) and one `client_{i}_update.pt` per participating
client (`{"client_id", "num_examples", "parameters", "metrics"}`).
FedEraser consumes both.

**Manifest** — `{"forgotten_client", "base_model", "oracle_model",
"unlearned_models": {method: path}}`. `run_evaluation` reads it and builds
`{"pre_unlearning", "oracle", **unlearned_models}`. `run_unlearning` merges
into an existing manifest so methods can be run separately.

## Key design decisions

**Spatial (not dense) autoencoder.** The model is Baur's fully-convolutional
variant: a 256×8×8 spatial bottleneck, no dense layer. This is a strong
reconstruction-based localiser and avoids a fully-connected bottleneck's
parameter blow-up at 128×128.

**FedAvg with per-round checkpointing.** Checkpointing every round is what
makes FedEraser's calibrated replay possible without re-running the
federation. BatchNorm running statistics are aggregated along with weights
(the simple FedAvg choice); FedBN argues against this under non-IID, which
is a reasonable alternative to explore.

**Federated KD.** The student is a federated model trained over the
remaining clients (local match+push objective, FedAvg each round), not a
centrally-pooled student. This keeps client data local and makes the
comparison with the federated oracle fair. The push divergence is capped
at `tau_push`, an active cap at the AE's error scale.

**FedEraser calibration.** Each replayed round threads the running
unlearned model forward: a cheap local probe (a few SGD steps per
remaining client from the current model) estimates the magnitude the
client would move, and the cached update *direction* is rescaled to that
magnitude before aggregation. The sequential dependence across rounds is
essential — using cached updates as absolute weights would be incoherent,
because each was computed from a different input model.

**Hyperparameter calibration to the AE error scale.** The published
classifier-domain settings for gradient ascent and KD overshoot on a
generative AE (reconstruction errors live on a much smaller scale than
classifier logits). `configs/base.yaml` documents the rescaled values used
in the paper inline.

## Evaluation metrics

- **Anomaly detection** (utility): per-pixel reconstruction error scored
  against tumour masks → pixel AUROC and best-F1 Dice, on tumour test
  slices, separately for remaining and forgotten clients (Baur 2021).
- **Unlearning efficacy**: 1-Wasserstein and KS distance between the
  unlearned and oracle per-slice reconstruction-error distributions on the
  forgotten client's held-out healthy slices. Smaller = closer to oracle.
- **Membership inference**: loss-threshold attack with the forgotten
  client's train slices as members and its test slices as non-members;
  attack AUROC near 0.5 means members are not identifiable.

## Reproducibility notes

- Seeds 42, 123, 2024 are the paper's three seeds; set
  `experiment.seed` per run. `set_global_seed` (in `federated/seeding.py`) is
  called at the start of every stochastic stage, so a given seed reproduces
  model initialisation and aggregation order exactly.
- Determinism is now full when `experiment.deterministic: true` (the default):
  DataLoader shuffling is driven by a `torch.Generator` seeded from
  `derive_seed(run_seed, stream, client_id, round)`, workers are seeded via
  `worker_init_fn`, and cuDNN uses deterministic algorithms. Identical configs
  therefore produce identical results. `derive_seed` uses crc32 rather than
  `hash()` because Python randomises `hash()` per process, which would break
  reproducibility inside Ray workers.
- Shuffling still varies *across rounds* (the round index enters the derived
  seed), so training keeps its stochasticity while remaining reproducible.
- Set `deterministic: false` to recover throughput if you do not need
  bit-level reproducibility.
- Colab/transient environments: persist `outputs/` and `processed/` to
  durable storage between sessions, since the federation checkpoints are
  required by FedEraser and by evaluation.
