# Development notes

Notes I kept while building and debugging this codebase. Most of it concerns
things I got wrong the first time and had to go back and fix, several of which
changed the results. I am keeping it in the repository because a few of the
fixes matter for interpreting the numbers in the paper, and because anyone
comparing this code against the original method papers will hit the same
questions I did.

Entries are roughly in the order I worked through them.

---

## Rewriting the unlearning methods

Three problems in my first implementation were serious enough to invalidate
results, so I rewrote the affected methods and re-ran everything.

### Knowledge distillation was not actually federated

My first version trained the student centrally, pooling the remaining clients'
data into one process. That is a much easier problem than the one the oracle
solves. The oracle is trained federated, with data kept local, so a centralised
student gets an information advantage the oracle never has, and any comparison
between them is measuring centralisation as much as unlearning. It also quietly
breaks the premise of the whole setup, which is that client data does not move.

The student is now a federated model. Every remaining client computes the
match and push objective on its own data, and the server FedAvg-aggregates the
student each round. This changed the KD numbers substantially.

### FedEraser was not doing a sequential replay

Two things were wrong. The replay loop overwrote its running model every round
instead of threading it forward, so there was no reconstruction of a trajectory
at all. And the calibration step compared two nearly identical full weight
vectors, which meant its scale factor came out at roughly 1.0 every time and
did nothing.

What was left was effectively "average the remaining clients' last-round
updates". Any variance I measured from that was noise from a broken
implementation rather than a property of the method, so I threw those results
away.

The replay now threads a running model through the retained rounds. At each
round a short local probe from the current model estimates how far each
remaining client would move, and the cached update direction is rescaled to that
magnitude before aggregation.

### The released config would not reproduce the paper

The config I first pushed carried the published classifier-domain
hyperparameters, which diverge on an autoencoder. Anyone cloning the repository
would have reproduced the divergence rather than my results. The config now has
the values I actually used, with a note next to each rescaled one explaining why
it was rescaled.

---

## FedEraser stability

The first real run produced NaN weights. Tracing it back: replaying 99 rounds
from an under-trained starting checkpoint compounds magnitude errors until the
weights blow up, and scaling the BatchNorm variance buffers along with
everything else can push a variance negative, which is enough on its own.

Four changes stopped the NaNs:

- Normalisation buffers are no longer perturbed during replay. Their deltas are
  zeroed and the calibration retraining recomputes them afterwards.
- The per-round magnitude ratio is clipped to [1/10, 10], so one noisy probe
  cannot inject a huge update that then compounds.
- `probe_steps` went from 1 to 5. A single Adam step, with its moment estimates
  cold, is too noisy a magnitude estimate on a high-loss model.
- A guard stops the replay and reverts to the last finite state if the running
  model goes non-finite, rather than silently saving a NaN model.

After all of that FedEraser is finite but still fails, which is the point: the
failure survives four different attempts to stabilise it, so it is a property of
the method here and not something I mis-tuned.

I also fixed a smaller thing while in there. The running model is initialised to
the round-1 global model, but the loop was also replaying round 1, applying it
twice. The loop now starts at round 2.

Calibration magnitudes are computed over learnable parameters only. Including
the BatchNorm running statistics in that norm lets buffer movement distort the
scale.

---

## Seeding and reproducibility

The code had no seeding at all initially, which made the three-seed comparison
meaningless as a controlled experiment. `federated/seeding.py` now provides
`set_global_seed`, called at the start of every stochastic stage from
`config["experiment"]["seed"]`.

Later I went further, because I hit a case where two configurations that were
mathematically identical produced results differing by about 0.0007 in
Wasserstein distance, which was the same size as an effect I wanted to claim.
Three things were responsible:

The dataloaders were unseeded. They now take a `torch.Generator` seeded from
`derive_seed(run_seed, stream, client_id, round)`, with a `worker_init_fn` that
seeds numpy and `random` inside each worker process. `derive_seed` uses crc32
rather than Python's `hash()`, because `hash()` is randomised per process and
would give different values inside every Ray worker.

Ray returns client results in completion order, and FedAvg was summing them in
whatever order they arrived. Floating point addition is not associative, so the
aggregate differed slightly between runs and the difference compounded.
`aggregate_fit` and `aggregate_evaluate` now sort by client id first.

Determinism was only being enabled in the driver process, which is not where
training happens. `enable_determinism()` is now called inside the clients too,
before anything reaches the GPU, and it sets `CUBLAS_WORKSPACE_CONFIG` at module
import so Ray workers inherit it.

Shuffling still varies across rounds, since the round index goes into the derived
seed. Training keeps its stochasticity; the sequence is just reproducible.

The runs reported in the paper predate this work, so they are statistically but
not bitwise reproducible. Rather than assume the variation away I measured it and
reported effect sizes against it.

---

## Scaling the distillation push weight

This was an attempt to fix the KD failure rather than only diagnose it. The idea
was to scale the push weight by how much the teacher had actually memorised the
forgotten client, measured from the teacher's own reconstruction error, so that
the push would be strong where there was something to remove and weak where there
was not. `unlearning/push_scaling.py` implements several variants of that signal.

It did not work out, for a reason worth recording.

The estimator itself had a bug. `_mean_recon_error` was taking the first N
batches with `shuffle=False`, which for a large client means the first quarter of
the cases in file order rather than a sample. The held-out side, being smaller,
was covered in full. So the two ends of the generalisation gap were measured on
non-comparable data, and the same client and seed gave opposite signs depending
on the batch cap. The full split is now evaluated by default, any cap draws a
seeded random subsample, and means are weighted by slice count.

The deeper problem is not fixable in code. Measuring a generalisation gap needs
enough held-out data, and the small clients in this partition hold about 32 cases
each, which leaves roughly five held-out cases. When I changed the held-out split
from test to validation, four of six clients reversed their verdict. One client
went from a scale of 1.000 to 0.141 and another from 0.000 to 1.000. Only the
dominant client, with around a thousand held-out slices, gave a stable estimate.

So there are two usable measurement points on this dataset, which is not enough
to demonstrate that anything adapts to client size. I added a guard that refuses
to push when the held-out set is below `min_heldout_slices`, on the grounds that
an unreliable signal is worse than none, and dropped the adaptivity claim from
the paper.

The module is still in the repository. It is not part of the paper's method, and
Table 2's `alpha_p = 0` arm should be reproduced by setting `alpha_push: 0.0`
directly. What survives from this work is the direct ablation, which does not
depend on the signal at all: turning the push off improves fidelity by 63% and
82% in the two regimes, several times the measured noise floor.

---

## Smaller fixes

**The autoencoder was mislabelled.** I had been calling it Baur's dense variant.
Baur's dense AE has a fully-connected low-dimensional bottleneck; this model is
fully convolutional with a 256x8x8 bottleneck, which is his spatial variant. The
architecture never changed, only the description.

**Preprocessing pointed at the wrong partitioning file.** The docstring and CLI
examples said `partitioning_2.csv`. Everything was run with `partitioning_1.csv`.

**The unlearning manifest overwrote instead of merging.** Running a single method
on its own discarded the paths recorded for the others, which then broke
evaluation. It merges now, so methods can be run separately.

**The KD client was holding a DataLoader in a closure.** The forgotten client's
loader was built once and captured in `client_fn`, so Ray serialised it to every
worker. The client now takes a path and builds its own loader, which is what the
base federation already did.

**Documentation.** The repository started life as a set of course-style notes and
still read that way. The guide is now a description of the architecture and the
design decisions, and the README is written for someone trying to reproduce the
paper.
