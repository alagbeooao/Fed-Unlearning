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

Three things in it need explanation, and all three are labelled in place.

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

### One cell missing from the ablation aggregate

The push-weight ablation was run by a batch whose job list covers five of the
six cells. Seed 123, client 2 is absent from it, so `results_seed123_forget2.zip`
was never written and the aggregation cell in `01_experiments_as_run.ipynb`
reports five cells, labelling its client 2 summary as two seeds. That cell's
ablation was run separately and its archive was not kept.

Table 2 of the paper is a three-seed mean in both regimes. Tables 1 and 3 are
unaffected: they cover all six cells and three seeds in both regimes, and are
harvested from the full `outputs_seed*_forget*.zip` archives.
