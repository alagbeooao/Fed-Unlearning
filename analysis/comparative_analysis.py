"""Rebuild the paper's tables from the three-seed results.

The per-run numbers are held as literals below, transcribed from the
evaluation_results_forgot_*.json files in results/. Those JSONs are the
primary record; the literals are here so the tables can be rebuilt without
them present.

Figures are not produced here. analysis/make_paper_figures.py draws the three
figures exactly as they appear in the paper.

Run: python analysis/comparative_analysis.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# RAW RESULTS  (seed -> method -> dict of metrics)
# metrics: auroc_rem, dice_rem, w_oracle, ks, mia
# ----------------------------------------------------------------------
def row(a, d, w, ks, mia):
    return {"auroc": a, "dice": d, "w": w, "ks": ks, "mia": mia}

CLIENT0 = {
 42:   {"pre":row(0.8951,0.2697,0.0042,0.4017,0.5021),"oracle":row(0.8463,0.1602,0,0,0.4794),
        "ga":row(0.8627,0.1890,0.0032,0.3180,0.4983),"kd":row(0.9526,0.5051,0.0047,0.3703,0.4927),
        "fe":row(0.8370,0.1538,0.0093,0.5683,0.4731)},
 123:  {"pre":row(0.8608,0.1900,0.0043,0.3912,0.4989),"oracle":row(0.8510,0.1693,0,0,0.4754),
        "ga":row(0.8711,0.2091,0.0036,0.3403,0.4961),"kd":row(0.9530,0.5075,0.0076,0.4798,0.5038),
        "fe":row(0.8475,0.1665,0.0128,0.6576,0.4735)},
 2024: {"pre":row(0.8716,0.2091,0.0030,0.2992,0.4976),"oracle":row(0.8578,0.1772,0,0,0.4774),
        "ga":row(0.8542,0.1725,0.0028,0.2873,0.4958),"kd":row(0.9081,0.3311,0.0003,0.0509,0.4903),
        "fe":row(0.8357,0.1563,0.0147,0.7476,0.4733)},
}
CLIENT2 = {
 42:   {"pre":row(0.8729,0.2138,0.0011,0.3455,0.2938),"oracle":row(0.8628,0.1840,0,0,0.2926),
        "ga":row(0.8602,0.1791,0.0007,0.2182,0.2907),"kd":row(0.9338,0.3984,0.0040,0.7273,0.2940),
        "fe":row(0.8719,0.2143,0.0014,0.4000,0.2867)},
 123:  {"pre":row(0.8756,0.2188,0.0004,0.1636,0.2998),"oracle":row(0.8682,0.1990,0,0,0.2921),
        "ga":row(0.8545,0.1707,0.0002,0.1091,0.2937),"kd":row(0.9502,0.4700,0.0118,1.0000,0.3133),
        "fe":row(0.8553,0.1772,0.0001,0.0909,0.2902)},
 2024: {"pre":row(0.8852,0.2430,0.0006,0.2182,0.2979),"oracle":row(0.8890,0.2568,0,0,0.2997),
        "ga":row(0.8830,0.2384,0.0003,0.1818,0.2943),"kd":row(0.9138,0.3224,0.0050,0.7818,0.2761),
        "fe":row(0.8744,0.2173,0.0007,0.2182,0.2870)},
}
SEEDS = (42, 123, 2024)
METHODS = ["pre", "oracle", "ga", "kd", "fe"]
NAMES = {"pre":"Pre-unlearning","oracle":"Oracle","ga":"Grad. ascent","kd":"KD","fe":"FedEraser"}


def ms(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def agg(data, method, key):
    return ms([data[s][method][key] for s in SEEDS])


def w_improvement(data, method):
    imps = [100*(data[s]["pre"]["w"] - data[s][method]["w"]) / data[s]["pre"]["w"]
            for s in SEEDS]
    return imps, ms(imps)


# ----------------------------------------------------------------------
# CONSOLE TABLES
# ----------------------------------------------------------------------
def print_table(data, title):
    print(f"\n=== {title} (mean ± std, 3 seeds) ===")
    print(f"{'method':<15}{'AUROC_rem':>16}{'Dice_rem':>16}{'W_oracle':>17}{'MIA':>16}")
    for m in METHODS:
        cells = []
        for key in ("auroc", "dice", "w", "mia"):
            mean, sd = agg(data, m, key)
            cells.append(f"{mean:.4f}±{sd:.4f}")
        print(f"{NAMES[m]:<15}{cells[0]:>16}{cells[1]:>16}{cells[2]:>17}{cells[3]:>16}")


print("="*70)
print("COMPARATIVE RESULTS: regime dependence of federated unlearning")
print("="*70)
print_table(CLIENT0, "CLIENT 0 — dominant forgotten client (50% of data)")
print_table(CLIENT2, "CLIENT 2 — minor forgotten client (4% of data)")

print("\n--- Wasserstein improvement vs pre-unlearning baseline ---")
for tag, data in [("CLIENT 0", CLIENT0), ("CLIENT 2", CLIENT2)]:
    print(f"\n{tag}:")
    for m in ("ga", "kd", "fe"):
        imps, (mean, sd) = w_improvement(data, m)
        print(f"  {NAMES[m]:<13}{[f'{x:+.0f}%' for x in imps]}  "
              f"mean {mean:+.1f}% ± {sd:.1f}%")

print("\n--- KD utility advantage over oracle (AUROC_rem) ---")
for tag, data in [("CLIENT 0", CLIENT0), ("CLIENT 2", CLIENT2)]:
    diffs = [data[s]["kd"]["auroc"] - data[s]["oracle"]["auroc"] for s in SEEDS]
    mean, sd = ms(diffs)
    print(f"  {tag}: {[f'{x:+.3f}' for x in diffs]}  mean {mean:+.3f} ± {sd:.3f}")

print("\n--- KD fidelity to oracle (KS distance, lower=closer) ---")
for tag, data in [("CLIENT 0", CLIENT0), ("CLIENT 2", CLIENT2)]:
    ks = [data[s]["kd"]["ks"] for s in SEEDS]
    mean, sd = ms(ks)
    print(f"  {tag}: {[f'{x:.3f}' for x in ks]}  mean {mean:.3f} ± {sd:.3f}")

# ----------------------------------------------------------------------
# Save the aggregated numbers as JSON for the paper
summary = {}
for tag, data in [("client0", CLIENT0), ("client2", CLIENT2)]:
    summary[tag] = {}
    for m in METHODS:
        summary[tag][m] = {k: list(agg(data, m, k)) for k in ("auroc", "dice", "w", "ks", "mia")}
        if m in ("ga", "kd", "fe"):
            _, wi = w_improvement(data, m)
            summary[tag][m]["w_improvement_pct"] = list(wi)
json.dump(summary, open(OUT/"aggregated_results.json", "w"), indent=2)

print("\n" + "="*70)
print(f"aggregated_results.json written to {OUT}/")
print("="*70)
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
