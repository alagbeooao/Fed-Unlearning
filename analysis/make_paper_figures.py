"""Regenerate Figures 1 to 3 of the paper at LNCS column width.

The earlier versions were drawn on an 8-inch canvas and then scaled into a
4.8-inch column, which pushed the tick labels down to 2-4pt. Springer wants
nothing below 6pt. Drawing at the final width means the font sizes set here
are the sizes that reach the page. Everything is also distinguishable in
black and white: marker shape, hatch and line style all carry the series,
not colour alone.
"""
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

HERE = Path(__file__).resolve().parent

# Take the per-seed results from comparative_analysis.py rather than keeping a
# second copy. Its printing is suppressed so this script's output stays clean.
_src = (HERE / "comparative_analysis.py").read_text()
_ns = {"__file__": str(HERE / "comparative_analysis.py"), "__name__": "_data"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(_src, "comparative_analysis.py", "exec"), _ns)
CLIENT0, CLIENT2, SEEDS = _ns["CLIENT0"], _ns["CLIENT2"], _ns["SEEDS"]

TW = 4.82                      # LNCS \textwidth in inches
OUT = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "pdf.fonttype": 42,
})

ORDER = ["pre", "oracle", "ga", "kd", "fe"]
NAMES = {"pre": "Pre-unlearning", "oracle": "Oracle", "ga": "Grad. ascent",
         "kd": "KD", "fe": "FedEraser"}
MARK = {"pre": "o", "oracle": "*", "ga": "^", "kd": "s", "fe": "v"}
GREY = {"pre": "0.45", "oracle": "0.0", "ga": "0.25", "kd": "0.6", "fe": "0.75"}


def agg(data, m, k):
    v = [data[s][m][k] for s in SEEDS]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


# Push-weight ablation, matched runs (paper Table 2): (W1, AUROC) at alpha_p = 0
def save(fig, name):
    """Write the PDF the paper uses, plus a PNG for on-screen viewing."""
    for ext, dpi in ((".pdf", None), (".png", 200)):
        fig.savefig(OUT / (name + ext), bbox_inches="tight", pad_inches=0.02, dpi=dpi)


ABLATED = {"c0": (0.0020, 0.872), "c2": (0.0012, 0.902)}

# ---------------------------------------------------------------- figure 1 --
fig, axes = plt.subplots(1, 2, figsize=(TW, 2.25))
for ax, (data, tag, abl) in zip(axes, [(CLIENT0, "Client 0 (dominant, 51%)", "c0"),
                                       (CLIENT2, "Client 2 (minor, 2.6%)", "c2")]):
    for m in ORDER:
        w = agg(data, m, "w")[0]
        a = agg(data, m, "auroc")[0]
        ax.scatter(w, a, marker=MARK[m], s=46, c=GREY[m], edgecolors="k",
                   linewidths=0.6, zorder=3, label=NAMES[m])
    aw, aa = ABLATED[abl]
    ax.scatter(aw, aa, marker="D", s=40, facecolors="white", edgecolors="k",
               linewidths=0.9, zorder=4, label="KD, push off")
    ax.annotate("", xy=(aw, aa),
                xytext=(agg(data, "kd", "w")[0], agg(data, "kd", "auroc")[0]),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35",
                                shrinkA=4, shrinkB=4), zorder=2)
    ax.set_title(tag)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.margins(0.18)
axes[0].set_ylabel("Pixel AUROC\n(higher = more useful)")
fig.supxlabel("Wasserstein distance to oracle (lower = more compliant)", fontsize=7.5, y=0.04)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.13), handletextpad=0.3, columnspacing=0.9)
fig.tight_layout()
save(fig, "fig1_utility_vs_fidelity")
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
fig, ax = plt.subplots(figsize=(TW, 1.95))
x = np.arange(len(ORDER)); width = 0.36
for tag, data, off, hatch, col in [
        ("Client 0 (dominant)", CLIENT0, -width / 2, "", "0.55"),
        ("Client 2 (minor)", CLIENT2, width / 2, "///", "0.85")]:
    means = [agg(data, m, "auroc")[0] for m in ORDER]
    sds = [agg(data, m, "auroc")[1] for m in ORDER]
    ax.bar(x + off, means, width, yerr=sds, capsize=2, label=tag, color=col,
           hatch=hatch, edgecolor="k", linewidth=0.6, error_kw={"lw": 0.6})
ax.axhline(agg(CLIENT0, "oracle", "auroc")[0], ls=":", lw=0.8, c="k")
ax.set_xticks(x); ax.set_xticklabels([NAMES[m] for m in ORDER])
ax.set_ylabel("Pixel AUROC,\nremaining clients")
ax.set_ylim(0.80, 0.99)
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.4)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig2_auroc_both_regimes")
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
fig, ax = plt.subplots(figsize=(TW * 0.74, 1.95))
regimes = ["Client 0\n(dominant, 51%)", "Client 2\n(minor, 2.6%)"]
xx = np.arange(2)
for m, ls in (("ga", "-"), ("kd", "--")):
    means = [agg(CLIENT0, m, "ks")[0], agg(CLIENT2, m, "ks")[0]]
    sds = [agg(CLIENT0, m, "ks")[1], agg(CLIENT2, m, "ks")[1]]
    ax.errorbar(xx, means, yerr=sds, marker=MARK[m], markersize=5, capsize=3,
                linestyle=ls, color="k", markerfacecolor=GREY[m],
                markeredgecolor="k", label=NAMES[m], elinewidth=0.7)
ax.set_xticks(xx); ax.set_xticklabels(regimes)
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("KS distance to oracle\n(lower = closer)")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper left", handlelength=2.0)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig3_fidelity_divergence")
plt.close(fig)
print("written to", OUT)
