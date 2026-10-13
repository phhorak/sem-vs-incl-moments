"""SEM cocktail <M_X^6> vs E_l cut, stacked by category with measurements."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).parent
CATS = {"D": r"$D$", "D*": r"$D^*$", "D**": r"$D^{**}$", "D(*) pi": r"$D^{(*)}\pi$",
        "D(*) pi pi": r"$D^{(*)}\pi\pi$", "Ds(*) K": r"$D_s^{(*)}K$"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
EXP = {"babar": ("BaBar", "#8c564b", "D")}

S = json.load(open(OUT / "stack_moments.json"))
data = json.load(open(HERE / "output" / "5" / "sem_moments_raw.json"))["experimental_data_raw"]["mx_3"]
cuts = np.array(S["cuts"]["el"])
stk = np.nan_to_num([S["mx_3"][c] for c in CATS])

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=250)
ax.stackplot(cuts, stk, colors=COLORS, alpha=0.55, lw=0)
ax.plot(cuts, stk.sum(0), color="0.25", lw=0.9)
for exp, (lbl, col, mk) in EXP.items():
    e = data[exp]
    ax.errorbar(e["cuts"], e["values"], yerr=e["errors"], ls="", marker=mk, color=col, mec="black", ecolor="black",
                lw=1.2, ms=7, capsize=2, zorder=20)
ax.set_xlim(0, 2.4)
ax.set_ylim(0, None)
ax.set_xlabel(r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$", fontsize=20)
ax.set_ylabel(r"$\langle M_X^6\rangle\ [\mathrm{GeV}^6]$", fontsize=20)
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=15)
h = [Patch(color=c, alpha=0.55, lw=0, label=l) for c, l, s in zip(COLORS, CATS.values(), stk) if s.max() > 1]
h += [Line2D([], [], marker="D", color="#8c564b", mec="black", ls="", ms=8, label="BaBar")]
ax.legend(handles=h, ncol=2, frameon=False, fontsize=13, loc="upper right", handlelength=1.2, columnspacing=1.0)
fig.tight_layout()
fig.savefig(OUT / "sem_stack_mx3.pdf", bbox_inches="tight")
