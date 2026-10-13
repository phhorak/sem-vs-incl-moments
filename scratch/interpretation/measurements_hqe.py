"""Measured raw moments vs threshold with the HQE fit (Kolya) extrapolated to threshold 0."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from lib.bundle import Bundle, default_path

import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP = {"babar": ("BaBar", "#8c564b", "D"), "belle": ("Belle", "#1f77b4", "o"),
       "belleII": ("Belle II", "#ff7f0e", "s"), "cleo": ("CLEO", "#2ca02c", "^")}
HQE = "#d62728"
ROWS = [("mx", r"$\langle M_X^{2n}\rangle$", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
        ("el", r"$\langle E_\ell^n\rangle$", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
        ("q2", r"$\langle q^{2n}\rangle$", r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$")]

data = json.load(open(HERE / "output" / "5" / "sem_moments_raw.json"))["experimental_data_raw"]
hc, ht = Bundle(default_path(HERE)).hqe
herr = ht.std(0)

fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), dpi=250)
for r, (v, ylabel, xlabel) in enumerate(ROWS):
    for c in range(3):
        ax = axes[r, c]
        for exp, e in data[f"{v}_{c + 1}"].items():
            if exp in EXP:
                _, col, mk = EXP[exp]
                ax.errorbar(e["cuts"], e["values"], yerr=e["errors"], ls="", marker=mk, color=col, mec="black",
                            ecolor=col, lw=1.6, ms=7, capsize=2, zorder=20)
        ax.errorbar([0.0], [hc[3 * r + c]], yerr=[herr[3 * r + c]], ls="", marker="*", color=HQE, mec="black",
                    ecolor=HQE, lw=2, ms=20, capsize=3, zorder=25)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=30, labelpad=8)
        ax.set_xlabel(xlabel, fontsize=28, labelpad=8)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=22, pad=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.margins(x=0.08, y=0.15)
        ax.text(0.06 if r else 0.94, 0.94, rf"$n={c + 1}$", transform=ax.transAxes, fontsize=32, color="0.15",
                ha="left" if r else "right", va="top")
h = [Line2D([], [], marker=mk, color=col, mec="black", ls="", ms=10, label=l) for l, col, mk in EXP.values()]
h.append(Line2D([], [], marker="*", color=HQE, mec="black", ls="", ms=22, label=r"HQE fit, $\mathrm{cut}=0$"))
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=24)
fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.135, wspace=0.30, hspace=0.35)
fig.savefig(Path(__file__).parent / "measurements_hqe_3x3.pdf", bbox_inches="tight")
