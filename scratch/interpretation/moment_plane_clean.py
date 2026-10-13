"""Mean vs variance of the threshold-0 gap (HQE route) against SEM, HQE and candidate gap modes.

Reads the step-6 bundle; candidate moments come from output/3 and are cached in moment_plane_modes.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.bundle import Bundle, default_path
from lib.systematics import c_true

OUT = Path(__file__).resolve().parent
B = Bundle(default_path(HERE))
cfg, T = B.cfg, B.toys
SUP = cfg["data"]["support"]
OBS = ["mx2", "el", "q2"]
MODES = {  # sample: (label, colour, marker)
    "Dp1DstEta": (r"$D_1'\to D^*\eta$", "#1f77b4", "o"),
    "D0pipipipenu": (r"$D\pi\pi\pi$", "#9467bd", "^"),
    "Dst0pipipipenu": (r"$D^*\pi\pi\pi$", "#c49cde", "^"),
    "DsKenu": (r"$D_sK$", "#2ca02c", "s"),
    "LcPenu": (r"$\Lambda_c\bar p$", "#d62728", "P"),
}

cache = OUT / "moment_plane_modes.json"
mom = json.load(open(cache)) if cache.exists() else {}
for s in set(MODES) - set(mom):
    d = pd.read_parquet(HERE / "output" / "3" / f"{s}.parquet", columns=["Mx", "El_B", "q2", "total_weight"])
    d = d[np.isfinite(d.to_numpy()).all(axis=1) & (d.total_weight > 0)]
    w = d.total_weight.to_numpy() / d.total_weight.sum()
    x = {"mx2": d.Mx.to_numpy() ** 2, "el": d.El_B.to_numpy(), "q2": d.q2.to_numpy()}
    mom[s] = {o: [float(np.sum(w * x[o] ** k)) for k in (1, 2)] for o in OBS}
    json.dump(mom, open(cache, "w"), indent=1)

bf_gap = float(T["bf_gap"])
c0, n = float(c_true(cfg["systematics"], bf_gap)), len(T["hqe_toys"])
rng = np.random.default_rng(11)
js = rng.choice(len(T["z_gap"]), n)
c = c_true(cfg["systematics"], bf_gap, T["z_gap"][js])[:, None]
mv = lambda m: np.stack([m[..., 0], m[..., 1] - m[..., 0] ** 2], -1)
pts = {}
for k, o in enumerate(OBS):
    hc, ht = T["hqe_central"][3 * k:3 * k + 2], T["hqe_toys"][:, 3 * k:3 * k + 2]
    sc, st = T["nom_sem0"][MAX_ORDER * k:MAX_ORDER * k + 2], T["sem0_stat_ff_bfmode"][js, MAX_ORDER * k:MAX_ORDER * k + 2]
    pts[o] = dict(gap=mv(c0 * hc - (c0 - 1) * sc), gap_toys=mv(c * ht - (c - 1) * st), sem=mv(sc), hqe=mv(hc))

import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

GAP = "#ff7f0e"
LABEL = {"mx2": (r"$\langle M_X^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(M_X^2)\ [\mathrm{GeV}^4]$"),
         "el": (r"$\langle E_\ell^B\rangle\ [\mathrm{GeV}]$", r"$\mathrm{Var}(E_\ell^B)\ [\mathrm{GeV}^2]$"),
         "q2": (r"$\langle q^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(q^2)\ [\mathrm{GeV}^4]$")}
fig, axes = plt.subplots(1, 3, figsize=(24, 7.4), dpi=200)
for ax, o in zip(axes, OBS):
    P, lo, hi = pts[o], *SUP[o]
    ref = np.array([P["gap"], P["sem"], P["hqe"]] + [mv(np.array(mom[s][o])) for s in MODES])
    q = np.percentile(P["gap_toys"], [1, 99], axis=0)
    (x0, y0), (x1, y1) = np.minimum(ref.min(0), q[0]), np.maximum(ref.max(0), q[1])
    y0 = min(y0, 0)
    px, py = 0.08 * (x1 - x0), 0.1 * (y1 - y0)
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + 1.5 * py
    xg = np.linspace(x0, x1, 400)
    ax.fill_between(xg, y0, 0, color="0.88", lw=0, zorder=0)
    ax.fill_between(xg, np.maximum((xg - lo) * (hi - xg), y0), y1, color="0.88", lw=0, zorder=0)
    mu, cov = P["gap_toys"].mean(0), np.cov(P["gap_toys"], rowvar=False)
    w, v = np.linalg.eigh(cov)
    ang = np.linspace(0, 2 * np.pi, 200)
    for chi2, a in ((5.991, 0.2), (2.279, 0.45)):
        xy = mu[:, None] + v @ (np.sqrt(np.clip(w, 0, None) * chi2)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
        ax.fill(*xy, color=GAP, alpha=a, lw=0, zorder=2)
    ax.plot(*P["gap"], "o", color=GAP, ms=13, mec="k", mew=1.4, zorder=6)
    ax.plot(*P["sem"], "s", color="k", ms=12, zorder=6)
    ax.plot(*P["hqe"], "D", color="0.45", ms=12, zorder=6)
    for s, (_, col, mk) in MODES.items():
        ax.plot(*mv(np.array(mom[s][o])), mk, color=col, ms=13, mec="k", mew=0.8, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(LABEL[o][0], fontsize=26)
    ax.set_ylabel(LABEL[o][1], fontsize=26)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=19)
h = [Patch(color=GAP, alpha=0.5, lw=0, label="Gap (68%, 95%)"),
     Line2D([], [], marker="s", color="k", ls="", ms=12, label="SEM"),
     Line2D([], [], marker="D", color="0.45", ls="", ms=12, label="Inclusive (HQE)"),
     Patch(color="0.88", lw=0, label="No positive density")]
h += [Line2D([], [], marker=mk, color=col, mec="k", mew=0.8, ls="", ms=13, label=l) for l, col, mk in MODES.values()]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=20, bbox_to_anchor=(0.5, -0.2),
           handlelength=1.2, columnspacing=1.4)
fig.tight_layout()
fig.savefig(OUT / "moment_plane_clean.pdf", bbox_inches="tight")
print({o: {k: np.round(P[k], 3).tolist() for k in ("gap", "sem", "hqe")} for o, P in pts.items()})
