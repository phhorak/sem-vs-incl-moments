"""Interpretation plots for the data gap at threshold 0 (HQE route, moments m1..m3).

  moment_plane     mean vs variance per observable: data gap (68/95%), candidate gap modes,
                   SEM and inclusive references, region no positive density can reach
  cdf_bounds       model-independent bounds on P(X <= x) over all densities with the measured
                   moments (linear programs), nominal and with uncertainties, MaxEnt overlaid
  admissibility    fraction of admissible pseudo-experiments vs the assumed gap fraction
  candidates.json  chi2 of each candidate's moments against the data gap
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.maxent import MaxEnt, hausdorff_check, raw_to_mu01
from lib.systematics import c_true, load_hqe_raw

OUT = HERE / "scratch" / "interpretation"
cfg = yaml.safe_load(open(HERE / "config.yaml"))
mc, sysc, SUP = cfg["maxent"], cfg["systematics"], cfg["data"]["support"]
OBS = ["mx2", "el", "q2"]
T = dict(np.load(Path(cfg["paths"]["output"]) / "5" / "toys.npz"))
bf_gap, n = float(T["bf_gap"]), len(T["z_gap"])
f_nom = bf_gap / sysc["bf_incl"]
rng = np.random.default_rng(11)

# ── inputs: inclusive (HQE), SEM, residual, per observable, orders 1-3 ──────────
hqe_c, hqe_t = load_hqe_raw(HERE / "inputs" / "hqe_likelihood_toys" / "likelihood_toys.h5")
ih = rng.choice(len(hqe_t), n, replace=len(hqe_t) < n)
incl = {o: (hqe_c[3 * k:3 * k + 3], hqe_t[ih, 3 * k:3 * k + 3]) for k, o in enumerate(OBS)}
sem = {o: (T["nom_sem0"][MAX_ORDER * k:MAX_ORDER * k + 3], T["sem0_stat_ff_bfmode"][:, MAX_ORDER * k:MAX_ORDER * k + 3])
       for k, o in enumerate(OBS)}


def residual(o, f=None, toys=True, with_gap_draw=True):
    """Gap raw moments (orders 1-3) for gap fraction f (None: nominal)."""
    c0 = 1.0 / (f if f is not None else f_nom)
    (ic, it), (sc, st) = incl[o], sem[o]
    if not toys:
        return c0 * ic - (c0 - 1) * sc
    if with_gap_draw and f is None:
        c = c_true(sysc, bf_gap, T["z_gap"])[:, None]
    else:
        c = np.full((n, 1), c0)
    return c * it - (c - 1) * st


R_nom = {o: residual(o, toys=False) for o in OBS}
R_toy = {o: residual(o) for o in OBS}

# ── candidates ────────────────────────────────────────────────────────────────
CAND = {  # sample: (label, family)
    "D2stDeta": (r"$D_2^*\!\to\! D\eta$", "Deta"), "Dp1DstEta": (r"$D_1'\!\to\! D^*\eta$", "Deta"),
    "D0stDeta": (r"$D_0^*\!\to\! D\eta$", "Deta"), "D1_2550Deta": (r"$D_1(2550)\!\to\! D\eta$", "Deta"),
    "DummyDeta": (r"$D\eta$ (NR)", "Deta"), "D1Dgamma": (r"$D_1\!\to\! D\gamma$", "Dgamma"),
    "D0pipipipenu": (r"$D\pi\pi\pi$", "D3pi"), "Dst0pipipipenu": (r"$D^*\pi\pi\pi$", "D3pi"),
    "DsKenu": (r"$D_sK$", "DsK"), "DsStarKenu": (r"$D_s^*K$", "DsK"), "DsKStarenu": (r"$D_sK^*$", "DsK"),
    "DsStarKStarenu": (r"$D_s^*K^*$", "DsK"), "LcPenu": (r"$\Lambda_c\bar p$", "baryon"),
}
FAMILY_COLOR = {"Deta": "#1f77b4", "Dgamma": "#7f7f7f", "D3pi": "#9467bd", "DsK": "#2ca02c", "baryon": "#d62728"}
FAMILY_LABEL = {"Deta": r"$D^{(*)}\eta$", "Dgamma": r"$D\gamma$", "D3pi": r"$D^{(*)}\pi\pi\pi$", "DsK": r"$D_s^{(*)}K^{(*)}$",
                "baryon": r"$\Lambda_c\bar p$"}
cand_mom = {}
for s in CAND:
    d = pd.read_parquet(Path(cfg["paths"]["output"]) / "3" / f"{s}.parquet", columns=["Mx", "El_B", "q2", "total_weight"])
    d = d[np.isfinite(d.to_numpy()).all(axis=1) & (d.total_weight > 0)]
    w = d.total_weight.to_numpy() / d.total_weight.sum()
    x = {"mx2": d.Mx.to_numpy() ** 2, "el": d.El_B.to_numpy(), "q2": d.q2.to_numpy()}
    cand_mom[s] = {o: np.array([np.sum(w * x[o] ** k) for k in (1, 2, 3)]) for o in OBS}
print("candidates loaded", flush=True)

# chi2 of each candidate vs the data gap (all 9 moments; and per observable)
Rn = np.concatenate([R_nom[o] for o in OBS])
Rt = np.concatenate([R_toy[o] for o in OBS], axis=1)
C = np.cov(Rt, rowvar=False)
chi = {}
for s in CAND:
    dv = np.concatenate([cand_mom[s][o] for o in OBS]) - Rn
    per = {}
    for k, o in enumerate(OBS):
        sl = slice(3 * k, 3 * k + 3)
        per[o] = float(dv[sl] @ np.linalg.solve(C[sl, sl], dv[sl]))
    chi[s] = dict(total=float(dv @ np.linalg.solve(C, dv)), **per)
order = sorted(chi, key=lambda s: chi[s]["total"])
print("candidate chi2 (9 moments / per observable):")
for s in order:
    print(f"  {s:16s} {chi[s]['total']:9.1f}   mx2 {chi[s]['mx2']:8.1f}  el {chi[s]['el']:8.1f}  q2 {chi[s]['q2']:8.1f}")

# ── plotting setup ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FS_LABEL, FS_TICK, FS_LEG = 26, 19, 20
DATA_COLOR = "#ff7f0e"


def style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(True)


UNIT = {"mx2": (r"$\langle M_X^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(M_X^2)\ [\mathrm{GeV}^4]$"),
        "el": (r"$\langle E_\ell^B\rangle\ [\mathrm{GeV}]$", r"$\mathrm{Var}(E_\ell^B)\ [\mathrm{GeV}^2]$"),
        "q2": (r"$\langle q^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(q^2)\ [\mathrm{GeV}^4]$")}


def ellipse(ax, pts, color):
    mu, cov = pts.mean(0), np.cov(pts, rowvar=False)
    w, v = np.linalg.eigh(cov)
    ang = np.linspace(0, 2 * np.pi, 200)
    for q, a in ((5.991, 0.18), (2.279, 0.4)):
        xy = mu[:, None] + v @ (np.sqrt(np.clip(w, 0, None) * q)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
        ax.fill(xy[0], xy[1], color=color, alpha=a, lw=0, zorder=2)


# ── 1. moment plane ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 7.4), dpi=200)
for ax, o in zip(axes, OBS):
    lo, hi = SUP[o]
    mv = lambda m: (m[..., 0], m[..., 1] - m[..., 0] ** 2)
    pts = np.stack(mv(R_toy[o]), 1)
    xs = [mv(R_nom[o])[0], mv(incl[o][0])[0], mv(sem[o][0])[0]] + [mv(cand_mom[s][o])[0] for s in CAND]
    ys = [mv(R_nom[o])[1], mv(incl[o][0])[1], mv(sem[o][0])[1]] + [mv(cand_mom[s][o])[1] for s in CAND]
    x0, x1 = min(xs + list(np.percentile(pts[:, 0], [1]))), max(xs + list(np.percentile(pts[:, 0], [99])))
    y0, y1 = min(ys + list(np.percentile(pts[:, 1], [1])) + [0]), max(ys + list(np.percentile(pts[:, 1], [99])))
    px, py = 0.08 * (x1 - x0), 0.12 * (y1 - y0)
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + 1.6 * py
    xg = np.linspace(x0, x1, 400)
    ax.fill_between(xg, y0, np.minimum(0, y1), color="0.85", lw=0, zorder=0)
    ax.fill_between(xg, np.maximum((xg - lo) * (hi - xg), y0), y1, color="0.85", lw=0, zorder=0)
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ellipse(ax, pts, DATA_COLOR)
    ax.plot(*mv(R_nom[o]), "o", color=DATA_COLOR, ms=11, mec="k", mew=1.2, zorder=5)
    ax.plot(*mv(sem[o][0]), "s", color="k", ms=10, zorder=5)
    ax.plot(*mv(incl[o][0]), "D", color="0.45", ms=10, zorder=5)
    for s, (lbl, fam) in CAND.items():
        x, y = mv(cand_mom[s][o])
        ax.plot(x, y, "o", color=FAMILY_COLOR[fam], ms=8, mec="white", mew=0.8, zorder=4)
        ax.annotate(lbl, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=12, color=FAMILY_COLOR[fam], zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(UNIT[o][0], fontsize=FS_LABEL)
    ax.set_ylabel(UNIT[o][1], fontsize=FS_LABEL)
    style(ax)
h = [Patch(color=DATA_COLOR, alpha=0.5, lw=0, label="Data gap (68%, 95%)"),
     Line2D([], [], marker="s", color="k", ls="", ms=10, label="SEM"),
     Line2D([], [], marker="D", color="0.45", ls="", ms=10, label="Inclusive (HQE)")]
h += [Line2D([], [], marker="o", color=c, ls="", ms=9, label=FAMILY_LABEL[f]) for f, c in FAMILY_COLOR.items()]
h += [Patch(color="0.85", lw=0, label="No positive density")]
fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=17, bbox_to_anchor=(0.5, -0.12),
           handlelength=1.4, columnspacing=1.2)
fig.tight_layout()
fig.savefig(OUT / "moment_plane.pdf", bbox_inches="tight")
plt.close(fig)
print("moment_plane done", flush=True)

# ── 2. CDF bounds (linear programs) ──────────────────────────────────────────
tg = np.linspace(0, 1, 401)
A = np.stack([tg ** k for k in range(4)])


def cdf_bounds(mu, xs):
    lo_b, hi_b = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        ind = (tg <= x).astype(float)
        r1 = linprog(ind, A_eq=A, b_eq=mu, bounds=(0, None), method="highs")
        r2 = linprog(-ind, A_eq=A, b_eq=mu, bounds=(0, None), method="highs")
        if r1.status == 0 and r2.status == 0:
            lo_b[i], hi_b[i] = r1.fun, -r2.fun
    return lo_b, hi_b


xt = np.linspace(0, 1, 51)
bounds = {}
for o in ("el", "q2"):
    lo, hi = SUP[o]
    mu = raw_to_mu01(np.r_[1.0, R_nom[o]], lo, hi)
    nom = cdf_bounds(mu, xt)
    toys = [cdf_bounds(raw_to_mu01(np.r_[1.0, r], lo, hi), xt) for r in R_toy[o][:200]
            if hausdorff_check(raw_to_mu01(np.r_[1.0, r], lo, hi))[0]]
    L = np.array([t[0] for t in toys])
    H = np.array([t[1] for t in toys])
    me = MaxEnt(tg, mom_tol=mc["mom_tol"]).solve(mu, mc["alpha"][o], mc["beta"][o])
    cdf = np.concatenate([[0], np.cumsum(0.5 * (me.f[1:] + me.f[:-1]) * np.diff(tg))])
    bounds[o] = dict(x=lo + xt * (hi - lo), nom=nom, lo68=np.nanpercentile(L, 16, 0), hi68=np.nanpercentile(H, 84, 0),
                     me_x=lo + tg * (hi - lo), me=cdf, n_toys=len(toys))
    print(f"bounds {o}: {len(toys)} admissible toys", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), dpi=200)
XL = {"el": r"$E_\ell^B\ [\mathrm{GeV}]$", "q2": r"$q^2\ [\mathrm{GeV}^2]$"}
YL = {"el": r"$P(E_\ell^B \leq x)$", "q2": r"$P(q^2 \leq x)$"}
for ax, o in zip(axes, ("el", "q2")):
    b = bounds[o]
    ax.fill_between(b["x"], b["lo68"], b["hi68"], color=DATA_COLOR, alpha=0.18, lw=0, step="post")
    ax.fill_between(b["x"], b["nom"][0], b["nom"][1], color=DATA_COLOR, alpha=0.45, lw=0, step="post")
    ax.plot(b["me_x"], b["me"], color="#1f77b4", lw=2.6)
    ax.set_xlim(b["x"][0], b["x"][-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(XL[o], fontsize=FS_LABEL)
    ax.set_ylabel(YL[o], fontsize=FS_LABEL)
    style(ax)
h = [Patch(color=DATA_COLOR, alpha=0.45, lw=0, label="Allowed (measured moments)"),
     Patch(color=DATA_COLOR, alpha=0.18, lw=0, label="incl. uncertainties (68%)"),
     Line2D([], [], color="#1f77b4", lw=2.6, label="MaxEnt")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEG, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
fig.savefig(OUT / "cdf_bounds.pdf", bbox_inches="tight")
plt.close(fig)

# ── 3. admissibility vs assumed gap fraction ────────────────────────────────
fs = np.linspace(0.06, 0.40, 69)
adm = {o: [] for o in OBS}
for f in fs:
    for o in OBS:
        lo, hi = SUP[o]
        rt = residual(o, f=f)[:600]
        adm[o].append(np.mean([hausdorff_check(raw_to_mu01(np.r_[1.0, r], lo, hi))[0] for r in rt]))
fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
sig = f_nom * 0.154
ax.axvspan(f_nom - sig, f_nom + sig, color="0.88", lw=0)
ax.axvline(f_nom, color="0.5", lw=1.2, ls="--")
LBL = {"mx2": r"$M_X^2$", "el": r"$E_\ell$", "q2": r"$q^2$"}
COL = {"mx2": "#1f77b4", "el": "#ff7f0e", "q2": "#2ca02c"}
for o in OBS:
    ax.plot(fs, adm[o], color=COL[o], lw=2.8, label=LBL[o])
ax.text(f_nom + 0.004, 0.5, "Table 1", rotation=90, fontsize=16, color="0.4", va="center")
ax.set_xlim(fs[0], fs[-1])
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"assumed $\mathcal{B}_\mathrm{gap}/\mathcal{B}_\mathrm{incl}$", fontsize=FS_LABEL)
ax.set_ylabel("admissible fraction", fontsize=FS_LABEL)
style(ax)
ax.legend(fontsize=FS_LEG, frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(OUT / "admissibility_vs_gap.pdf", bbox_inches="tight")
plt.close(fig)

json.dump(dict(f_nom=f_nom, chi2=chi, order=order,
               admissibility={o: dict(f=fs.tolist(), frac=adm[o]) for o in OBS},
               bounds={o: dict(x=bounds[o]["x"].tolist(), lo=bounds[o]["nom"][0].tolist(), hi=bounds[o]["nom"][1].tolist(),
                               lo68=bounds[o]["lo68"].tolist(), hi68=bounds[o]["hi68"].tolist()) for o in bounds},
               residual_nominal={o: R_nom[o].tolist() for o in OBS}),
          open(OUT / "interpretation.json", "w"), indent=1)
print("done", flush=True)
