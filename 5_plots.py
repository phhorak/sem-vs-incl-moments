#!/usr/bin/env python3
"""Step 5: Produce moment plots from cocktail + gap mode parquets.

Outputs:
  figures/5/distributions_3panel.png
  figures/5/distributions_gap_modes_3panel.png
  figures/5/sem_vs_data_central_3x3.png
  figures/5/sem_vs_data_raw_3x3.png
  figures/5/stacked_central_3x3.png
  figures/5/stacked_raw_3x3.png
  figures/5/gap_modes_overlay_3x3.png
  figures/5/gap_modes_with_sm_central_3x3.png
  figures/5/gap_modes_with_sm_raw_3x3.png
  output/5/sem_moments_raw.json
"""
import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import plothist

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "lib"))

from lib.gap_modes import GAP_MODES
from lib.moments import (
    EXP_STYLE, YLABELS, YLABELS_RAW, ROW_TITLES,
    build_exp_moments_df, compute_curves_np, df_to_plot_dict,
)

p    = cfg["plots"]
out5 = Path(cfg["paths"]["output"]) / "5"
fig5 = Path(cfg["paths"]["figures"]) / "5"
out5.mkdir(parents=True, exist_ok=True)
fig5.mkdir(parents=True, exist_ok=True)

n_toys   = p.get("n_toys", 500)
toy_seed = p.get("toy_seed", 42)

EL_CUTS = np.linspace(0.0, 2.5, 50)
Q2_CUTS = np.linspace(0.0, 10.0, 45)

# Local grid with actual arrays (GRID in lib/moments.py has string placeholders)
_GRID = [
    ("mx", ["mx_1", "mx_2", "mx_3"], EL_CUTS, r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("el", ["el_1", "el_2", "el_3"], EL_CUTS, r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("q2", ["q2_1", "q2_2", "q2_3"], Q2_CUTS, r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$"),
]

CATEGORIES = ["D", "D*", "D**", "D(*) pi", "D(*) pi pi", "Ds(*) K"]
CAT_LABELS = {
    "D":            r"$B \to D\,\ell\nu$",
    "D*":           r"$B \to D^*\ell\nu$",
    "D**":          r"$B \to D^{**}\ell\nu$",
    "D(*) pi":      r"$B \to D^{(*)}\pi\,\ell\nu$",
    "D(*) pi pi":   r"$B \to D^{(*)}\pi\pi\,\ell\nu$",
    "Ds(*) K":      r"$B \to D_s^{(*)}K\,\ell\nu$",
}
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MKEYS = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]


def _plot_exp(ax, key, exp_data, seen):
    for exp, entry in exp_data.get(key, {}).items():
        st  = EXP_STYLE.get(exp, dict(color="grey", marker="o", label=exp))
        lbl = st["label"] if st["label"] not in seen else None
        if lbl:
            seen.add(st["label"])
        ax.errorbar(entry["cuts"], entry["values"], yerr=entry["errors"],
                    ls="", marker=st["marker"], color=st["color"],
                    mec="black", ecolor=st["color"], lw=1, ms=5, capsize=2,
                    zorder=20, label=lbl)

def _plot_avg(ax, key, avg_data, show_label=False):
    if not avg_data or key not in avg_data:
        return
    entry = avg_data[key]
    cuts = np.asarray(entry["cuts"], dtype=float)
    vals = np.asarray(entry["values"], dtype=float)
    errs = np.asarray(entry["errors"], dtype=float)
    ax.plot(cuts, vals, color="black", lw=1.8, zorder=24,
            label="Experimental average" if show_label else None)
    ax.fill_between(cuts, vals - errs, vals + errs, color="black", alpha=0.18, zorder=23,
                    label=r"Average $\pm1\sigma$" if show_label else None)


def _spectral(n):
    cmap = plt.get_cmap("Spectral")
    return [cmap(1.0 - i / max(n - 1, 1)) for i in range(n)]


# ── A. Load cocktail + gap mode parquets ──────────────────────────────────────

print("[1/7] Loading parquets …")
cocktail = pd.read_parquet(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet")
if "total_weight" not in cocktail.columns:
    cocktail["total_weight"] = cocktail["weight"] * cocktail["mc_weight"] * cocktail["ff_weight"]

gap_frames = {}
for meta in GAP_MODES:
    pq = Path(cfg["paths"]["output"]) / "3" / f"{meta['mode']}.parquet"
    if pq.exists():
        gdf = pd.read_parquet(pq)
        if "total_weight" not in gdf.columns:
            gdf["total_weight"] = gdf["weight"] * gdf["mc_weight"] * gdf["ff_weight"]
        gap_frames[meta["mode"]] = gdf
    else:
        print(f"  warning: {pq} not found — skipping {meta['mode']}")

# ── B. Load experimental data ─────────────────────────────────────────────────

print("[2/7] Loading experimental data …")
exp_central: dict = {}
exp_raw: dict = {}
avg_central: dict = {}
avg_raw: dict = {}
tar_path = Path(p.get("exp_data_tar", "inputs/sem.tar.gz"))
if tar_path.exists():
    with tempfile.TemporaryDirectory(prefix="sem_expdata_") as tmp:
        with tarfile.open(tar_path) as tf:
            try:
                tf.extractall(tmp, filter="data")
            except TypeError:
                tf.extractall(tmp)
        exp_df = build_exp_moments_df(Path(tmp))
    exp_central = df_to_plot_dict(exp_df, "measured")
    exp_raw     = df_to_plot_dict(exp_df, "calculated")
else:
    print(f"  warning: {tar_path} not found — no experimental data overlay")

avg_path = Path(cfg["paths"]["output"]) / "4" / "experimental_average.json"
if avg_path.exists():
    try:
        avg_payload = json.loads(avg_path.read_text())
        avg_raw = avg_payload.get("average_raw", avg_payload.get("average", {}))
        avg_central = avg_payload.get("average_central", {})
    except Exception as e:
        print(f"  warning: failed to read {avg_path}: {e}")
else:
    print(f"  note: {avg_path} not found — run 4_average.py first for averaged overlays")

# ── C. Compute nominal SEM moment curves (central + raw) ──────────────────────

print("[3/7] Computing nominal moment curves …")
_ok = (
    np.isfinite(cocktail["Mx"].to_numpy())
    & np.isfinite(cocktail["El_B"].to_numpy())
    & np.isfinite(cocktail["q2"].to_numpy())
    & (cocktail["total_weight"].to_numpy() > 0)
)
_mx2 = np.square(cocktail.loc[_ok, "Mx"].to_numpy(dtype=float))
_el  = cocktail.loc[_ok, "El_B"].to_numpy(dtype=float)
_q2  = cocktail.loc[_ok, "q2"].to_numpy(dtype=float)
_w   = cocktail.loc[_ok, "total_weight"].to_numpy(dtype=float)

sem_cen_nom = compute_curves_np(_mx2, _el, _q2, _w, EL_CUTS, Q2_CUTS)

# Raw moments: E[X^n] instead of central moments
_el_ord = np.argsort(_el); _q2_ord = np.argsort(_q2)
_el_s = _el[_el_ord]; _mx2_es = _mx2[_el_ord]; _w_es = _w[_el_ord]
_q2_s = _q2[_q2_ord]; _w_qs   = _w[_q2_ord]
_cw   = np.concatenate(([0.], np.cumsum(_w_es)))
_cmx2 = np.concatenate(([0.], np.cumsum(_w_es * _mx2_es)))
_cmx4 = np.concatenate(([0.], np.cumsum(_w_es * _mx2_es**2)))
_cmx6 = np.concatenate(([0.], np.cumsum(_w_es * _mx2_es**3)))
_cel  = np.concatenate(([0.], np.cumsum(_w_es * _el_s)))
_cel2 = np.concatenate(([0.], np.cumsum(_w_es * _el_s**2)))
_cel3 = np.concatenate(([0.], np.cumsum(_w_es * _el_s**3)))
_Ne   = len(_el_s)
_cw2  = np.concatenate(([0.], np.cumsum(_w_qs)))
_cq2  = np.concatenate(([0.], np.cumsum(_w_qs * _q2_s)))
_cq4  = np.concatenate(([0.], np.cumsum(_w_qs * _q2_s**2)))
_cq6  = np.concatenate(([0.], np.cumsum(_w_qs * _q2_s**3)))
_Nq   = len(_q2_s)

def _sf(c, k, N): return float(c[N] - c[k])

sem_raw_nom: dict[str, list] = {k: [] for k in MKEYS}
for cut in EL_CUTS:
    k = int(np.searchsorted(_el_s, cut, side="right"))
    sw = _sf(_cw, k, _Ne)
    if sw <= 0:
        for key in ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3"]:
            sem_raw_nom[key].append(float("nan"))
        continue
    sem_raw_nom["mx_1"].append(_sf(_cmx2, k, _Ne) / sw)
    sem_raw_nom["mx_2"].append(_sf(_cmx4, k, _Ne) / sw)
    sem_raw_nom["mx_3"].append(_sf(_cmx6, k, _Ne) / sw)
    sem_raw_nom["el_1"].append(_sf(_cel,  k, _Ne) / sw)
    sem_raw_nom["el_2"].append(_sf(_cel2, k, _Ne) / sw)
    sem_raw_nom["el_3"].append(_sf(_cel3, k, _Ne) / sw)
for cut in Q2_CUTS:
    k = int(np.searchsorted(_q2_s, cut, side="right"))
    sw = _sf(_cw2, k, _Nq)
    if sw <= 0:
        for key in ["q2_1", "q2_2", "q2_3"]:
            sem_raw_nom[key].append(float("nan"))
        continue
    sem_raw_nom["q2_1"].append(_sf(_cq2, k, _Nq) / sw)
    sem_raw_nom["q2_2"].append(_sf(_cq4, k, _Nq) / sw)
    sem_raw_nom["q2_3"].append(_sf(_cq6, k, _Nq) / sw)
sem_raw_nom = {k: np.asarray(v, dtype=float) for k, v in sem_raw_nom.items()}

# ── D. Toy MC for uncertainty bands (central moments) ─────────────────────────

print(f"[4/7] Running {n_toys} toys …")
t0 = time.monotonic()
rng = np.random.default_rng(toy_seed)

base_w     = cocktail.loc[_ok, "weight"].to_numpy(dtype=float) \
           * cocktail.loc[_ok, "mc_weight"].to_numpy(dtype=float)
bf         = cocktail.loc[_ok, "bf"].to_numpy(dtype=float)
bf_unc     = cocktail.loc[_ok, "bf_unc"].to_numpy(dtype=float)
rel_bf_unc = np.divide(bf_unc, bf, out=np.zeros_like(bf_unc), where=bf > 0)
decay_codes, _ = pd.factorize(cocktail.loc[_ok, "decay_name"], sort=True)
n_decays       = int(decay_codes.max() + 1) if len(decay_codes) else 0
decay_codes    = decay_codes.astype(np.int32)

ff_central = cocktail.loc[_ok, "ff_weight"].to_numpy(dtype=float)
ff_up_cols = sorted(c for c in cocktail.columns if c.startswith("ff_weight_up"))
ff_dn_cols = sorted(c for c in cocktail.columns if c.startswith("ff_weight_down"))
_var_ids   = sorted(
    set(int(c.replace("ff_weight_up", "")) for c in ff_up_cols)
    & set(int(c.replace("ff_weight_down", "")) for c in ff_dn_cols)
)
ff_up_diff = [cocktail.loc[_ok, f"ff_weight_up{i}"].to_numpy(dtype=float) - ff_central for i in _var_ids]
ff_dn_diff = [ff_central - cocktail.loc[_ok, f"ff_weight_down{i}"].to_numpy(dtype=float) for i in _var_ids]

N_ok = int(_ok.sum())
toy_store: dict[str, list] = {k: [] for k in MKEYS}

for i_toy in range(n_toys):
    idx    = rng.integers(0, N_ok, size=N_ok)
    counts = np.bincount(idx, minlength=N_ok).astype(float)
    z_br   = rng.standard_normal(size=n_decays)
    br_f   = np.clip(1.0 + z_br[decay_codes] * rel_bf_unc, 0.0, None)
    ff_s   = ff_central.copy()
    if _var_ids and n_decays:
        z_ff = rng.standard_normal(size=(n_decays, len(_var_ids)))
        for j in range(len(_var_ids)):
            z = z_ff[decay_codes, j]
            ff_s += np.where(z >= 0.0, z * ff_up_diff[j], z * ff_dn_diff[j])
    ff_s = np.clip(ff_s, 0.0, None)
    w_t  = base_w * br_f * ff_s * counts
    w_t  = np.where(np.isfinite(w_t) & (w_t > 0), w_t, 0.0)
    cur  = compute_curves_np(_mx2, _el, _q2, w_t, EL_CUTS, Q2_CUTS)
    for k in MKEYS:
        toy_store[k].append(cur[k])

    if (i_toy + 1) % max(1, n_toys // 10) == 0:
        dt = time.monotonic() - t0
        print(f"  toys {i_toy+1}/{n_toys}  {dt:.1f}s  ETA {dt/(i_toy+1)*(n_toys-i_toy-1):.1f}s",
              flush=True)

sem_std = {k: np.nanstd(np.stack(toy_store[k]), axis=0, ddof=1) for k in MKEYS}
print(f"  done ({time.monotonic()-t0:.1f}s)")

# ── E. Distributions per category (3-panel) ───────────────────────────────────

print("[5/7] Category distribution plots …")
df_plot = cocktail.loc[_ok].copy()
VARS = [
    ("Mx",   r"$M_X\ [\mathrm{GeV}]$",     np.linspace(1.6, 5.5, 120)),
    ("El_B", r"$E_\ell^B\ [\mathrm{GeV}]$", np.linspace(0.0, 2.8, 120)),
    ("q2",   r"$q^2\ [\mathrm{GeV}^2]$",    np.linspace(0.0, 12.5, 120)),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
for ax, (col, xlabel, bins) in zip(axes, VARS):
    db = bins[1] - bins[0]
    for cat, color in zip(CATEGORIES, CAT_COLORS):
        m = df_plot["category"] == cat
        if not m.any():
            continue
        h, _ = np.histogram(df_plot.loc[m, col].to_numpy(float), bins=bins,
                            weights=df_plot.loc[m, "total_weight"].to_numpy(float))
        area = h.sum() * db
        if area > 0:
            ax.stairs(h / area, bins, lw=1.8, color=color, label=CAT_LABELS[cat])
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel("Normalised entries / bin", fontsize=10)
    ax.grid(alpha=0.25); ax.set_xlim(float(bins[0]), float(bins[-1])); ax.set_ylim(bottom=0)
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Kinematic distributions by decay category", fontsize=13)
fig.tight_layout()
fig.savefig(fig5 / "distributions_3panel.png"); plt.close(fig)
print("  Wrote distributions_3panel.png")

# Gap-mode-only distributions (same 3-panel format)
active_gap_for_dist = [m for m in GAP_MODES if m["mode"] in gap_frames]
if active_gap_for_dist:
    fig_g, axes_g = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for ax, (col, xlabel, bins) in zip(axes_g, VARS):
        db = bins[1] - bins[0]
        for meta in active_gap_for_dist:
            gdf = gap_frames[meta["mode"]]
            m = np.isfinite(gdf[col].to_numpy(dtype=float))
            if not np.any(m):
                continue
            vals = gdf.loc[m, col].to_numpy(dtype=float)
            w = gdf.loc[m, "total_weight"].to_numpy(dtype=float)
            h, _ = np.histogram(vals, bins=bins, weights=w)
            area = h.sum() * db
            if area > 0:
                ax.stairs(h / area, bins, lw=meta["lw"], ls=meta["ls"],
                          color=meta["color"], label=meta["label"])
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel("Normalised entries / bin", fontsize=10)
        ax.grid(alpha=0.25); ax.set_xlim(float(bins[0]), float(bins[-1])); ax.set_ylim(bottom=0)
    axes_g[0].legend(fontsize=9, loc="upper right")
    fig_g.suptitle("Kinematic distributions of active gap modes", fontsize=13)
    fig_g.tight_layout()
    fig_g.savefig(fig5 / "distributions_gap_modes_3panel.png"); plt.close(fig_g)
    print("  Wrote distributions_gap_modes_3panel.png")
else:
    print("  note: no active gap-mode parquets found for distributions_gap_modes_3panel.png")


# ── F. SEM vs data plots (central + raw) ──────────────────────────────────────

def _sem_vs_data_plot(sem_nom, sem_std, exp_data, avg_data, outpath, moment_space):
    ylabels = YLABELS_RAW if moment_space == "raw" else YLABELS
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
    seen: set[str] = set()
    for r, (_, keys, cuts, xlabel) in enumerate(_GRID):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            ax.plot(cuts, sem_nom[key], lw=1.8,
                    label="SEM (weighted)" if (r == c == 0) else None)
            if np.any(np.isfinite(sem_std[key])):
                ax.fill_between(cuts, sem_nom[key] - sem_std[key], sem_nom[key] + sem_std[key],
                                alpha=0.25, lw=0, label="SEM uncertainty" if (r == c == 0) else None)
            _plot_avg(ax, key, avg_data, show_label=(r == 0 and c == 0))
            _plot_exp(ax, key, exp_data, seen)
            ax.set_ylabel(ylabels[key]); ax.set_xlabel(xlabel)
            x0, x1 = float(cuts[0]), float(cuts[-1])
            ax.set_xlim(x0 - 0.02*(x1-x0), x1 + 0.02*(x1-x0))
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][c])
        axes[r, 0].text(-0.35, 0.5, ROW_TITLES[r], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center")
    handles, labels = [], []; seen2: set[str] = set()
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen2:
                seen2.add(l); handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
                   bbox_to_anchor=(0.5, 0.998), fontsize=9)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.92])
    fig.savefig(outpath); plt.close(fig)
    print(f"  Wrote {outpath.name}")

_sem_vs_data_plot(sem_cen_nom, sem_std, exp_central, avg_central,
                  fig5 / "sem_vs_data_central_3x3.png", "central")
_sem_vs_data_plot(sem_raw_nom, sem_std, exp_raw, avg_raw,
                  fig5 / "sem_vs_data_raw_3x3.png", "raw")


# ── G. Stacked category contributions ─────────────────────────────────────────

print("[6/7] Stacked + gap mode plots …")


def _stacked_plot(df, outpath, moment_space):
    w_all   = df["total_weight"].to_numpy(dtype=float)
    mx2_all = np.square(df["Mx"].to_numpy(dtype=float))
    el_all  = df["El_B"].to_numpy(dtype=float)
    q2_all  = df["q2"].to_numpy(dtype=float)
    cat_all = df["category"].to_numpy()
    ok      = np.isfinite(mx2_all) & np.isfinite(el_all) & np.isfinite(q2_all) & (w_all > 0)
    raw     = moment_space == "raw"

    def gm(x, w): d = np.sum(w); return float(np.sum(w * x) / d) if d > 0 else 0.0
    def wm(x, w, n, mu=0.): d = np.sum(w); return float(np.sum(w*(x-mu)**n)/d) if d > 0 else 0.

    contribs: dict[str, dict[str, list]] = {k: {c: [] for c in CATEGORIES} for k in MKEYS}

    for cut in EL_CUTS:
        m = ok & (el_all > cut)
        w = w_all[m]; mx2 = mx2_all[m]; el = el_all[m]; cat = cat_all[m]
        W = np.sum(w); mu_mx2 = gm(mx2, w); mu_el = gm(el, w)
        for c in CATEGORIES:
            cm = cat == c; wc = w[cm]; fc = np.sum(wc) / W if W > 0 else 0.0
            mu_m = 0. if raw else mu_mx2; mu_e = 0. if raw else mu_el
            contribs["mx_1"][c].append(fc * wm(mx2[cm], wc, 1))
            contribs["mx_2"][c].append(fc * wm(mx2[cm], wc, 2, mu_m))
            contribs["mx_3"][c].append(fc * wm(mx2[cm], wc, 3, mu_m))
            contribs["el_1"][c].append(fc * wm(el[cm],  wc, 1))
            contribs["el_2"][c].append(fc * wm(el[cm],  wc, 2, mu_e))
            contribs["el_3"][c].append(fc * wm(el[cm],  wc, 3, mu_e))

    for cut in Q2_CUTS:
        m = ok & (q2_all > cut)
        w = w_all[m]; q2 = q2_all[m]; cat = cat_all[m]
        W = np.sum(w); mu_q2 = gm(q2, w); mu_q = 0. if raw else mu_q2
        for c in CATEGORIES:
            cm = cat == c; wc = w[cm]; fc = np.sum(wc) / W if W > 0 else 0.0
            contribs["q2_1"][c].append(fc * wm(q2[cm], wc, 1))
            contribs["q2_2"][c].append(fc * wm(q2[cm], wc, 2, mu_q))
            contribs["q2_3"][c].append(fc * wm(q2[cm], wc, 3, mu_q))

    ylabels = YLABELS_RAW if raw else YLABELS
    colors  = _spectral(len(CATEGORIES))
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), dpi=150)
    seen: set[str] = set()

    for r, (_, keys, cuts, xlabel) in enumerate(_GRID):
        for ci, key in enumerate(keys):
            ax   = axes[r, ci]
            stks = np.stack([np.asarray(contribs[key][cat]) for cat in CATEGORIES])
            lbls = [CAT_LABELS[cat] if (r == 0 and ci == 0) else None for cat in CATEGORIES]
            ax.stackplot(cuts, np.maximum(stks, 0.), labels=lbls, colors=colors, alpha=0.85)
            if np.any(stks < 0):
                ax.stackplot(cuts, np.minimum(stks, 0.), colors=colors, alpha=0.85)
                ax.plot(cuts, stks.sum(axis=0), color="black", lw=1.2, ls="--", zorder=10)
                ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.4)
            ax.set_ylabel(ylabels[key]); ax.set_xlabel(xlabel)
            ax.set_xlim(float(cuts[0]), float(cuts[-1])); ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][ci])
        axes[r, 0].text(-0.38, 0.5, ROW_TITLES[r], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center")

    cat_handles, cat_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(cat_handles, cat_labels, loc="upper center", ncol=min(len(cat_handles), 8),
               frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=14)
    fig.tight_layout(rect=[0.04, 0.02, 0.99, 0.95])
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    print(f"  Wrote {outpath.name}")


_stacked_plot(cocktail.loc[_ok], fig5 / "stacked_central_3x3.png", "central")
_stacked_plot(cocktail.loc[_ok], fig5 / "stacked_raw_3x3.png", "raw")


# ── H. Gap mode overlay + SM+gap+data plots ───────────────────────────────────

REQUIRED_COLS = ["Mx", "El_B", "q2", "total_weight", "category"]
sm_df = cocktail[REQUIRED_COLS].copy()
sm_df.loc[sm_df["category"] == "Ds(*) K", "total_weight"] = 0.0

active_gap     = [m for m in GAP_MODES if m["mode"] in gap_frames]
gap_categories = [m["category"] for m in active_gap]

all_frames = [sm_df]
for meta in active_gap:
    gdf = gap_frames[meta["mode"]][REQUIRED_COLS].copy()
    gdf["category"] = meta["category"]
    all_frames.append(gdf)
combined = pd.concat(all_frames, ignore_index=True)

w_c    = combined["total_weight"].to_numpy(dtype=float)
mx2_c  = np.square(combined["Mx"].to_numpy(dtype=float))
el_c   = combined["El_B"].to_numpy(dtype=float)
q2_c   = combined["q2"].to_numpy(dtype=float)
cat_c  = combined["category"].to_numpy()
is_gap = np.isin(cat_c, gap_categories)
ok_c   = np.isfinite(mx2_c) & np.isfinite(el_c) & np.isfinite(q2_c) & (w_c > 0)


def _compute_gap_all(moment_space):
    raw = moment_space == "raw"
    def gm(x, w): d = np.sum(w); return float(np.sum(w*x)/d) if d > 0 else 0.
    def wm(x, w, n, mu=0.): d = np.sum(w); return float(np.sum(w*(x-mu)**n)/d) if d > 0 else 0.

    sm_total: dict[str, list] = {k: [] for k in MKEYS}
    gap_out:  dict[str, dict[str, list]] = {k: {c: [] for c in gap_categories} for k in MKEYS}

    for cut in EL_CUTS:
        m = ok_c & (el_c > cut)
        w = w_c[m]; mx2 = mx2_c[m]; el = el_c[m]; cat = cat_c[m]; gap = is_gap[m]
        W = np.sum(w)
        w_sm = w[~gap]; mx2_sm = mx2[~gap]; el_sm = el[~gap]
        mu_mx2_sm = gm(mx2_sm, w_sm); mu_el_sm = gm(el_sm, w_sm)
        mu_m = 0. if raw else mu_mx2_sm; mu_e = 0. if raw else mu_el_sm
        sm_total["mx_1"].append(wm(mx2_sm, w_sm, 1))
        sm_total["mx_2"].append(wm(mx2_sm, w_sm, 2, mu_m))
        sm_total["mx_3"].append(wm(mx2_sm, w_sm, 3, mu_m))
        sm_total["el_1"].append(wm(el_sm,  w_sm, 1))
        sm_total["el_2"].append(wm(el_sm,  w_sm, 2, mu_e))
        sm_total["el_3"].append(wm(el_sm,  w_sm, 3, mu_e))
        mu_mx2_all = gm(mx2, w); mu_el_all = gm(el, w)
        mu_ma = 0. if raw else mu_mx2_all; mu_ea = 0. if raw else mu_el_all
        for gc in gap_categories:
            cm = cat == gc; wc = w[cm]; fc = np.sum(wc) / W if W > 0 else 0.
            gap_out["mx_1"][gc].append(fc * wm(mx2[cm], wc, 1))
            gap_out["mx_2"][gc].append(fc * wm(mx2[cm], wc, 2, mu_ma))
            gap_out["mx_3"][gc].append(fc * wm(mx2[cm], wc, 3, mu_ma))
            gap_out["el_1"][gc].append(fc * wm(el[cm], wc, 1))
            gap_out["el_2"][gc].append(fc * wm(el[cm], wc, 2, mu_ea))
            gap_out["el_3"][gc].append(fc * wm(el[cm], wc, 3, mu_ea))

    for cut in Q2_CUTS:
        m = ok_c & (q2_c > cut)
        w = w_c[m]; q2 = q2_c[m]; cat = cat_c[m]; gap = is_gap[m]
        W = np.sum(w)
        w_sm = w[~gap]; q2_sm = q2[~gap]; mu_q2_sm = gm(q2_sm, w_sm)
        mu_q = 0. if raw else mu_q2_sm
        sm_total["q2_1"].append(wm(q2_sm, w_sm, 1))
        sm_total["q2_2"].append(wm(q2_sm, w_sm, 2, mu_q))
        sm_total["q2_3"].append(wm(q2_sm, w_sm, 3, mu_q))
        mu_q2_all = gm(q2, w); mu_qa = 0. if raw else mu_q2_all
        for gc in gap_categories:
            cm = cat == gc; wc = w[cm]; fc = np.sum(wc) / W if W > 0 else 0.
            gap_out["q2_1"][gc].append(fc * wm(q2[cm], wc, 1))
            gap_out["q2_2"][gc].append(fc * wm(q2[cm], wc, 2, mu_qa))
            gap_out["q2_3"][gc].append(fc * wm(q2[cm], wc, 3, mu_qa))

    sm_arr  = {k: np.asarray(v, dtype=float) for k, v in sm_total.items()}
    gap_arr = {k: {c: np.asarray(v, dtype=float) for c, v in cats.items()}
               for k, cats in gap_out.items()}
    return sm_arr, gap_arr


# Gap overlay (central only: shows additive contribution of each gap mode)
_, gap_ov = _compute_gap_all("central")
YLABELS_OV = {
    "mx_1": r"$\mathcal{B}_{\rm gap}\langle M_X^2 \rangle$",
    "mx_2": r"$\mathcal{B}_{\rm gap}\langle (M_X^2-\mu)^2 \rangle$",
    "mx_3": r"$\mathcal{B}_{\rm gap}\langle (M_X^2-\mu)^3 \rangle$",
    "el_1": r"$\mathcal{B}_{\rm gap}\langle E_\ell \rangle$",
    "el_2": r"$\mathcal{B}_{\rm gap}\langle (E_\ell-\mu)^2 \rangle$",
    "el_3": r"$\mathcal{B}_{\rm gap}\langle (E_\ell-\mu)^3 \rangle$",
    "q2_1": r"$\mathcal{B}_{\rm gap}\langle q^2 \rangle$",
    "q2_2": r"$\mathcal{B}_{\rm gap}\langle (q^2-\mu)^2 \rangle$",
    "q2_3": r"$\mathcal{B}_{\rm gap}\langle (q^2-\mu)^3 \rangle$",
}
fig_ov, axes_ov = plt.subplots(3, 3, figsize=(18, 10), dpi=150)
for r, (_, keys, cuts, xlabel) in enumerate(_GRID):
    for ci, key in enumerate(keys):
        ax = axes_ov[r, ci]
        for meta in active_gap:
            ax.plot(cuts, gap_ov[key][meta["category"]],
                    color=meta["color"], ls=meta["ls"], lw=meta["lw"],
                    label=meta["label"] if (r == 0 and ci == 0) else None)
        ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.4)
        ax.set_ylabel(YLABELS_OV[key]); ax.set_xlabel(xlabel)
        ax.set_xlim(float(cuts[0]), float(cuts[-1])); ax.grid(alpha=0.2)
        if r == 0: ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][ci])
    axes_ov[r, 0].text(-0.38, 0.5, ROW_TITLES[r], transform=axes_ov[r, 0].transAxes,
                       rotation=90, va="center", ha="center")
ov_h = [mlines.Line2D([], [], color=m["color"], ls=m["ls"], lw=m["lw"], label=m["label"])
        for m in active_gap]
fig_ov.legend(ov_h, [m["label"] for m in active_gap],
              loc="upper center", ncol=len(ov_h), frameon=False,
              bbox_to_anchor=(0.5, 0.998), fontsize=11)
fig_ov.tight_layout(rect=[0.04, 0.02, 0.99, 0.95])
fig_ov.savefig(fig5 / "gap_modes_overlay_3x3.png", bbox_inches="tight"); plt.close(fig_ov)
print("  Wrote gap_modes_overlay_3x3.png")


# SM + gap + data plots for both central and raw
for space in ("central", "raw"):
    sm_arr, gap_arr = _compute_gap_all(space)
    ylabels  = YLABELS_RAW if space == "raw" else YLABELS
    exp_data = exp_raw if space == "raw" else exp_central
    avg_data = avg_raw if space == "raw" else avg_central
    suffix   = "_raw" if space == "raw" else "_central"
    fig_sm, axes_sm = plt.subplots(3, 3, figsize=(18, 10), dpi=150)
    seen_sm: set[str] = set()
    for r, (_, keys, cuts, xlabel) in enumerate(_GRID):
        for ci, key in enumerate(keys):
            ax = axes_sm[r, ci]
            ax.plot(cuts, sm_arr[key], color="0.45", lw=2.2, zorder=5,
                    label="SM cocktail" if (r == ci == 0) else None)
            for meta in active_gap:
                ax.plot(cuts, gap_arr[key][meta["category"]],
                        color=meta["color"], ls=meta["ls"], lw=meta["lw"], zorder=6,
                        label=meta["label"] if (r == ci == 0) else None)
            _plot_avg(ax, key, avg_data, show_label=(r == 0 and ci == 0))
            _plot_exp(ax, key, exp_data, seen_sm)
            ax.axhline(0, color="black", lw=0.4, ls=":", alpha=0.4)
            ax.set_ylabel(ylabels[key]); ax.set_xlabel(xlabel)
            ax.set_xlim(float(cuts[0]), float(cuts[-1])); ax.grid(alpha=0.2)
            if r == 0: ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][ci])
        axes_sm[r, 0].text(-0.38, 0.5, ROW_TITLES[r], transform=axes_sm[r, 0].transAxes,
                           rotation=90, va="center", ha="center")
    sm_h   = mlines.Line2D([], [], color="0.45", lw=2.2, label="SM cocktail")
    gap_h2 = [mlines.Line2D([], [], color=m["color"], ls=m["ls"], lw=m["lw"],
                             label=m["label"]) for m in active_gap]
    exp_h, exp_l = [], []
    for ax in fig_sm.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            known = {"SM cocktail"} | {m["label"] for m in active_gap}
            if l and l not in known and l not in exp_l:
                exp_h.append(h); exp_l.append(l)
    fig_sm.legend([sm_h] + gap_h2 + exp_h, ["SM cocktail"] + [m["label"] for m in active_gap] + exp_l,
                  loc="upper center", ncol=len(active_gap) + 1 + len(exp_h),
                  frameon=False, bbox_to_anchor=(0.5, 0.998), fontsize=10)
    fig_sm.tight_layout(rect=[0.04, 0.02, 0.99, 0.95])
    fname = f"gap_modes_with_sm{suffix}_3x3.png"
    fig_sm.savefig(fig5 / fname, bbox_inches="tight"); plt.close(fig_sm)
    print(f"  Wrote {fname}")


# ── I. Write sem_moments_raw.json ─────────────────────────────────────────────

print("[7/7] Writing output/5/sem_moments_raw.json …")
payload = {
    "cuts": {"el": EL_CUTS.tolist(), "q2": Q2_CUTS.tolist()},
    "sem_nominal_central": {k: v.tolist() for k, v in sem_cen_nom.items()},
    "sem_std_central":     {k: v.tolist() for k, v in sem_std.items()},
    "sem_nominal_raw":     {k: v.tolist() for k, v in sem_raw_nom.items()},
    "sem_std_raw":         {k: v.tolist() for k, v in sem_std.items()},
    "experimental_data":   exp_central,
    "experimental_data_raw": exp_raw,
}
(out5 / "sem_moments_raw.json").write_text(json.dumps(payload, indent=2))
print("Done.")
