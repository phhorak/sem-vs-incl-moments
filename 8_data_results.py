#!/usr/bin/env python3
"""Step 8: gap density from the measured moments.

  Joint fit     El, q2: one density f(t) ~ t^a (1-t)^b exp(sum_k c_k t^k) fit by chi2 to the
                conditional residual moments at all thresholds, weighted by the step-6 covariance.
  Mx            separate fits per El threshold (Mx moments are El-selected sub-populations).
                Both fits only with config data.joint_fit (or --joint-fit); off by default.
  HQE           exact Hausdorff + MaxEnt inversion of the HQE-fit moments at threshold 0
                (Markus Prim's likelihood toys), with and without SEM subtraction.
  Feasibility   Hausdorff check + MaxEnt inversion per threshold (TeX table).
  Gap modes     candidate gap-mode shapes (SHOW_MODES) drawn over the gap density.

All bands are 68% toy intervals: the step-6 residual toys ("all" source: exp + SEM stat + FF +
BF-mode + B_gap) refit per toy; for HQE, HQE toys paired with step-6 SEM toys.

Everything is read from the step-6 bundle (lib/bundle.py); the MC samples are not needed, except
the step-3 gap-mode samples for the gap-mode overlay (skipped where they are missing).

Usage:
  python3 8_data_results.py [--bundle FILE] [--n-toys 1000]
  python3 8_data_results.py --submit [--after JOB]
Outputs go to <output>/8 (the configured output directory, else ./output) and figures/8.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import yaml

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from lib.asimov import COLS, MAX_ORDER, smooth_truth
from lib.jointfit import VAR, M_JOINT, JointFit, band, joint_fit, mx_fits
from lib.maxent import MaxEnt, hausdorff_check, raw_to_mu01
from lib.bundle import Bundle, default_path, output_root
from lib.systematics import c_true, write_budget_tex

SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode", "all"]
OBS = ["Mx", "El", "Q2"]
# ── Joint and per-threshold fits (lib/jointfit, shared with step 7) ─────────────

def joint_results(cfg, B, n_toys, rng):
    res = {}
    for obs in ("El", "Q2"):
        R = B.residual(obs, "all")
        r = joint_fit(cfg, obs, R, n_toys, rng)
        asimov = JointFit(cfg, obs, R["thr"], None, True, r["lo"], r["hi"])
        c_a, _ = asimov.fit(R["nominal"], M_JOINT)
        per_source = {}
        for src in SOURCES:
            S = B.residual(obs, src)
            j = JointFit(cfg, obs, S["thr"], S["cov"], True, r["lo"], r["hi"])
            c, _ = j.fit(S["nominal"], M_JOINT)
            per_source[src] = j.chi2(c, j.target(S["nominal"]), M_JOINT) / r["dof"]
        print(f"[{obs}] joint fit chi2/ndf = {r['chi2']:.1f}/{r['dof']}; {r['n_ok']}/{r['n_toys']} toys refit "
              f"in {r['seconds']:.0f}s; per source {per_source}", flush=True)
        res[obs] = r | dict(asimov_chi2=asimov.chi2(c_a, asimov.target(R["nominal"]), M_JOINT),
                            chi2_per_dof_by_source=per_source)
    return res


def mx_results(cfg, B, n_toys, rng):
    out = mx_fits(cfg, B.residual("Mx", "all"), n_toys, rng)
    print(f"[Mx] {len(out)} per-threshold fits", flush=True)
    return out


# ── HQE inversion ────────────────────────────────────────────────────────────

def hqe_results(cfg, T, n_toys, rng, subtract):
    """Exact inversion at threshold 0. subtract=True: gap density (HQE toy x SEM toy x B_gap);
    False: full inclusive density, HQE toys only."""
    raw_c9, raw_t9 = T["hqe_central"], T["hqe_toys"]
    mc, sysc, sup = cfg["maxent"], cfg["systematics"], cfg["data"]["support"]
    solver = MaxEnt(np.linspace(0, 1, 400), mom_tol=mc["mom_tol"])
    bf_gap = float(T["bf_gap"])
    out = {}
    for k, obs in enumerate(OBS):
        v = VAR[obs]
        lo, hi = sup[v]
        a, b = mc["alpha"][v], mc["beta"][v]

        def invert(raw):
            m = raw_to_mu01(np.r_[1.0, raw], lo, hi)
            if not hausdorff_check(m)[0]:
                return None, False
            r = solver.solve(m, a, b)
            return r.f / (hi - lo), r.converged

        sem_nom, sem_toys = T["nom_sem0"][MAX_ORDER * k:MAX_ORDER * k + 3], \
            T["sem0_stat_ff_bfmode"][:, MAX_ORDER * k:MAX_ORDER * k + 3]
        c0 = float(c_true(sysc, bf_gap))
        raw_c, hqe = raw_c9[3 * k:3 * k + 3], raw_t9[:, 3 * k:3 * k + 3]
        f_c, ok_c = invert(c0 * raw_c - (c0 - 1) * sem_nom if subtract else raw_c)
        n_hqe = len(hqe)
        n = min(n_toys, n_hqe, len(sem_toys))
        ih, js = rng.choice(n_hqe, n, replace=False), rng.choice(len(sem_toys), n, replace=False)
        curves, n_haus = [], 0
        for i, j in zip(ih, js):
            raw = hqe[i]
            if subtract:
                c = float(c_true(sysc, bf_gap, T["z_gap"][j]))
                raw = c * raw - (c - 1) * sem_toys[j]
            f, ok = invert(raw)
            n_haus += f is not None
            if ok:
                curves.append(f)
        out[obs] = dict(x=lo + solver.t * (hi - lo), f=f_c if ok_c else None, band=band(curves),
                        hausdorff_frac=n_haus / n, converged_frac=len(curves) / n)
        print(f"[HQE {'gap' if subtract else 'full'} {obs}] central {'ok' if ok_c else 'FAILED'}, "
              f"Hausdorff {n_haus}/{n}, converged {len(curves)}/{n}", flush=True)
    return out


def hqe_budget(cfg, T, orders=(1, 2, 3)):
    """Relative uncertainty [%] of the threshold-0 residual gap moments per source (HQE toys,
    SEM template, B_gap), from all toys; FF and BF-mode are taken on top of SEM stat."""
    sysc, bf_gap = cfg["systematics"], float(T["bf_gap"])
    raw_c9, raw_t9 = T["hqe_central"], T["hqe_toys"]
    n = len(T["z_gap"])
    raw_h = raw_t9[np.random.default_rng(7).choice(len(raw_t9), n, replace=n > len(raw_t9))]
    c0, ct = float(c_true(sysc, bf_gap)), c_true(sysc, bf_gap, T["z_gap"])[:, None]
    sem_cols = [MAX_ORDER * k + o - 1 for k in range(3) for o in orders]
    hqe_cols = [3 * k + o - 1 for k in range(3) for o in orders]
    sem_nom, raw_c = T["nom_sem0"][sem_cols], raw_c9[hqe_cols]
    sem = lambda v: T[f"sem0_{v}"][:, sem_cols]
    nom = c0 * raw_c - (c0 - 1) * sem_nom
    var = lambda r: r.var(0)
    v_stat = var(c0 * raw_c - (c0 - 1) * sem("stat"))
    rel = lambda v: 100 * np.sqrt(np.clip(v, 0, None)) / np.abs(nom)
    rows = {"hqe": rel(var(c0 * raw_h[:, hqe_cols] - (c0 - 1) * sem_nom)),
            "stat": rel(v_stat),
            "ff": rel(var(c0 * raw_c - (c0 - 1) * sem("stat_ff")) - v_stat),
            "bf_mode": rel(var(c0 * raw_c - (c0 - 1) * sem("stat_bfmode")) - v_stat),
            "bf_gap": rel(var(ct * raw_c - (ct - 1) * sem_nom))}
    rows["quadrature"] = np.sqrt(sum(r ** 2 for r in rows.values()))
    rows["total"] = rel(var(ct * raw_h[:, hqe_cols] - (ct - 1) * sem("stat_ff_bfmode")))
    return {k: v.tolist() for k, v in rows.items()}


BUDGET_LABELS = {"hqe": "HQE fit", "stat": "MC stat.", "ff": "Form factors", "bf_mode": r"$\mathcal{B}$ per mode",
                 "bf_gap": r"$\mathcal{B}_\mathrm{gap}$", "quadrature": "Quadrature sum", "total": "Total"}


# ── Feasibility table ────────────────────────────────────────────────────────

def feasibility(cfg, B):
    """Per threshold: residual [0,1] moments on the data support, Hausdorff and MaxEnt (N=4)."""
    mc, sup = cfg["maxent"], cfg["data"]["support"]
    solver = MaxEnt(np.linspace(0, 1, mc["grid_size"]), mom_tol=mc["mom_tol"])
    rows = []
    for obs in OBS:
        v = VAR[obs]
        lo, hi = sup[v]
        R = B.residual(obs, "all")
        for i, thr in enumerate(R["thr"]):
            m = raw_to_mu01(np.r_[1.0, R["nominal"][3 * i:3 * i + 3]], lo, hi)
            haus = hausdorff_check(m)[0]
            conv = haus and solver.solve(m, mc["alpha"][v], mc["beta"][v]).converged
            rows.append(dict(obs=obs, thr=thr, m=m[1:].tolist(), var=float(m[2] - m[1] ** 2),
                             hausdorff=bool(haus), maxent=bool(conv)))
    return rows


def write_feasibility_tex(rows, path):
    name = {"Mx": r"$M_X^2$", "El": r"$E_\ell$", "Q2": r"$q^2$"}
    ck = lambda b: r"$\checkmark$" if b else r"$\times$"
    lines = [r"\begin{tabular}{cccccccc}", r"\toprule",
             r"Observable & Threshold & $m_1^\text{gap}$ & $m_2^\text{gap}$ & $m_3^\text{gap}$ & "
             r"$\mathrm{Var}[t]$ & Hausdorff & MaxEnt \\", r"\midrule"]
    for r in rows:
        lines.append(f"{name[r['obs']]} & {r['thr']:.2f} & " + " & ".join(f"{x:.4f}" for x in r["m"])
                     + f" & {r['var']:+.2e} & {ck(r['hausdorff'])} & {ck(r['maxent'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


# ── Candidate gap modes ──────────────────────────────────────────────────────

# (gap mode, legend, color, linestyle) drawn over the gap density in gap_density_modes.pdf
SHOW_MODES = [("DsKenu", r"$D_sK$", "#6a3d9a", "--"),
              ("Dp1DstEta", r"$D_1'\to D^*\eta$", "#b15928", "-."),
              ("LcPenu", r"$\Lambda_c\bar{p}$", "#33a02c", ":"),
              # Mx, El, q2 are set by B -> D1 l nu alone: the D1 -> D gamma sample stands for every D1 decay
              ("D1Dgamma", r"$B\to D_1(2420)\ell\nu$", "#ff7f00", "-")]
# (key, legend, color, linestyle, cocktail decays) SEM modes drawn for comparison in
# gap_density_modes_El_q2.pdf; e and mu, B0 and B+
COCKTAIL_MODES = [("Dlnu", r"$B\to D\ell\nu$", "#17becf", "-",
                   ["Bp_Denu", "B0_Denu", "Bp_Dmunu", "B0_Dmunu"]),
                  ("D0stlnu", r"$B\to D_0^*\ell\nu$", "#7f7f7f", "--",
                   ["Bp_D0stenu", "B0_D0stenu", "Bp_D0stmunu", "B0_D0stmunu"])]
MODE_BINS = {"Mx": 400, "El": 300, "Q2": 300}


def shape(cfg, d, el_cut, q2_lo):
    """Smoothed densities of one sample (B0 and B+ weighted equally), unit area on the data support:
    Mx for El > el_cut, q2 above q2_lo. {obs: (x, f)}."""
    sup, sm = cfg["data"]["support"], cfg["asimov"]["truth_smoothing"]
    ranges = {"Mx": tuple(np.sqrt(sup["mx2"])), "El": tuple(sup["el"]), "Q2": (q2_lo, sup["q2"][1])}
    d = d[np.isfinite(d[COLS[:4]].to_numpy(float)).all(axis=1) & (d["total_weight"] > 0)]
    w = d["total_weight"].to_numpy(float).copy()
    species = d["decay_name"].astype(str).str[:2].to_numpy()
    for s in np.unique(species):
        w[species == s] /= w[species == s].sum()
    x = {"Mx": d["Mx"].to_numpy(float), "El": d["El_B"].to_numpy(float), "Q2": d["q2"].to_numpy(float)}
    out = {}
    for obs in OBS:
        sel = x["El"] > el_cut if obs == "Mx" else slice(None)
        h, e = np.histogram(x[obs][sel], MODE_BINS[obs], ranges[obs], weights=w[sel])
        db = e[1] - e[0]
        f = smooth_truth(h / db, sm[obs.lower()] / db, finite_lo=obs == "Q2")
        out[obs] = (0.5 * (e[1:] + e[:-1]), f / (f.sum() * db))
    print(f"[modes] {len(w)} events, species {sorted(np.unique(species))}", flush=True)
    return out


def mode_shapes(cfg, out3, el_cut, q2_lo, cocktail=False):
    """shape() of the SHOW_MODES samples (step 3) and, with cocktail, of the COCKTAIL_MODES decays
    of the SEM cocktail. {mode: {obs: (x, f)}}; missing samples are left out."""
    import pandas as pd
    out = {}
    for mode, *_ in SHOW_MODES:
        path = out3 / f"{mode}.parquet"
        if not path.exists():
            print(f"[modes] {path} missing, {mode} not drawn", flush=True)
            continue
        print(f"[modes] {mode}:", end=" ")
        out[mode] = shape(cfg, pd.read_parquet(path, columns=COLS), el_cut, q2_lo)
    path = out3 / "cocktail.parquet"
    if cocktail and path.exists():
        names = [n for *_, decays in COCKTAIL_MODES for n in decays]
        ck = pd.read_parquet(path, columns=COLS, filters=[("decay_name", "in", names)])
        ck["decay_name"] = ck["decay_name"].astype(str)
        for key, *_, decays in COCKTAIL_MODES:
            print(f"[modes] {key}:", end=" ")
            out[key] = shape(cfg, ck[ck["decay_name"].isin(decays)], el_cut, q2_lo)
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

FIT_COLOR, HQE_COLOR = "#1f77b4", "#d62728"
CATEGORIES = ["D", "D*", "D**", "D(*) pi", "D(*) pi pi", "Ds(*) K"]
CAT_LABELS = [r"$D$", r"$D^*$", r"$D^{**}$", r"$D^{(*)}\pi$", r"$D^{(*)}\pi\pi$", r"$D_s^{(*)}K$"]
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
XLABEL = {"Mx": r"$M_X\ [\mathrm{GeV}]$", "El": r"$E_\ell^B\ [\mathrm{GeV}]$", "Q2": r"$q^2\ [\mathrm{GeV}^2]$"}
YLABEL = {"Mx": r"$f(M_X)\ [\mathrm{GeV}^{-1}]$", "El": r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$",
          "Q2": r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"}
FS_LABEL, FS_TICK, FS_LEG = 26, 19, 20


def style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(True)


def to_mx(x, *ys):
    """Mx2 grid and densities -> Mx grid and densities (Jacobian 2 Mx)."""
    mx = np.sqrt(x)
    return (mx, *[None if y is None else y * 2 * mx for y in ys])


def figure(plt):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.2), dpi=200)
    for ax, obs in zip(axes, OBS):
        ax.set_xlabel(XLABEL[obs], fontsize=FS_LABEL)
        ax.set_ylabel(YLABEL[obs], fontsize=FS_LABEL)
        style(ax)
    return fig, axes


def finish(fig, axes, handles, path, ymax=None):
    for i, ax in enumerate(axes):
        ax.set_ylim(0, None if ymax is None else ymax[i])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FS_LEG,
               bbox_to_anchor=(0.5, -0.11), handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def draw_curve(ax, x, f, bnd, color, obs, ls="-"):
    if obs == "Mx":
        x, f, lo_, hi_ = to_mx(x, f, *(bnd if bnd is not None else (None, None)))
        bnd = None if lo_ is None else (lo_, hi_)
    if bnd is not None:
        ax.fill_between(x, bnd[0], bnd[1], color=color, alpha=0.3, lw=0)
    if f is not None:
        ax.plot(x, f, color=color, lw=2.6, ls=ls)
    return x


def peak(r, jac=1.0):
    """Largest value of a curve and its band (empty if the curve is missing)."""
    return [float(np.max(y * jac)) for y in [r["f"], *(r["band"] if r["band"] is not None else [])] if y is not None]


def infeasible_note(ax, hqe_gap, y=0.74):
    if hqe_gap["Mx"]["f"] is None:
        ax.text(0.95, y, "HQE moments not\nHausdorff-feasible", transform=ax.transAxes,
                fontsize=17, color=HQE_COLOR, ha="right", va="top")


def plot_gap(plt, cfg, joint, mx, hqe_gap, path):
    """HQE inversion, with the joint and per-El-cut fits where they were run (joint, mx non-empty)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib as mpl
    fig, axes = figure(plt)
    if mx:
        cmap, thrs = plt.get_cmap("viridis"), sorted(mx)
        norm = mpl.colors.Normalize(min(thrs), max(thrs))
        for thr in thrs:
            m = mx[thr]
            xg, fg = to_mx(m["x"], m["f"])
            axes[0].plot(xg, fg, color=cmap(norm(thr)), lw=1.6, alpha=0.9)
        cax = axes[0].inset_axes([0.55, 0.9, 0.4, 0.04])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
        cb.set_label(r"$E_\ell$ cut [GeV]", fontsize=16)
        cb.ax.tick_params(labelsize=14)
    axes[0].set_xlim(1.75, 3.0)
    sup = cfg["data"]["support"]
    for ax, obs in zip(axes, OBS):
        h = hqe_gap[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], HQE_COLOR, obs)
        if obs in joint:
            j = joint[obs]
            draw_curve(ax, j["x"], j["f"], j["band"], FIT_COLOR, obs)
            ax.text(0.06, 0.94, rf"$\chi^2/\mathrm{{ndf}} = {j['chi2']:.1f}/{j['dof']}$", transform=ax.transAxes,
                    fontsize=20, color=FIT_COLOR, ha="left", va="top")
        if obs != "Mx":
            ax.set_xlim(*sup[VAR[obs]])
    infeasible_note(axes[0], hqe_gap, 0.74 if mx else 0.94)
    sel = lambda x: (x >= 1.75) & (x <= 3.0)
    ymx = max((float(np.max((m["f"] * 2 * np.sqrt(m["x"]))[sel(np.sqrt(m["x"]))])) for m in mx.values()), default=None)
    ymx = ymx if ymx is not None else max(peak(hqe_gap["Mx"], 2 * np.sqrt(hqe_gap["Mx"]["x"])), default=None)
    yel = max(peak(hqe_gap["El"]) + (peak(joint["El"]) if "El" in joint else []))
    handles = ([Line2D([], [], color=FIT_COLOR, lw=2.6, label=r"Joint $\chi^2$ fit")] if joint else []) \
        + ([Line2D([], [], color="0.35", lw=1.6, label=r"Per-$E_\ell$-cut fits ($M_X$)")] if mx else []) \
        + [Line2D([], [], color=HQE_COLOR, lw=2.6, label="HQE inversion"),
           Patch(color="0.5", alpha=0.3, lw=0, label="68% (toys)")]
    finish(fig, axes, handles, path, ymax=[None if ymx is None else 1.35 * ymx, 1.3 * yel, 0.7])
    plt.close(fig)


def plot_hqe_full(plt, hqe_full, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = figure(plt)
    for ax, obs in zip(axes, OBS):
        h = hqe_full[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], FIT_COLOR, obs)
        ax.set_xlim(h["x"][0], h["x"][-1])
    axes[0].set_xlim(1.6, 4.0)
    handles = [Line2D([], [], color=FIT_COLOR, lw=2.6, label="HQE inversion (no SEM subtraction)"),
               Patch(color=FIT_COLOR, alpha=0.3, lw=0, label="68% (HQE toys)")]
    finish(fig, axes, handles, path)
    plt.close(fig)


def plot_hqe_vs_stack(plt, T, hqe_full, smoothing, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = [str(c) for c in T["stack_categories"]]
    fig, axes = figure(plt)
    for ax, obs in zip(axes, OBS):
        # stacked, smoothed categories as in step 7; single-hadron categories (D, D* in Mx) as lines
        e = T[f"stackedges_{obs}"]
        xc, sig = 0.5 * (e[1:] + e[:-1]), smoothing[obs.lower()] / (e[1] - e[0])
        sm = lambda h: smooth_truth(h, sig, finite_lo=obs == "Q2")
        base, peaks = np.zeros(len(xc)), np.zeros(len(xc), bool)
        for cat, color in zip(CATEGORIES, CAT_COLORS):
            h = T[f"stack_nominal_{obs}"][names.index(cat)] if cat in names else np.zeros(len(xc))
            if obs == "Mx" and 0 < h.sum() < np.sort(h)[-2:].sum() / 0.95:
                ax.axvline(np.average(xc, weights=h), color=color, lw=3.5, alpha=0.9)
                peaks |= h > 0
                continue
            ax.fill_between(xc, base, base + sm(h), color=color, alpha=0.55, lw=0)
            base = base + sm(h)
        ax.plot(xc, base, color="0.25", lw=0.9)
        # toys hold the total density: leave out the bins the smoothed lines would spill into
        near = np.convolve(peaks, np.ones(2 * int(4 * sig) + 3), "same") > 0
        lo_, hi_ = np.percentile([sm(h) for h in T[f"stack_{obs}"]], [16, 84], axis=0)
        ax.fill_between(xc, lo_, hi_, where=~near, color="0.15", alpha=0.45, lw=0, zorder=4)
        h = hqe_full[obs]
        x, f, b0, b1 = h["x"], h["f"], *(h["band"] if h["band"] is not None else (None, None))
        if obs == "Mx":
            x, f, b0, b1 = to_mx(x, f, b0, b1)
        if b0 is not None:
            ax.fill_between(x, b0, b1, facecolor="none", edgecolor="black", hatch="///", lw=0, zorder=5)
        if f is not None:
            ax.plot(x, f, color="black", lw=2.6, zorder=6)
        ax.set_xlim(e[0], e[-1])
    axes[0].set_xlim(1.6, 4.0)
    hm = hqe_full["Mx"]
    top = [y for y in (hm["f"], None if hm["band"] is None else hm["band"][1]) if y is not None]
    ymax = [1.4 * max(float(np.max(y * 2 * np.sqrt(hm["x"]))) for y in top) if top else None, None, None]
    handles = [Patch(color=c, alpha=0.55, lw=0, label=l) for cat, c, l in zip(CATEGORIES, CAT_COLORS, CAT_LABELS)
               if cat in names and T["stack_nominal_El"][names.index(cat)].sum() > 0]
    handles += [Patch(color="0.15", alpha=0.45, lw=0, label="SEM 68%"),
                Line2D([], [], color="black", lw=2.6, label="HQE inversion"),
                Patch(facecolor="none", edgecolor="black", hatch="///", label="HQE 68%")]
    finish(fig, axes, handles, path, ymax=ymax)
    plt.close(fig)


def plot_gap_modes(plt, cfg, joint, mx, hqe_gap, modes, path):
    """The gap density of plot_gap (Mx fit at the loosest El cut, where the fits were run) with the
    candidate gap-mode shapes."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = figure(plt)
    if mx:
        thr0 = min(mx)
        for thr in sorted(mx)[1:]:
            axes[0].plot(*to_mx(mx[thr]["x"], mx[thr]["f"]), color="0.75", lw=1.0)
        draw_curve(axes[0], mx[thr0]["x"], mx[thr0]["f"], mx[thr0]["band"], FIT_COLOR, "Mx")
        axes[0].text(0.06, 0.94, rf"$E_\ell > {thr0:.1f}\ \mathrm{{GeV}}$", transform=axes[0].transAxes,
                     fontsize=20, color=FIT_COLOR, ha="left", va="top")
    sup = cfg["data"]["support"]
    for ax, obs in zip(axes, OBS):
        h = hqe_gap[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], HQE_COLOR, obs)
        if obs in joint:
            j = joint[obs]
            draw_curve(ax, j["x"], j["f"], j["band"], FIT_COLOR, obs)
            ax.text(0.06, 0.94, rf"$\chi^2/\mathrm{{ndf}} = {j['chi2']:.1f}/{j['dof']}$", transform=ax.transAxes,
                    fontsize=20, color=FIT_COLOR, ha="left", va="top")
        if obs != "Mx":
            ax.set_xlim(*sup[VAR[obs]])
        for mode, _, color, ls in SHOW_MODES:
            if mode in modes:
                ax.plot(*modes[mode][obs], color=color, ls=ls, lw=2.2)
    axes[0].set_xlim(1.75, 4.0)
    infeasible_note(axes[0], hqe_gap, 0.94)
    # y range from the fits and the HQE curve (Mx without fits: the typical mode peak); narrow mode
    # peaks may run off the top
    y_mx = peak(mx[thr0], 2 * np.sqrt(mx[thr0]["x"])) if mx else \
        [float(np.median([np.max(modes[m]["Mx"][1]) for m in modes]))]
    ymax = [1.4 * max(y_mx)] + [1.3 * max(peak(hqe_gap[o]) + (peak(joint[o]) if o in joint else []))
                                for o in ("El", "Q2")]
    handles = ([Line2D([], [], color=FIT_COLOR, lw=2.6, label=r"$\chi^2$ fit")] if joint else []) \
        + ([Line2D([], [], color="0.75", lw=1.0, label=r"Tighter $E_\ell$ cuts ($M_X$)")] if mx else []) \
        + [Line2D([], [], color=HQE_COLOR, lw=2.6, label="HQE inversion"),
           Patch(color="0.5", alpha=0.3, lw=0, label="68% (toys)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=2.2, label=l) for m, l, c, ls in SHOW_MODES if m in modes]
    for ax, y in zip(axes, ymax):
        ax.set_ylim(0, y)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEG,
               bbox_to_anchor=(0.5, -0.2), handlelength=2.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_modes_el_q2(plt, cfg, hqe_gap, modes, path):
    """El and q2: the HQE gap inversion (68% band) with the candidate gap modes and, for scale,
    the COCKTAIL_MODES SEM shapes; all of unit area on the full support (threshold 0)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), dpi=200)
    sup = cfg["data"]["support"]
    styles = [(m, l, c, ls) for m, l, c, ls in SHOW_MODES] + [(m, l, c, ls) for m, l, c, ls, _ in COCKTAIL_MODES]
    for ax, obs in zip(axes, ("El", "Q2")):
        h = hqe_gap[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], HQE_COLOR, obs)
        top = peak(h)
        for mode, _, color, ls in styles:
            if mode in modes:
                x, f = modes[mode][obs]
                ax.plot(x, f, color=color, ls=ls, lw=2.2)
                top.append(float(np.max(f)))
        ax.set_xlim(*sup[VAR[obs]])
        ax.set_ylim(0, 1.1 * max(top))
        ax.set_xlabel(XLABEL[obs], fontsize=FS_LABEL)
        ax.set_ylabel(YLABEL[obs], fontsize=FS_LABEL)
        style(ax)
    handles = [Line2D([], [], color=HQE_COLOR, lw=2.6, label="HQE inversion"),
               Patch(color=HQE_COLOR, alpha=0.3, lw=0, label="68% (toys)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=2.2, label=l) for m, l, c, ls in styles if m in modes]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEG,
               bbox_to_anchor=(0.5, -0.2), handlelength=2.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(B, n_toys, seed, joint_on=None):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401  (house style)
    import matplotlib.pyplot as plt

    cfg, T = B.cfg, B.toys
    local = yaml.safe_load(open(HERE / "config.yaml"))   # plotting settings may be newer than the bundle
    od, fd = output_root(local, HERE) / "8", HERE / "figures" / "8"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    budget = hqe_budget(cfg, T)
    write_budget_tex(budget, od / "moment_budget_data_thr0.tex", BUDGET_LABELS)
    rows = feasibility(cfg, B)
    write_feasibility_tex(rows, od / "feasibility_table.tex")
    if joint_on is None:
        joint_on = local["data"].get("joint_fit", True)
    joint = joint_results(cfg, B, n_toys, rng) if joint_on else {}
    mx = mx_results(cfg, B, n_toys, rng) if joint_on else {}
    hqe_gap = hqe_results(cfg, T, n_toys, rng, subtract=True)
    hqe_full = hqe_results(cfg, T, n_toys, rng, subtract=False)

    plot_gap(plt, cfg, joint, mx, hqe_gap, fd / "gap_density.pdf")
    # mode shapes normalized like the curves they are compared with: with the fits, Mx above the
    # loosest El cut and q2 above the joint-fit start; without, threshold 0 on the full support
    out3, q2_0 = output_root(local, HERE) / "3", cfg["data"]["support"]["q2"][0]
    modes0 = mode_shapes(local, out3, 0.0, q2_0, cocktail=True)
    modes = mode_shapes(local, out3, min(mx), joint["Q2"]["x"][0]) if joint_on else modes0
    if modes:
        plot_gap_modes(plt, cfg, joint, mx, hqe_gap, modes, fd / "gap_density_modes.pdf")
        plot_gap_modes_el_q2(plt, cfg, hqe_gap, modes0, fd / "gap_density_modes_El_q2.pdf")
    plot_hqe_full(plt, hqe_full, fd / "hqe_inclusive_density.pdf")
    plot_hqe_vs_stack(plt, T, hqe_full, local["data"]["stack_smoothing"], fd / "hqe_vs_sem_stack.pdf")

    summary = dict(
        joint={o: {k: v for k, v in r.items() if k not in ("x", "f", "band", "lo", "hi", "seconds")}
               for o, r in joint.items()},
        mx={f"{t:.2f}": dict(chi2=r["chi2"]) for t, r in mx.items()},
        hqe_gap={o: {k: r[k] for k in ("hausdorff_frac", "converged_frac")} | {"central_ok": r["f"] is not None}
                 for o, r in hqe_gap.items()},
        hqe_full={o: {k: r[k] for k in ("hausdorff_frac", "converged_frac")} | {"central_ok": r["f"] is not None}
                  for o, r in hqe_full.items()},
        moment_budget_data_thr0_pct=budget, feasibility=rows)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
    print(f"-> {fd}, {od}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", help="step-6 bundle (default: <output>/6/ if built here, else data/)")
    p.add_argument("--n-toys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after", help="LSF job name the submitted job waits for")
    p.add_argument("--joint-fit", action=argparse.BooleanOptionalAction, default=None,
                   help="joint and per-El-cut fits on/off (default: config data.joint_fit)")
    args = p.parse_args()
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    bundle = Path(args.bundle).resolve() if args.bundle else default_path(HERE)
    if args.submit:
        logs = HERE / "logs" / "8"
        logs.mkdir(parents=True, exist_ok=True)
        dep = f' -w "done({args.after})"' if args.after else ""
        jf = "" if args.joint_fit is None else " --joint-fit" if args.joint_fit else " --no-joint-fit"
        cmd = (f'bsub -q {cfg["generation"]["queue"]} -env all -J s8data{dep} -n 4 -oo {logs}/run.log '
               f'"cd {HERE} && python3 8_data_results.py --bundle {bundle} --n-toys {args.n_toys}{jf}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        run(Bundle(bundle), args.n_toys, args.seed, args.joint_fit)


if __name__ == "__main__":
    main()
