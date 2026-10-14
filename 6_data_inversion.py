#!/usr/bin/env python3
"""Step 6: threshold-0 spectra from the HQE moments (MaxEnt, illustration only) and the gap-moment uncertainty budget.

  HQE inversion   exact Hausdorff check + MaxEnt of the HQE-fit moments (Markus Prim's likelihood toys), without
                  SEM subtraction (inclusive spectrum, against the SEM stack) and with it (gap spectrum)
  Budget          relative uncertainty of the gap moments per source (TeX table)

Bands are 68% intervals over the first --n-toys toys of the shared pairing (lib/gap.py). Reads the step-4 bundle.

Usage:
  python3 6_data_inversion.py [--bundle FILE] [--n-toys 1000]
  python3 6_data_inversion.py --submit [--after JOB]
Outputs go to <output>/6 (the configured output directory, else ./output) and figures/6.
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
from lib.asimov import smooth_truth
from lib.bundle import Bundle, default_path, output_root
from lib.gap import Gap
from lib.maxent import MaxEnt, hausdorff_check, raw_to_mu01
from lib.systematics import write_budget_tex

OBS = ["Mx", "El", "Q2"]
VAR = {"Mx": "mx2", "El": "el", "Q2": "q2"}


def hqe_results(G, n_toys, subtract):
    """Exact inversion at threshold 0. subtract=True: gap density (HQE toy x SEM toy x B_gap); False: inclusive."""
    mc = G.cfg["maxent"]
    solver = MaxEnt(np.linspace(0, 1, 400), mom_tol=mc["mom_tol"])
    n = min(n_toys, G.n)
    c = G.c()[:n]
    raw_c = G.c0 * G.inc_c - (G.c0 - 1) * G.sem_c if subtract else G.inc_c
    raw_t = c * G.inc_t[:n] - (c - 1) * G.sem_t[:n] if subtract else G.inc_t[:n]
    out = {}
    for k, obs in enumerate(OBS):
        v = VAR[obs]
        lo, hi = G.sup[v]

        def invert(raw):
            m = raw_to_mu01(np.r_[1.0, raw], lo, hi)
            if not hausdorff_check(m)[0]:
                return None, False
            r = solver.solve(m, mc["alpha"][v], mc["beta"][v])
            return r.f / (hi - lo), r.converged

        f_c, ok_c = invert(raw_c[3 * k:3 * k + 3])
        res = [invert(r[3 * k:3 * k + 3]) for r in raw_t]
        curves = [f for f, ok in res if ok]
        out[obs] = dict(x=lo + solver.t * (hi - lo), f=f_c if ok_c else None,
                        band=np.percentile(curves, [16, 84], axis=0) if len(curves) > 10 else None,
                        hausdorff_frac=sum(f is not None for f, _ in res) / n, converged_frac=len(curves) / n)
        print(f"[HQE {'gap' if subtract else 'full'} {obs}] central {'ok' if ok_c else 'FAILED'}, "
              f"Hausdorff {out[obs]['hausdorff_frac']:.3f}, converged {out[obs]['converged_frac']:.3f}", flush=True)
    return out


def hqe_budget(G):
    """Relative uncertainty [%] of the threshold-0 gap moments m1..m3 per source; FF and BF-mode on top of SEM stat."""
    c0, ct = G.c0, G.c()
    nom = c0 * G.inc_c - (c0 - 1) * G.sem_c
    var = lambda r: r.var(0)
    rel = lambda v: 100 * np.sqrt(np.clip(v, 0, None)) / np.abs(nom)
    v_stat = var(c0 * G.inc_c - (c0 - 1) * G.sem_v["stat"])
    rows = {"hqe": rel(var(c0 * G.inc_t - (c0 - 1) * G.sem_c)), "stat": rel(v_stat),
            "ff": rel(var(c0 * G.inc_c - (c0 - 1) * G.sem_v["stat_ff"]) - v_stat),
            "bf_mode": rel(var(c0 * G.inc_c - (c0 - 1) * G.sem_v["stat_bfmode"]) - v_stat),
            "bf_gap": rel(var(ct * G.inc_c - (ct - 1) * G.sem_c))}
    rows["quadrature"] = np.sqrt(sum(r ** 2 for r in rows.values()))
    rows["total"] = rel(var(ct * G.inc_t - (ct - 1) * G.sem_t))
    return {k: v.tolist() for k, v in rows.items()}


BUDGET_LABELS = {"hqe": "HQE fit", "stat": "MC stat.", "ff": "Form factors", "bf_mode": r"$\mathcal{B}$ per mode",
                 "bf_gap": r"$\mathcal{B}_\mathrm{gap}$", "quadrature": "Quadrature sum", "total": "Total"}


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


def plot_gap(plt, cfg, hqe_gap, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = figure(plt)
    axes[0].set_xlim(1.75, 3.0)
    for ax, obs in zip(axes, OBS):
        h = hqe_gap[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], HQE_COLOR, obs)
        if obs != "Mx":
            ax.set_xlim(*cfg["data"]["support"][VAR[obs]])
    infeasible_note(axes[0], hqe_gap, 0.94)
    ymx = max(peak(hqe_gap["Mx"], 2 * np.sqrt(hqe_gap["Mx"]["x"])), default=None)
    handles = [Line2D([], [], color=HQE_COLOR, lw=2.6, label="HQE inversion"),
               Patch(color="0.5", alpha=0.3, lw=0, label="68% (toys)")]
    finish(fig, axes, handles, path, ymax=[None if ymx is None else 1.35 * ymx, 1.3 * max(peak(hqe_gap["El"])), 0.7])
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
        # stacked, smoothed categories as in step 5; single-hadron categories (D, D* in Mx) as lines
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


def run(B, n_toys):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401  (house style)
    import matplotlib.pyplot as plt

    G, T = Gap(B), B.toys
    local = yaml.safe_load(open(HERE / "config.yaml"))   # plotting settings may be newer than the bundle
    od, fd = output_root(local, HERE) / "6", HERE / "figures" / "6"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    budget = hqe_budget(G)
    write_budget_tex(budget, od / "moment_budget_data_thr0.tex", BUDGET_LABELS)
    hqe_gap, hqe_full = hqe_results(G, n_toys, subtract=True), hqe_results(G, n_toys, subtract=False)
    plot_gap(plt, G.cfg, hqe_gap, fd / "gap_density.pdf")
    plot_hqe_full(plt, hqe_full, fd / "hqe_inclusive_density.pdf")
    plot_hqe_vs_stack(plt, T, hqe_full, local["data"]["stack_smoothing"], fd / "hqe_vs_sem_stack.pdf")
    keep = lambda r: {k: r[k] for k in ("hausdorff_frac", "converged_frac")} | {"central_ok": r["f"] is not None}
    json.dump(dict(hqe_gap={o: keep(r) for o, r in hqe_gap.items()}, hqe_full={o: keep(r) for o, r in hqe_full.items()},
                   moment_budget_data_thr0_pct=budget), open(od / "summary.json", "w"), indent=1)
    print(f"-> {fd}, {od}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", help="step-4 bundle (default: <output>/4/ if built here, else data/)")
    p.add_argument("--n-toys", type=int, default=1000)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after", help="LSF job name the submitted job waits for")
    args = p.parse_args()
    bundle = Path(args.bundle).resolve() if args.bundle else default_path(HERE)
    if args.submit:
        logs = HERE / "logs" / "6"
        logs.mkdir(parents=True, exist_ok=True)
        dep = f' -w "done({args.after})"' if args.after else ""
        cmd = (f'bsub -q {yaml.safe_load(open(HERE / "config.yaml"))["batch"]["queue"]} -env all -J s6inversion{dep} -n 2 -oo {logs}/run.log '
               f'"cd {HERE} && python3 6_data_inversion.py --bundle {bundle} --n-toys {args.n_toys}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        run(Bundle(bundle), args.n_toys)


if __name__ == "__main__":
    main()
