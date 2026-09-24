#!/usr/bin/env python3
"""Step 7: Asimov closure test of the moment inversion.

Each scenario in config `asimov.scenarios` is a known gap truth, injected into a pseudo-inclusive
sample (1-f)*cocktail + f*truth with f = B_gap/B_incl, and recovered from the residual moments
c_true*m_incl - c_sem*m_SEM (Hausdorff check + MaxEnt) using moments m_1..m_N, N in maxent.n_moments.

Bands come from the step-6 toys, recombined per systematic source (each on top of stat):
  stat          cocktail bootstraps A and B, bootstrap of the gap truth
  ff, bf_mode   SEM-template variations
  incl_moments  HQE-fit toys at threshold 0, as relative shifts of the pseudo-inclusive moments
  bf_gap        assumed gap budget in c_true
  total         all of the above

Everything is read from the step-6 bundle (lib/bundle.py), which also carries the configuration
the toys were made with; the MC samples are not needed.

Usage:
  python3 7_asimov_closure.py --bundle FILE --run         # all inversions and plots, locally
  python3 7_asimov_closure.py --submit [--after JOB]      # inversion jobs + dependent plot job (LSF)
  python3 7_asimov_closure.py --invert SCENARIO SOURCE
  python3 7_asimov_closure.py --plot
Outputs go to <output>/7 (the configured output directory, else ./output) and figures/7.
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
from lib.asimov import MAX_ORDER
from lib.bundle import Bundle, default_path, output_root
from lib.maxent import MaxEnt, hausdorff_check, raw_to_mu01
from lib.systematics import c_true, hqe_incl_deviations, write_budget_tex

SOURCES = ["stat", "ff", "bf_mode", "incl_moments", "bf_gap", "total"]
OBS = ["mx2", "el", "q2"]
PCTS = [2.5, 16, 50, 84, 97.5]

OUT = output_root(yaml.safe_load(open(HERE / "config.yaml")), HERE) / "7"   # local setting, not the bundle's


def load(B):
    """Configuration, step-6 toys, and per toy the HQE relative deviations of the inclusive moments."""
    cfg, T = B.cfg, B.toys
    dev = hqe_incl_deviations(*B.hqe, len(T["z_gap"]), np.random.default_rng([cfg["toys"]["seed"], 6]), MAX_ORDER)
    return cfg, T, dev




def residual(cfg, T, dev, scen, src):
    """Raw residual moments, orders 1..MAX_ORDER of mx2, el, q2, per toy (src=None: nominal)."""
    sysc, bf_gap = cfg["systematics"], float(T["bf_gap"])
    f = bf_gap / sysc["bf_incl"]
    if src is None:
        incl, sem, c = (1 - f) * T["nom_sem0"] + f * T[f"nom_gap_{scen}"], T["nom_sem0"], c_true(sysc, bf_gap)
        return c * incl - (c - 1) * sem
    incl = (1 - f) * T["ck_B"] + f * T[f"gap_{scen}"]
    if src in ("incl_moments", "total"):
        incl = incl * (1 + dev)
    variant = {"ff": "stat_ff", "bf_mode": "stat_bfmode", "total": "stat_ff_bfmode"}.get(src, "stat")
    sem = T[f"sem0_{variant}"]
    c = c_true(sysc, bf_gap, T["z_gap"] if src in ("bf_gap", "total") else 0.0)
    c = np.broadcast_to(c, (len(sem),))[:, None]
    return c * incl - (c - 1) * sem


def mu01(raw, obs, sup):
    k = OBS.index(obs) * MAX_ORDER
    return raw_to_mu01(np.concatenate([[1.0], raw[k:k + MAX_ORDER]]), *sup[obs])


def all_n(mc):
    return sorted(set(mc["n_moments"]) | set(mc["n_moments_extended"]))


def scenario_support(T, scen):
    return {o: tuple(T[f"support_{scen}"][i]) for i, o in enumerate(OBS)}


# ── Inversion job ────────────────────────────────────────────────────────────

def run_invert(B, scen, src):
    cfg, T, dev = load(B)
    mc = cfg["maxent"]
    sup = scenario_support(T, scen)
    solver = MaxEnt(np.linspace(0, 1, mc["grid_size"]), mom_tol=mc["mom_tol"])
    nom = residual(cfg, T, dev, scen, None)
    toys = residual(cfg, T, dev, scen, src)
    out = {}
    for obs in OBS:
        a, b = mc["alpha"][obs], mc["beta"][obs]
        span = sup[obs][1] - sup[obs][0]
        m_nom = mu01(nom, obs, sup)
        for N in (all_n(mc) if src == "total" else mc["n_moments"]):
            lam0 = solver.solve(m_nom[:N + 1], a, b).lam
            f, haus, err = [], np.zeros(len(toys), bool), np.full(len(toys), np.nan)
            for i, raw in enumerate(toys):
                m = mu01(raw, obs, sup)[:N + 1]
                haus[i] = hausdorff_check(m)[0]
                if not haus[i]:
                    continue
                r = solver.solve(m, a, b, lam0=lam0)
                err[i] = r.mom_err
                if r.converged:
                    f.append(r.f / span)
            key = f"{obs}_N{N}"
            out[f"pct_{key}"] = np.percentile(f, PCTS, axis=0) if len(f) > 10 else np.full((5, len(solver.t)), np.nan)
            out[f"haus_{key}"], out[f"err_{key}"] = haus, err
            out[f"nconv_{key}"] = len(f)
            print(f"{scen}/{src} {key}: Hausdorff {haus.mean():.3f}, converged {len(f)}/{len(toys)}",
                  flush=True)
    od = OUT
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"bands_{scen}_{src}.npz", **out)


# ── Plotting ─────────────────────────────────────────────────────────────────

CON_COLOR, UNCON_COLOR = "#ff7f0e", "#6a51a3"
COMP_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
SOURCE_COLORS = {"stat": "0.55", "ff": "#1b9e77", "bf_mode": "#d95f02", "incl_moments": "#7570b3",
                 "bf_gap": "#e7298a", "total": "black"}
SOURCE_LABELS = {"stat": "MC stat.", "ff": "Form factors", "bf_mode": r"$\mathcal{B}$ per mode",
                 "incl_moments": "Inclusive moments", "bf_gap": r"$\mathcal{B}_\mathrm{gap}$",
                 "total": "Total"}
XLABEL = {"mx2": r"$M_X\ [\mathrm{GeV}]$", "el": r"$E_\ell^B\ [\mathrm{GeV}]$", "q2": r"$q^2\ [\mathrm{GeV}^2]$"}
YLABEL = {"mx2": r"$f(M_X)\ [\mathrm{GeV}^{-1}]$", "el": r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$",
          "q2": r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"}
FS_LABEL, FS_TICK, FS_LEG = 26, 19, 17


def style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(True)


def smooth_truth(h, sigma_bins, finite_lo=False):
    """Gaussian smoothing that keeps the endpoint behaviour: odd reflection where the density
    vanishes (forces zero at the edge), even reflection where it is finite (q2 -> 0)."""
    from scipy.ndimage import gaussian_filter1d
    n = min(int(4 * sigma_bins) + 1, len(h) - 1)
    lo = h[1:n + 1][::-1] * (1 if finite_lo else -1)
    hi = -h[-n - 1:-1][::-1]
    s = gaussian_filter1d(np.concatenate([lo, h, hi]), sigma_bins, mode="constant")[n:n + len(h)]
    return np.clip(s, 0.0, None)


class Panels:
    """Everything one scenario's figures need, in display coordinates (Mx rather than Mx2)."""

    def __init__(self, cfg, T, dev, scen, truth):
        mc, sc = cfg["maxent"], cfg["asimov"]["scenarios"][scen]
        self.scen, self.Ns, self.Ns_ext = scen, list(mc["n_moments"]), list(mc["n_moments_extended"])
        self.legends = [c["legend"] for c in sc["components"]]
        self.prior_free = {o: mc["alpha"][o] == 0 and mc["beta"][o] == 0 for o in OBS}
        sup = scenario_support(T, scen)
        t = np.linspace(0, 1, mc["grid_size"])
        solver = MaxEnt(t, mom_tol=mc["mom_tol"])
        nom = residual(cfg, T, dev, scen, None)
        self.closure = float(np.max(np.abs(nom - T[f"nom_gap_{scen}"])))
        self.d = {}
        for obs in OBS:
            lo, hi = sup[obs]
            x = lo + t * (hi - lo)
            jac = 2 * np.sqrt(x) if obs == "mx2" else np.ones_like(x)
            xd = np.sqrt(x) if obs == "mx2" else x
            xlim = tuple(truth["mxrange"]) if obs == "mx2" else (lo, hi)
            m = mu01(nom, obs, sup)
            curves = {}
            for N in all_n(mc):
                con = solver.solve(m[:N + 1], mc["alpha"][obs], mc["beta"][obs])
                unc = solver.solve(m[:N + 1], 0.0, 0.0)
                curves[N] = dict(con=con.f / (hi - lo) * jac, unc=unc.f / (hi - lo) * jac,
                                 haus=hausdorff_check(m[:N + 1])[0], err=(con.mom_err, unc.mom_err),
                                 f_x=(con.f / (hi - lo), unc.f / (hi - lo)))
            bands = {}
            for src in SOURCES:
                p = OUT / f"bands_{scen}_{src}.npz"
                if p.exists():
                    B = np.load(p)
                    bands[src] = {N: dict(pct=B[f"pct_{obs}_N{N}"] * jac, haus=B[f"haus_{obs}_N{N}"],
                                          nconv=int(B[f"nconv_{obs}_N{N}"]), err=B[f"err_{obs}_N{N}"])
                                  for N in all_n(mc) if f"pct_{obs}_N{N}" in B}
            # truth: stacked, smoothed components (single-hadron components drawn as a line)
            edges = truth[f"{obs}_edges"]
            db = edges[1] - edges[0]
            sig = cfg["asimov"]["truth_smoothing"]["mx" if obs == "mx2" else obs]
            dens, delta = [], []
            for i, (wsum, mu, sd) in enumerate(truth["stats"]):
                delta.append(obs == "mx2" and sd < 0.02)
                h = truth[f"{obs}_dens"][i]
                dens.append((mu, wsum) if delta[-1] else smooth_truth(h, sig / db, finite_lo=obs == "q2"))
            # L1 distance to the truth, on the reconstruction variable (Mx2, El, q2)
            h_x = truth[f"{obs}_l1"]
            e_x = np.linspace(lo, hi, len(h_x) + 1)
            xc = 0.5 * (e_x[1:] + e_x[:-1])
            for N, c in curves.items():
                c["l1"] = [float(np.sum(np.abs(np.interp(xc, x, fx) - h_x)) * np.diff(e_x)[0]) for fx in c["f_x"]]
            self.d[obs] = dict(xd=xd, xlim=xlim, edges=edges, dens=dens, delta=delta, curves=curves, bands=bands)

    def ymax(self, obs, Ns):
        """Largest truth or nominal curve value in view (as the reference figure; bands may clip),
        ignoring curves within 0.15 of a delta-like truth component."""
        D = self.d[obs]
        sel = (D["xd"] >= D["xlim"][0]) & (D["xd"] <= D["xlim"][1])
        for d, dl in zip(D["dens"], D["delta"]):
            if dl:
                sel &= np.abs(D["xd"] - d[0]) > 0.15
        kinds = ("con",) if self.prior_free[obs] else ("con", "unc")
        vals = [D["curves"][N][k][sel].max() for N in Ns for k in kinds]
        truth = sum(d for d, dl in zip(D["dens"], D["delta"]) if not dl)
        return max(vals + [float(np.max(truth))])

    def draw(self, axes, obs, band, Ns, headroom=1.65):
        D = self.d[obs]
        centers = 0.5 * (D["edges"][1:] + D["edges"][:-1])
        ytop = headroom * self.ymax(obs, Ns)
        for ax, N in zip(axes.flat, Ns):
            base = np.zeros_like(centers)
            for i, (d, dl) in enumerate(zip(D["dens"], D["delta"])):
                if dl:
                    ax.vlines(d[0], 0, ytop, color=COMP_COLORS[i], lw=3.5, alpha=0.9, label=self.legends[i])
                    continue
                ax.fill_between(centers, base, base + d, color=COMP_COLORS[i], alpha=0.55, lw=0,
                                label=self.legends[i])
                base = base + d
            ax.plot(centers, base, color="0.25", lw=0.9)
            c = D["curves"][N]
            if band and N in D["bands"].get("total", {}):
                b = D["bands"]["total"][N]
                ax.fill_between(D["xd"], b["pct"][1], b["pct"][3], color=CON_COLOR, alpha=0.35, lw=0,
                                label="68%")
                ax.text(0.06, 0.82, f"feasible {b['nconv'] / len(b['haus']):.0%}", transform=ax.transAxes,
                        fontsize=FS_LEG, color="0.35", ha="left", va="top")
            if not self.prior_free[obs]:
                ax.plot(D["xd"], c["unc"], color=UNCON_COLOR, lw=2.2, ls="--", label="Unconstrained")
            ax.plot(D["xd"], c["con"], color=CON_COLOR, lw=2.4, label="Constrained")
            ax.text(0.06, 0.94, rf"$N={N}$", transform=ax.transAxes, fontsize=FS_LABEL, color="0.15",
                    ha="left", va="top")
            ax.set_xlim(*D["xlim"])
            style(ax)
        axes.flat[0].set_ylim(0, ytop)
        for ax in axes[-1, :]:
            ax.set_xlabel(XLABEL[obs], fontsize=FS_LABEL)
        for ax in axes[:, 0]:
            ax.set_ylabel(YLABEL[obs], fontsize=FS_LABEL)

    def legend_handles(self, band, obs=None):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        h = [Patch(color=COMP_COLORS[i], alpha=0.55, lw=0, label=l) for i, l in enumerate(self.legends)]
        if obs is None or not self.prior_free[obs]:
            h += [Line2D([], [], color=UNCON_COLOR, lw=2.2, ls="--", label="Unconstrained")]
        h += [Line2D([], [], color=CON_COLOR, lw=2.4,
                     label="MaxEnt" if obs is not None and self.prior_free[obs] else "Constrained")]
        if band:
            h += [Patch(color=CON_COLOR, alpha=0.35, lw=0, label="68% (toys)")]
        return h


def run_plot(B):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401  (house style)
    import matplotlib.pyplot as plt

    cfg, T, dev = load(B)
    fig_root = HERE / "figures" / "7"

    summary = {}
    for scen in cfg["asimov"]["scenarios"]:
        fig_dir = fig_root / scen
        fig_dir.mkdir(parents=True, exist_ok=True)
        P = Panels(cfg, T, dev, scen, B.truth(scen))
        print(f"{scen}: nominal closure max|residual - truth| = {P.closure:.1e}")
        for band in (False, True):
            tag = "_band" if band else ""
            for Ns, ncol, suffix in ((P.Ns, 2, ""), (P.Ns_ext, 3, f"_N{P.Ns_ext[0]}to{P.Ns_ext[-1]}")):
                for obs in OBS:
                    fig, axes = plt.subplots(2, ncol, figsize=(4.75 * ncol, 9.0), dpi=250, sharex=True,
                                             sharey=True)
                    P.draw(axes, obs, band, Ns)
                    axes[0, 0].legend(handles=P.legend_handles(band, obs), fontsize=FS_LEG - 2,
                                      loc="upper right", frameon=False, handlelength=1.4, labelspacing=0.4,
                                      borderaxespad=0.3)
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.06, hspace=0.08)
                    fig.savefig(fig_dir / f"convergence_{obs}{suffix}{tag}.pdf", bbox_inches="tight")
                    plt.close(fig)
                fig = plt.figure(figsize=(14 * ncol, 9.2), dpi=200)
                for sf, obs in zip(fig.subfigures(1, 3, wspace=0.02), OBS):
                    axes = sf.subplots(2, ncol, sharex=True, sharey=True)
                    P.draw(axes, obs, band, Ns, headroom=1.3)
                    sf.subplots_adjust(left=0.19 / (ncol / 2), right=0.985, bottom=0.13, top=0.985,
                                       wspace=0.06, hspace=0.08)
                h = P.legend_handles(band)
                fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=22,
                           bbox_to_anchor=(0.5, -0.045), handlelength=1.8)
                fig.savefig(fig_dir / f"convergence_all3{suffix}{tag}.pdf", bbox_inches="tight")
                plt.close(fig)
        plot_breakdown(plt, P, fig_dir / "systematics_breakdown.pdf", N=3)
        summary[scen] = scenario_summary(P)
        summary[scen]["moment_budget_pct"] = moment_budget(cfg, T, dev, scen)
        write_budget_tex(summary[scen]["moment_budget_pct"],
                         OUT / f"moment_budget_{scen}.tex", BUDGET_LABELS)
        print(f"  -> {fig_dir}")
    od = OUT
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
    write_tex(summary, od / "closure_table.tex")


def plot_breakdown(plt, P, path, N):
    """68% half-width of the constrained band per source, relative to the nominal peak."""
    if not any(P.d[o]["bands"] for o in OBS):
        return
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6), dpi=200)
    for ax, obs in zip(axes, OBS):
        D = P.d[obs]
        nom = D["curves"][N]["con"]
        core = (nom > 0.05 * nom.max()) & (D["xd"] >= D["xlim"][0]) & (D["xd"] <= D["xlim"][1])
        top = 0.0
        for src in SOURCES:
            if src not in D["bands"]:
                continue
            pct = D["bands"][src][N]["pct"]
            hw = 0.5 * (pct[3] - pct[1]) / nom.max()
            top = max(top, float(np.nanmax(hw[core])))
            ax.plot(D["xd"], hw, color=SOURCE_COLORS[src], lw=3.0 if src == "total" else 2.2,
                    label=SOURCE_LABELS[src])
        ax.set_xlim(*D["xlim"])
        ax.set_ylim(0, 1.25 * top)
        ax.set_xlabel(XLABEL[obs], fontsize=FS_LABEL)
        style(ax)
    axes[0].set_ylabel(r"$\sigma_{68\%}(f)\,/\,f_\mathrm{max}$", fontsize=FS_LABEL)
    axes[0].text(0.94, 0.94, rf"$N={N}$", transform=axes[0].transAxes, fontsize=FS_LABEL, color="0.15",
                 ha="right", va="top")
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=20,
                     bbox_to_anchor=(0.5, -0.1), handlelength=2.2)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scenario_summary(P):
    out = {"nominal_closure": P.closure}
    for obs in OBS:
        D = P.d[obs]
        for N, c in D["curves"].items():
            row = dict(hausdorff_nominal=bool(c["haus"]), mom_err_nominal=list(c["err"]),
                       l1_to_truth_constrained=c["l1"][0], l1_to_truth_unconstrained=c["l1"][1])
            for src, b in D["bands"].items():
                if N not in b:
                    continue
                n = len(b[N]["haus"])
                err = b[N]["err"][np.isfinite(b[N]["err"])]
                row[src] = dict(hausdorff_pass=float(b[N]["haus"].mean()), converged=b[N]["nconv"] / n,
                                mom_err_p50=float(np.median(err)) if len(err) else None,
                                mom_err_p99=float(np.percentile(err, 99)) if len(err) else None,
                                band68_halfwidth_rel=float(np.nanmean(0.5 * (b[N]["pct"][3] - b[N]["pct"][1]))
                                                           / c["con"].max()))
            out[f"{obs}_N{N}"] = row
    return out


def moment_budget(cfg, T, dev, scen, orders=(1, 2, 3)):
    """Relative uncertainty [%] of the residual gap moments per source, from all toys (no
    inversion, so no feasibility selection). Each source row is its own contribution on top of
    stat: sqrt(var(stat + source) - var(stat))."""
    nom = residual(cfg, T, dev, scen, None)
    var = {src: residual(cfg, T, dev, scen, src).var(0) for src in SOURCES}
    cols = [OBS.index(o) * MAX_ORDER + k - 1 for o in OBS for k in orders]
    rel = lambda v: 100 * np.sqrt(np.clip(v, 0, None))[cols] / np.abs(nom[cols])
    rows = {"stat": rel(var["stat"])}
    rows.update({s: rel(var[s] - var["stat"]) for s in SOURCES if s not in ("stat", "total")})
    rows["quadrature"] = np.sqrt(sum(r ** 2 for r in rows.values()))
    rows["total"] = rel(var["total"])
    return {k: v.tolist() for k, v in rows.items()}


BUDGET_LABELS = dict(SOURCE_LABELS, quadrature="Quadrature sum")


def write_tex(summary, path):
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Scenario & Observable & $N$ & Hausdorff & Converged & $L_1$ to truth \\", r"\midrule"]
    names = {"mx2": r"$M_X^2$", "el": r"$E_\ell$", "q2": r"$q^2$"}
    for scen, S in summary.items():
        for obs in OBS:
            for key in sorted(k for k in S if k.startswith(obs + "_N")):
                r = S[key]
                tot = r.get("total", {})
                lines.append(f"{scen} & {names[obs]} & {key.split('_N')[1]} & "
                             f"{tot.get('hausdorff_pass', float('nan')):.3f} & "
                             f"{tot.get('converged', float('nan')):.3f} & {r['l1_to_truth_constrained']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


# ── Submit ───────────────────────────────────────────────────────────────────

def run_submit(cfg, bundle, dry_run, after=None):
    logs = HERE / "logs" / "7"
    logs.mkdir(parents=True, exist_ok=True)
    queue, tag = cfg["generation"]["queue"], "s7inv"
    py = f"cd {HERE} && python3 7_asimov_closure.py --bundle {bundle}"
    dep = f' -w "done({after})"' if after else ""
    cmds = [f'bsub -q {queue} -env all -J {tag}_{s}_{src}{dep} -oo {logs}/invert_{s}_{src}.log "{py} --invert {s} {src}"'
            for s in cfg["asimov"]["scenarios"] for src in SOURCES]
    cmds.append(f'bsub -q {queue} -env all -J {tag}_plot -w "ended({tag}_*)" -n 2 -oo {logs}/plot.log "{py} --plot"')
    for cmd in cmds:
        print(cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", help="step-6 bundle (default: <output>/6/ if built here, else data/)")
    p.add_argument("--run", action="store_true", help="all inversions, then the plots, in this process")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--invert", nargs=2, metavar=("SCENARIO", "SOURCE"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--after", help="LSF job name the submitted jobs wait for")
    args = p.parse_args()
    bundle = Path(args.bundle) if args.bundle else default_path(HERE)
    if args.submit:
        run_submit(yaml.safe_load(open(HERE / "config.yaml")), bundle.resolve(), args.dry_run, args.after)
        return
    if not (args.run or args.invert or args.plot):
        p.print_help()
        return
    B = Bundle(bundle)
    if args.invert:
        run_invert(B, *args.invert)
    elif args.plot:
        run_plot(B)
    else:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(scen, src) for scen in B.cfg["asimov"]["scenarios"] for src in SOURCES]
        with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as ex:
            list(ex.map(run_invert, [B] * len(jobs), *zip(*jobs)))
        run_plot(B)


if __name__ == "__main__":
    main()
