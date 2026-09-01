#!/usr/bin/env python3
"""
8_hausdorff_data.py

Subtract cocktail SEM raw moments from the experimentally averaged moments
to obtain residual moments, then plot the 3x3 panel (rows: Mx, El, q2 /
cols: 1st, 2nd, 3rd moment) as a function of the lepton-energy or q² cut.
Also overlays the nominal toy gap curve from the MC for reference.

Usage:
  cd clean && python3 8_hausdorff_data.py --config config.yaml
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import json
from math import comb

# Avoid thread-spawn failures on constrained batch/login nodes when many --toy-job chunks run
# concurrently via bsub (mirrors 7_hausdorff_toy.py, which now shares the toy per-event
# systematics machinery with this script).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.moments import compute_raw_curves_np
from lib import systematics as syst
from lib.systematics import (
    read_parquet_downcast, MX2_LO_MIN, MX2_HI_MAX, EL_HI_MAX, Q2_HI_MAX,
)
from lib import joint_maxent as jm


MOMENT_KEYS = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]

YLABELS = {
    "mx_1": r"$\Delta\langle M_X^2 \rangle\ [\mathrm{GeV}^2]$",
    "mx_2": r"$\Delta\langle (M_X^2)^2 \rangle\ [\mathrm{GeV}^4]$",
    "mx_3": r"$\Delta\langle (M_X^2)^3 \rangle\ [\mathrm{GeV}^6]$",
    "el_1": r"$\Delta\langle E_\ell \rangle\ [\mathrm{GeV}]$",
    "el_2": r"$\Delta\langle E_\ell^2 \rangle\ [\mathrm{GeV}^2]$",
    "el_3": r"$\Delta\langle E_\ell^3 \rangle\ [\mathrm{GeV}^3]$",
    "q2_1": r"$\Delta\langle q^2 \rangle\ [\mathrm{GeV}^2]$",
    "q2_2": r"$\Delta\langle (q^2)^2 \rangle\ [\mathrm{GeV}^4]$",
    "q2_3": r"$\Delta\langle (q^2)^3 \rangle\ [\mathrm{GeV}^6]$",
}

ROW_TITLES = [r"$M_X^2$", r"$E_\ell$", r"$q^2$"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--submit",    action="store_true",
                   help="Submit toy chunk jobs to bsub instead of running locally")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print bsub commands without executing (use with --submit)")
    p.add_argument("--toy-job",   action="store_true",
                   help="Run a single toy chunk (used by batch jobs)")
    p.add_argument("--chunk",     type=int, default=0,
                   help="Chunk index (0-based, used with --toy-job)")
    p.add_argument("--n-chunks",  type=int, default=1,
                   help="Total number of chunks (used with --toy-job)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    hd   = cfg.get("hausdorff_data", {})
    ht   = cfg.get("hausdorff_toy", {})

    out_dir = Path(cfg["paths"]["output"]) / "8"
    fig_dir = Path(cfg["paths"]["figures"]) / "8"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cocktail_path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"

    # ── Load averaged experimental data and SEM cocktail moments ─────────────
    exp_path = Path(cfg["paths"]["output"]) / "4" / "experimental_average.json"
    sem_path = Path(cfg["paths"]["output"]) / "5" / "sem_moments_raw.json"

    print("Loading experimental average and SEM moments …")
    with open(exp_path) as f:
        exp_avg = json.load(f)
    with open(sem_path) as f:
        sem_raw = json.load(f)

    # Raw moments, not central: these are what raw_to_mu01()/hausdorff_moment_check()
    # actually consume below, and matching that keeps this residual plot consistent
    # with figures/5/sem_vs_data_raw_3x3.png (central moments subtract mean-dependent
    # cross terms, so a central-moment residual's sign at order >= 2 need not agree
    # with the raw-moment residual's sign, even though only the raw one is physical here).
    avg = exp_avg["average_raw"]
    sem_nom = sem_raw["sem_nominal_raw"]
    sem_el_cuts = np.array(sem_raw["cuts"]["el"])
    sem_q2_cuts = np.array(sem_raw["cuts"]["q2"])

    # Support bounds (always needed)
    mx2_lo, mx2_hi = float(hd["mx_support"][0]), float(hd["mx_support"][1])
    el_lo,  el_hi  = float(hd["el_support"][0]),  float(hd["el_support"][1])
    q2_lo,  q2_hi  = float(hd["q2_support"][0]),  float(hd["q2_support"][1])

    if not (args.toy_job or args.submit):
        # ── Compute residual moments: exp − SEM ──────────────────────────────
        el_keys = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3"]

        residuals = {}
        for key in MOMENT_KEYS:
            entry = avg[key]
            exp_cuts = np.array(entry["cuts"])
            exp_vals = np.array(entry["values"])
            exp_errs = np.array(entry["errors"])

            sem_cuts_ref = sem_el_cuts if key in el_keys else sem_q2_cuts
            sem_vals_interp = np.interp(exp_cuts, sem_cuts_ref, np.array(sem_nom[key]))

            residuals[key] = {
                "cuts": exp_cuts,
                "values": exp_vals - sem_vals_interp,
                "errors": exp_errs,
            }

        res_out = {k: {kk: v.tolist() for kk, v in vd.items()} for k, vd in residuals.items()}
        with open(out_dir / "residual_moments.json", "w") as f:
            json.dump(res_out, f, indent=2)
        print(f"  → {out_dir / 'residual_moments.json'}")

        # ── Load MC cocktail and all gap modes for reference curves ──────────
        print("Loading MC cocktail for reference curves …")
        gap_modes = sorted(ht.get("gap_modes",
            [p.stem for p in (Path(cfg["paths"]["output"]) / "3").glob("*.parquet")
             if p.stem != "cocktail"]))

        cols = ["Mx", "El_B", "q2", "total_weight"]

        def _clean(df):
            arr = df[cols].to_numpy(float)
            ok = np.all(np.isfinite(arr), axis=1) & (arr[:, 3] > 0)
            return df.loc[ok, cols].reset_index(drop=True)

        df_sem_clean = _clean(pd.read_parquet(cocktail_path))
        mx_e  = df_sem_clean["Mx"].to_numpy(float) ** 2
        el_e  = df_sem_clean["El_B"].to_numpy(float)
        q2_e  = df_sem_clean["q2"].to_numpy(float)
        w_e   = df_sem_clean["total_weight"].to_numpy(float)

        el_dense = np.linspace(0.0, 2.0, 200)
        q2_dense = np.linspace(1.5, 10.0, 200)
        curves_excl = compute_raw_curves_np(mx_e, el_e, q2_e, w_e, el_dense, q2_dense)

        gap_curves = {}
        for gm in gap_modes:
            gp = Path(cfg["paths"]["output"]) / "3" / f"{gm}.parquet"
            if not gp.exists():
                print(f"  skip {gm} (no parquet)")
                continue
            df_g = _clean(pd.read_parquet(gp))
            if len(df_g) == 0:
                continue
            df_full = pd.concat([df_sem_clean, df_g], ignore_index=True)
            mx_i = df_full["Mx"].to_numpy(float) ** 2
            el_i = df_full["El_B"].to_numpy(float)
            q2_i = df_full["q2"].to_numpy(float)
            w_i  = df_full["total_weight"].to_numpy(float)
            ci = compute_raw_curves_np(mx_i, el_i, q2_i, w_i, el_dense, q2_dense)
            gap_curves[gm] = {k: ci[k] - curves_excl[k] for k in MOMENT_KEYS}
            print(f"  computed gap curve for {gm}")

        # ── 3x3 panel: residual moments with all gap references ──────────────
        print("Plotting 3x3 residual moment panel …")

        panel_layout = [
            (["mx_1", "mx_2", "mx_3"], "el"),
            (["el_1", "el_2", "el_3"], "el"),
            (["q2_1", "q2_2", "q2_3"], "q2"),
        ]
        xlabels = {
            "el": r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$",
            "q2": r"$q^2_\mathrm{cut}\ [\mathrm{GeV}^2]$",
        }
        toy_cuts = {"el": el_dense, "q2": q2_dense}

        cmap = plt.get_cmap("tab10")
        gm_list = list(gap_curves.keys())
        gm_colors = {gm: cmap(i % 10) for i, gm in enumerate(gm_list)}

        fig, axes = plt.subplots(3, 3, figsize=(13, 9), dpi=150)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.08, hspace=0.40, wspace=0.30)

        for row, (keys, cut_type) in enumerate(panel_layout):
            for col, key in enumerate(keys):
                ax = axes[row, col]
                res = residuals[key]
                cuts = res["cuts"]
                vals = res["values"]
                errs = res["errors"]

                ax.axhline(0, color="k", lw=0.7, linestyle=":")

                for gm, gc in gap_curves.items():
                    tc = toy_cuts[cut_type]
                    tg = gc[key]
                    fin = np.isfinite(tg)
                    ax.plot(tc[fin], tg[fin], color=gm_colors[gm], lw=1.2,
                            linestyle="--", alpha=0.7,
                            label=gm if (row == 0 and col == 0) else "_")

                ax.errorbar(cuts, vals, yerr=errs, fmt="o", color="k",
                            markersize=4, linewidth=1.2, capsize=3, zorder=5,
                            label="Data − SEM" if (row == 0 and col == 0) else "_")

                ax.set_ylabel(YLABELS[key], fontsize=7.5)
                ax.set_xlabel(xlabels[cut_type], fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)

                if col == 0:
                    ax.text(-0.42, 0.5, ROW_TITLES[row], transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5,
                   frameon=True, bbox_to_anchor=(0.5, 0.995))

        fig.suptitle(
            r"Residual moments: $\langle X\rangle_\mathrm{data} - \langle X\rangle_\mathrm{SEM}$"
            "   (MC gap modes)",
            fontsize=11, y=0.925)

        out_fig = fig_dir / "residual_moments_3x3.png"
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"  → {out_fig}")

    # Helper functions always needed (Hausdorff check + binomial map + MaxEnt)
    def hausdorff_moment_check(mu, tol=1e-8):
        mu = np.asarray(mu, dtype=float)
        N = len(mu) - 1
        worst = np.inf
        for n in range(N + 1):
            row = mu[n:N + 1].copy()
            worst = min(worst, row[0])
            for k in range(1, N - n + 1):
                row = row[1:] - row[:-1]
                worst = min(worst, ((-1) ** k) * row[0])
        return (worst >= -tol), float(worst)

    def raw_to_mu01(raw, lo, hi):
        span = hi - lo
        nmax = len(raw)
        mu = np.zeros(nmax)
        for k in range(nmax):
            mu[k] = sum(comb(k, j) * (-lo)**(k - j) * raw[j]
                        for j in range(k + 1)) / span**k
        return mu

    grid_size = int(hd.get("grid_size", 2000))
    t_grid    = np.linspace(0., 1., grid_size)
    tp_grid   = np.stack([t_grid**k for k in range(5)])

    _ba = hd.get("boundary_alpha", {})
    _bb = hd.get("boundary_beta",  {})
    obs_alpha = {"Mx": float(_ba.get("mx2", 0)), "El": float(_ba.get("el", 0)), "Q2": float(_ba.get("q2", 0))}
    obs_beta  = {"Mx": float(_bb.get("mx2", 0)), "El": float(_bb.get("el", 0)), "Q2": float(_bb.get("q2", 0))}

    def maxent_pdf(mu01, t, lam0=None, tp_full=None, alpha=0.0, beta=0.0):
        N  = len(mu01) - 1
        tp = tp_full[:N+1] if tp_full is not None else np.stack([t**k for k in range(N+1)])
        eps   = 1e-300
        prior = (np.maximum(t, eps) ** alpha) * (np.maximum(1.0 - t, eps) ** beta)
        def _g(lam):
            lf = lam @ tp; lf -= lf.max()
            return prior * np.exp(lf)
        def dual(lam):
            lf0 = lam @ tp
            return np.log(np.trapz(prior * np.exp(lf0 - lf0.max()), t)) + lf0.max() - lam @ mu01
        def grad(lam):
            g = _g(lam); Z = np.trapz(g, t); gn = g / Z
            return np.array([np.trapz(tp[k]*gn, t) - mu01[k] for k in range(N+1)])
        x0  = np.zeros(N+1) if lam0 is None else lam0.copy()
        res = minimize(dual, x0, jac=grad, method="L-BFGS-B",
                       options={"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12})
        fn = _g(res.x); Z = np.trapz(fn, t); fn /= Z
        mom_err = max(abs(np.trapz(tp[k]*fn, t) - mu01[k]) for k in range(N+1))
        return fn, res.x, bool(res.success), float(mom_err)

    avg_raw = exp_avg["average_raw"]
    sem_nom_raw = sem_raw["sem_nominal_raw"]

    # bf_incl/bf_gap: same config-driven values and same compute_bf_gap() convention as
    # 7_hausdorff_toy.py, rather than separately hardcoded constants -- bf_gap_central is
    # derived fresh from the actual per-mode BFs in the cocktail, not assumed.
    sysc        = cfg.get("systematics", {})
    bf_incl     = float(sysc.get("bf_incl", 0.1105))
    bf_incl_unc = float(sysc.get("bf_incl_unc", 0.0016))
    df_bf = pd.read_parquet(cocktail_path, columns=["decay_name", "bf", "bf_unc"])
    bf_sem_computed, bf_gap_central, sigma_bf_gap = syst.compute_bf_gap(df_bf, bf_incl, bf_incl_unc)
    del df_bf
    bf_gap  = bf_gap_central
    bf_sem  = bf_incl - bf_gap
    r_exp   = bf_incl / bf_gap
    r_sem   = bf_sem  / bf_gap
    print(f"  bf_gap: computed central={bf_gap*100:.3f}%  sigma={sigma_bf_gap*100:.3f}%"
          f"  (bf_sem_computed={bf_sem_computed*100:.3f}%, bf_incl={bf_incl*100:.3f}%)")

    def raw_gap_at_threshold(keys, sem_cuts_ref, thr):
        # exp/SEM raw moments interpolated to thr, combined via the r_exp/r_sem BF-budget
        # identity -- shared by the Hausdorff check, Mx2 support scan, and MaxEnt inversion.
        raw_gap = []
        for key in keys:
            key_cuts = np.array(avg_raw[key]["cuts"])
            key_vals = np.array(avg_raw[key]["values"])
            exp_val = np.interp(thr, key_cuts, key_vals)
            sem_val = np.interp(thr, sem_cuts_ref, np.array(sem_nom_raw[key]))
            raw_gap.append(r_exp * exp_val - r_sem * sem_val)
        return raw_gap

    # Systematics active for the data toy loop: the full combined set of everything enabled in
    # config (unlike 7_hausdorff_toy.py's solo+combined run-label registry, the data bands only
    # need the one "everything on" band -- there's no MC truth to check solo impacts against).
    active = [n for n in syst.SYSTEMATIC_NAMES if sysc.get("enable", {}).get(n, True)]
    print(f"  Active systematics for data toy loop: {active}")

    obs_cfg = {
        "Mx": (["mx_1", "mx_2", "mx_3"], sem_el_cuts, mx2_lo, mx2_hi),
        "El": (["el_1", "el_2", "el_3"], sem_el_cuts, el_lo,  el_hi),
        "Q2": (["q2_1", "q2_2", "q2_3"], sem_q2_cuts, q2_lo,  q2_hi),
    }

    obs_support = {
        "Mx": (mx2_lo, mx2_hi, r"$M_X^2\ [\mathrm{GeV}^2]$", r"$f(M_X^2)\ [\mathrm{GeV}^{-2}]$"),
        "El": (el_lo,  el_hi,  r"$E_\ell\ [\mathrm{GeV}]$",  r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$"),
        "Q2": (q2_lo,  q2_hi,  r"$q^2\ [\mathrm{GeV}^2]$",   r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"),
    }

    obs_title  = {"Mx": r"$M_X^2$", "El": r"$E_\ell$", "Q2": r"$q^2$"}
    obs_xlabel = {
        "Mx": r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$",
        "El": r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$",
        "Q2": r"$q^2_\mathrm{cut}\ [\mathrm{GeV}^2]$",
    }

    # ── Joint multi-threshold MaxEnt (lib/joint_maxent.py), opt-in via config -- see
    # 7_hausdorff_toy.py for the same wiring on the Asimov side and scratch/joint_slack_*.py /
    # scratch/joint_conditional_asimov_test.py for the closure-test validation.
    #
    # This pipeline uses the CONDITIONAL-moment formulation (maxent_joint_soft_conditional),
    # not the plain-x^k/global-normalization one 7_hausdorff_toy.py uses -- the measured exp/SEM
    # moment tables only report CONDITIONAL shape moments per threshold (mu_0=1 by construction
    # of the r_exp/r_sem identity), with no cross-threshold yield/survival-fraction information,
    # unlike 7_'s full MC event-level truth. An earlier version approximated the missing
    # survival fraction from the SEM cocktail's own yield curve; that badly distorted the fit
    # (SEM and the true gap signal are different populations). The conditional formulation needs
    # no such estimate at all: E[X^k|X>=thr] = m_k is used directly via the algebraically
    # equivalent linear constraint integral_thr^hi (x^k - m_k) f dx = 0 (see lib/joint_maxent.py
    # module docstring for the derivation) -- exactly what's measured, no proxy.
    #
    # Its continuity-penalty gammas were tuned SEPARATELY from 7_'s (see gammas_conditional in
    # config.yaml) -- the same gamma values that work for 7_'s formulation do not transfer here.
    jmc = cfg.get("joint_maxent", {})
    joint_enable = bool(jmc.get("enable", False))
    joint_sigma_rel = float(jmc.get("sigma_rel", 0.03))
    joint_gammas = {}
    for _obs in ("El", "Q2"):
        _g = jmc.get("gammas_conditional", {}).get(_obs, [5.0, 5.0] if _obs == "El" else [0.0, 0.0])
        joint_gammas[_obs] = tuple(float(x) for x in _g)
    # Clipped to what the tabulated moments can actually support: obs_cfg's keys lists only go
    # up to the 3rd raw moment (mu_0=1 implicit + 3 measured => N<=4), unlike the Asimov toy
    # which can compute any order directly from MC truth.
    joint_n = {}
    for _obs in ("El", "Q2"):
        _n_cfg = int(jmc.get("n_moments", {}).get(_obs, 5 if _obs == "El" else 3))
        _n_max = len(obs_cfg[_obs][0]) + 1
        joint_n[_obs] = min(_n_cfg, _n_max)
        if joint_enable and _n_cfg > _n_max:
            print(f"  joint_maxent: {_obs} n_moments={_n_cfg} > available {_n_max} "
                  f"(only {len(obs_cfg[_obs][0])} raw moments tabulated) -- clipped to {_n_max}")
    joint_cuts = {"El": [float(c) for c in jmc.get("el_cuts", [0.0, 0.4, 0.8, 1.2, 1.6])],
                  "Q2": [float(c) for c in jmc.get("q2_cuts", [0.5, 2.0, 3.5, 5.0, 6.5])]}
    joint_prior = {}
    for _obs in ("El", "Q2"):
        _a, _b = obs_alpha[_obs], obs_beta[_obs]
        _eps = 1e-300
        joint_prior[_obs] = (np.maximum(t_grid, _eps) ** _a) * (np.maximum(1.0 - t_grid, _eps) ** _b)
    # Per-observable opt-in: Q2's conditional-moment continuity penalty doesn't work well yet
    # (see SESSION_NOTES_2026-07-28.md, "Known limitation 2") -- default to both for
    # backwards compat, but let config restrict to just the validated observable(s).
    joint_observables = [o for o in jmc.get("observables", ["El", "Q2"]) if o in ("El", "Q2")]
    if joint_enable:
        print(f"  Joint MaxEnt: ENABLED  observables={joint_observables}  n_moments={joint_n}  "
              f"gammas={joint_gammas}  sigma_rel={joint_sigma_rel}")

    if args.toy_job or args.submit:
        # Load previously saved results (toy jobs skip the heavy sections above)
        with open(out_dir / "hausdorff_data_check.json") as f:
            haus_results = json.load(f)
        with open(out_dir / "maxent_data_inversions.json") as f:
            maxent_out = json.load(f)
        print("Loaded saved Hausdorff and MaxEnt results.")
    else:
        # ── Moment consistency check across thresholds ───────────────────────
        print("Plotting moment consistency check …")

        consist_obs = [
            ("El", ["el_1", "el_2", "el_3"], sem_el_cuts,
             r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$"),
            ("Q2", ["q2_1", "q2_2", "q2_3"], sem_q2_cuts,
             r"$q^2_\mathrm{cut}\ [\mathrm{GeV}^2]$"),
        ]

        fig_cc, axes_cc = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig_cc.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.13, wspace=0.30)

        for ax_cc, (obs, keys, sem_cuts_ref, xlabel_cc) in zip(axes_cc, consist_obs):
            cuts = np.array(avg_raw[keys[0]]["cuts"])
            kc  = np.array(avg_raw[keys[0]]["cuts"])
            kv  = np.array(avg_raw[keys[0]]["values"])
            ke  = np.array(avg_raw[keys[0]]["errors"])
            sem_v = np.interp(cuts, sem_cuts_ref, np.array(sem_nom_raw[keys[0]]))
            mean_x     = r_exp * np.interp(cuts, kc, kv) - r_sem * sem_v
            mean_x_err = r_exp * np.interp(cuts, kc, ke)

            ax_cc.axhline(0, color="k", lw=0.7, ls=":")
            ax_cc.errorbar(cuts, mean_x, yerr=mean_x_err,
                           fmt="o", color="steelblue", markersize=5,
                           linewidth=1.2, capsize=3, label=r"$\langle x\rangle_\mathrm{gap}(t)$")

            ax_cc2 = ax_cc.twinx()
            delta     = np.diff(mean_x)
            delta_err = np.sqrt(mean_x_err[1:]**2 + mean_x_err[:-1]**2)
            mid_cuts  = 0.5 * (cuts[:-1] + cuts[1:])
            ax_cc2.axhline(0, color="crimson", lw=0.7, ls="--", alpha=0.5)
            ax_cc2.errorbar(mid_cuts, delta, yerr=delta_err,
                            fmt="s", color="crimson", markersize=4,
                            linewidth=1.0, capsize=3, alpha=0.8,
                            label=r"$\Delta\langle x\rangle$ (must be $\geq 0$)")
            ax_cc2.set_ylabel(r"$\Delta\langle x\rangle$", fontsize=9, color="crimson")
            ax_cc2.tick_params(labelsize=8, colors="crimson")
            ax_cc2.legend(fontsize=8, loc="upper right")

            ax_cc.set_xlabel(xlabel_cc, fontsize=10)
            ax_cc.set_ylabel(r"$\langle x\rangle_\mathrm{gap}$", fontsize=9)
            ax_cc.set_title(obs, fontsize=11)
            ax_cc.tick_params(labelsize=8)
            ax_cc.grid(alpha=0.25)
            ax_cc.legend(fontsize=8, loc="upper left")

        fig_cc.suptitle(
            r"Moment consistency: $\langle x\rangle_\mathrm{gap}(t)$ must be non-decreasing in $t$"
            "\n(violation → moment data inconsistent with single underlying distribution)",
            fontsize=10)
        out_cc = fig_dir / "moment_consistency.pdf"
        fig_cc.savefig(out_cc)
        plt.close(fig_cc)
        print(f"  → {out_cc}")

        # ── Hausdorff check at every threshold (from data) ────────────────────
        print("Hausdorff check at every threshold (data) …")

        max_el_cut = float(hd.get("max_el_cut", np.inf))
        max_q2_cut = float(hd.get("max_q2_cut", np.inf))

        haus_results = {}
        for obs, (keys, sem_cuts_ref, lo, hi) in obs_cfg.items():
            cuts = np.array(avg_raw[keys[0]]["cuts"])
            if obs in ("Mx", "El"):
                cuts = cuts[cuts < max_el_cut]
            elif obs == "Q2":
                cuts = cuts[cuts < max_q2_cut]
            haus_results[obs] = {"thresholds": cuts.tolist(), "worst": [], "pass": []}

            for thr in cuts:
                lo_thr = float(thr) if obs in ("El", "Q2") else lo

                raw_full = np.array([1.0] + raw_gap_at_threshold(keys, sem_cuts_ref, thr))
                mu01 = raw_to_mu01(raw_full, lo_thr, hi)
                ok, worst = hausdorff_moment_check(mu01)
                haus_results[obs]["worst"].append(float(worst))
                haus_results[obs]["pass"].append(bool(ok))

        with open(out_dir / "hausdorff_data_check.json", "w") as f:
            json.dump(haus_results, f, indent=2)
        print(f"  → {out_dir / 'hausdorff_data_check.json'}")

        # ── Plot: Hausdorff summary ───────────────────────────────────────────
        print("Plotting Hausdorff summary …")

        fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
        fig2.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

        for ax, obs in zip(axes2, ["Mx", "El", "Q2"]):
            res_obs = haus_results[obs]
            thresholds = np.array(res_obs["thresholds"])
            worst = np.array(res_obs["worst"])
            pass_flags = np.array(res_obs["pass"])

            ax.axhline(0, color="k", lw=1.0, linestyle="--", alpha=0.5)
            ax.plot(thresholds[pass_flags],  worst[pass_flags],
                    "o", color="steelblue", markersize=6, label="pass")
            ax.plot(thresholds[~pass_flags], worst[~pass_flags],
                    "o", color="crimson",   markersize=6, label="fail")
            ax.plot(thresholds, worst, "-", color="steelblue", lw=1.2, alpha=0.6)

            ax.set_title(obs_title[obs], fontsize=12)
            ax.set_xlabel(obs_xlabel[obs], fontsize=10)
            ax.set_ylabel("Hausdorff worst value", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

        fig2.suptitle(
            r"Hausdorff criterion on data residuals (worst value, $\geq 0$ = valid)"
            "\n" r"$[\mu_0,\mu_1,\mu_2,\mu_3]$ from raw moments (data $-$ SEM), per threshold",
            fontsize=10)

        out_fig2 = fig_dir / "hausdorff_data_summary.pdf"
        fig2.savefig(out_fig2)
        plt.close(fig2)
        print(f"  → {out_fig2}")

        # ── Mx² support scan ─────────────────────────────────────────────────
        print("Plotting Mx² support scan …")

        mx_el_cuts = np.array(avg_raw["mx_1"]["cuts"])
        lo_scan    = np.linspace(0.0, 5.0, 100)
        hi_scan    = np.linspace(5.5, 30.0, 100)
        n_panels   = len(mx_el_cuts)
        ncols_s    = 7
        nrows_s    = int(np.ceil(n_panels / ncols_s))

        fig_s = plt.figure(figsize=(ncols_s * 3.4, nrows_s * 3.2 + 1.2))
        gs_s  = matplotlib.gridspec.GridSpec(
            nrows_s, ncols_s + 1, figure=fig_s,
            width_ratios=[1] * ncols_s + [0.06],
            hspace=0.55, wspace=0.35,
            left=0.05, right=0.94, top=0.84, bottom=0.08)

        im_s = None
        for ax_idx, thr in enumerate(mx_el_cuts):
            row, col = divmod(ax_idx, ncols_s)
            ax = fig_s.add_subplot(gs_s[row, col])

            raw_gap_s = raw_gap_at_threshold(["mx_1", "mx_2", "mx_3"], sem_el_cuts, thr)
            raw_full_s = np.array([1.0] + raw_gap_s)
            mean_mx2   = raw_gap_s[0]

            W = np.full((len(lo_scan), len(hi_scan)), np.nan)
            for i, lo in enumerate(lo_scan):
                for j, hi in enumerate(hi_scan):
                    if lo < mean_mx2 and hi > mean_mx2:
                        W[i, j] = hausdorff_moment_check(
                            raw_to_mu01(raw_full_s, lo, hi))[1]

            im_s = ax.pcolormesh(hi_scan, lo_scan, np.clip(W, -0.1, 0.1),
                                 cmap="RdBu", vmin=-0.1, vmax=0.1, shading="auto")
            ax.contour(hi_scan, lo_scan, W, levels=[0], colors="k", linewidths=1.5)
            ax.axhline(mean_mx2, color="limegreen", lw=1.5, linestyle="--")
            ax.set_title(r"$E_\ell > " + f"{thr:.1f}" + r"\ \mathrm{GeV}$", fontsize=10, pad=4)
            ax.set_xlabel(r"$\mathrm{hi}\ [\mathrm{GeV}^2]$", fontsize=9)
            ax.set_ylabel(r"$\mathrm{lo}\ [\mathrm{GeV}^2]$", fontsize=9)
            ax.tick_params(labelsize=8)

        cax_s = fig_s.add_subplot(gs_s[:, ncols_s])
        cb_s  = fig_s.colorbar(im_s, cax=cax_s)
        cb_s.set_label(r"Hausdorff worst value (clipped to $[-0.1,\,0.1]$)", fontsize=10)
        cb_s.ax.tick_params(labelsize=9)

        fig_s.text(0.495, 0.95,
            r"Hausdorff worst value vs $M_X^2$ support $[\mathrm{lo},\,\mathrm{hi}]$"
            r" — one panel per $E_{\ell,\mathrm{cut}}$ threshold",
            ha="center", va="bottom", fontsize=12)
        fig_s.text(0.495, 0.91,
            r"Black contour: pass boundary (worst $\geq 0$). "
            r"Green dashed: gap mean $\langle M_X^2\rangle_\mathrm{gap}$.",
            ha="center", va="bottom", fontsize=10)

        out_figs = fig_dir / "hausdorff_mx2_support_scan.pdf"
        fig_s.savefig(out_figs)
        plt.close(fig_s)
        print(f"  → {out_figs}")

        # ── MaxEnt inversion for all passing thresholds ───────────────────────
        print("MaxEnt inversion for passing thresholds …")

        maxent_out = {}
        for obs, (keys, sem_cuts_ref, lo, hi) in obs_cfg.items():
            res_obs    = haus_results[obs]
            thresholds = np.array(res_obs["thresholds"])
            pass_flags = np.array(res_obs["pass"])
            passing    = thresholds[pass_flags]

            if len(passing) == 0:
                print(f"  {obs}: no passing thresholds, skip")
                continue

            xlo_base, xhi, xlabel, ylabel = obs_support[obs]
            maxent_out[obs] = {"thresholds": {}}

            ncols = min(len(passing), 5)
            nrows = int(np.ceil(len(passing) / ncols))
            fig3, axes3 = plt.subplots(nrows, ncols,
                                       figsize=(ncols * 3.5, nrows * 3.2),
                                       squeeze=False)

            lam_warm = None
            for idx, thr in enumerate(passing):
                row, col = divmod(idx, ncols)
                ax = axes3[row, col]

                xlo = float(thr) if obs in ("El", "Q2") else xlo_base
                span   = xhi - xlo
                x_grid = xlo + t_grid * span

                raw_gap = raw_gap_at_threshold(keys, sem_cuts_ref, thr)
                raw_full = np.array([1.0] + raw_gap)
                mu01     = raw_to_mu01(raw_full, xlo, xhi)

                entry = {"threshold": float(thr), "support": [xlo, xhi],
                         "mu01": mu01.tolist(), "raw_gap": raw_gap,
                         "converged": False, "mom_err": None,
                         "x_grid": x_grid.tolist(), "f_phys": None}
                try:
                    f_t, lam_warm, ok_opt, mom_err = maxent_pdf(mu01, t_grid,
                                                                  lam0=lam_warm,
                                                                  tp_full=tp_grid,
                                                                  alpha=obs_alpha[obs],
                                                                  beta=obs_beta[obs])
                    f_phys = f_t / span
                    status = f"mom err {mom_err:.1e}" + ("" if ok_opt else " (!)")
                    if mom_err < 1e-4:
                        ax.plot(x_grid, f_phys, color="steelblue", lw=2)
                        entry.update({"converged": ok_opt, "mom_err": mom_err,
                                      "f_phys": f_phys.tolist()})
                    else:
                        lam_warm = None
                        status = f"bad mom err {mom_err:.1e}"
                except Exception as e:
                    lam_warm = None
                    status = f"failed: {e}"
                maxent_out[obs]["thresholds"][f"{thr:.2f}"] = entry

                cut_label = (r"$E_\ell > " + f"{thr:.1f}" + r"\ \mathrm{GeV}$"
                             if obs != "Q2" else
                             r"$q^2 > " + f"{thr:.1f}" + r"\ \mathrm{GeV}^2$")
                ax.set_title(cut_label + f"\n{status}", fontsize=9)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.25)

            for idx in range(len(passing), nrows * ncols):
                axes3[divmod(idx, ncols)].set_visible(False)

            fig3.suptitle(
                r"MaxEnt inversion of data residual — " + obs_title[obs]
                + "\n" + r"(passing Hausdorff thresholds, $\alpha=\beta=0$)",
                fontsize=11)
            fig3.tight_layout()
            out_fig3 = fig_dir / f"maxent_data_{obs}.pdf"
            fig3.savefig(out_fig3)
            plt.close(fig3)
            print(f"  → {out_fig3}")

            # ── Overlay plot: all curves on common scale ──────────────────────
            # Collect good curves in ascending threshold order
            good = []
            for thr in passing:
                e = maxent_out[obs]["thresholds"][f"{thr:.2f}"]
                if e["f_phys"] is not None:
                    xlo = float(thr) if obs in ("El", "Q2") else xlo_base
                    good.append((float(thr), xlo, np.array(e["x_grid"]), np.array(e["f_phys"])))

            if len(good) >= 2:
                fig4, ax4 = plt.subplots(figsize=(7, 4.5))
                cmap4 = plt.get_cmap("viridis")
                colors4 = [cmap4(i / (len(good) - 1)) for i in range(len(good))]

                # Highest threshold normalized to 1; each lower curve normalized
                # so its integral above the next curve's threshold is also 1.
                scaled_curves = [None] * len(good)
                for i in range(len(good) - 1, -1, -1):
                    thr, xlo, xg, fp = good[i]
                    if i == len(good) - 1:
                        norm = np.trapz(fp, xg)
                    else:
                        _, _, xg_next, fp_next_s = scaled_curves[i + 1]
                        ref   = np.trapz(fp_next_s, xg_next)
                        above = np.trapz(fp[xg >= good[i + 1][0]], xg[xg >= good[i + 1][0]])
                        norm  = above / ref if ref > 0 else 1.0
                    scaled_curves[i] = (thr, xlo, xg, fp / norm if norm > 0 else fp)

                ymax = np.percentile(np.concatenate([fp for _, _, xg, fp in scaled_curves]), 99)

                for i, (thr, _, xg, fp_s) in enumerate(scaled_curves):
                    cut_label = (r"$E_\ell > " + f"{thr:.1f}" + r"\ \mathrm{GeV}$"
                                 if obs != "Q2" else
                                 r"$q^2 > " + f"{thr:.1f}" + r"\ \mathrm{GeV}^2$")
                    ax4.plot(xg, fp_s, color=colors4[i], lw=1.8, label=cut_label)

                ax4.set_xlabel(xlabel, fontsize=10)
                ax4.set_ylabel("rescaled " + ylabel, fontsize=10)
                ax4.set_xlim(xlo_base, xhi)
                ax4.set_ylim(0, 1.2 * ymax)
                ax4.tick_params(labelsize=9)
                ax4.grid(alpha=0.25)
                ax4.legend(fontsize=8, loc="best")
                fig4.suptitle(
                    r"MaxEnt inversions overlaid — " + obs_title[obs]
                    + "\n" + r"(each curve rescaled: $\int_{\mathrm{thr}_{i+1}}^{\mathrm{hi}} f\,dx = 1$)",
                    fontsize=10)
                fig4.tight_layout()
                out_fig4 = fig_dir / f"maxent_data_{obs}_overlay.pdf"
                fig4.savefig(out_fig4)
                plt.close(fig4)
                print(f"  → {out_fig4}")

        with open(out_dir / "maxent_data_inversions.json", "w") as f:
            json.dump(maxent_out, f, indent=2)
        print(f"  → {out_dir / 'maxent_data_inversions.json'}")

    if not (args.toy_job or args.submit):
        print("Done.")
        return

    # ── Toy chunk job or batch submission ─────────────────────────────────────
    n_toys_band = int(ht.get("n_toys_band", 500))
    n_chunks    = int(ht.get("n_toy_chunks", 20))
    seed_base   = int(ht.get("seed", 42)) + 1
    queue       = cfg["generation"]["queue"]
    here        = Path(__file__).parent.resolve()
    toy_dir     = out_dir / "toys"
    log_dir     = here / "logs" / "8"
    toy_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        # Dispatch one bsub job per chunk
        n_toys_per_chunk = (n_toys_band + n_chunks - 1) // n_chunks
        for chunk_idx in range(n_chunks):
            log_file = log_dir / f"toys_{chunk_idx:04d}.log"
            job_cmd  = (f"cd {here} && "
                        f"python3 8_hausdorff_data.py --config config.yaml "
                        f"--toy-job --chunk {chunk_idx} --n-chunks {n_chunks}")
            # Each toy chunk loads the full cocktail (~24M events) into memory; measured peak
            # RSS ~9.6GB via `/usr/bin/time -v --toy-job --chunk 0`. LSF caps memory at 4GB/slot
            # (hard, -M cannot raise it), so request 3 slots (12GB) for headroom.
            bsub_cmd = f'bsub -q {queue} -env all -n "3" -oo "{log_file}" "{job_cmd}"'
            print(bsub_cmd)
            if not args.dry_run:
                subprocess.run(bsub_cmd, shell=True, check=True)
        if args.dry_run:
            print("(dry run — no jobs submitted)")
        print("Done.")
        return

    if not args.toy_job:
        print("Done.")
        return

    # ── Single toy chunk (--toy-job) ──────────────────────────────────────────
    chunk_idx        = args.chunk
    n_chunks_total   = args.n_chunks
    n_toys_per_chunk = (n_toys_band + n_chunks_total - 1) // n_chunks_total
    i_start = chunk_idx * n_toys_per_chunk
    i_end   = min(i_start + n_toys_per_chunk, n_toys_band)
    n_local = i_end - i_start
    if n_local <= 0:
        print(f"Chunk {chunk_idx}: nothing to do (i_start={i_start} >= {n_toys_band})")
        return

    print(f"Toy chunk {chunk_idx}/{n_chunks_total}: toys {i_start}–{i_end-1} ({n_local} toys)")

    rng = np.random.default_rng(seed_base + chunk_idx * 1000)

    need_ff      = "ff"            in active
    need_bf_mode = "bf_mode"       in active
    need_bf_gap  = "bf_gap"        in active
    need_incl    = "incl_moments"  in active

    bounds_sig     = float(ht.get("bounds_sigma", 0.0))
    span_El_nom_hd = el_hi - el_lo
    span_Q2_nom_hd = q2_hi - q2_lo
    q2_bsig        = bounds_sig * (span_Q2_nom_hd / span_El_nom_hd) if span_El_nom_hd > 0 else bounds_sig

    # ── Cocktail (SEM-side) event data for the toy loop: independent stat bootstrap + ff +
    # bf_mode systematics, the same treatment 7_hausdorff_toy.py applies to its pseudo-SEM
    # side, applied here directly to the real cocktail that sem_nominal_raw came from ──
    _base_cols = ["Mx", "El_B", "q2", "total_weight", "bf", "bf_unc", "decay_name"]
    ff_cols: list[str] = []
    if need_ff:
        _schema_cols = pq.ParquetFile(cocktail_path).schema_arrow.names
        ff_cols = ["ff_weight"] + [c for c in _schema_cols
                                    if c.startswith("ff_weight_up") or c.startswith("ff_weight_down")]
    df_cocktail = read_parquet_downcast(cocktail_path, _base_cols + ff_cols, float32_cols=ff_cols)
    ok_c = (
        np.isfinite(df_cocktail["Mx"].to_numpy()) & np.isfinite(df_cocktail["El_B"].to_numpy())
        & np.isfinite(df_cocktail["q2"].to_numpy()) & np.isfinite(df_cocktail["total_weight"].to_numpy())
        & (df_cocktail["total_weight"].to_numpy() > 0)
    )
    df_cocktail = df_cocktail.loc[ok_c].reset_index(drop=True)
    mx2_c    = df_cocktail["Mx"].to_numpy(float) ** 2
    el_c     = df_cocktail["El_B"].to_numpy(float)
    q2_c     = df_cocktail["q2"].to_numpy(float)
    w_c_base = df_cocktail["total_weight"].to_numpy(float)
    N_c      = len(df_cocktail)

    ff_slope_matrix, ff_nuisance_names = (
        syst.build_ff_slope_matrix(df_cocktail) if need_ff else (np.zeros((N_c, 0)), []))
    bf_mode_codes, bf_rel_unc_per_mode = syst.bf_mode_setup(
        df_cocktail["decay_name"].to_numpy(), df_cocktail["bf"].to_numpy(),
        df_cocktail["bf_unc"].to_numpy())
    del df_cocktail
    print(f"  Cocktail toy loop: {N_c:,} events  (need_ff={need_ff}, "
          f"FF nuisances={len(ff_nuisance_names)})")

    cov_data   = exp_avg["average_raw_cov"]
    cov_points = cov_data["points"]
    cov_matrix = np.array(cov_data["cov"])
    cov_mean   = np.array([
        np.interp(p["cut"], avg_raw[p["key"]]["cuts"], avg_raw[p["key"]]["values"])
        for p in cov_points])

    def toy_exp_val(sample, key, thr):
        pts = [(float(p["cut"]), sample[i])
               for i, p in enumerate(cov_points) if p["key"] == key]
        if not pts:
            return np.nan
        pts.sort()
        cuts_k, vals_k = zip(*pts)
        return float(np.interp(thr, cuts_k, vals_k))

    MOM_ERR_TOL  = 1e-4
    # toy_curves[obs][thr_str] = list of f_phys arrays (good toys only), always re-gridded onto
    # the NOMINAL (un-smeared) support before storing -- see f_common below -- so downstream
    # code (9_hausdorff_bands.py) can keep treating every toy curve as living on one common
    # physical grid per threshold, exactly as 7_hausdorff_toy.py does for its own toy_f arrays.
    toy_curves = {obs: {} for obs in obs_cfg}
    toy_curves_joint = {obs: [] for obs in ("El", "Q2")}  # one curve per toy (NaN if not converged)
    t_toys_start = time.time()

    for toy_idx in range(n_local):
        # 1) Experimental-side draw: correlated Gaussian from the real measured raw-moment
        # covariance (the data-driven realization of "incl_moments" -- unlike the MC toy, which
        # has no such covariance and must approximate it, real data already has one).
        exp_sample = syst.sample_from_cov(cov_mean, cov_matrix, rng) if need_incl else cov_mean

        # 2) SEM/cocktail-side draw: independent stat bootstrap (always on) + ff + bf_mode.
        idx_c    = rng.integers(0, N_c, size=N_c)
        counts_c = np.bincount(idx_c, minlength=N_c).astype(float)
        w_c      = w_c_base * counts_c
        if need_ff:
            w_c = w_c * syst.sample_ff_multiplier_from_matrix(ff_slope_matrix, rng)
        if need_bf_mode:
            w_c = w_c * syst.sample_bf_multiplier_from_codes(bf_mode_codes, bf_rel_unc_per_mode, rng)
        w_c = np.where(np.isfinite(w_c) & (w_c > 0), w_c, 0.)
        sem_raw_toy = compute_raw_curves_np(mx2_c, el_c, q2_c, w_c, sem_el_cuts, sem_q2_cuts)

        # 3) bf_gap: perturb the assumed gap-mode BR budget (off by default in config -- see
        # SYSTEMATICS_NOTES); bf_incl itself stays fixed, its uncertainty is already folded
        # into sigma_bf_gap via compute_bf_gap().
        if need_bf_gap:
            bf_gap_t = syst.sample_bf_gap(bf_gap_central, sigma_bf_gap, rng)
            r_exp_t  = bf_incl / bf_gap_t
            r_sem_t  = (bf_incl - bf_gap_t) / bf_gap_t
        else:
            r_exp_t, r_sem_t = r_exp, r_sem

        # 3b) Joint multi-threshold MaxEnt (opt-in): one fit per obs using every Hausdorff-
        # passing candidate threshold as a simultaneous constraint on one global density.
        # Conditional-moment formulation (lib/joint_maxent.py): the measured constraint
        # E[X^k|X>=thr] = m_k is used directly via integral_{thr}^{hi}(x^k - m_k) f dx = 0,
        # which needs NO survival-fraction estimate (an earlier version approximated it from
        # the SEM cocktail's own yield curve -- found to badly distort the fit, since SEM and
        # the true gap signal are different populations with no reason to share a survival
        # curve; this formulation has no such term at all). Nominal (un-smeared) support only,
        # matching 7_hausdorff_toy.py's scope.
        if joint_enable:
            for obs in joint_observables:
                keys, sem_cuts_ref, lo_nom, hi_nom = obs_cfg[obs]
                N = joint_n[obs]
                raw_by_thr = {}
                for thr in joint_cuts[obs]:
                    raw_gap_j = []
                    for key in keys[:N - 1]:
                        exp_val = toy_exp_val(exp_sample, key, thr)
                        sem_val = float(np.interp(thr, sem_cuts_ref, sem_raw_toy[key]))
                        raw_gap_j.append(r_exp_t * exp_val - r_sem_t * sem_val)
                    raw_by_thr[thr] = np.array([1.0] + raw_gap_j)

                passing_j = jm.passing_thresholds_from_conditional_raw(raw_by_thr, hi_nom)
                if len(passing_j) < 2:
                    toy_curves_joint[obs].append(np.full(grid_size, np.nan))
                    continue
                span_j = hi_nom - lo_nom
                m_by_thr = {thr: raw_by_thr[thr][1:] for thr in passing_j}
                basis_j, _ = jm.build_basis_conditional(t_grid, lo_nom, hi_nom, passing_j, N, m_by_thr)
                try:
                    f_j, ok_j, err_j = jm.maxent_joint_soft_conditional(
                        basis_j, passing_j, N, span_j, t_grid, joint_prior[obs], m_by_thr,
                        sigma_rel=joint_sigma_rel, gammas=joint_gammas[obs])
                except Exception:
                    ok_j, f_j, err_j = False, None, np.inf
                if ok_j and err_j < 0.2 and jm.sane_density(f_j):
                    toy_curves_joint[obs].append(f_j / span_j)
                else:
                    toy_curves_joint[obs].append(np.full(grid_size, np.nan))

        # 4) Support-bound smearing (always-on, magnitude set by hausdorff_toy.bounds_sigma;
        # 0 by default -- a no-op). Only the physically-uncertain endpoint is smeared: for
        # El/Q2 the lower bound is the (fixed) data threshold itself, so only the upper bound
        # moves; for Mx2 (no threshold) both endpoints move, mirroring 7_hausdorff_toy.py.
        mx2_lo_t = np.clip(mx2_lo + rng.normal(0., bounds_sig * 2 * np.sqrt(mx2_lo)),
                            MX2_LO_MIN, mx2_hi - 1.0)
        mx2_hi_t = np.clip(mx2_hi + rng.normal(0., bounds_sig * 2 * np.sqrt(mx2_hi)),
                            mx2_lo_t + 1.0, MX2_HI_MAX)
        el_hi_t  = np.clip(el_hi + rng.normal(0., bounds_sig), el_lo + 0.3, EL_HI_MAX)
        q2_hi_t  = np.clip(q2_hi + rng.normal(0., q2_bsig),    q2_lo + 1.0, Q2_HI_MAX)
        xhi_t      = {"Mx": mx2_hi_t, "El": el_hi_t, "Q2": q2_hi_t}
        xlo_base_t = {"Mx": mx2_lo_t, "El": el_lo,   "Q2": q2_lo}

        for obs, (keys, sem_cuts_ref, lo_nom, hi_nom) in obs_cfg.items():
            thresholds = np.array(haus_results[obs]["thresholds"])
            pass_flags = np.array(haus_results[obs]["pass"])
            passing    = thresholds[pass_flags]
            xhi_o      = xhi_t[obs]
            xlo_base_o = xlo_base_t[obs]

            lam_toy = None
            for thr in passing:
                thr_str = f"{thr:.2f}"
                xlo  = float(thr) if obs in ("El", "Q2") else xlo_base_o
                if xlo >= xhi_o:
                    lam_toy = None
                    continue
                span = xhi_o - xlo

                raw_gap = []
                for key in keys:
                    exp_val = toy_exp_val(exp_sample, key, thr)
                    sem_val = float(np.interp(thr, sem_cuts_ref, sem_raw_toy[key]))
                    raw_gap.append(r_exp_t * exp_val - r_sem_t * sem_val)

                raw_full = np.array([1.0] + raw_gap)
                mu01     = raw_to_mu01(raw_full, xlo, xhi_o)
                ok_h, _ = hausdorff_moment_check(mu01)
                if not ok_h:
                    lam_toy = None
                    continue
                try:
                    f_t, lam_toy, ok_opt, mom_err = maxent_pdf(
                        mu01, t_grid, lam0=lam_toy, tp_full=tp_grid,
                        alpha=obs_alpha[obs], beta=obs_beta[obs])
                    if mom_err < MOM_ERR_TOL:
                        # Re-grid from this toy's own (possibly smeared) support onto the
                        # nominal support at the same threshold, so every stored curve shares
                        # one common physical grid regardless of bounds_sigma.
                        f_phys_t   = f_t / span
                        x_grid_t   = xlo + t_grid * span
                        span_nom   = hi_nom - xlo
                        x_grid_nom = xlo + t_grid * span_nom
                        f_common   = np.interp(x_grid_nom, x_grid_t, f_phys_t, left=0., right=0.)
                        toy_curves[obs].setdefault(thr_str, []).append(f_common.tolist())
                    else:
                        lam_toy = None
                except Exception:
                    lam_toy = None

        if (toy_idx + 1) % 10 == 0:
            elapsed   = time.time() - t_toys_start
            rate      = (toy_idx + 1) / elapsed
            remaining = (n_local - toy_idx - 1) / rate
            print(f"  {toy_idx+1}/{n_local} toys  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    out_chunk = toy_dir / f"chunk_{chunk_idx:04d}.json"
    payload = {"chunk": chunk_idx, "i_start": i_start, "i_end": i_end, "toy_curves": toy_curves}
    if joint_enable:
        payload["toy_curves_joint"] = {
            obs: [np.asarray(c).tolist() for c in curves]
            for obs, curves in toy_curves_joint.items()
        }
    with open(out_chunk, "w") as f:
        json.dump(payload, f)
    print(f"  → {out_chunk}")
    print("Done.")


if __name__ == "__main__":
    main()
