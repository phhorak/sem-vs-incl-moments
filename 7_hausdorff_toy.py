#!/usr/bin/env python3
"""
7_hausdorff_toy.py

Simulation closure toy for the Hausdorff + MaxEnt gap reconstruction approach.

Setup:
  pseudo-inclusive  = full cocktail (cocktail.parquet)
  pseudo-SEM        = cocktail minus the chosen gap mode
  residual moments  = inclusive − SEM, mapped to [0,1] via binomial change-of-variables
  → Hausdorff check → MaxEnt inversion → compare to truth

Smearing per toy (always on, every run):
  - independent bootstrap resample of inclusive and SEM
  - Gaussian variation of support bounds (Mx, El, q²)

Named systematics (each individually toggleable via config `systematics:`, run solo + combined):
  - incl_moments: correlated relative perturbation of raw moments, sized like the real
    experimental average
  - ff: Hammer FF eigenvariation nuisances, grouped D / D* / D** (each internally correlated)
  - bf_mode: per-decay-mode branching-fraction Gaussian nuisance (single shared draw, applied
    identically to both the pseudo-inclusive and pseudo-SEM cocktail copies)
  - bf_gap: inclusive-BR uncertainty propagated into the assumed gap-mode BR budget

Usage:
  cd clean && python3 7_hausdorff_toy.py --config config.yaml
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from math import comb
from pathlib import Path

# Avoid thread-spawn failures on constrained batch/login nodes (and when running many
# --toy-job chunks concurrently, e.g. via bsub or a local parallel launch).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.optimize import minimize
from tqdm import tqdm
import plothist

sys.path.insert(0, str(Path(__file__).parent))
from lib.moments import compute_curves_np, build_curve_context, compute_curves_from_context
from lib import systematics as syst
from lib.systematics import (
    read_parquet_downcast, MX2_LO_MIN, MX2_HI_MAX, EL_LO_MIN, EL_HI_MAX, Q2_LO_MIN, Q2_HI_MAX,
)
from lib import joint_maxent as jm

# Truth-vs-reconstruction color pair (tab10 blue/orange -- colorblind-safe, matches the palette
# already used for the decomposition/n-moments-comparison plots in this file).
TRUTH_COLOR = "#1f77b4"
RECON_COLOR = "#ff7f0e"

# ── Numerical routines (adapted from hausdorff/hausdorff_gap_simulation.py) ───

def hausdorff_moment_check(mu, tol=1e-8):
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or len(mu) == 0 or np.any(~np.isfinite(mu)):
        return False, -np.inf
    N = len(mu) - 1
    worst = np.inf
    for n in range(N + 1):
        max_k = N - n
        row = mu[n:N + 1].copy()
        worst = min(worst, row[0])
        for k in range(1, max_k + 1):
            row = row[1:] - row[:-1]
            val = ((-1) ** k) * row[0]
            worst = min(worst, val)
    return (worst >= -tol), float(worst)


def maxent_pdf(mu01, t, lam0=None, return_meta=False, tp_full=None, opt_options=None,
               alpha=0.0, beta=0.0, sigma_rel=0.0):
    """MaxEnt density on [0,1] with optional boundary constraints.

    The unnormalized density is  g(t) = t^alpha * (1-t)^beta * exp(lambda . p(t)),
    enforcing g(0)=0 when alpha>0 and g(1)=0 when beta>0.
    Moments are matched as  integral t^k g(t) dt / Z  =  mu01[k].

    sigma_rel > 0 relaxes the exact-equality dual to a Gaussian-relaxed ("soft") one, adding
    0.5*sum((sigma_j*lam_j)^2) with sigma_j = sigma_rel*max(|mu01[j]|, 1e-6) -- same mechanism
    as lib/joint_maxent.py's soft constraints (see its module docstring), but for a single
    threshold's ordinary moment vector rather than a joint multi-threshold fit. sigma_j is
    slack, not a measured uncertainty: it exists so the dual stays bounded (has a minimum at
    all) when mu01 falls just outside the achievable moment polytope for a bounded density,
    which the exact dual (sigma_rel=0, the default) has no solution for.
    """
    N = len(mu01) - 1
    tp = tp_full[:N + 1] if tp_full is not None else np.stack([t ** k for k in range(N + 1)])

    # prior shape: t^alpha * (1-t)^beta, precomputed once
    # small epsilon to avoid 0^0 issues at endpoints
    eps = 1e-300
    prior = (np.maximum(t, eps) ** alpha) * (np.maximum(1.0 - t, eps) ** beta)
    sigma = sigma_rel * np.maximum(np.abs(mu01), 1e-6) if sigma_rel > 0 else None

    def _fg(lam):
        lf = lam @ tp;  lf -= lf.max()
        return prior * np.exp(lf)

    def dual(lam):
        lf0 = lam @ tp
        val = np.log(np.trapz(prior * np.exp(lf0 - lf0.max()), t)) + lf0.max() - lam @ mu01
        if sigma is not None:
            val += 0.5 * np.sum((sigma * lam) ** 2)
        return val

    def grad(lam):
        g = _fg(lam);  Z = np.trapz(g, t);  gn = g / Z
        base = np.array([np.trapz(tp[k] * gn, t) - mu01[k] for k in range(N + 1)])
        if sigma is not None:
            base = base + (sigma ** 2) * lam
        return base

    x0 = np.zeros(N + 1) if lam0 is None else lam0.copy()
    opts = opt_options or {"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12}
    res = minimize(dual, x0, jac=grad, method="L-BFGS-B", options=opts)
    fn = _fg(res.x);  Z = np.trapz(fn, t);  fn = fn / Z
    if not return_meta:
        return fn, res.x
    mom_fit = np.array([np.trapz(tp[k] * fn, t) for k in range(N + 1)])
    return fn, res.x, {
        "opt_success": bool(res.success),
        "opt_message": str(res.message),
        "opt_nit": int(res.nit),
        "mom_err": float(np.max(np.abs(mom_fit - mu01))),
    }


def maxent_solution_ok(f, t, mu_target, mom_tol=5e-4, min_iqr_t=0.02, max_peak_t=80.0):
    if f.ndim != 1 or len(f) != len(t):
        return False
    if len(f) < 3 or np.any(~np.isfinite(f)) or np.any(f < 0):
        return False
    Z = np.trapz(f, t)
    if not np.isfinite(Z) or Z <= 0:
        return False
    fn = f / Z
    mom_fit = np.array([np.trapz((t ** k) * fn, t) for k in range(len(mu_target))])
    if np.max(np.abs(mom_fit - mu_target)) > mom_tol:
        return False
    dt = np.diff(t)
    cdf = np.empty_like(t);  cdf[0] = 0.
    cdf[1:] = np.cumsum(0.5 * (fn[1:] + fn[:-1]) * dt)
    if cdf[-1] <= 0 or not np.isfinite(cdf[-1]):
        return False
    cdf /= cdf[-1]
    iqr = float(np.interp(0.75, cdf, t)) - float(np.interp(0.25, cdf, t))
    if iqr < min_iqr_t:
        return False
    if np.max(fn) > max_peak_t:
        return False
    return True


def maxent_joint(constraints, t, prior=None):
    """Joint MaxEnt inversion using moment constraints from multiple thresholds.

    constraints: list of (mask_on_t_grid, target_value) where mask selects
                 the sub-interval [thr, hi] and target is the unnormalized
                 raw moment ∫_{thr}^{hi} t^k f(t) dt (in [0,1] coords).
    Returns f on t (normalized so total integral = 1).
    """
    if prior is None:
        prior = np.ones_like(t)
    n = len(constraints)

    def _g(lam):
        # sum lambda_j * basis_j(t), where basis_j = mask_j * t^k_j
        lf = sum(lam[j] * constraints[j][0] for j in range(n))
        lf -= lf.max()
        return prior * np.exp(lf)

    def dual(lam):
        lf = sum(lam[j] * constraints[j][0] for j in range(n))
        lf_m = lf.max()
        g = prior * np.exp(lf - lf_m)
        Z = np.trapz(g, t)
        return np.log(Z) + lf_m - sum(lam[j] * constraints[j][1] for j in range(n))

    def grad(lam):
        g = _g(lam); Z = np.trapz(g, t); gn = g / Z
        return np.array([np.trapz(constraints[j][0] * gn, t) - constraints[j][1]
                         for j in range(n)])

    res = minimize(dual, np.zeros(n), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-12})
    g = _g(res.x); Z = np.trapz(g, t); fn = g / Z
    mom_err = max(abs(np.trapz(constraints[j][0] * fn, t) - constraints[j][1])
                  for j in range(n))
    return fn, bool(res.success), float(mom_err)


def residual_mu01(x_i, w_i, x_e, w_e, lo, span, max_mom):
    """Residual moments on [0,1] via binomial change-of-variables."""
    Br = w_i.sum() - w_e.sum()
    raw = np.array([((w_i * x_i ** n).sum() - (w_e * x_e ** n).sum()) / Br
                    for n in range(max_mom)])
    mu = np.zeros(max_mom)
    for k in range(max_mom):
        mu[k] = sum(comb(k, j) * (-lo) ** (k - j) * raw[j] for j in range(k + 1)) / span ** k
    return mu


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--submit",    action="store_true",
                   help="Submit toy-chunk jobs to bsub for every active systematics run-label")
    p.add_argument("--toy-job",   action="store_true",
                   help="Run a single toy chunk for one run-label (used by batch jobs)")
    p.add_argument("--merge",     action="store_true",
                   help="Merge previously-submitted toy chunks and write summary/figures")
    p.add_argument("--run-label", default=None,
                   help="Run-label this chunk covers (used with --toy-job)")
    p.add_argument("--chunk",     type=int, default=0, help="Chunk index (0-based, --toy-job)")
    p.add_argument("--n-chunks",  type=int, default=1, help="Total chunks (--toy-job/--submit)")
    p.add_argument("--dry-run",   action="store_true", help="Print bsub commands without running")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    ht   = cfg.get("hausdorff_toy", {})

    gap_mode    = str(ht.get("gap_mode", "D2stDeta"))
    n_toys      = int(ht.get("n_toys", 200))
    n_moments   = [int(x) for x in ht.get("n_moments", [4, 6, 8])]
    grid_size   = int(ht.get("grid_size", 2000))
    bounds_sig  = float(ht.get("bounds_sigma", 0.15))
    el_cuts     = np.array(ht.get("el_cuts",  [0.0, 0.6, 1.0]), dtype=float)
    q2_cuts     = np.array(ht.get("q2_cuts",  [1.5, 3.0, 6.0]), dtype=float)
    _ba = ht.get("boundary_alpha", {})
    _bb = ht.get("boundary_beta",  {})
    alpha = {"Mx": float(_ba.get("mx2", 0)), "El": float(_ba.get("el", 0)), "Q2": float(_ba.get("q2", 0))}
    beta  = {"Mx": float(_bb.get("mx2", 0)), "El": float(_bb.get("el", 0)), "Q2": float(_bb.get("q2", 0))}
    mx_support  = list(ht.get("mx_support", [1.8, 2.7]))
    el_support  = list(ht.get("el_support", [0.2, 2.4]))
    q2_support  = list(ht.get("q2_support", [0.0, 12.0]))
    seed        = int(ht.get("seed", 42))

    out_dir  = Path(cfg["paths"]["output"]) / "7"
    fig_dir  = Path(cfg["paths"]["figures"]) / "7"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --toy-job/--merge/--submit only care about the toy loop -- skip the (slow, standalone)
    # nominal-diagnostic figure sections that a plain local run also produces.
    skip_diagnostics = bool(args.toy_job or args.merge or args.submit)

    # ── Systematics run registry (needed before data loading to decide which columns/
    # structures are actually required -- matters for --toy-job chunk jobs, which run on
    # LSF's default queue with a hard ~4GB memory cap that a full FF setup can exceed) ──
    sysc = cfg.get("systematics", {})
    runs = syst.active_systematics(sysc)
    # "all_soft": same active systematics as "all", but _run_toy_loop below switches to a
    # Gaussian-relaxed MaxEnt dual for any run_label ending in "_soft" instead of requiring
    # exact moment equality -- see maxent_pdf's sigma_rel docstring.
    if "all" in runs and bool(ht.get("soft_all_enable", False)):
        runs["all_soft"] = list(runs["all"])
    if args.toy_job:
        need_ff = "ff" in runs.get(args.run_label, [])
    elif args.merge or args.submit:
        need_ff = False  # neither mode runs the toy loop directly
    else:
        need_ff = any("ff" in active for active in runs.values())  # local mode: all labels

    # ── Joint multi-threshold MaxEnt (lib/joint_maxent.py), opt-in via config -- see
    # scratch/joint_slack_test.py / joint_slack_toy_test.py for the closure-test validation
    # behind these defaults. Only El/Q2 (Mx isn't cut on itself).
    jmc = cfg.get("joint_maxent", {})
    joint_enable = bool(jmc.get("enable", False))
    joint_sigma_rel = float(jmc.get("sigma_rel", 0.03))
    joint_gammas = (float(jmc.get("gamma_value", 30.0)), float(jmc.get("gamma_slope", 30.0)))
    joint_n = {"El": int(jmc.get("n_moments", {}).get("El", 5)),
               "Q2": int(jmc.get("n_moments", {}).get("Q2", 3))}
    joint_cuts = {"El": [float(c) for c in jmc.get("el_cuts", [0.0, 0.4, 0.8, 1.2, 1.6])],
                  "Q2": [float(c) for c in jmc.get("q2_cuts", [0.5, 2.0, 3.5, 5.0, 6.5])]}
    if joint_enable:
        print(f"  Joint MaxEnt: ENABLED  n_moments={joint_n}  gammas={joint_gammas}  "
              f"sigma_rel={joint_sigma_rel}")

    # ── Load data ─────────────────────────────────────────────────────────────
    # pseudo-inclusive = SEM cocktail + gap mode
    # pseudo-SEM       = SEM cocktail only
    print("Loading cocktail …")
    cocktail_path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    gap_path      = Path(cfg["paths"]["output"]) / "3" / f"{gap_mode}.parquet"
    # cocktail.parquet has ~109 columns (FF eigenvariations, Hammer rate diagnostics, etc.);
    # only read what's actually used, and only read the (~16) FF eigenvariation columns when
    # an active run actually needs the "ff" systematic.
    _base_cols = ["Mx", "El_B", "q2", "total_weight", "bf", "bf_unc", "decay_name"]
    ff_cols: list[str] = []
    if need_ff:
        _cocktail_schema_cols = pq.ParquetFile(cocktail_path).schema_arrow.names
        ff_cols = ["ff_weight"] + [c for c in _cocktail_schema_cols
                                    if c.startswith("ff_weight_up") or c.startswith("ff_weight_down")]
    # FF columns are deliberately kept OUT of df_full/the big concat below: df_gap has no
    # ff_weight_up*/down* columns, so concatenating them in would force a reindex-with-NaN
    # that upcasts back to float64 (pandas requires a common dtype across the concat), which
    # defeats the float32 downcast and re-creates the >4GB peak we're trying to avoid. Instead
    # ff_cols stay on df_sem alone (cocktail-only) and get filtered/reused directly below.
    df_sem = read_parquet_downcast(cocktail_path, _base_cols + ff_cols, float32_cols=ff_cols)
    df_gap = pd.read_parquet(gap_path, columns=_base_cols)
    gap_decay_names = set(df_gap["decay_name"].unique())
    print(f"  SEM cocktail: {len(df_sem):,} rows  |  gap mode '{gap_mode}': {len(df_gap):,} rows")
    print(f"  gap decay_names: {gap_decay_names}")

    # Concatenate to form pseudo-inclusive, label gap events (base columns only)
    df_gap = df_gap.assign(_is_gap=True)
    df_sem = df_sem.assign(_is_gap=False)
    cols = _base_cols + ["_is_gap"]
    df_full = pd.concat([df_sem[cols], df_gap[cols]], ignore_index=True)

    ok = (
        np.isfinite(df_full["Mx"].to_numpy()) & np.isfinite(df_full["El_B"].to_numpy())
        & np.isfinite(df_full["q2"].to_numpy()) & np.isfinite(df_full["total_weight"].to_numpy())
        & (df_full["total_weight"].to_numpy() > 0)
    )
    df_full = df_full.loc[ok].reset_index(drop=True)

    # Same filter, applied directly to df_sem, gives exactly the cocktail rows that survive in
    # df_full (df_sem's rows come first, in the same order, in the pd.concat above) -- this is
    # what feeds the FF slope matrix, without ever routing FF columns through the big concat.
    if ff_cols:
        ok_sem = (
            np.isfinite(df_sem["Mx"].to_numpy()) & np.isfinite(df_sem["El_B"].to_numpy())
            & np.isfinite(df_sem["q2"].to_numpy()) & np.isfinite(df_sem["total_weight"].to_numpy())
            & (df_sem["total_weight"].to_numpy() > 0)
        )
        df_cocktail_ff = df_sem.loc[ok_sem, ["decay_name", "bf", "bf_unc"] + ff_cols].reset_index(drop=True)
    else:
        df_cocktail_ff = None
    del df_sem, df_gap
    gc.collect()

    mx  = df_full["Mx"].to_numpy(float)
    mx2 = mx ** 2
    el  = df_full["El_B"].to_numpy(float)
    q2  = df_full["q2"].to_numpy(float)
    w   = df_full["total_weight"].to_numpy(float)
    is_gap = df_full["_is_gap"].to_numpy(bool)

    # ── Systematics setup (sysc/runs/need_ff already computed above, before data loading) ──
    print(f"  Systematics runs: {list(runs.keys())}  (need_ff={need_ff})")

    exp_avg_path = Path(cfg["paths"]["output"]) / "4" / "experimental_average.json"
    exp_avg = json.load(open(exp_avg_path)) if exp_avg_path.exists() else {}
    # Orders 1-3 use the real measured rel_unc/correlation; order 4 (needed once n_moments
    # includes 5, which uses orders 0-4) is not measured -- extrapolate it (ad-hoc, see
    # extrapolate_incl_moment_order4's docstring) rather than leaving it as an unperturbed
    # anchor against a strongly-correlated shift in orders 1-3.
    _ext_keys, incl_rel_unc, _ext_corr = syst.extrapolate_incl_moment_order4(exp_avg)
    incl_corr_by_obs = {
        "Mx": (["mx_1", "mx_2", "mx_3", "mx_4"], _ext_corr[0:4, 0:4]),
        "El": (["el_1", "el_2", "el_3", "el_4"], _ext_corr[4:8, 4:8]),
        "Q2": (["q2_1", "q2_2", "q2_3", "q2_4"], _ext_corr[8:12, 8:12]),
    }

    bf_incl     = float(sysc.get("bf_incl", 0.1105))
    bf_incl_unc = float(sysc.get("bf_incl_unc", 0.0016))

    # FF/BF nuisance structure, precomputed once on the cocktail-only rows (indices into the
    # length-len(df_full) arrays), reused every toy -- only the theta draw changes.
    # df_cocktail (base columns, from df_full) and df_cocktail_ff (FF columns, built directly
    # from df_sem above) are row-for-row aligned: both are filtered with the identical `ok`
    # criteria, and df_sem's rows are the first block in the df_full concat, so the same rows
    # survive in the same relative order in both.
    cocktail_idx = np.flatnonzero(~is_gap)
    df_cocktail = df_full.iloc[cocktail_idx].reset_index(drop=True)
    bf_sem_computed, bf_gap_central, sigma_bf_gap = syst.compute_bf_gap(df_cocktail, bf_incl, bf_incl_unc)
    print(f"  bf_gap: computed central={bf_gap_central*100:.3f}%  sigma={sigma_bf_gap*100:.3f}%"
          f"  (bf_sem_computed={bf_sem_computed*100:.3f}%, bf_incl={bf_incl*100:.3f}%)")

    if need_ff:
        assert len(df_cocktail_ff) == len(df_cocktail), (
            f"df_cocktail_ff/df_cocktail row-count mismatch: "
            f"{len(df_cocktail_ff)} vs {len(df_cocktail)} -- alignment assumption broken")
        ff_slope_matrix, ff_nuisance_names = syst.build_ff_slope_matrix(df_cocktail_ff)
    else:
        ff_slope_matrix, ff_nuisance_names = np.zeros((len(df_cocktail), 0)), []
    bf_mode_codes, bf_rel_unc_per_mode = syst.bf_mode_setup(
        df_cocktail["decay_name"].to_numpy(), df_cocktail["bf"].to_numpy(),
        df_cocktail["bf_unc"].to_numpy())
    print(f"  FF nuisances: {len(ff_nuisance_names)} ({ff_nuisance_names})")
    print(f"  BF-mode nuisances: {len(bf_rel_unc_per_mode)} unique decay modes")

    # Nothing downstream needs df_cocktail or the raw FF eigenvariation columns anymore --
    # only ff_slope_matrix (already extracted, float32). df_full never carried ff_cols (kept
    # out of the concat above) and df_sem/df_gap were already freed after loading.
    del df_cocktail
    if need_ff:
        del df_cocktail_ff
    gc.collect()

    # Support bounds in Mx² (GeV²), El (GeV), q² (GeV²)
    Mx2_lo_nom = float(mx_support[0]) if mx_support else float(mx2[is_gap].min() * 0.999)
    Mx2_hi_nom = float(mx_support[1]) if mx_support else float(mx2[is_gap].max() * 1.001)
    El_lo_nom  = float(el_support[0]) if el_support else float(el[is_gap].min() * 0.999)
    El_hi_nom  = float(el_support[1]) if el_support else float(el[is_gap].max() * 1.001)
    Q2_lo_nom  = float(q2_support[0]) if q2_support else float(q2[is_gap].min() * 0.999)
    Q2_hi_nom  = float(q2_support[1]) if q2_support else float(q2[is_gap].max() * 1.001)
    span_nom    = Mx2_hi_nom - Mx2_lo_nom
    span_El_nom = El_hi_nom - El_lo_nom
    span_Q2_nom = Q2_hi_nom - Q2_lo_nom

    print(f"  Mx² support: [{Mx2_lo_nom:.3f}, {Mx2_hi_nom:.3f}] GeV²  span={span_nom:.3f}")
    print(f"  El  support: [{El_lo_nom:.3f}, {El_hi_nom:.3f}] GeV   span={span_El_nom:.3f}")
    print(f"  q²  support: [{Q2_lo_nom:.3f}, {Q2_hi_nom:.3f}] GeV²  span={span_Q2_nom:.3f}")

    # ── Common grids ──────────────────────────────────────────────────────────
    t_grid  = np.linspace(0., 1., grid_size)
    tp_grid = np.stack([t_grid ** k for k in range(max(max(n_moments), 10) + 1)])
    Mx2_grid = Mx2_lo_nom + t_grid * span_nom
    El_grid  = El_lo_nom  + t_grid * span_El_nom
    Q2_grid  = Q2_lo_nom  + t_grid * span_Q2_nom

    MAX_MOM = max(n_moments) + 1

    # ── Nominal residual moments ───────────────────────────────────────────────
    print("Computing nominal residual moments …")
    w_incl = w.copy()
    w_excl = np.where(~is_gap, w, 0.)

    mu_nom_Mx2 = residual_mu01(mx2, w_incl, mx2, w_excl, Mx2_lo_nom, span_nom, MAX_MOM)
    mu_nom_El  = residual_mu01(el,  w_incl, el,  w_excl, El_lo_nom,  span_El_nom, MAX_MOM)
    mu_nom_Q2  = residual_mu01(q2,  w_incl, q2,  w_excl, Q2_lo_nom,  span_Q2_nom, MAX_MOM)

    # ── Truth histograms ───────────────────────────────────────────────────────
    w_gap = np.where(is_gap, w, 0.)
    bins_Mx2 = np.linspace(Mx2_lo_nom, Mx2_hi_nom, 70)
    bins_El  = np.linspace(El_lo_nom,  El_hi_nom,  70)
    bins_Q2  = np.linspace(Q2_lo_nom,  Q2_hi_nom,  70)
    bct_Mx2  = 0.5 * (bins_Mx2[:-1] + bins_Mx2[1:]);  db_Mx2 = bins_Mx2[1] - bins_Mx2[0]
    bct_El   = 0.5 * (bins_El[:-1]  + bins_El[1:]);   db_El  = bins_El[1]  - bins_El[0]
    bct_Q2   = 0.5 * (bins_Q2[:-1]  + bins_Q2[1:]);   db_Q2  = bins_Q2[1]  - bins_Q2[0]
    norm_gap = w_gap.sum()
    h_Mx2, _ = np.histogram(mx2, bins=bins_Mx2, weights=w_gap);  h_Mx2 = h_Mx2 / (norm_gap * db_Mx2)
    h_El,  _ = np.histogram(el,  bins=bins_El,  weights=w_gap);  h_El  = h_El  / (norm_gap * db_El)
    h_Q2,  _ = np.histogram(q2,  bins=bins_Q2,  weights=w_gap);  h_Q2  = h_Q2  / (norm_gap * db_Q2)

    # Mx-domain truth histogram (Mx = sqrt(Mx²)), built directly from MC truth rather than by
    # transforming the Mx² histogram -- used for plotting reconstructions on an Mx axis.
    Mx_lo_nom, Mx_hi_nom = float(np.sqrt(Mx2_lo_nom)), float(np.sqrt(Mx2_hi_nom))
    bins_Mx = np.linspace(Mx_lo_nom, Mx_hi_nom, 350)
    bct_Mx  = 0.5 * (bins_Mx[:-1] + bins_Mx[1:]);  db_Mx = bins_Mx[1] - bins_Mx[0]
    h_Mx, _ = np.histogram(mx, bins=bins_Mx, weights=w_gap);  h_Mx = h_Mx / (norm_gap * db_Mx)

    def _pctile_range(x, w, lo_pct=0.5, hi_pct=99.5, margin=0.05):
        """Percentile-based DISPLAY range (not used for the actual reconstruction, which needs
        the full physical support) -- the true kinematic endpoints can leave a long near-empty
        tail that isn't useful to plot (e.g. Mx's support goes well past where the gap mode's
        density has dropped to ~0)."""
        order = np.argsort(x);  xs = x[order];  ws = w[order]
        cw = np.cumsum(ws);  cw /= cw[-1]
        lo = float(np.interp(lo_pct / 100, cw, xs))
        hi = float(np.interp(hi_pct / 100, cw, xs))
        pad = margin * (hi - lo)
        return lo - pad, hi + pad

    is_gap_mask = is_gap  # w_gap is already zero outside is_gap, but argsort needs finite x too
    Mx_disp  = _pctile_range(mx[is_gap_mask],  w[is_gap_mask])
    El_disp  = _pctile_range(el[is_gap_mask],  w[is_gap_mask])
    Q2_disp  = _pctile_range(q2[is_gap_mask],  w[is_gap_mask])
    DISP_RANGE = {"Mx": Mx_disp, "El": El_disp, "Q2": Q2_disp}

    # ── Figure: nominal MaxEnt convergence (no smearing) ─────────────────────
    print("Figure: nominal MaxEnt convergence …")
    N_LIST_NOM = sorted(set(n_moments))
    obs_nom_cfg = [
        ("Mx2", "Mx", Mx2_grid, bct_Mx2, h_Mx2, db_Mx2, Mx2_lo_nom, Mx2_hi_nom,
         mu_nom_Mx2, r"$M_X^2\ [\mathrm{GeV}^2]$", r"$f(M_X^2)\ [\mathrm{GeV}^{-2}]$",
         "maxent_convergence_Mx2.png"),
        ("El",  "El", El_grid,  bct_El,  h_El,  db_El,  El_lo_nom,  El_hi_nom,
         mu_nom_El,  r"$E_\ell^B\ [\mathrm{GeV}]$", r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$",
         "maxent_convergence_El.png"),
        ("Q2",  "Q2", Q2_grid,  bct_Q2,  h_Q2,  db_Q2,  Q2_lo_nom,  Q2_hi_nom,
         mu_nom_Q2,  r"$q^2\ [\mathrm{GeV}^2]$",   r"$f(q^2)\ [\mathrm{GeV}^{-2}]$",
         "maxent_convergence_Q2.png"),
    ]
    if skip_diagnostics:
        obs_nom_cfg = []  # --toy-job/--merge/--submit: skip standalone diagnostic figures
    for obs_lbl, obs_key, xgrid, bct, h_true, db, xlo, xhi, mu_nom_obs, xlabel, ylabel, fname in obs_nom_cfg:
        fig_c, axes_c = plt.subplots(1, len(N_LIST_NOM),
                                     figsize=(3.8 * len(N_LIST_NOM), 4), dpi=150, sharey=True)
        if len(N_LIST_NOM) == 1:
            axes_c = [axes_c]
        span_obs = xhi - xlo
        a, b = alpha[obs_key], beta[obs_key]

        def _draw_convergence(axes_list, al, be, title_suffix):
            for ax, n in zip(axes_list, N_LIST_NOM):
                ok_h, worst = hausdorff_moment_check(mu_nom_obs[:n])
                f_t, _ = maxent_pdf(mu_nom_obs[:n], t_grid, tp_full=tp_grid, alpha=al, beta=be)
                f_phys = f_t / span_obs
                mom_err = max(abs(np.trapz(t_grid**k * f_t, t_grid) - mu_nom_obs[k])
                             for k in range(n))
                ax.bar(bct, h_true, width=db, color=TRUTH_COLOR, alpha=0.55, label=f"True {gap_mode}")
                ax.plot(xgrid, f_phys, color=RECON_COLOR, lw=2.2,
                        label=f"MaxEnt ({n} mom.)\nerr={mom_err:.1e}")
                ax.set_title(f"{n} moments  Hausdorff {'OK' if ok_h else 'FAIL'} "
                             f"({worst:.2e}){title_suffix}", fontsize=12)
                ax.set_xlabel(xlabel, fontsize=13)
                ax.tick_params(labelsize=11)
                ax.set_xlim(xlo, xhi);  ax.set_ylim(bottom=0);  ax.grid(alpha=0.25)
                ax.legend(fontsize=11, loc="upper right")
            axes_list[0].set_ylabel(ylabel, fontsize=13)

        _draw_convergence(axes_c, a, b, "")
        axes_c[0].set_ylabel(ylabel, fontsize=13)
        fig_c.suptitle(f"Nominal MaxEnt convergence ({obs_lbl}) — gap: {gap_mode}", fontsize=13)
        fig_c.tight_layout()
        fig_c.savefig(fig_dir / fname);  plt.close(fig_c)
        print(f"  → {fig_dir / fname}")

        # For El: also save unconstrained version as separate figure
        if obs_key == "El" and (a != 0 or b != 0):
            fig_u, axes_u = plt.subplots(1, len(N_LIST_NOM),
                                         figsize=(3.8 * len(N_LIST_NOM), 4), dpi=150, sharey=True)
            if len(N_LIST_NOM) == 1:
                axes_u = [axes_u]
            _draw_convergence(axes_u, 0., 0., " (unconstrained)")
            fig_u.suptitle(f"Nominal MaxEnt convergence ({obs_lbl}, unconstrained) — gap: {gap_mode}",
                           fontsize=13)
            fig_u.tight_layout()
            unc_fname = fname.replace(".png", "_unconstrained.png")
            fig_u.savefig(fig_dir / unc_fname);  plt.close(fig_u)
            print(f"  → {fig_dir / unc_fname}")

    # ── Base weight for the toy loop (systematics multiply this, see below) ────
    N_ev        = len(df_full)
    w_base      = df_full["total_weight"].to_numpy(float)

    # ── Overlay check: MaxEnt at multiple thresholds, normalized for consistency ─
    print("Figure: multi-threshold MaxEnt overlay check …")

    N_overlay = max(n_moments)

    # For Mx: cuts are on El, support bounds stay fixed; mask events by el >= thr
    # For El/Q2: cuts are on the same variable, support lower bound shifts with thr
    overlay_obs = [
        ("Mx", mx2, el,  w_incl, w_excl, Mx2_lo_nom, Mx2_hi_nom, Mx2_grid, el_cuts,
         True,  r"$E_\ell^B\ [\mathrm{GeV}]$", r"$M_X^2\ [\mathrm{GeV}^2]$",
         r"$f(M_X^2)\ [\mathrm{GeV}^{-2}]$"),
        ("El",  el,  el,  w_incl, w_excl, El_lo_nom,  El_hi_nom,  El_grid,  el_cuts,
         False, r"$E_\ell^B\ [\mathrm{GeV}]$", r"$E_\ell^B\ [\mathrm{GeV}]$",
         r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$"),
        ("Q2",  q2,  q2,  w_incl, w_excl, Q2_lo_nom,  Q2_hi_nom,  Q2_grid,  q2_cuts,
         False, r"$q^2\ [\mathrm{GeV}^2]$",    r"$q^2\ [\mathrm{GeV}^2]$",
         r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"),
    ]
    if skip_diagnostics:
        overlay_obs = []

    for obs, x_arr, cut_arr, wi, we, xlo_base, xhi, xgrid, cuts, fixed_support, cut_xlabel, xlabel, ylabel in overlay_obs:
        good = []        # (thr, xgrid_thr, f_phys)
        truth = []       # single entry: lowest-cut truth histogram
        w_gap_ov = np.where(is_gap, w, 0.)
        for thr in sorted(cuts):
            if fixed_support:
                # Mx: mask on El, Mx² support unchanged
                xlo_thr, span_thr = xlo_base, xhi - xlo_base
                xgrid_thr = xgrid
                mask = cut_arr >= thr
            else:
                # El/Q2: support lower bound = threshold
                xlo_thr = float(thr)
                span_thr = xhi - xlo_thr
                xgrid_thr = xlo_thr + t_grid * span_thr
                mask = cut_arr >= thr
            wi_cut = np.where(mask, wi, 0.)
            we_cut = np.where(mask, we, 0.)
            if wi_cut.sum() <= 0:
                continue
            mu = residual_mu01(x_arr, wi_cut, x_arr, we_cut, xlo_thr, span_thr, N_overlay)
            ok_h, _ = hausdorff_moment_check(mu[:N_overlay])
            if not ok_h:
                print(f"  {obs} El>{thr:.1f}: Hausdorff fail")
                continue
            f_t, _ = maxent_pdf(mu[:N_overlay], t_grid, tp_full=tp_grid,
                                alpha=alpha[obs], beta=beta[obs])
            f_phys = f_t / span_thr
            mom_err = max(abs(np.trapz(tp_grid[k]*f_t, t_grid) - mu[k])
                          for k in range(N_overlay))
            if mom_err >= 1e-4:
                print(f"  {obs} El>{thr:.1f}: bad mom_err {mom_err:.1e}")
                continue
            print(f"  {obs} El>{thr:.1f}: ok, mom_err={mom_err:.1e}")
            good.append((float(thr), xgrid_thr, f_phys))
            if not truth:
                w_gap_cut = np.where(mask, w_gap_ov, 0.)
                norm_gap_cut = w_gap_cut.sum()
                bins_t = np.linspace(xlo_thr, xhi, 50)
                bct_t  = 0.5 * (bins_t[:-1] + bins_t[1:])
                db_t   = bins_t[1] - bins_t[0]
                h_t, _ = np.histogram(x_arr, bins=bins_t, weights=w_gap_cut)
                h_phys = h_t / (norm_gap_cut * db_t) if norm_gap_cut > 0 else h_t * 0
                truth.append((bct_t, h_phys, db_t))

        if len(good) < 2:
            print(f"  {obs}: fewer than 2 good curves, skip overlay")
            continue

        def _overlay_panel(ax, good_curves, truth_lowest=None):
            # Normalize from highest threshold down:
            # - highest curve: normalize to 1 over its own support
            # - each lower curve i: scale so ∫_{thr_{i+1}}^hi f_i dx = total of scaled curve i+1
            #   → lower curve has same integral above the next cut, extra area below is the
            #   acceptance ratio
            if len(good_curves) < 2:
                return
            sc = [None] * len(good_curves)
            norm0 = 1.0  # normalization of the lowest-cut curve (index 0)
            for i in range(len(good_curves) - 1, -1, -1):
                thr, xg, fp = good_curves[i]
                if i == len(good_curves) - 1:
                    n = np.trapz(fp, xg)
                else:
                    thr_next, xg_next, fp_next_s = sc[i + 1]
                    ref   = np.trapz(fp_next_s, xg_next)
                    above = np.trapz(fp[xg >= thr_next], xg[xg >= thr_next])
                    n  = above / ref if ref > 0 else 1.0
                n = n if n > 0 else 1.0
                if i == 0:
                    norm0 = n
                sc[i] = (thr, xg, fp / n)
            cmap_ov = plt.get_cmap("viridis")
            cols_ov = [cmap_ov(i / (len(sc) - 1)) for i in range(len(sc))]
            for i, (thr, xg, fps) in enumerate(sc):
                ax.plot(xg, fps, color=cols_ov[i], lw=1.8,
                        label=cut_xlabel.split("[")[0].strip() + f" > {thr:.1f}")
            if truth_lowest is not None:
                bct_t, h_phys, db_t = truth_lowest
                ax.bar(bct_t, h_phys / norm0, width=db_t,
                       color=RECON_COLOR, alpha=0.25, linewidth=0, label=f"truth {gap_mode}")
            ymax = np.percentile(np.concatenate([fp for _, _, fp in sc]), 99)
            ax.set_ylim(0, 1.2 * ymax)
            ax.set_xlim(good_curves[0][1][0], xhi)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("rescaled " + ylabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)

        # Single-panel plot with N_overlay moments
        fig_ov, ax_ov = plt.subplots(figsize=(7, 4.5), dpi=150)
        _overlay_panel(ax_ov, good, truth_lowest=truth[0] if truth else None)
        fig_ov.suptitle(
            f"MC MaxEnt overlay — {obs}  ({N_overlay} moments, gap: {gap_mode})\n"
            r"each curve: $\int_{\mathrm{thr}_{i+1}}^{\mathrm{hi}} f\,dx = 1$",
            fontsize=10)
        fig_ov.tight_layout()
        out_ov = fig_dir / f"maxent_overlay_{obs}.pdf"
        fig_ov.savefig(out_ov); plt.close(fig_ov)
        print(f"  → {out_ov}")

        # Multi-panel: one panel per moment count
        n_list_ov = [2, 3, 4, 7, 10]
        fig_mp, axes_mp = plt.subplots(1, len(n_list_ov),
                                       figsize=(4.5 * len(n_list_ov), 4.5), dpi=150)
        for ax_mp, n_ov in zip(axes_mp, n_list_ov):
            good_n = []
            truth_n = []
            for thr in sorted(cuts):
                if fixed_support:
                    xlo_thr_n, span_thr_n = xlo_base, xhi - xlo_base
                    xgrid_thr_n = xgrid
                    mask_n = cut_arr >= thr
                else:
                    xlo_thr_n = float(thr); span_thr_n = xhi - xlo_thr_n
                    xgrid_thr_n = xlo_thr_n + t_grid * span_thr_n
                    mask_n = cut_arr >= thr
                wi_cut = np.where(mask_n, wi, 0.)
                we_cut = np.where(mask_n, we, 0.)
                if wi_cut.sum() <= 0:
                    continue
                mu_n = residual_mu01(x_arr, wi_cut, x_arr, we_cut, xlo_thr_n, span_thr_n, n_ov)
                ok_h, _ = hausdorff_moment_check(mu_n[:n_ov])
                if not ok_h:
                    continue
                try:
                    f_t, _ = maxent_pdf(mu_n[:n_ov], t_grid, tp_full=tp_grid,
                                        alpha=alpha[obs], beta=beta[obs])
                except Exception:
                    continue
                mom_err = max(abs(np.trapz(tp_grid[k]*f_t, t_grid) - mu_n[k])
                              for k in range(n_ov))
                if mom_err >= 1e-4:
                    continue
                good_n.append((float(thr), xgrid_thr_n, f_t / span_thr_n))
                if not truth_n:
                    w_gap_cut_n = np.where(mask_n, w_gap_ov, 0.)
                    norm_g = w_gap_cut_n.sum()
                    bins_tn = np.linspace(xlo_thr_n, xhi, 50)
                    bct_tn  = 0.5 * (bins_tn[:-1] + bins_tn[1:])
                    db_tn   = bins_tn[1] - bins_tn[0]
                    h_tn, _ = np.histogram(x_arr, bins=bins_tn, weights=w_gap_cut_n)
                    h_tn = h_tn / (norm_g * db_tn) if norm_g > 0 else h_tn * 0
                    truth_n.append((bct_tn, h_tn, db_tn))
            _overlay_panel(ax_mp, good_n, truth_lowest=truth_n[0] if truth_n else None)
            ax_mp.set_title(f"{n_ov} moments", fontsize=10)
        fig_mp.suptitle(
            f"MC MaxEnt overlay — {obs}, varying moments  (gap: {gap_mode})\n"
            r"each curve: $\int_{\mathrm{thr}_{i+1}}^{\mathrm{hi}} f\,dx = 1$",
            fontsize=10)
        fig_mp.tight_layout()
        out_mp = fig_dir / f"maxent_overlay_{obs}_moments.pdf"
        fig_mp.savefig(out_mp); plt.close(fig_mp)
        print(f"  → {out_mp}")

    # ── Joint MaxEnt inversion across all thresholds ──────────────────────────
    print("Figure: joint MaxEnt inversion …")

    joint_obs = [
        ("El", el,  w_incl, w_excl, El_lo_nom,  El_hi_nom,  el_cuts,
         r"$E_\ell^B\ [\mathrm{GeV}]$",  r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$"),
        ("Q2", q2,  w_incl, w_excl, Q2_lo_nom,  Q2_hi_nom,  q2_cuts,
         r"$q^2\ [\mathrm{GeV}^2]$",     r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"),
    ]
    if skip_diagnostics:
        joint_obs = []

    for obs, x_arr, wi, we, xlo, xhi, cuts, xlabel, ylabel in joint_obs:
        span_obs = xhi - xlo
        xgrid = xlo + t_grid * span_obs  # physical grid on full support
        w_gap_arr = wi - we

        # Build constraints: for each threshold and each moment order k=0..N_overlay-1
        # target = ∫_{thr}^{hi} x^k * f(x) dx  (unnormalized, in physical coords)
        # In [0,1] coords: x = xlo + t*span, dx = span*dt
        # ∫_{thr}^{hi} x^k f(x) dx = span * ∫_{t_thr}^{1} x(t)^k * f(xlo+t*span)/span dt
        #                           = ∫_{t_thr}^{1} x(t)^k * f_t(t) dt   (f_t = f_phys * span)
        # So in t-space the unnormalized raw moment is:
        #   M_k(thr) = ∫_{t_thr}^{1} (xlo + t*span)^k * g(t) dt   where g = f_t (unnorm)
        # We match these to the MC:  M_k(thr) = (∑_{x>=thr} w_gap * x^k) / (∑_all w_gap)
        # The last factor normalizes g so that ∫_0^1 g dt = 1 (total probability = 1).

        norm_total = w_gap_arr[x_arr >= xlo].sum()
        if norm_total <= 0:
            print(f"  {obs}: no gap weight, skip joint"); continue

        constraints = []
        for thr in sorted(cuts):
            mask_phys = x_arr >= thr
            t_thr = (float(thr) - xlo) / span_obs
            mask_t = t_grid >= t_thr   # sub-interval mask on t_grid
            x_grid_k = xgrid           # physical x on full t_grid
            for k in range(N_overlay):
                target = (w_gap_arr * np.where(mask_phys, x_arr**k, 0.)).sum() / norm_total
                basis  = np.where(mask_t, x_grid_k**k, 0.)
                constraints.append((basis, target))

        eps = 1e-300
        prior_j = (np.maximum(t_grid, eps) ** alpha[obs]) * (np.maximum(1.0 - t_grid, eps) ** beta[obs])

        f_joint, ok_j, mom_err_j = maxent_joint(constraints, t_grid, prior=prior_j)
        f_joint_phys = f_joint / span_obs
        print(f"  {obs}: joint ok={ok_j}, mom_err={mom_err_j:.1e}")

        # truth histogram on full support
        w_gap_full = np.where(is_gap, w, 0.)
        bins_j = np.linspace(xlo, xhi, 70)
        bct_j  = 0.5 * (bins_j[:-1] + bins_j[1:]); db_j = bins_j[1] - bins_j[0]
        h_j, _ = np.histogram(x_arr, bins=bins_j, weights=w_gap_full)
        h_j = h_j / (w_gap_full.sum() * db_j)

        fig_j, ax_j = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax_j.bar(bct_j, h_j, width=db_j, color=TRUTH_COLOR, alpha=0.55, label=f"True {gap_mode}")
        ax_j.plot(xgrid, f_joint_phys, color=RECON_COLOR, lw=2, label="Joint MaxEnt")
        ax_j.set_xlim(xlo, xhi); ax_j.set_ylim(bottom=0)
        ax_j.set_xlabel(xlabel, fontsize=10); ax_j.set_ylabel(ylabel, fontsize=10)
        ax_j.legend(fontsize=9); ax_j.grid(alpha=0.25)
        fig_j.suptitle(
            f"Joint MaxEnt — {obs}  ({len(cuts)} thresholds × {N_overlay} moments"
            f" = {len(constraints)} constraints, gap: {gap_mode})",
            fontsize=10)
        fig_j.tight_layout()
        out_j = fig_dir / f"maxent_joint_{obs}.pdf"
        fig_j.savefig(out_j); plt.close(fig_j)
        print(f"  → {out_j}")

    # ── Toy loop, run once per systematics run-label ───────────────────────────
    OBS = ["Mx", "El", "Q2"]
    SEM_KEYS = ["mx_1","mx_2","mx_3","el_1","el_2","el_3","q2_1","q2_2","q2_3"]
    toy_opt_options = {"maxiter": 6000, "ftol": 1e-11, "gtol": 1e-9}

    # Nominal SEM-style gap curves (truth: incl - excl), unaffected by systematics.
    n_el_plot = 35;  n_q2_plot = 35
    el_plot_cuts = np.linspace(0.0, 1.7, n_el_plot)
    q2_plot_cuts = np.linspace(0.0, 10.0, n_q2_plot)
    gap_nom_i = compute_curves_np(mx ** 2, el, q2, w_incl, el_plot_cuts, q2_plot_cuts)
    gap_nom_e = compute_curves_np(mx ** 2, el, q2, w_excl, el_plot_cuts, q2_plot_cuts)

    # el/q2 values (and hence their sort order) never change across toys -- only the weight
    # vector does. Precompute the sort/power context once; the toy loop then pays only the
    # O(N) weight-dependent part instead of a fresh O(N log N) argsort on every call.
    curve_ctx = build_curve_context(mx ** 2, el, q2)
    # mx2**k / el**k / q2**k don't depend on the toy's weight vector either -- precompute once.
    mx2_pows = [mx2 ** k for k in range(MAX_MOM)]
    el_pows  = [el  ** k for k in range(MAX_MOM)]
    q2_pows  = [q2  ** k for k in range(MAX_MOM)]

    obs_cfg = [
        ("Mx", Mx2_grid, bct_Mx2, h_Mx2, db_Mx2, Mx2_lo_nom, Mx2_hi_nom,
         mu_nom_Mx2, r"$M_X^2\ [\mathrm{GeV}^2]$",  r"$f(M_X^2)\ [\mathrm{GeV}^{-2}]$"),
        ("El", El_grid,  bct_El,  h_El,  db_El,  El_lo_nom,  El_hi_nom,
         mu_nom_El,  r"$E_\ell^B\ [\mathrm{GeV}]$",  r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$"),
        ("Q2", Q2_grid,  bct_Q2,  h_Q2,  db_Q2,  Q2_lo_nom,  Q2_hi_nom,
         mu_nom_Q2,  r"$q^2\ [\mathrm{GeV}^2]$",     r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"),
    ]

    # Priors for the joint fit (same alpha/beta boundary convention as the legacy path), built
    # once -- El/Q2 only, on the fixed nominal support (the joint path doesn't (yet) thread
    # through the support-bound smearing the legacy path applies to Mx2/El/Q2 separately).
    eps = 1e-300
    joint_prior = {
        obs: (np.maximum(t_grid, eps) ** alpha[obs]) * (np.maximum(1.0 - t_grid, eps) ** beta[obs])
        for obs in ("El", "Q2")
    }
    joint_support = {"El": (El_lo_nom, El_hi_nom), "Q2": (Q2_lo_nom, Q2_hi_nom)}

    def _run_toy_loop(run_label, active, rng, n_toys_local):
        """Run n_toys_local toys for one systematics run-label; returns the raw accumulators
        (toy_f, haus_pass, conv_pass, toy_gap_curves, br_nonpos) -- used directly by the local
        (in-process) mode, and by --toy-job to produce one chunk's worth of toys."""
        # A "*_soft" run-label reuses its base label's active systematics list (see runs["all_soft"]
        # above) but relaxes the per-threshold MaxEnt dual instead of requiring exact moment
        # equality: the Hausdorff pre-check no longer skips the fit outright (it's still recorded
        # as a diagnostic), and maxent_pdf gets sigma_rel > 0. Reuses joint_maxent's sigma_rel as
        # the slack scale for consistency; the moment-match acceptance tolerance widens to match
        # (an exact match is no longer the point).
        is_soft = run_label.endswith("_soft")
        soft_sigma_rel = joint_sigma_rel if is_soft else 0.0
        soft_mom_tol = 10.0 * joint_sigma_rel if is_soft else 5e-4
        toy_f     = {obs: {n: [] for n in n_moments} for obs in OBS}
        haus_pass = {obs: {n: [] for n in n_moments} for obs in OBS}
        conv_pass = {obs: {n: [] for n in n_moments} for obs in OBS}
        lam_warm  = {obs: {n: None for n in n_moments} for obs in OBS}
        toy_gap_curves = {k: [] for k in SEM_KEYS}
        toy_moments = {obs: [] for obs in OBS}  # raw residual moments mu_0..mu_{MAX_MOM-1}, per toy
        toy_f_joint  = {obs: [] for obs in ("El", "Q2")}  # joint-fit curve per toy (NaN if not converged)
        joint_conv   = {obs: [] for obs in ("El", "Q2")}
        br_nonpos = 0

        # wi_eff/we/Br/rng below are set fresh each toy iteration in the loop; these two
        # helpers close over them (Python resolves free variables at call time, so this is
        # safe) to avoid repeating the same per-observable logic for Mx2/El/Q2 three times.
        def _compute_residual_moments(obs, pows, lo_t, span_t):
            """Residual moments for one observable, from precomputed power arrays (faster than
            calling residual_mu01, which recomputes x**k every toy) -- with the optional
            incl_moments perturbation of the inclusive-side raw moment sums."""
            S_incl = np.array([(wi_eff * pows[k]).sum() for k in range(MAX_MOM)])
            S_excl = np.array([(we * pows[k]).sum()     for k in range(MAX_MOM)])
            if "incl_moments" in active:
                # Perturb the INCLUSIVE-side raw moment (pre-subtraction), matching how the
                # experimental covariance enters the data pipeline (8_hausdorff_data.py) --
                # perturbing the already-subtracted residual directly would apply an
                # inclusive-sized relative uncertainty to a much smaller quantity and wildly
                # overstate the effect (see SYSTEMATICS_NOTES.md).
                N_incl = wi_eff.sum()
                keys3, corr3 = incl_corr_by_obs[obs]
                m_incl3 = {k: S_incl[idx3 + 1] / N_incl for idx3, k in enumerate(keys3)}
                pert3 = syst.sample_incl_moment_perturbation(m_incl3, keys3, incl_rel_unc, corr3, rng)
                for idx3, k in enumerate(keys3):
                    S_incl[idx3 + 1] = pert3[k] * N_incl
            raw = (S_incl - S_excl) / Br
            mu_t = np.zeros(MAX_MOM)
            for k in range(MAX_MOM):
                mu_t[k] = sum(comb(k, j) * (-lo_t) ** (k - j) * raw[j] for j in range(k + 1)) / span_t ** k
            toy_moments[obs].append(mu_t.copy())
            return mu_t

        def _fit_and_record(obs, mu_t, lo_t, span_t, common_grid):
            """Hausdorff check + MaxEnt fit for one observable, at every requested n_moments,
            recording pass/converged flags and the fitted curve interpolated onto common_grid."""
            for n in n_moments:
                ok_h, _ = hausdorff_moment_check(mu_t[:n])
                haus_pass[obs][n].append(bool(ok_h))
                if not ok_h and not is_soft:
                    lam_warm[obs][n] = None
                    toy_f[obs][n].append(np.full(grid_size, np.nan))
                    conv_pass[obs][n].append(False)
                    continue
                try:
                    f_t, lam, meta = maxent_pdf(mu_t[:n], t_grid, lam0=lam_warm[obs][n],
                                                 return_meta=True, tp_full=tp_grid,
                                                 opt_options=toy_opt_options,
                                                 alpha=alpha[obs], beta=beta[obs],
                                                 sigma_rel=soft_sigma_rel)
                    if not meta["opt_success"] or not maxent_solution_ok(
                            f_t, t_grid, mu_t[:n], mom_tol=soft_mom_tol):
                        raise RuntimeError
                    lam_warm[obs][n] = lam
                    grid_t = lo_t + t_grid * span_t
                    f_common = np.interp(common_grid, grid_t, f_t / span_t, left=0., right=0.)
                    toy_f[obs][n].append(f_common)
                    conv_pass[obs][n].append(True)
                except Exception:
                    lam_warm[obs][n] = None
                    toy_f[obs][n].append(np.full(grid_size, np.nan))
                    conv_pass[obs][n].append(False)

        t0 = time.monotonic()
        for i in tqdm(range(n_toys_local), desc=run_label):
            # Base weights: two SEPARATE copies, since in the real analysis the inclusive
            # measurement and the SEM (sum-of-exclusive) template are independent -- unlike
            # this toy's convenience construction (pseudo-inclusive = SEM cocktail + gap),
            # a real SEM-modeling uncertainty (FF, per-mode BF) has no reason to cancel
            # against the inclusive side, and shouldn't in this toy either. So:
            #   - ff / bf_mode (SEM template modeling uncertainty): perturb w_excl_base only.
            #   - incl_moments (independent experimental measurement uncertainty): perturbs
            #     only the inclusive-side raw moment sums directly (see below), not weights.
            #   - bf_gap (assumed gap-mode normalization): only ever affects gap rows, which
            #     `we` always zeroes out anyway -- naturally inclusive-side-only already.
            w_incl_base = w_base.copy()
            w_excl_base = w_base.copy()
            if "ff" in active:
                w_excl_base[cocktail_idx] *= syst.sample_ff_multiplier_from_matrix(ff_slope_matrix, rng)
            if "bf_mode" in active:
                w_excl_base[cocktail_idx] *= syst.sample_bf_multiplier_from_codes(
                    bf_mode_codes, bf_rel_unc_per_mode, rng)
            if "bf_gap" in active:
                bf_gap_toy = syst.sample_bf_gap(bf_gap_central, sigma_bf_gap, rng)
                gap_scale = bf_gap_toy / bf_gap_central if bf_gap_central != 0 else 1.0
                w_incl_base[is_gap] *= gap_scale

            # 1) Inclusive toy: always-on stat bootstrap
            idx_i    = rng.integers(0, N_ev, size=N_ev)
            counts_i = np.bincount(idx_i, minlength=N_ev).astype(float)
            wi       = w_incl_base * counts_i
            wi_eff   = np.where((wi > 0) & np.isfinite(wi), wi, 0.)

            # 2) Exclusive (SEM) toy: independent stat bootstrap
            idx_e    = rng.integers(0, N_ev, size=N_ev)
            counts_e = np.bincount(idx_e, minlength=N_ev).astype(float)
            we_all   = w_excl_base * counts_e
            we_all   = np.where((we_all > 0) & np.isfinite(we_all), we_all, 0.)
            we       = np.where(~is_gap, we_all, 0.)

            # SEM-style gap curves (incl - excl) for this toy
            mi_toy = compute_curves_from_context(curve_ctx, wi_eff, el_plot_cuts, q2_plot_cuts)
            me_toy = compute_curves_from_context(curve_ctx, we,     el_plot_cuts, q2_plot_cuts)
            for k in SEM_KEYS:
                toy_gap_curves[k].append(mi_toy[k] - me_toy[k])

            # Joint multi-threshold MaxEnt (opt-in): one fit per obs using every Hausdorff-
            # passing candidate threshold as a simultaneous constraint, instead of the legacy
            # per-n-moments single-threshold fits below. Uses the SAME toy wi_eff/we this toy's
            # legacy path uses, so it's directly comparable toy-for-toy.
            if joint_enable:
                for obs, x_arr in (("El", el), ("Q2", q2)):
                    xlo_j, xhi_j = joint_support[obs]
                    f_j, ok_j, _err_j, _passing_j = jm.fit_joint_events(
                        x_arr, wi_eff, we, xlo_j, xhi_j, joint_cuts[obs], joint_n[obs],
                        t_grid, joint_prior[obs], sigma_rel=joint_sigma_rel, gammas=joint_gammas)
                    toy_f_joint[obs].append(f_j if ok_j else np.full(grid_size, np.nan))
                    joint_conv[obs].append(bool(ok_j))

            Br = wi_eff.sum() - we.sum()
            if Br <= 0:
                br_nonpos += 1
                for obs in OBS:
                    for n in n_moments:
                        haus_pass[obs][n].append(False)
                        conv_pass[obs][n].append(False)
                        toy_f[obs][n].append(np.full(grid_size, np.nan))
                    toy_moments[obs].append(np.full(MAX_MOM, np.nan))
                continue

            # 3) Mx² residual moments with (always-on) support variation
            Mx2_lo_t = np.clip(Mx2_lo_nom + rng.normal(0., bounds_sig * 2 * np.sqrt(Mx2_lo_nom)),
                               MX2_LO_MIN, Mx2_hi_nom - 1.0)
            Mx2_hi_t = np.clip(Mx2_hi_nom + rng.normal(0., bounds_sig * 2 * np.sqrt(Mx2_hi_nom)),
                               Mx2_lo_t + 1.0, MX2_HI_MAX)
            span_t   = Mx2_hi_t - Mx2_lo_t
            mu_Mx2_t = _compute_residual_moments("Mx", mx2_pows, Mx2_lo_t, span_t)
            _fit_and_record("Mx", mu_Mx2_t, Mx2_lo_t, span_t, Mx2_grid)

            # 4) El residual moments with (always-on) support variation
            El_lo_t = np.clip(El_lo_nom + rng.normal(0., bounds_sig), EL_LO_MIN, El_hi_nom - 0.3)
            El_hi_t = np.clip(El_hi_nom + rng.normal(0., bounds_sig), El_lo_t + 0.3, EL_HI_MAX)
            span_El_t = El_hi_t - El_lo_t
            mu_El_t = _compute_residual_moments("El", el_pows, El_lo_t, span_El_t)
            _fit_and_record("El", mu_El_t, El_lo_t, span_El_t, El_grid)

            # 5) q² residual moments with (always-on) support variation (sigma scaled to q² range)
            q2_bsig = bounds_sig * (span_Q2_nom / span_El_nom)  # scale relative to El span
            Q2_lo_t = np.clip(Q2_lo_nom + rng.normal(0., q2_bsig), Q2_LO_MIN, Q2_hi_nom - 1.0)
            Q2_hi_t = np.clip(Q2_hi_nom + rng.normal(0., q2_bsig), Q2_lo_t + 1.0, Q2_HI_MAX)
            span_Q2_t = Q2_hi_t - Q2_lo_t
            mu_Q2_t = _compute_residual_moments("Q2", q2_pows, Q2_lo_t, span_Q2_t)
            _fit_and_record("Q2", mu_Q2_t, Q2_lo_t, span_Q2_t, Q2_grid)

            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / (i + 1) * (n_toys_local - i - 1)
                print(f"    [{run_label}] toy {i+1}/{n_toys_local}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

        elapsed = time.monotonic() - t0
        print(f"  [{run_label}] {n_toys_local} toys done in {elapsed:.1f}s")
        return (toy_f, haus_pass, conv_pass, toy_gap_curves, toy_moments, br_nonpos,
                toy_f_joint, joint_conv)

    def _write_summary_and_figures(run_label, active, toy_f, haus_pass, conv_pass,
                                    toy_gap_curves, toy_moments, br_nonpos,
                                    toy_f_joint=None, joint_conv=None):
        """Write systematics_summary.json + the 3 diagnostic figure sets for one run-label,
        from already-computed (possibly chunk-merged) toy accumulators."""
        run_out_dir = out_dir / gap_mode / "systematics" / run_label
        run_fig_dir = fig_dir / gap_mode / "systematics" / run_label
        run_out_dir.mkdir(parents=True, exist_ok=True)
        run_fig_dir.mkdir(parents=True, exist_ok=True)
        n_toys = len(haus_pass[OBS[0]][n_moments[0]])  # actual toy count (may differ from
                                                         # config if chunk-merged / a chunk failed)

        # Includes bias/impact diagnostics on the reconstructed PDF itself (not just raw
        # moments), for the "check if something looks off" sanity pass: rel_68width_to_peak
        # (band width normalized to the nominal peak height, comparable across observables),
        # and fractional L1 distance of the toy-median curve to both the truth histogram and
        # the no-smearing nominal MaxEnt reconstruction (bias check).
        obs_cfg_by_name = {c[0]: c for c in obs_cfg}
        summary = {"gap_mode": gap_mode, "run_label": run_label, "active_systematics": active,
                   "n_toys": n_toys, "br_nonpos": br_nonpos, "by_obs": {}}
        recon_cache = {}  # (obs, n) -> dict of arrays, for the combined 3x3 grid figure below
        for obs in OBS:
            _, xgrid_o, bct_o, h_true_o, db_o, xlo_o, xhi_o, mu_nom_o, _, _ = obs_cfg_by_name[obs]
            span_obs_o = xhi_o - xlo_o
            summary["by_obs"][obs] = {}
            for n in n_moments:
                hp = np.array(haus_pass[obs][n], dtype=bool)
                cp = np.array(conv_pass[obs][n], dtype=bool)
                haus_frac = float(hp.mean()) if len(hp) > 0 else 0.
                conv_frac = float(cp.mean()) if len(cp) > 0 else 0.
                arr = np.array(toy_f[obs][n])
                ok_ = np.all(np.isfinite(arr), axis=1) if arr.size else np.zeros(0, dtype=bool)
                arr_ok = arr[ok_]
                band68 = rel_band68 = bias_vs_truth = bias_vs_nominal = mean_rel68_core = None
                band_area_68_pct = None
                if arr_ok.shape[0] >= 4:
                    lo1, hi1, med = np.percentile(arr_ok, [16, 84, 50], axis=0)
                    band68 = float(np.median(hi1 - lo1))
                    f_nom_n, _ = maxent_pdf(mu_nom_o[:n], t_grid, tp_full=tp_grid,
                                             alpha=alpha[obs], beta=beta[obs])
                    f_nom_phys = f_nom_n / span_obs_o
                    h_true_interp = np.interp(xgrid_o, bct_o, h_true_o)
                    peak = max(float(np.max(f_nom_phys)), 1e-300)
                    rel_band68 = band68 / peak
                    truth_norm = max(float(np.trapz(h_true_interp, xgrid_o)), 1e-300)
                    nom_norm = max(float(np.trapz(f_nom_phys, xgrid_o)), 1e-300)
                    bias_vs_truth = float(np.trapz(np.abs(med - h_true_interp), xgrid_o)) / truth_norm
                    bias_vs_nominal = float(np.trapz(np.abs(med - f_nom_phys), xgrid_o)) / nom_norm
                    # Mean, over converged toys, of the per-bin 68% band width as a percentage
                    # of the nominal (no-smearing) curve at that bin -- restricted to a "core"
                    # region (>=5% of peak) so near-zero tail bins don't blow up the ratio.
                    core = f_nom_phys >= 0.05 * peak
                    if core.any():
                        rel_width_per_bin = (hi1[core] - lo1[core]) / f_nom_phys[core]
                        mean_rel68_core = float(np.mean(rel_width_per_bin)) * 100.0
                    # Systematics-table metric: integrated 68% band area (hi1-lo1 integrated
                    # over the full support), as a % of the total probability mass. Since
                    # f_nom_phys is a normalized density (integrates to ~1), this is a single,
                    # unit-consistent number directly comparable across observables and across
                    # systematics -- unlike a per-bin ratio, it isn't sensitive to a "core
                    # region" cutoff choice, and it's invariant under the Mx<->Mx² change of
                    # variables (same integral either way), so it's computed once here, before
                    # the optional Mx Jacobian transform below.
                    band_area_68_pct = float(np.trapz(hi1 - lo1, xgrid_o)) * 100.0
                    lo2, hi2 = np.percentile(arr_ok, [2.5, 97.5], axis=0)
                    if obs == "Mx":
                        # Translate Mx² -> Mx via the standard Jacobian f(Mx) = f(Mx²)*2*Mx,
                        # using the Mx-domain truth histogram built directly from MC truth
                        # (not by transforming the Mx² histogram).
                        xgrid_mx = np.sqrt(xgrid_o)
                        jac = 2.0 * xgrid_mx
                        recon_cache[(obs, n)] = dict(
                            xgrid=xgrid_mx, f_nom_phys=f_nom_phys * jac, h_true=h_Mx,
                            bct=bct_Mx, db=db_Mx, xlo=Mx_lo_nom, xhi=Mx_hi_nom,
                            lo1=lo1 * jac, hi1=hi1 * jac, lo2=lo2 * jac, hi2=hi2 * jac,
                            med=med * jac, n_conv=int(cp.sum()))
                    else:
                        recon_cache[(obs, n)] = dict(
                            xgrid=xgrid_o, f_nom_phys=f_nom_phys, h_true=h_true_o,
                            bct=bct_o, db=db_o, xlo=xlo_o, xhi=xhi_o,
                            lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2, med=med, n_conv=int(cp.sum()))
                summary["by_obs"][obs][str(n)] = {
                    "hausdorff_pass_frac": haus_frac,
                    "convergence_pass_frac": conv_frac,
                    "n_converged": int(cp.sum()),
                    "median_68width": band68,
                    "rel_68width_to_peak": rel_band68,
                    "bias_L1_vs_truth_frac": bias_vs_truth,
                    "bias_L1_vs_nominal_frac": bias_vs_nominal,
                    "mean_rel68_core_pct": mean_rel68_core,
                    "band_area_68_pct": band_area_68_pct,
                }
                print(f"  [{run_label}] {obs} n={n}: Hausdorff {haus_frac*100:.1f}%  "
                      f"converged {cp.sum()}/{n_toys}  68%w={band68}  rel68w={rel_band68}  "
                      f"bias_vs_truth={bias_vs_truth}  bias_vs_nominal={bias_vs_nominal}  "
                      f"mean_rel68_core_pct={mean_rel68_core}  band_area_68_pct={band_area_68_pct}")

        with open(run_out_dir / "systematics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → {run_out_dir / 'systematics_summary.json'}")

        # ── Figure 0: combined grid (rows=Mx/El/Q2, cols=n=2,3,4,5) ────────────
        N_GRID = [n for n in (2, 3, 4, 5) if n in n_moments]
        GRID_XLABEL = {"Mx": r"$M_X\ [\mathrm{GeV}]$", "El": r"$E_\ell\ [\mathrm{GeV}]$",
                       "Q2": r"$q^2\ [\mathrm{GeV}^2]$"}
        GRID_YLABEL = {"Mx": r"$f(M_X)\ [\mathrm{GeV}^{-1}]$", "El": r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$",
                       "Q2": r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"}
        if N_GRID:
            fig0, axes0 = plt.subplots(len(OBS), len(N_GRID),
                                        figsize=(4.6 * len(N_GRID), 3.8 * len(OBS)),
                                        dpi=150, sharex="row")
            if len(OBS) == 1:
                axes0 = axes0[np.newaxis, :]
            if len(N_GRID) == 1:
                axes0 = axes0[:, np.newaxis]
            for row, obs in enumerate(OBS):
                for col, n in enumerate(N_GRID):
                    ax = axes0[row, col]
                    c = recon_cache.get((obs, n))
                    if c is None:
                        ax.text(0.5, 0.5, "no converged toys", ha="center", va="center",
                                 transform=ax.transAxes, fontsize=9, color="crimson")
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    ax.bar(c["bct"], c["h_true"], width=c["db"], color=TRUTH_COLOR, alpha=0.45,
                            label=f"True {gap_mode}" if (row == 0 and col == 0) else "_")
                    ax.fill_between(c["xgrid"], c["lo2"], c["hi2"], color=RECON_COLOR, alpha=0.15,
                                     label="95% band" if (row == 0 and col == 0) else "_")
                    ax.fill_between(c["xgrid"], c["lo1"], c["hi1"], color=RECON_COLOR, alpha=0.30,
                                     label="68% band" if (row == 0 and col == 0) else "_")
                    ax.plot(c["xgrid"], c["f_nom_phys"], color=RECON_COLOR, lw=2.0,
                             label="Nominal MaxEnt" if (row == 0 and col == 0) else "_")
                    ax.plot(c["xgrid"], c["med"], color=RECON_COLOR, lw=1.5, linestyle="--",
                             label="Toy median" if (row == 0 and col == 0) else "_")
                    # Display range: 0.5-99.5th percentile of the true gap-mode distribution
                    # (+5% margin), not the full physical support -- the support can have a
                    # long near-empty tail (e.g. Mx) that isn't useful to actually look at.
                    disp_lo, disp_hi = DISP_RANGE[obs]
                    if obs == "Mx":
                        disp_hi = min(disp_hi, 3.0)
                    ax.set_xlim(max(c["xlo"], disp_lo), min(c["xhi"], disp_hi))
                    # ylim from the truth histogram, not the toy bands -- a poorly-converged
                    # 68%/95% band can otherwise blow up the scale and hide everything else.
                    ax.set_ylim(0, 1.5 * max(float(np.max(c["h_true"])), 1e-300))
                    ax.grid(alpha=0.25)
                    ax.tick_params(labelsize=8)
                    ax.set_xlabel(GRID_XLABEL[obs], fontsize=9)
                    if col == 0:
                        ax.set_ylabel(GRID_YLABEL[obs], fontsize=9)
                    ax.set_title(f"{obs}, n={n}  ({c['n_conv']}/{n_toys} conv.)", fontsize=9)
            h0, l0 = axes0[0, 0].get_legend_handles_labels()
            fig0.legend(h0, l0, loc="upper center", ncol=4, frameon=False,
                        bbox_to_anchor=(0.5, 1.0), fontsize=10)
            fig0.suptitle(
                f"Gap reconstruction — {gap_mode}  run: {run_label}  ({n_toys} toys)\n"
                f"systematics: {active or 'none (stat + support only)'}",
                fontsize=11, y=1.05)
            fig0.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
            fig0.savefig(run_fig_dir / "reconstruction_grid_3x3.png", bbox_inches="tight")
            plt.close(fig0)
            print(f"  → {run_fig_dir / 'reconstruction_grid_3x3.png'}")

        # ── Figure 0b: residual-moment distributions (incl - SEM, per toy) ───────
        MOM_LABEL = {"Mx": r"M_X^2", "El": r"E_\ell", "Q2": r"q^2"}
        n_mom_plot = MAX_MOM - 1  # skip mu_0, which is trivially 1
        fig0b, axes0b = plt.subplots(len(OBS), n_mom_plot,
                                      figsize=(3.2 * n_mom_plot, 3.2 * len(OBS)),
                                      dpi=150)
        if len(OBS) == 1:
            axes0b = axes0b[np.newaxis, :]
        if n_mom_plot == 1:
            axes0b = axes0b[:, np.newaxis]
        for row, obs in enumerate(OBS):
            arr = np.array(toy_moments[obs])  # (n_toys, MAX_MOM)
            ok_m = np.all(np.isfinite(arr), axis=1) if arr.size else np.zeros(0, dtype=bool)
            arr_ok = arr[ok_m]
            for col in range(n_mom_plot):
                k = col + 1
                ax = axes0b[row, col]
                if arr_ok.shape[0] > 0:
                    vals = arr_ok[:, k]
                    ax.hist(vals, bins=40, color=RECON_COLOR, alpha=0.7)
                    ax.axvline(float(np.median(vals)), color="k", lw=1.0, linestyle="--")
                ax.set_title(rf"$\mu_{k}({MOM_LABEL[obs]})$", fontsize=9)
                ax.tick_params(labelsize=7)
                if col == 0:
                    ax.set_ylabel(obs, fontsize=9)
        fig0b.suptitle(
            f"Residual moment distributions (incl − SEM) — {gap_mode}  run: {run_label}  "
            f"({n_toys} toys)",
            fontsize=11, y=1.02)
        fig0b.tight_layout()
        fig0b.savefig(run_fig_dir / "residual_moment_distributions.png", bbox_inches="tight")
        plt.close(fig0b)
        print(f"  → {run_fig_dir / 'residual_moment_distributions.png'}")

        # ── Figure 1: Hausdorff pass rate bar chart ──────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, sharey=True)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for ax, obs in zip(axes, OBS):
            fracs  = [np.array(haus_pass[obs][n], dtype=bool).mean() * 100 for n in n_moments]
            labels = [str(n) for n in n_moments]
            bars   = ax.bar(labels, fracs, color=colors[:len(n_moments)], alpha=0.8)
            for bar, frac in zip(bars, fracs):
                ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.5, f"{frac:.1f}%",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title(obs, fontsize=11)
            ax.set_xlabel("n moments", fontsize=10)
            ax.set_ylim(0, 110)
            ax.axhline(100, color="k", lw=0.8, linestyle=":")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Hausdorff pass rate [%]", fontsize=10)
        fig.suptitle(f"Hausdorff pass rate — gap: {gap_mode}  run: {run_label}  ({n_toys} toys)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(run_fig_dir / "hausdorff_passrate.png");  plt.close(fig)

        # ── Figure 2: SEM-style gap moment curves with bands ─────────────────────
        YLABELS = {
            "mx_1": r"$\langle M_X^2 \rangle$", "mx_2": r"$\mu_2(M_X^2)$", "mx_3": r"$\mu_3(M_X^2)$",
            "el_1": r"$\langle E_\ell \rangle$", "el_2": r"$\mu_2(E_\ell)$", "el_3": r"$\mu_3(E_\ell)$",
            "q2_1": r"$\langle q^2 \rangle$", "q2_2": r"$\mu_2(q^2)$", "q2_3": r"$\mu_3(q^2)$",
        }
        plot_grid = [
            (["mx_1","mx_2","mx_3"], el_plot_cuts, r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$",   r"$M_X^2$ gap"),
            (["el_1","el_2","el_3"], el_plot_cuts, r"$E_{\ell,\mathrm{cut}}\ [\mathrm{GeV}]$",   r"$E_\ell$ gap"),
            (["q2_1","q2_2","q2_3"], q2_plot_cuts, r"$q^2_\mathrm{cut}\ [\mathrm{GeV}^2]$",     r"$q^2$ gap"),
        ]
        fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
        for row, (keys, cuts, xlabel, row_title) in enumerate(plot_grid):
            for col, key in enumerate(keys):
                ax = axes2[row, col]
                arr = np.array(toy_gap_curves[key])
                fin = np.all(np.isfinite(arr), axis=1)
                arr = arr[fin]
                nom = gap_nom_i[key] - gap_nom_e[key]
                if arr.shape[0] > 0:
                    lo2 = np.percentile(arr, 2.5,  axis=0)
                    lo1 = np.percentile(arr, 16,   axis=0)
                    hi1 = np.percentile(arr, 84,   axis=0)
                    hi2 = np.percentile(arr, 97.5, axis=0)
                    ax.fill_between(cuts, lo2, hi2, color=RECON_COLOR, alpha=0.15, label="95% band")
                    ax.fill_between(cuts, lo1, hi1, color=RECON_COLOR, alpha=0.30, label="68% band")
                    med = np.percentile(arr, 50, axis=0)
                    ax.plot(cuts, med, color=RECON_COLOR, lw=1.5, linestyle="--", label="Toy median")
                ax.plot(cuts, nom, color=RECON_COLOR, lw=2., label="Nominal gap")
                ax.axhline(0, color="k", lw=0.7, linestyle=":")
                ax.set_ylabel(YLABELS[key], fontsize=8)
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_xlim(float(cuts[0]), float(cuts[-1]))
                ax.grid(alpha=0.25);  ax.tick_params(labelsize=7)
                if col == 0:
                    ax.text(-0.38, 0.5, row_title, transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=9)
        h2, l2 = axes2[0,0].get_legend_handles_labels()
        fig2.legend(h2, l2, loc="upper center", ncol=4, frameon=False,
                    bbox_to_anchor=(0.5, 0.995), fontsize=10)
        fig2.suptitle(
            f"Gap moment curves: {gap_mode}  run: {run_label}  ({n_toys} toys: stat"
            f" + support ±{bounds_sig} GeV always-on; systematics: {active or 'none'})",
            fontsize=10, y=1.01)
        fig2.tight_layout(rect=[0.04, 0.01, 0.99, 0.97])
        fig2.savefig(run_fig_dir / "gap_moment_curves.png");  plt.close(fig2)

        # ── Figure 3+: MaxEnt reconstruction per observable ───────────────────────
        for obs, xgrid, bct, h_true, db, xlo, xhi, mu_nom, xlabel, ylabel in obs_cfg:
            fig3, axes3 = plt.subplots(1, len(n_moments), figsize=(5*len(n_moments), 5.5),
                                       dpi=150, sharey=True)
            if len(n_moments) == 1:
                axes3 = [axes3]
            for ax, n in zip(axes3, n_moments):
                arr = np.array(toy_f[obs][n])
                ok_ = np.all(np.isfinite(arr), axis=1)
                arr = arr[ok_]

                # nominal MaxEnt (no smearing)
                f_nom, _ = maxent_pdf(mu_nom[:n], t_grid, tp_full=tp_grid,
                                      alpha=alpha[obs], beta=beta[obs])
                span_obs  = xhi - xlo
                f_nom_phys = f_nom / span_obs

                ax.bar(bct, h_true, width=db, color=TRUTH_COLOR, alpha=0.55, label=f"True {gap_mode}")
                if arr.shape[0] >= 4:
                    lo2, lo1, med, hi1, hi2 = np.percentile(arr, [2.5, 16, 50, 84, 97.5], axis=0)
                    ax.fill_between(xgrid, lo2, hi2, color=RECON_COLOR, alpha=0.15, label="95% toy band")
                    ax.fill_between(xgrid, lo1, hi1, color=RECON_COLOR, alpha=0.30, label="68% toy band")
                    ax.plot(xgrid, med, color=RECON_COLOR, lw=1.5, linestyle="--", label="Toy median")
                ax.plot(xgrid, f_nom_phys, color=RECON_COLOR, lw=2.2, label="Nominal MaxEnt")
                ax.set_title(f"{n} moments  ({ok_.sum()}/{n_toys} converged)", fontsize=10)
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_xlim(xlo, xhi);  ax.set_ylim(bottom=0);  ax.grid(alpha=0.25)
                ax.legend(fontsize=7.5, loc="upper right")
            axes3[0].set_ylabel(ylabel, fontsize=11)
            fig3.suptitle(
                f"MaxEnt gap reconstruction in {obs}  run: {run_label}  ({n_toys} toys: stat"
                f" + support ±{bounds_sig:.2f} always-on; systematics: {active or 'none'})\n"
                f"Gap mode: {gap_mode}",
                fontsize=11)
            fig3.tight_layout()
            out_path = run_fig_dir / f"maxent_reconstruction_{obs}.png"
            fig3.savefig(out_path);  plt.close(fig3)

        # ── Figure 4 (opt-in): joint multi-threshold MaxEnt, replacing the legacy
        # per-n-moments single-threshold fits above with one fit per toy using every
        # Hausdorff-passing threshold simultaneously (lib/joint_maxent.py) ──────────
        # Guard against merging in chunks from before this feature existed (no jointf__*/
        # jointok__* keys saved) -- those leave toy_f_joint[obs] empty for this run-label even
        # with joint_enable on, which isn't the same as "ran but nothing converged".
        have_joint_data = joint_enable and toy_f_joint is not None and all(
            len(toy_f_joint[obs]) == n_toys for obs in ("El", "Q2"))
        if joint_enable and not have_joint_data:
            print(f"  [{run_label}] joint_maxent enabled but chunk data missing/incomplete "
                  f"(mixed old/new chunks?) -- skipping joint figure for this run-label")
        if have_joint_data:
            joint_summary = {}
            fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5.5), dpi=150)
            for ax, obs in zip(axes4, ("El", "Q2")):
                _, xgrid_o, bct_o, h_true_o, db_o, xlo_o, xhi_o, _, xlabel_o, ylabel_o = next(
                    c for c in obs_cfg if c[0] == obs)
                span_obs = xhi_o - xlo_o
                arr = np.array(toy_f_joint[obs])
                ok_ = np.array(joint_conv[obs], dtype=bool) & np.all(np.isfinite(arr), axis=1)
                arr_ok = arr[ok_]

                f_nom_j, ok_nom_j, err_nom_j, passing_nom_j = jm.fit_joint_events(
                    el if obs == "El" else q2, w_incl, w_excl, xlo_o, xhi_o,
                    joint_cuts[obs], joint_n[obs], t_grid, joint_prior[obs],
                    sigma_rel=joint_sigma_rel, gammas=joint_gammas)

                ax.bar(bct_o, h_true_o, width=db_o, color=TRUTH_COLOR, alpha=0.55,
                       label=f"True {gap_mode}")
                if arr_ok.shape[0] >= 4:
                    lo2, lo1, med, hi1, hi2 = np.percentile(arr_ok, [2.5, 16, 50, 84, 97.5], axis=0)
                    ax.fill_between(xgrid_o, lo2/span_obs, hi2/span_obs, color=RECON_COLOR,
                                     alpha=0.15, label="95% toy band")
                    ax.fill_between(xgrid_o, lo1/span_obs, hi1/span_obs, color=RECON_COLOR,
                                     alpha=0.30, label="68% toy band")
                    ax.plot(xgrid_o, med/span_obs, color=RECON_COLOR, lw=1.5, ls="--",
                            label="Toy median")
                if ok_nom_j:
                    ax.plot(xgrid_o, f_nom_j/span_obs, color=RECON_COLOR, lw=2.2,
                            label="Nominal joint MaxEnt")
                ax.set_title(f"{obs}  ({int(ok_.sum())}/{n_toys} converged, "
                             f"nominal: {len(passing_nom_j)} thr pass)", fontsize=10)
                ax.set_xlabel(xlabel_o, fontsize=11); ax.set_ylabel(ylabel_o, fontsize=11)
                ax.set_xlim(xlo_o, xhi_o); ax.set_ylim(bottom=0); ax.grid(alpha=0.25)
                ax.legend(fontsize=8, loc="upper right")

                joint_summary[obs] = {
                    "n_moments": joint_n[obs], "n_converged": int(ok_.sum()), "n_toys": n_toys,
                    "nominal_converged": bool(ok_nom_j), "nominal_thresholds_passing": passing_nom_j,
                }
            fig4.suptitle(
                f"Joint multi-threshold MaxEnt (opt-in)  run: {run_label}  ({n_toys} toys; "
                f"systematics: {active or 'none'})\nGap mode: {gap_mode}", fontsize=11)
            fig4.tight_layout()
            fig4.savefig(run_fig_dir / "joint_maxent_reconstruction.png"); plt.close(fig4)
            with open(run_out_dir / "joint_maxent_summary.json", "w") as f:
                json.dump(joint_summary, f, indent=2)
            print(f"  → {run_fig_dir / 'joint_maxent_reconstruction.png'}")

        print(f"  → figures in {run_fig_dir}")

    def _write_decomposition():
        """Quadrature variance decomposition of the systematic-uncertainty budget, cross-
        referencing all run-label systematics_summary.json files written by
        _write_summary_and_figures. Requires base + every solo_<systematic> label (and, if
        present, "all") to already have been written -- run once after all labels in this
        invocation are done, in both --merge and the default local-mode loop.

        Each solo_X run = base (stat+support) + one systematic X. Treating band_area_68_pct
        as a sigma-like quantity, the marginal variance contributed by X alone is
            Var_X = band_area_68_pct[solo_X]^2 - band_area_68_pct[base]^2   (clipped >= 0)
        Fractional contribution of X to the systematic (non-stat) budget:
            frac_X = Var_X / sum_j(Var_j)
        Consistency check: does the combined "all" run's variance match the quadrature-sum
        prediction Var[base] + sum_X Var_X? If not, the sources don't combine independently.
        """
        sys_out_dir = out_dir / gap_mode / "systematics"
        sys_fig_dir = fig_dir / gap_mode / "systematics"
        solo_labels = [f"solo_{name}" for name in syst.SYSTEMATIC_NAMES if f"solo_{name}" in runs]
        needed = ["base"] + solo_labels
        summaries = {}
        for label in needed + (["all"] if "all" in runs else []):
            p = sys_out_dir / label / "systematics_summary.json"
            if p.exists():
                summaries[label] = json.loads(p.read_text())
        missing = [l for l in needed if l not in summaries]
        if missing:
            print(f"\nSkipping variance decomposition: missing summaries for {missing} "
                  f"(run --merge/local mode with all systematics enabled first)")
            return
        solo_names = {f"solo_{n}": n for n in syst.SYSTEMATIC_NAMES}

        def get(label, obs, n, key):
            return summaries[label]["by_obs"][obs][str(n)].get(key)

        N_GRID_D = [n for n in (2, 3, 4) if n in n_moments]
        rows = []
        for obs in OBS:
            for n in N_GRID_D:
                base_band = get("base", obs, n, "band_area_68_pct")
                if base_band is None:
                    continue
                var_base = base_band ** 2
                var_x = {}
                for label in solo_labels:
                    band = get(label, obs, n, "band_area_68_pct")
                    var_x[label] = None if band is None else max(band ** 2 - var_base, 0.0)
                valid_vars = {k: v for k, v in var_x.items() if v is not None}
                var_sum = sum(valid_vars.values())
                predicted_band = float(np.sqrt(var_base + var_sum))
                actual_band = get("all", obs, n, "band_area_68_pct") if "all" in summaries else None
                rows.append(dict(
                    obs=obs, n=n, base_band_pct=base_band, base_conv=get("base", obs, n, "n_converged"),
                    solo_band_pct={l: get(l, obs, n, "band_area_68_pct") for l in solo_labels},
                    solo_conv={l: get(l, obs, n, "n_converged") for l in solo_labels},
                    var_x=var_x, frac_x={l: (100 * v / var_sum if v is not None and var_sum > 0 else None)
                                          for l, v in var_x.items()},
                    predicted_band_pct=predicted_band, actual_band_pct=actual_band,
                    actual_conv=get("all", obs, n, "n_converged") if "all" in summaries else None,
                    consistency_ratio=(actual_band / predicted_band
                                        if actual_band is not None and predicted_band > 0 else None),
                ))

        sys_out_dir.mkdir(parents=True, exist_ok=True)
        sys_fig_dir.mkdir(parents=True, exist_ok=True)
        with open(sys_out_dir / "decomposition.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  → {sys_out_dir / 'decomposition.json'}")

        from plothist import get_color_palette
        _pal = [mcolors.to_hex(c) for c in get_color_palette("cubehelix", len(solo_labels))]
        colors = dict(zip(solo_labels, _pal))
        fig, axes = plt.subplots(1, len(OBS), figsize=(5 * len(OBS), 5), dpi=150, sharey=True)
        if len(OBS) == 1:
            axes = [axes]
        for ax, obs in zip(axes, OBS):
            obs_rows = [r for r in rows if r["obs"] == obs]
            ns = [r["n"] for r in obs_rows]
            bottoms = np.zeros(len(obs_rows))
            for label in solo_labels:
                vals = np.array([r["frac_x"][label] if r["frac_x"][label] is not None else 0.0
                                  for r in obs_rows])
                ax.bar([str(n) for n in ns], vals, bottom=bottoms,
                       color=colors.get(label, "gray"),
                       label=solo_names[label] if obs == OBS[0] else "_")
                bottoms += vals
            for i, r in enumerate(obs_rows):
                if r["consistency_ratio"] is not None:
                    ax.text(i, 103, f"{r['consistency_ratio']:.1f}x", ha="center", fontsize=8)
                else:
                    ax.text(i, 103, "n/a", ha="center", fontsize=8, color="crimson")
            ax.set_ylim(0, 115)
            ax.set_xlabel("n moments", fontsize=10)
            ax.set_title(obs, fontsize=12)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("fraction of systematic (non-stat) variance [%]", fontsize=10)
        fig.legend(loc="upper center", ncol=len(solo_labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.08), fontsize=9)
        fig.suptitle(f"Systematic variance budget decomposition ({gap_mode}, {n_toys} toys)\n"
                     "labels above bars: actual/predicted ratio for the combined 'all' run "
                     "(quadrature-sum consistency check)", fontsize=11, y=1.16)
        fig.tight_layout()
        fig.savefig(sys_fig_dir / "decomposition.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  → {sys_fig_dir / 'decomposition.png'}")

    # ── Mode dispatch: --submit / --toy-job / --merge / local (default) ────────
    run_labels_ordered = list(runs.keys())  # deterministic order, used for reproducible seeding

    if args.submit:
        if n_toys == 0:
            print("n_toys=0: nothing to submit."); return
        queue = cfg["generation"]["queue"]
        here  = Path(__file__).parent.resolve()
        n_chunks = args.n_chunks if args.n_chunks > 1 else int(ht.get("n_toy_chunks", 20))
        config_abs = str(Path(args.config).resolve())
        # bsub only forwards a minimal environment by default; "-env all" forwards the full
        # environment of the submitting shell (must have the Belle II env already sourced,
        # e.g. via the sb2ln alias, before invoking --submit) to the execution host.
        for run_label in run_labels_ordered:
            log_dir = here / "logs" / "7" / gap_mode / run_label
            log_dir.mkdir(parents=True, exist_ok=True)
            # Unlike 8_hausdorff_data.py's --toy-job (which only reads small JSON files), every
            # chunk here loads the full cocktail.parquet (13.8GB on disk with the full-isospin/
            # lepton generalization). Queue "s" has a hard ~4GB-per-slot memory cap that can't be
            # raised via -M directly; -n N requests N slots (~4GB each), trading lower scheduling
            # priority for headroom. Measured actual peak RSS (2026-08-31, post decay_name
            # categorical-dtype fix in lib.systematics.read_parquet_downcast): base/solo_bf_mode/
            # solo_incl_moments (no FF columns) ~14.8GB -> 5 slots; all/solo_ff (+ FF
            # eigenvariation columns) ~20.2GB -> 6 slots. (Note the earlier "-n 4" here, and no
            # slots flag at all for the non-FF labels, is exactly why base/solo_bf_mode/
            # solo_incl_moments failed 500/500 and all/solo_ff failed 380/500 in the first
            # full-isospin submission.)
            label_needs_ff = "ff" in runs.get(run_label, [])
            slots_flag = '-n "6" ' if label_needs_ff else '-n "5" '
            print(f"\n{run_label}: submitting {n_chunks} chunk job(s)"
                  f"{'  (ff: 4 slots/job)' if label_needs_ff else ''}")
            for chunk_idx in range(n_chunks):
                log_file = log_dir / f"chunk_{chunk_idx:04d}.log"
                job_cmd  = (f"cd {here} && python3 7_hausdorff_toy.py "
                            f"--config {config_abs} "
                            f"--toy-job --run-label {run_label} --chunk {chunk_idx} "
                            f"--n-chunks {n_chunks}")
                bsub_cmd = f'bsub -q {queue} -env all {slots_flag}-oo "{log_file}" "{job_cmd}"'
                print(bsub_cmd)
                if not args.dry_run:
                    subprocess.run(bsub_cmd, shell=True, check=True)
        if args.dry_run:
            print("\n(dry run — no jobs submitted)")
        print("\nAll chunk jobs submitted. Once they finish, run with --merge to write "
              "summary/figures.")
        print("Done.")
        return

    if args.toy_job:
        if not args.run_label or args.run_label not in runs:
            print(f"error: --run-label must be one of {run_labels_ordered}")
            sys.exit(1)
        active = runs[args.run_label]
        n_chunks_total  = max(args.n_chunks, 1)
        n_toys_per_chunk = (n_toys + n_chunks_total - 1) // n_chunks_total
        i_start = args.chunk * n_toys_per_chunk
        i_end   = min(i_start + n_toys_per_chunk, n_toys)
        n_local = i_end - i_start
        if n_local <= 0:
            print(f"Chunk {args.chunk}: nothing to do (i_start={i_start} >= {n_toys})")
            print("Done."); return
        print(f"Toy chunk {args.chunk}/{n_chunks_total} for run '{args.run_label}': "
              f"toys {i_start}-{i_end-1} ({n_local} toys)")
        # Deterministic, collision-free seed per (run_label, chunk): run-label position in the
        # (stable-ordered) runs dict * a large stride + chunk index -- NOT hash(run_label),
        # which is randomized per-process by default and would break reproducibility across
        # separately-launched bsub jobs.
        run_idx_seed = run_labels_ordered.index(args.run_label)
        rng = np.random.default_rng(seed + run_idx_seed * 1_000_000 + args.chunk)
        (toy_f, haus_pass, conv_pass, toy_gap_curves, toy_moments, br_nonpos,
         toy_f_joint, joint_conv) = _run_toy_loop(args.run_label, active, rng, n_local)
        chunk_dir = out_dir / gap_mode / "systematics" / args.run_label / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"chunk_{args.chunk:04d}.npz"
        save_kwargs = {"br_nonpos": np.int64(br_nonpos)}
        for obs in OBS:
            for n in n_moments:
                save_kwargs[f"haus__{obs}__{n}"] = np.array(haus_pass[obs][n], dtype=bool)
                save_kwargs[f"conv__{obs}__{n}"] = np.array(conv_pass[obs][n], dtype=bool)
                save_kwargs[f"toyf__{obs}__{n}"] = np.array(toy_f[obs][n], dtype=float)
            save_kwargs[f"mu__{obs}"] = np.array(toy_moments[obs], dtype=float)
        for k in SEM_KEYS:
            save_kwargs[f"gap__{k}"] = np.array(toy_gap_curves[k], dtype=float)
        if joint_enable:
            for obs in ("El", "Q2"):
                save_kwargs[f"jointf__{obs}"] = np.array(toy_f_joint[obs], dtype=float)
                save_kwargs[f"jointok__{obs}"] = np.array(joint_conv[obs], dtype=bool)
        np.savez_compressed(chunk_path, **save_kwargs)
        print(f"  → {chunk_path}")
        print("Done.")
        return

    if args.merge:
        for run_label, active in runs.items():
            chunk_dir = out_dir / gap_mode / "systematics" / run_label / "chunks"
            chunk_paths = sorted(chunk_dir.glob("chunk_*.npz")) if chunk_dir.exists() else []
            # Skip zero-byte chunks left behind by killed/preempted bsub jobs.
            chunk_paths = [cp for cp in chunk_paths if cp.stat().st_size > 0]
            if not chunk_paths:
                print(f"  {run_label}: no chunks found in {chunk_dir}, skip")
                continue
            print(f"\n=== Merging '{run_label}': {len(chunk_paths)} chunk file(s) ===")
            toy_f     = {obs: {n: [] for n in n_moments} for obs in OBS}
            haus_pass = {obs: {n: [] for n in n_moments} for obs in OBS}
            conv_pass = {obs: {n: [] for n in n_moments} for obs in OBS}
            toy_gap_curves = {k: [] for k in SEM_KEYS}
            toy_moments = {obs: [] for obs in OBS}
            toy_f_joint = {obs: [] for obs in ("El", "Q2")}
            joint_conv  = {obs: [] for obs in ("El", "Q2")}
            br_nonpos = 0
            for cp in chunk_paths:
                with np.load(cp) as z:
                    br_nonpos += int(z["br_nonpos"])
                    for obs in OBS:
                        for n in n_moments:
                            haus_pass[obs][n].extend(z[f"haus__{obs}__{n}"].tolist())
                            conv_pass[obs][n].extend(z[f"conv__{obs}__{n}"].tolist())
                            toy_f[obs][n].extend(list(z[f"toyf__{obs}__{n}"]))
                        toy_moments[obs].extend(list(z[f"mu__{obs}"]))
                    for k in SEM_KEYS:
                        toy_gap_curves[k].extend(list(z[f"gap__{k}"]))
                    if joint_enable and f"jointf__El" in z:
                        for obs in ("El", "Q2"):
                            toy_f_joint[obs].extend(list(z[f"jointf__{obs}"]))
                            joint_conv[obs].extend(z[f"jointok__{obs}"].tolist())
            print(f"  merged {len(haus_pass[OBS[0]][n_moments[0]])} toys")
            _write_summary_and_figures(run_label, active, toy_f, haus_pass, conv_pass,
                                        toy_gap_curves, toy_moments, br_nonpos,
                                        toy_f_joint, joint_conv)
        _write_decomposition()
        print("Done.")
        return

    # Default: run everything in-process (fine for small/interactive n_toys; use
    # --submit/--merge for large production runs).
    if n_toys == 0:
        print("n_toys=0: skipping toy loop, only nominal figures produced.")
        return
    for run_idx, run_label in enumerate(run_labels_ordered):
        active = runs[run_label]
        print(f"\n=== Toy run '{run_label}'  (systematics: {active or 'none (stat+support only)'}) ===")
        rng = np.random.default_rng(seed + run_idx)
        (toy_f, haus_pass, conv_pass, toy_gap_curves, toy_moments, br_nonpos,
         toy_f_joint, joint_conv) = _run_toy_loop(run_label, active, rng, n_toys)
        _write_summary_and_figures(run_label, active, toy_f, haus_pass, conv_pass,
                                    toy_gap_curves, toy_moments, br_nonpos,
                                    toy_f_joint, joint_conv)
    _write_decomposition()
    print("Done.")


if __name__ == "__main__":
    main()
