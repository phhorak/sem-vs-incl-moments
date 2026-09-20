#!/usr/bin/env python3
"""
6b_kolya_vs_joint.py

Step 6b: the two "Kolya vs joint chi^2 fit" comparison plots, with ALL uncertainties
(exp measurement + SEM stat + FF + BR-mode systematics, and for Kolya also the HQE
global-fit's own likelihood-toy uncertainty) propagated by TOY REFIT rather than the
delta-method linear propagation 6_joint_fit.py uses for its own (unrelated) figures.

Reuses step 5's already-generated per-toy raw_gap ensembles (output/5/toy_ensemble_*.npz,
output/5/sem0_stat_ff_bfmode.npz -- see 5_residual_covariance.py) and step 6's fit
machinery (imported from 6_joint_fit.py, not duplicated) -- this file only adds:
  (a) a warm-started BFGS-only refit (~10-15x faster than 6_joint_fit.py's cold
      Nelder-Mead+BFGS -- benchmarked at ~0.15-0.25s/toy vs ~2-3s/toy), reusing the
      SAME fixed nominal precision matrix across all toys (standard toy-refit practice:
      only the target moments vary per toy, not the weighting), warm-started from the
      nominal fit's own c_best;
  (b) "Kolya": pairs Markus Prim's HQE-fit likelihood toys (inputs/hqe_likelihood_toys/,
      see its README) with our own SEM stat+FF+BR-mode toy variation at threshold=0
      (output/5/sem0_stat_ff_bfmode.npz -- generated for free alongside the normal
      per-threshold toy loop, see 5_residual_covariance.py's el0_idx/q20_idx), combined
      via the same r_exp/r_sem BF-budget identity used everywhere else in this pipeline,
      then exactly inverted (Hausdorff check + MaxEnt) per pair -- same machinery as the
      per-threshold exact inversion, just with a random reshuffle of the two independent
      toy sources per band.

Produces two 3-panel PDFs in figures/6b/:
  - kolya_vs_joint_gap.pdf: Kolya (SEM-subtracted) vs the joint chi^2 fit, El/Q2 with
    ALL-uncertainty toy bands + chi^2/ndf in the legend, Mx with Kolya vs the per-
    El-threshold inversions (no single joint fit exists for Mx, see 6_joint_fit.py's
    own docstring for why not).
  - kolya_no_sem.pdf: Kolya WITHOUT SEM subtraction (full phase space), band from the
    HQE toys alone (that's already its complete/only uncertainty source).

Usage:
  python3 6b_kolya_vs_joint.py --config config.yaml
  python3 6b_kolya_vs_joint.py --config config.yaml --n-toys 300   # faster/rougher band
"""
import argparse
import importlib.util
import json
import sys
import time
from math import comb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401 (house style, matches 6_joint_fit.py's convention)
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from lib.moments import central_to_raw

# 6_joint_fit.py's filename isn't a valid module name -- import it by path instead of
# duplicating its OBS_CFG/PHYS_BOUNDARY/raw_to_mu01/load_common/load_source_cov logic.
_spec = importlib.util.spec_from_file_location("joint_fit_mod", HERE / "6_joint_fit.py")
jf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jf)

H5_PATH = HERE / "inputs" / "hqe_likelihood_toys" / "likelihood_toys.h5"


def compute_r_exp_r_sem(cfg):
    """jf.load_common() (6_joint_fit.py) does NOT return r_exp/r_sem -- step 5/6's json outputs
    already have them baked in. Recompute the same BF-budget identity ourselves (identical to
    5_residual_covariance.py's load_common) since Kolya needs it directly."""
    from lib.systematics import compute_bf_gap
    sysc = cfg.get("systematics", {})
    bf_incl = float(sysc.get("bf_incl", 0.1105))
    bf_incl_unc = float(sysc.get("bf_incl_unc", 0.0016))
    cocktail_path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    df_bf = pd.read_parquet(cocktail_path, columns=["decay_name", "bf", "bf_unc"])
    _, bf_gap_central, _ = compute_bf_gap(df_bf, bf_incl, bf_incl_unc)
    r_exp = bf_incl / bf_gap_central
    r_sem = (bf_incl - bf_gap_central) / bf_gap_central
    return r_exp, r_sem


# ── Warm-started refit (fast path for the toy loop; 6_joint_fit.py's own fit_maxent is the
# cold Nelder-Mead+BFGS version used once for the nominal fit / c_best / Cinv) ────────────────
def build_fit_context(obs, C, thresholds, M, t_grid_n=400):
    cfg_o = C["OBS_CFG"][obs]
    lo, hi, alpha, beta, masked = cfg_o["lo"], cfg_o["hi"], cfg_o["alpha"], cfg_o["beta"], cfg_o["masked"]
    span = hi - lo
    t_grid = np.linspace(0.0, 1.0, t_grid_n)
    eps = 1e-300
    prior = (np.maximum(t_grid, eps) ** alpha) * (np.maximum(1.0 - t_grid, eps) ** beta)
    mask_by_thr = {thr: (t_grid >= (thr - lo) / span).astype(float) for thr in thresholds} if masked \
        else {thr: np.ones_like(t_grid) for thr in thresholds}
    labels = [(thr, k) for thr in thresholds for k in range(1, 4)]
    tp = np.stack([t_grid ** k for k in range(1, M + 1)])
    return dict(lo=lo, hi=hi, span=span, t_grid=t_grid, prior=prior, mask_by_thr=mask_by_thr,
                labels=labels, tp=tp, M=M)


def model_moments_ctx(ctx, c):
    expo = c @ ctx["tp"]; expo -= expo.max()
    f = ctx["prior"] * np.exp(expo)
    out = np.empty(len(ctx["labels"]))
    for i, (thr, k) in enumerate(ctx["labels"]):
        mask = ctx["mask_by_thr"][thr]
        denom = np.trapz(f * mask, ctx["t_grid"])
        numer = np.trapz((ctx["t_grid"] ** k) * f * mask, ctx["t_grid"])
        out[i] = numer / denom if denom > 0 else np.nan
    return out


def f_phys_of_c_ctx(ctx, c):
    expo = c @ ctx["tp"]; expo -= expo.max()
    f = ctx["prior"] * np.exp(expo)
    Z = np.trapz(f, ctx["t_grid"])
    return (f / Z) / ctx["span"]


def regularized_cinv(cov_raw, lo, hi, n_thr, rel_floor=1e-3):
    """Exactly 6_joint_fit.py's fit_maxent internal weighting. This is NOT simply "eigenvalue-
    floor cov_raw and invert" -- fit_maxent first maps the RAW-moment covariance into MU01-space
    (the space the chi^2 actually operates in) via the raw_to_mu01 Jacobian, block-by-block
    across thresholds, and only THEN eigenvalue-floors and inverts THAT transformed matrix.
    An earlier version of this function skipped the Jacobian transform and floored cov_raw
    directly -- verified this gives a totally different, wrong-unit-space weight matrix
    (refitting the exact nominal Q2 target with that wrong Cinv found chi2~0.0025 at wildly
    different coefficients, vs the true chi2=33.5 fit_maxent actually reports -- the two aren't
    even the same optimization problem). That was the real cause of the toy band not bracketing
    the nominal curve, not the q2=0 domain extrapolation."""
    N = 4  # orders 0..3 (order 0 implicit) -- must match fit_maxent's own N
    J_full = jf.raw_to_mu01_jacobian(lo, hi, N)
    J_sub = J_full[1:, 1:]
    Ct = np.zeros((3 * n_thr, 3 * n_thr))
    for i in range(n_thr):
        for j in range(n_thr):
            Ct[3*i:3*i+3, 3*j:3*j+3] = J_sub @ cov_raw[3*i:3*i+3, 3*j:3*j+3] @ J_sub.T
    w_eig, v_eig = np.linalg.eigh(Ct)
    floor_val = rel_floor * w_eig.max()
    w_reg = np.maximum(w_eig, floor_val)
    return (v_eig / w_reg) @ v_eig.T


def refit_warm(ctx, raw_gap, thresholds, lo, hi, Cinv, c0, l2=1e-4):
    """One toy's raw_gap (flat, order 1..3 per threshold) -> f(t) via warm-started BFGS,
    reusing the FIXED nominal Cinv (standard toy-refit practice: only the target varies).

    Guards against the runaway-coefficient degeneracy 6_joint_fit.py's own fit_maxent already
    documents for Mx's single-threshold fits (l2=1e-4 is "negligible for well-constrained fits
    ... only bites the runaway case") -- BFGS-only from a warm start is more prone to actually
    landing in that degenerate regime than the slower cold Nelder-Mead+BFGS 6_joint_fit.py uses
    for the nominal fit, since it never explores away from c0's basin. Falls back to a fresh
    Nelder-Mead+BFGS cold start (same recipe as the nominal fit) if the warm BFGS result doesn't
    actually close the moments; still MUCH cheaper than doing that for every toy since it only
    triggers on the minority that diverge."""
    mu_cond_t = {thr: jf.raw_to_mu01(np.concatenate([[1.0], raw_gap[3*i:3*i+3]]), lo, hi)[1:]
                 for i, thr in enumerate(thresholds)}
    target = np.concatenate([mu_cond_t[thr] for thr in thresholds])

    def chi2(c):
        r = model_moments_ctx(ctx, c) - target
        return float(r @ Cinv @ r) + l2 * float(np.sum(c ** 2))

    def mom_closure(c):
        return float(np.max(np.abs(model_moments_ctx(ctx, c) - target)))

    res = minimize(chi2, c0, method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    c_final = res.x
    if mom_closure(c_final) > 0.3:  # bad closure -> warm BFGS likely ran into a degenerate basin
        res_nm = minimize(chi2, np.zeros_like(c0), method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True})
        res_bfgs = minimize(chi2, res_nm.x, method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
        c_final = res_bfgs.x if res_bfgs.fun < res_nm.fun else res_nm.x
    ok = mom_closure(c_final) <= 0.3
    return f_phys_of_c_ctx(ctx, c_final), c_final, ok


# ── Kolya (exact inversion) machinery, same as earlier standalone scripts ─────────────────────
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


def maxent_pdf(mu01, t, alpha=0.0, beta=0.0, lam0=None):
    N = len(mu01) - 1
    tp = np.stack([t ** k for k in range(N + 1)])
    eps = 1e-300
    prior = (np.maximum(t, eps) ** alpha) * (np.maximum(1.0 - t, eps) ** beta)

    def _g(lam):
        lf = lam @ tp; lf -= lf.max()
        return prior * np.exp(lf)

    def dual(lam):
        lf0 = lam @ tp
        return np.log(np.trapz(prior * np.exp(lf0 - lf0.max()), t)) + lf0.max() - lam @ mu01

    def grad(lam):
        g = _g(lam); Z = np.trapz(g, t); gn = g / Z
        return np.array([np.trapz(tp[k] * gn, t) - mu01[k] for k in range(N + 1)])

    x0 = np.zeros(N + 1) if lam0 is None else lam0.copy()
    res = minimize(dual, x0, jac=grad, method="L-BFGS-B",
                    options={"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12})
    fn = _g(res.x)
    Z = np.trapz(fn, t)
    fn /= Z
    mom_err = max(abs(np.trapz(tp[k] * fn, t) - mu01[k]) for k in range(N + 1))
    return fn, res.x, bool(res.success), float(mom_err)


# q2 hi: config.yaml's hausdorff_data.q2_support upper bound is 12.5, but the actual cocktail
# MC's own q2 never exceeds ~11.608 (checked directly: matches the physical B->D endpoint
# (M_B-M_D)^2~11.66) -- unlike mx_support's 16.0, which carries a "# from support scan" comment
# showing it was deliberately validated, q2_support has no such annotation and is just ~0.9 GeV^2
# past the true edge. That slack lets the beta=1.5 endpoint suppression anchor at the wrong
# point, leaving room for the reconstructed density to not actually vanish at the true edge.
# Fixed locally here (11.7, a small margin above the measured 11.608) rather than in the shared
# config.yaml, which other pipeline steps also read -- flagged in SESSION_NOTES for a real fix.
Q2_HI = 11.7
SUPPORT = {
    "mx": (3.2, 16.0, r"$M_X^2\ [\mathrm{GeV}^2]$", r"$f(M_X^2)$"),
    "el": (0.0, 2.3,  r"$E_\ell\ [\mathrm{GeV}]$",  r"$f(E_\ell)$"),
    "q2": (0.0, Q2_HI, r"$q^2\ [\mathrm{GeV}^2]$",   r"$f(q^2)$"),
}
BOUNDARY = {"mx": (0.0, 0.0), "el": (2.0, 3.5), "q2": (0.0, 1.5)}  # PHYS_BOUNDARY, gap-region
TITLE = {"mx": r"$M_X$", "el": r"$E_\ell$", "q2": r"$q^2$"}
OBS_KEY = {"el": "El", "q2": "Q2", "mx": "Mx"}


def load_h5_toys():
    with h5py.File(H5_PATH, "r") as f:
        names = [x.decode() for x in f["central/axis0"][:]]
        central = dict(zip(names, f["central/block0_values"][0]))
        toys = {n: f["toys/block0_values"][:, i] for i, n in enumerate(names)}
    return central, toys


def kolya_band(obs, r_exp, r_sem, sem_subtract, rng, t_grid, central, hqe_toys, sem0_toys, n_toys,
                sem_nom0=None):
    """sem_subtract=True: gap density (paired HQE toy x sem0 toy, gap-region boundary exponents).
    sem_subtract=False: full-phase-space density (HQE toy alone, flat alpha=beta=0).
    sem_nom0: the pipeline's actual nominal SEM raw moment at thr=0 for this obs's 3 keys
    (from output/5/sem_moments_raw.json, interpolated to 0.0) -- used for the CENTRAL curve,
    matching the nominal-value convention used everywhere else in this pipeline (the toy
    ensemble's own mean is a fine cross-check but not what "nominal" means elsewhere here)."""
    lo, hi, _, _ = SUPPORT[obs]
    # BOUNDARY's exponents (El: alpha=2 from massless-lepton E_l->0 phase-space counting, beta=3.5;
    # Q2: alpha=0 since rho(q2) is finite&nonzero at q2->0, beta=1.5; Mx: no derived exponent,
    # already (0,0)) are physical properties of the E_l->0 / q2->0 kinematic edge of the
    # UNDERLYING density itself -- they hold for the full inclusive spectrum exactly as much as
    # for the gap-region one (E_l=0 is E_l=0 either way), so they always apply here, not only
    # when sem_subtract=True.
    alpha, beta = BOUNDARY[obs]
    idx9 = {"mx": (0, 1, 2), "el": (3, 4, 5), "q2": (6, 7, 8)}[obs]

    c1 = central[f"{obs}_1_cut0.0"]; c2 = central[f"{obs}_2_cut0.0"]; c3 = central[f"{obs}_3_cut0.0"]
    raw_c = central_to_raw(np.array([c1, c2, c3]))
    if sem_subtract:
        raw_gap_c = r_exp * raw_c - r_sem * np.asarray(sem_nom0)
    else:
        raw_gap_c = raw_c
    mu01_c = jf.raw_to_mu01([1.0] + list(raw_gap_c), lo, hi)
    feas_c, worst_c = hausdorff_moment_check(mu01_c)
    fn_c = f_c = None
    if feas_c:
        fn_c, _, ok_c, _ = maxent_pdf(mu01_c, t_grid, alpha, beta)
        f_c = fn_c / (hi - lo)

    n_hqe = len(hqe_toys[f"{obs}_1_cut0.0"])
    n_sem = len(sem0_toys) if sem_subtract else n_hqe
    n_pairs = min(n_toys, n_hqe, n_sem)
    i_hqe = rng.choice(n_hqe, size=n_pairs, replace=False)
    i_sem = rng.choice(n_sem, size=n_pairs, replace=False) if sem_subtract else None

    f_toys = []
    n_haus_ok = 0
    for j in range(n_pairs):
        ih = i_hqe[j]
        raw_h = central_to_raw(np.array([hqe_toys[f"{obs}_1_cut0.0"][ih],
                                          hqe_toys[f"{obs}_2_cut0.0"][ih],
                                          hqe_toys[f"{obs}_3_cut0.0"][ih]]))
        if sem_subtract:
            sem_t = sem0_toys[i_sem[j]][list(idx9)]
            raw_gap = r_exp * raw_h - r_sem * sem_t
        else:
            raw_gap = raw_h
        mu01 = jf.raw_to_mu01([1.0] + list(raw_gap), lo, hi)
        ok_h, _ = hausdorff_moment_check(mu01)
        if not ok_h:
            continue
        n_haus_ok += 1
        fn, _, ok, _ = maxent_pdf(mu01, t_grid, alpha, beta)
        if ok:
            f_toys.append(fn / (hi - lo))
    f_toys = np.array(f_toys)
    conv_frac = len(f_toys) / max(n_pairs, 1)
    haus_frac = n_haus_ok / max(n_pairs, 1)
    band = np.percentile(f_toys, [16, 84], axis=0) if len(f_toys) > 5 else (None, None)
    return dict(x=lo + t_grid * (hi - lo), f_c=f_c, feas_c=feas_c, worst_c=worst_c,
                band=band, n_pairs=n_pairs, haus_frac=haus_frac, conv_frac=conv_frac)


def joint_fit_toy_band(obs, C, out5, n_toys, rng):
    jkey = OBS_KEY[obs]
    full = jf.load_source_cov(out5, jkey, "stat_ff_bfmode")
    thresholds = full["thresholds"]
    cfg_o = C["OBS_CFG"][jkey]
    lo, hi = cfg_o["lo"], cfg_o["hi"]

    nominal_fit = jf.fit_maxent(jkey, C, thresholds, full["nominal"], full["cov"], M=6)
    c_best, Cinv_nom = nominal_fit["c_best"], regularized_cinv(full["cov"], lo, hi, len(thresholds))
    ctx = build_fit_context(jkey, C, thresholds, M=6)

    ens = np.load(out5 / f"toy_ensemble_{jkey}_stat_ff_bfmode.npz")
    raw_gap_toys = ens["raw_gap"]
    n_avail = raw_gap_toys.shape[0]
    idx = rng.choice(n_avail, size=min(n_toys, n_avail), replace=False)

    f_toys = []
    n_bad = 0
    t0 = time.time()
    for i in idx:
        f_i, _, ok = refit_warm(ctx, raw_gap_toys[i], thresholds, lo, hi, Cinv_nom, c_best)
        if ok:
            f_toys.append(f_i)
        else:
            n_bad += 1
    f_toys = np.array(f_toys)
    print(f"  [{jkey}] joint-fit toy refit: {len(idx)} toys in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/max(len(idx),1):.2f}s/toy), {n_bad} discarded (bad moment closure)")
    band = np.percentile(f_toys, [16, 84], axis=0)
    return dict(x=nominal_fit["x_grid"], f_best=nominal_fit["f_best"], band=band,
                chi2=nominal_fit["chi2"], dof=nominal_fit["dof"], n_toys=len(idx))


def mx_perthreshold_toy_band(C, out5, n_toys, rng):
    full = jf.load_source_cov(out5, "Mx", "stat_ff_bfmode")
    thresholds = full["thresholds"]
    cfg_o = C["OBS_CFG"]["Mx"]
    lo, hi = cfg_o["lo"], cfg_o["hi"]
    ens = np.load(out5 / "toy_ensemble_Mx_stat_ff_bfmode.npz")
    raw_gap_toys = ens["raw_gap"]  # (n_avail, n_thr*3)
    n_avail = raw_gap_toys.shape[0]
    idx = rng.choice(n_avail, size=min(n_toys, n_avail), replace=False)

    out = {}
    t0 = time.time()
    for ti, thr in enumerate(thresholds):
        nom_i = full["nominal"][3*ti:3*ti+3]
        cov_i = full["cov"][3*ti:3*ti+3, 3*ti:3*ti+3]
        fit0 = jf.fit_maxent("Mx", C, [thr], nom_i, cov_i, M=2)
        Cinv_i = regularized_cinv(cov_i, lo, hi, 1)
        ctx = build_fit_context("Mx", C, [thr], M=2)
        f_toys = []
        for i in idx:
            f_i, _, ok = refit_warm(ctx, raw_gap_toys[i, 3*ti:3*ti+3], [thr], lo, hi, Cinv_i, fit0["c_best"])
            if ok:
                f_toys.append(f_i)
        f_toys = np.array(f_toys)
        band = np.percentile(f_toys, [16, 84], axis=0)
        out[thr] = dict(x=fit0["x_grid"], f_best=fit0["f_best"], band=band)
    print(f"  [Mx] {len(thresholds)} thresholds x {len(idx)} toys refit in {time.time()-t0:.0f}s")
    return out


# Same category breakdown/colors as 5_plots.py's distributions_3panel.png, reused here (not
# duplicated in spirit -- 5_plots.py doesn't export these as importable constants, so this is
# the smallest faithful copy).
CATEGORIES = ["D", "D*", "D**", "D(*) pi", "D(*) pi pi", "Ds(*) K"]
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
# Mx's SUPPORT bounds are in Mx^2 (matching the rest of this file's Hausdorff/MaxEnt machinery) --
# the stack histogram itself is built directly in Mx (GeV), so its edges are sqrt(lo)..sqrt(hi).
STACK_BINS = {o: np.linspace(*(np.sqrt(SUPPORT[o][:2]) if o == "mx" else SUPPORT[o][:2]), 80) for o in SUPPORT}


def sem_stack_toy_band(cfg, n_toys, seed, cache):
    """Per-bin densities of the TOTAL SEM stack under the combined SEM uncertainty (cocktail stat
    bootstrap x FF reweighting x BR-mode variation -- same toy construction as 5_residual_covariance's
    stat_ff_bfmode source, minus the exp side), each toy normalized to unit area like the nominal
    stack. Returns {obs: (n_toys, 80)}. Cached in `cache` (recomputing loads ~24M events)."""
    if cache.exists():
        d = np.load(cache)
        if len(d["mx"]) >= n_toys:
            return {o: d[o][:n_toys] for o in STACK_BINS}
    from lib.systematics import (read_parquet_downcast, build_ff_slope_matrix, sample_ff_multiplier_from_matrix,
                                 bf_mode_setup, sample_bf_multiplier_from_codes)
    path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    names = pq.ParquetFile(path).schema_arrow.names
    ff_cols = ["ff_weight"] + [c for c in names if c.startswith(("ff_weight_up", "ff_weight_down"))]
    df = read_parquet_downcast(path, ["Mx", "El_B", "q2", "total_weight", "decay_name", "bf", "bf_unc"] + ff_cols,
                               float32_cols=ff_cols)
    ok = np.isfinite(df[["Mx", "El_B", "q2", "total_weight"]].to_numpy()).all(axis=1) & (df["total_weight"].to_numpy() > 0)
    df = df.loc[ok].reset_index(drop=True)
    x = {"mx": df["Mx"].to_numpy(float), "el": df["El_B"].to_numpy(float), "q2": df["q2"].to_numpy(float)}
    w_all = df["total_weight"].to_numpy(float)
    ff_slopes, _ = build_ff_slope_matrix(df)
    bf_codes, bf_rel = bf_mode_setup(df["decay_name"].to_numpy(), df["bf"].to_numpy(), df["bf_unc"].to_numpy())
    del df
    rng = np.random.default_rng(seed)
    out = {o: np.empty((n_toys, len(b) - 1)) for o, b in STACK_BINS.items()}
    for it in range(n_toys):
        w = w_all * np.bincount(rng.integers(0, len(w_all), size=len(w_all)), minlength=len(w_all))
        w = w * sample_ff_multiplier_from_matrix(ff_slopes, rng) * sample_bf_multiplier_from_codes(bf_codes, bf_rel, rng)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.)
        for o, b in STACK_BINS.items():
            out[o][it] = np.histogram(x[o], bins=b, weights=w)[0] / (w.sum() * np.diff(b))
        if it % 20 == 0:
            print(f"  SEM stack toy {it}/{n_toys}", flush=True)
    np.savez(cache, **out)
    return out


def plot_kolya_vs_sem_stack(cfg, kolya_full, sem_band, fig_dir):
    """Overlay Kolya's full-phase-space (no SEM subtraction) inversion with a genuine STACKED
    histogram of our SEM cocktail broken down by decay category -- sanity check: does the
    HQE-extrapolated full inclusive density resemble the sum of the known exclusive channels
    the SEM cocktail is built from? Both normalized to unit area (stack normalized as a whole,
    not per-category) so they're directly comparable as densities. Kolya's own band (from
    kolya_full, computed with sem_subtract=False) is already ALL of its applicable
    uncertainty -- this quantity involves no SEM subtraction at all, so HQE-toy variation is
    its complete uncertainty budget, not a partial one. The SEM stack's own band (grey, on
    the stack's total) is the 68% central interval of `sem_band` (sem_stack_toy_band)."""
    cocktail_path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    df = pd.read_parquet(cocktail_path, columns=["Mx", "El_B", "q2", "total_weight", "category"])
    ok = (np.isfinite(df["Mx"]) & np.isfinite(df["El_B"]) & np.isfinite(df["q2"])
          & np.isfinite(df["total_weight"]) & (df["total_weight"] > 0))
    df = df.loc[ok].copy()
    col_map = {"mx": "Mx", "el": "El_B", "q2": "q2"}  # "Mx" is already Mx (GeV), not Mx^2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    for ax, obs in zip(axes, ["mx", "el", "q2"]):
        lo, hi, xlabel, _ = SUPPORT[obs]
        col = col_map[obs]
        bins = STACK_BINS[obs]
        db = bins[1] - bins[0]

        vals_list, w_list, colors, labels = [], [], [], []
        for cat, color in zip(CATEGORIES, CAT_COLORS):
            m = df["category"] == cat
            if not m.any():
                continue
            vals_list.append(df.loc[m, col].to_numpy(float))
            w_list.append(df.loc[m, "total_weight"].to_numpy(float))
            colors.append(color); labels.append(cat)
        total_w = sum(w.sum() for w in w_list)
        w_list_norm = [w / (total_w * db) for w in w_list]  # stacked heights integrate to 1
        ax.hist(vals_list, bins=bins, weights=w_list_norm, stacked=True, color=colors,
                label=labels, alpha=0.85, edgecolor="none")
        lo_b, hi_b = np.percentile(sem_band[obs], [16, 84], axis=0)
        ax.stairs(hi_b, bins, baseline=lo_b, fill=True, color="0.15", alpha=0.45, lw=0, zorder=4,
                  label="SEM uncertainty (68%)" if obs == "mx" else "_")

        kf = kolya_full[obs]
        if kf["feas_c"]:
            x, f_c, band = kf["x"], kf["f_c"], kf["band"]
            if obs == "mx":
                x, jac = np.sqrt(x), 2.0 * np.sqrt(x)
                f_c = f_c * jac
                band = (band[0] * jac, band[1] * jac) if band[0] is not None else (None, None)
                xlabel = r"$M_X\ [\mathrm{GeV}]$"
                ax.set_xlim(1.6, 5.5)
            if band[0] is not None:
                ax.fill_between(x, band[0], band[1], facecolor="none", edgecolor="black",
                                 hatch="///", linewidth=0.0, zorder=5,
                                 label="Kolya band (HQE toys)" if obs == "mx" else "_")
            ax.plot(x, f_c, color="black", lw=2.0, label="Kolya (full inversion)", zorder=6)

        ax.set_xlabel(xlabel); ax.set_ylabel("density")
        ax.set_ylim(bottom=0); ax.set_title(TITLE[obs]); ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = fig_dir / "kolya_vs_sem_stack.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n-toys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-sem-toys", type=int, default=200, help="SEM-stack histogram toys for its uncertainty band")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    C = jf.load_common(cfg)
    # Q2 domain for the JOINT FIT: [1.5, Q2_HI] -- 1.5 GeV^2 is the lowest measured threshold (below it
    # the fit has no data), hi is tightened from config.yaml's 12.5 to the cocktail MC's kinematic
    # edge. Kolya's own inversion still spans [0, Q2_HI] (SUPPORT).
    C["OBS_CFG"]["Q2"]["lo"] = 1.5
    C["OBS_CFG"]["Q2"]["hi"] = Q2_HI
    r_exp, r_sem = compute_r_exp_r_sem(cfg)
    C["r_exp"], C["r_sem"] = r_exp, r_sem
    out5 = Path(cfg["paths"]["output"]) / "5"
    fig_dir = Path(cfg["paths"]["figures"]) / "6b"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["paths"]["output"]) / "6b"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t_grid = np.linspace(0.0, 1.0, 400)

    central, hqe_toys = load_h5_toys()
    sem0_d = np.load(out5 / "sem0_stat_ff_bfmode.npz")
    sem0_toys = sem0_d["sem0"]

    # Actual nominal SEM raw moments at thr=0 (not the toy ensemble's mean) for Kolya's central
    # curve, matching the nominal-value convention used elsewhere in this pipeline.
    sem_raw_json = json.load(open(Path(cfg["paths"]["output"]) / "5" / "sem_moments_raw.json"))
    sem_nom_raw = sem_raw_json["sem_nominal_raw"]
    sem_el_cuts = np.array(sem_raw_json["cuts"]["el"])
    sem_q2_cuts = np.array(sem_raw_json["cuts"]["q2"])
    sem_nom0 = {}
    for obs, keys in (("mx", ["mx_1", "mx_2", "mx_3"]), ("el", ["el_1", "el_2", "el_3"]),
                       ("q2", ["q2_1", "q2_2", "q2_3"])):
        cuts = sem_el_cuts if obs in ("mx", "el") else sem_q2_cuts
        sem_nom0[obs] = np.array([np.interp(0.0, cuts, np.array(sem_nom_raw[k])) for k in keys])

    # r_exp/r_sem: same BF-budget identity as everywhere else, recomputed via load_common.
    r_exp, r_sem = C["r_exp"], C["r_sem"]
    print(f"r_exp={r_exp:.4f}  r_sem={r_sem:.4f}  sem0 toys available={len(sem0_toys)}  "
          f"HQE toys available={len(hqe_toys['el_1_cut0.0'])}")

    summary = {}

    # ── Plot A: Kolya (SEM-subtracted, ALL uncertainties) vs joint chi^2 fit (ALL unc., toy) ──
    print("Computing joint chi^2 fit toy bands (El, Q2) ...")
    joint = {obs: joint_fit_toy_band(obs, C, out5, args.n_toys, rng) for obs in ["el", "q2"]}
    print("Computing Mx per-threshold toy bands ...")
    mx_pt = mx_perthreshold_toy_band(C, out5, args.n_toys, rng)
    print("Computing Kolya (SEM-subtracted) toy bands ...")
    kolya_gap = {obs: kolya_band(obs, r_exp, r_sem, True, rng, t_grid, central, hqe_toys,
                                  sem0_toys, args.n_toys, sem_nom0=sem_nom0[obs])
                 for obs in ["mx", "el", "q2"]}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)

    ax = axes[0]
    kg = kolya_gap["mx"]
    if kg["feas_c"]:
        x_mx, jac = np.sqrt(kg["x"]), 2.0 * np.sqrt(kg["x"])
        ax.plot(x_mx, kg["f_c"] * jac, color="crimson", lw=1.8, label="Kolya", zorder=4)
        if kg["band"][0] is not None:
            ax.fill_between(x_mx, kg["band"][0] * jac, kg["band"][1] * jac, color="crimson",
                             alpha=0.35, label="Kolya band (ALL unc., toy)", zorder=2)
    else:
        ax.text(0.5, 0.92, "Kolya (SEM-subtracted): Hausdorff-INFEASIBLE",
                transform=ax.transAxes, ha="center", va="top", color="crimson", fontsize=8.5)
    cmap = plt.get_cmap("viridis")
    thrs = sorted(mx_pt.keys())
    for i, thr in enumerate(thrs):
        e = mx_pt[thr]
        xg = np.sqrt(e["x"]); jac_e = 2.0 * xg
        ax.plot(xg, e["f_best"] * jac_e, color=cmap(i / max(len(thrs) - 1, 1)), lw=1.0, alpha=0.85,
                label="per-threshold fit (El-cuts, toy band)" if i == 0 else "_", zorder=3)
        ax.fill_between(xg, e["band"][0] * jac_e, e["band"][1] * jac_e,
                         color=cmap(i / max(len(thrs) - 1, 1)), alpha=0.12, zorder=1)
    ax.set_xlabel(r"$M_X\ [\mathrm{GeV}]$"); ax.set_ylabel(r"$f(M_X)$")
    ax.set_xlim(1.75, 3.0); ax.set_ylim(bottom=0)
    ax.set_title(TITLE["mx"]); ax.grid(alpha=0.25); ax.legend(fontsize=6.5, loc="upper right")

    for ax, obs in zip(axes[1:], ["el", "q2"]):
        lo, hi, xlabel, ylabel = SUPPORT[obs]
        jkey = OBS_KEY[obs]
        jr = joint[obs]
        kg = kolya_gap[obs]
        ax.fill_between(jr["x"], jr["band"][0], jr["band"][1], color="steelblue", alpha=0.25,
                         label="joint fit band (ALL unc., toy)", zorder=1)
        if kg["band"][0] is not None:
            ax.fill_between(kg["x"], kg["band"][0], kg["band"][1], color="crimson", alpha=0.35,
                             label="Kolya band (ALL unc., toy)", zorder=2)
        ax.plot(jr["x"], jr["f_best"], color="steelblue", lw=1.8, zorder=3,
                label=rf"joint $\chi^2$ fit ($\chi^2$/ndf={jr['chi2']:.1f}/{jr['dof']}={jr['chi2']/jr['dof']:.2f})")
        if kg["feas_c"]:
            ax.plot(kg["x"], kg["f_c"], color="crimson", lw=1.8, label="Kolya", zorder=4)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xlim(lo, hi); ax.set_ylim(bottom=0, top=0.7 if obs == "q2" else None)
        ax.set_title(TITLE[obs]); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    out_a = fig_dir / "kolya_vs_joint_gap.pdf"
    fig.savefig(out_a)
    plt.close(fig)
    print(f"-> {out_a}")

    summary["kolya_vs_joint_gap"] = {
        "el": dict(chi2=joint["el"]["chi2"], dof=joint["el"]["dof"], n_toys=joint["el"]["n_toys"],
                   kolya_feasible=kolya_gap["el"]["feas_c"], kolya_haus_frac=kolya_gap["el"]["haus_frac"]),
        "q2": dict(chi2=joint["q2"]["chi2"], dof=joint["q2"]["dof"], n_toys=joint["q2"]["n_toys"],
                   kolya_feasible=kolya_gap["q2"]["feas_c"], kolya_haus_frac=kolya_gap["q2"]["haus_frac"]),
        "mx": dict(kolya_feasible=kolya_gap["mx"]["feas_c"], kolya_worst=kolya_gap["mx"]["worst_c"],
                   kolya_haus_frac=kolya_gap["mx"]["haus_frac"]),
    }

    # ── Plot B: Kolya, NOT SEM-subtracted (full phase space) ──────────────────────────────────
    print("Computing Kolya (no SEM subtraction) toy bands ...")
    kolya_full = {obs: kolya_band(obs, r_exp, r_sem, False, rng, t_grid, central, hqe_toys,
                                   sem0_toys, args.n_toys) for obs in ["mx", "el", "q2"]}

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.3), dpi=150)
    for ax, obs in zip(axes2, ["mx", "el", "q2"]):
        lo, hi, xlabel, ylabel = SUPPORT[obs]
        kf = kolya_full[obs]
        if not kf["feas_c"]:
            ax.text(0.5, 0.5, "central moments\nHausdorff-infeasible", transform=ax.transAxes,
                    ha="center", va="center", color="crimson")
            ax.set_title(TITLE[obs]); continue
        x, f_c = kf["x"], kf["f_c"]
        if obs == "mx":
            x, jac = np.sqrt(x), 2.0 * np.sqrt(x)
            f_c = f_c * jac
            band = (kf["band"][0] * jac, kf["band"][1] * jac) if kf["band"][0] is not None else (None, None)
            xlabel, ylabel = r"$M_X\ [\mathrm{GeV}]$", r"$f(M_X)$"
        else:
            band = kf["band"]
        if band[0] is not None:
            ax.fill_between(x, band[0], band[1], color="C0", alpha=0.25, label="toy band (HQE only)")
        ax.plot(x, f_c, color="C0", lw=1.8, label="MaxEnt (central)")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(TITLE[obs] + f"  ({kf['conv_frac']*100:.0f}% of toys converged)")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig2.tight_layout()
    out_b = fig_dir / "kolya_no_sem.pdf"
    fig2.savefig(out_b)
    plt.close(fig2)
    print(f"-> {out_b}")

    summary["kolya_no_sem"] = {obs: dict(feasible=kolya_full[obs]["feas_c"],
                                          conv_frac=kolya_full[obs]["conv_frac"])
                                for obs in ["mx", "el", "q2"]}

    # ── Plot C: Kolya (no SEM subtraction) vs our SEM cocktail, stacked by decay category ─────
    print("Building Kolya vs SEM-stacked-histogram overlay ...")
    sem_band = sem_stack_toy_band(cfg, args.n_sem_toys, args.seed, out_dir / "sem_stack_band.npz")
    plot_kolya_vs_sem_stack(cfg, kolya_full, sem_band, fig_dir)

    with open(out_dir / "summary.json", "w") as f:
        def _default(o):
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            raise TypeError(f"not serializable: {type(o)}")
        json.dump(summary, f, indent=2, default=_default)
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
