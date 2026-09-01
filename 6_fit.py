#!/usr/bin/env python3
"""Step 6: Fit gap mode signal strengths to experimental raw-moment average.

Reads:  output/4/experimental_average.json  (data points + errors)
        output/3/cocktail.parquet            (SM templates + nuisance templates)
        output/3/{mode}.parquet              (gap mode templates, one per mode)

Outputs:
  output/6/fit_results.json
  output/6/fit_results_individual.tex
  output/6/fit_results_unconstrained.tex
  figures/6/postfit_individual_{mode}.png  (one per gap mode)
  figures/6/postfit_unconstrained.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Avoid thread-spawn failures on constrained batch/login nodes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
try:
    import emcee as _emcee
    _EMCEE_AVAILABLE = True
except ImportError:
    _EMCEE_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "lib"))

from lib.gap_modes import GAP_MODES

out6 = Path(cfg["paths"]["output"]) / "6"
fig6 = Path(cfg["paths"]["figures"]) / "6"
out6.mkdir(parents=True, exist_ok=True)
fig6.mkdir(parents=True, exist_ok=True)

fc = cfg["fit"]
FIT_MODE        = fc.get("mode", "simultaneous")
ACTIVE_MODES    = fc.get("gap_modes", [m["mode"] for m in GAP_MODES])
INCLUDE_DSK_IN_SM = bool(fc.get("include_cocktail_dsk", True))
EL_PRECUT       = float(fc.get("el_cut_min", 0.5))
MX_UPPER_CUT    = fc.get("mx_upper_cut", None)
N_BR_MAX        = int(fc.get("n_br_nuisance", 20))
N_FF_MAX        = int(fc.get("n_ff_nuisance", 200))
DELTA_CHI2      = float(fc.get("delta_chi2", 2.71))
MU_SCAN_MAX     = float(fc.get("mu_total_scan_max", 10.0))
N_SCAN          = int(fc.get("n_scan", 400))
from lib import systematics as syst

# BR_REF = bf_gap_central: mu=1 now corresponds to the full measured inclusive-minus-
# exclusive gap (same compute_bf_gap() convention as 7_hausdorff_toy.py/8_hausdorff_data.py,
# derived fresh from the cocktail's per-mode BFs, not a hardcoded constant), so a single
# mode's mu_total_scan_max=1.2 comfortably covers "this one mode explains the whole gap"
# instead of pegging at the old br_ref=0.018 boundary (1.2 x 1.8% = 2.16%, far short of
# the ~6.26% measured gap) before the profile scan ever finds an upper limit.
_sysc        = cfg.get("systematics", {})
_bf_incl     = float(_sysc.get("bf_incl", 0.1105))
_bf_incl_unc = float(_sysc.get("bf_incl_unc", 0.0016))
_df_bf = pd.read_parquet(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet",
                          columns=["decay_name", "bf", "bf_unc"])
_bf_sem_computed, _bf_gap_central, _sigma_bf_gap = syst.compute_bf_gap(_df_bf, _bf_incl, _bf_incl_unc)
del _df_bf
BR_REF = _bf_gap_central
print(f"  BR_REF = bf_gap_central = {100*BR_REF:.3f}%  (sigma={100*_sigma_bf_gap:.3f}%)")
MCMC_N_BURN     = int(fc.get("mcmc_n_burn", 500))
MCMC_N_STEPS    = int(fc.get("mcmc_n_steps", 3000))
MCMC_N_WALKERS  = fc.get("mcmc_n_walkers", None)  # None = auto (4*ndim, min 128)

KEYS = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]
YLABELS_RAW = {
    "mx_1": r"$\langle M_X^2 \rangle$",      "mx_2": r"$\langle (M_X^2)^2 \rangle$",
    "mx_3": r"$\langle (M_X^2)^3 \rangle$",   "el_1": r"$\langle E_\ell \rangle$",
    "el_2": r"$\langle E_\ell^2 \rangle$",    "el_3": r"$\langle E_\ell^3 \rangle$",
    "q2_1": r"$\langle q^2 \rangle$",         "q2_2": r"$\langle (q^2)^2 \rangle$",
    "q2_3": r"$\langle (q^2)^3 \rangle$",
}
ROW_TITLES = [r"$M_X^2$ moments", r"$E_\ell$ moments", r"$q^2$ moments"]


# ── Core fit functions ────────────────────────────────────────────────────────

def _suffix_sum(sorted_sel, sorted_val, cuts):
    idx = np.searchsorted(sorted_sel, cuts, side="right")
    c   = np.concatenate(([0.], np.cumsum(sorted_val)))
    return c[-1] - c[idx]


def _compute_WN(mx2, el, q2, w, points, cuts_by_key):
    """Compute per-point W (total weight) and N (moment numerator) arrays."""
    ok   = np.isfinite(mx2) & np.isfinite(el) & np.isfinite(q2) & np.isfinite(w)
    if MX_UPPER_CUT is not None:
        ok &= (mx2 < float(MX_UPPER_CUT) ** 2)
    mx2  = mx2[ok]; el = el[ok]; q2 = q2[ok]; w = w[ok]

    ord_el = np.argsort(el, kind="stable")
    el_s   = el[ord_el]; w_el = w[ord_el]; mx2_s = mx2[ord_el]

    m_q    = el > EL_PRECUT
    q2_sub = q2[m_q]; w_sub = w[m_q]
    ord_q  = np.argsort(q2_sub, kind="stable")
    q2_s   = q2_sub[ord_q]; w_q = w_sub[ord_q]

    key_WN: dict[str, tuple] = {}
    for key, cuts in cuts_by_key.items():
        fam, n = key.split("_", 1); n = int(n)
        if fam == "mx":
            x = np.power(mx2_s, n)
            key_WN[key] = (_suffix_sum(el_s, w_el, cuts),
                           _suffix_sum(el_s, w_el * x, cuts))
        elif fam == "el":
            x = np.power(el_s, n)
            key_WN[key] = (_suffix_sum(el_s, w_el, cuts),
                           _suffix_sum(el_s, w_el * x, cuts))
        elif fam == "q2":
            x = np.power(q2_s, n)
            key_WN[key] = (_suffix_sum(q2_s, w_q, cuts),
                           _suffix_sum(q2_s, w_q * x, cuts))
        else:
            z = np.zeros(len(cuts), dtype=float)
            key_WN[key] = (z, z)

    Wv = np.zeros(len(points), dtype=float)
    Nv = np.zeros(len(points), dtype=float)
    for i, (key, cut_idx) in enumerate(points):
        Warr, Narr = key_WN[key]
        if cut_idx < len(Warr):
            Wv[i] = Warr[cut_idx]; Nv[i] = Narr[cut_idx]
    return Wv, Nv


def _suffix_sum_multi(sorted_sel, sorted_val_mat, cuts):
    """Vectorized suffix sums for many value columns at once."""
    idx = np.searchsorted(sorted_sel, cuts, side="right")
    c = np.vstack([
        np.zeros((1, sorted_val_mat.shape[1]), dtype=float),
        np.cumsum(sorted_val_mat, axis=0),
    ])
    return c[-1] - c[idx]


def _compute_WN_multi(mx2, el, q2, w_mat, points, cuts_by_key):
    """Compute per-point W/N for many weight columns in one pass.

    Args:
      w_mat: shape (N_events, N_templates)
    Returns:
      Wv, Nv with shape (N_points, N_templates)
    """
    ok = np.isfinite(mx2) & np.isfinite(el) & np.isfinite(q2)
    ok &= np.all(np.isfinite(w_mat), axis=1)
    if MX_UPPER_CUT is not None:
        ok &= (mx2 < float(MX_UPPER_CUT) ** 2)
    mx2 = mx2[ok]
    el = el[ok]
    q2 = q2[ok]
    w_mat = w_mat[ok]

    ord_el = np.argsort(el, kind="stable")
    el_s = el[ord_el]
    w_el = w_mat[ord_el]
    mx2_s = mx2[ord_el]

    m_q = el > EL_PRECUT
    q2_sub = q2[m_q]
    w_q = w_mat[m_q]
    ord_q = np.argsort(q2_sub, kind="stable")
    q2_s = q2_sub[ord_q]
    w_q = w_q[ord_q]

    J = w_mat.shape[1]
    key_WN = {}
    for key, cuts in cuts_by_key.items():
        fam, n = key.split("_", 1)
        n = int(n)
        if fam == "mx":
            x = np.power(mx2_s, n)[:, np.newaxis]
            key_WN[key] = (
                _suffix_sum_multi(el_s, w_el, cuts),
                _suffix_sum_multi(el_s, w_el * x, cuts),
            )
        elif fam == "el":
            x = np.power(el_s, n)[:, np.newaxis]
            key_WN[key] = (
                _suffix_sum_multi(el_s, w_el, cuts),
                _suffix_sum_multi(el_s, w_el * x, cuts),
            )
        elif fam == "q2":
            x = np.power(q2_s, n)[:, np.newaxis]
            key_WN[key] = (
                _suffix_sum_multi(q2_s, w_q, cuts),
                _suffix_sum_multi(q2_s, w_q * x, cuts),
            )
        else:
            z = np.zeros((len(cuts), J), dtype=float)
            key_WN[key] = (z, z)

    Wv = np.zeros((len(points), J), dtype=float)
    Nv = np.zeros((len(points), J), dtype=float)
    for i, (key, cut_idx) in enumerate(points):
        Warr, Narr = key_WN[key]
        if cut_idx < len(Warr):
            Wv[i, :] = Warr[cut_idx]
            Nv[i, :] = Narr[cut_idx]
    return Wv, Nv


def _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN):
    mu_k  = x[:K]; theta = x[K:]
    W     = W0 + (dW @ theta if dW.size else 0.) + Wg_all.T @ mu_k
    N     = N0 + (dN @ theta if dN.size else 0.) + Ng_all.T @ mu_k
    if np.any(W <= 0.):
        return 1e30, np.zeros_like(x)
    pred  = N / W
    r     = y - pred
    icr   = inv_cov @ r
    chi2  = float(r @ icr) + float(theta @ theta)
    dpd_mu = (Ng_all - pred[np.newaxis, :] * Wg_all) / W[np.newaxis, :]
    grad_mu = -2. * (dpd_mu @ icr)
    if dW.size:
        dpd_th  = (dN - pred[:, np.newaxis] * dW) / W[:, np.newaxis]
        grad_th = -2. * (dpd_th.T @ icr) + 2. * theta
    else:
        grad_th = 2. * theta
    return chi2, np.concatenate([grad_mu, grad_th])


def _nearest_psd(cov, rel_eps=1e-8):
    """Clip eigenvalues at rel_eps * (largest eigenvalue), not a fixed absolute
    floor: a covariance built by propagating a low-rank fit (e.g. the joint
    polynomial average in 4_average.py, where a handful of basis coefficients
    determine many output points) is *exactly* rank-deficient by construction,
    so most of its small eigenvalues are pure float64 roundoff clustered near
    the matrix's own scale, not close to any fixed absolute value like 1e-12."""
    w, v = np.linalg.eigh(cov)
    eps = rel_eps * max(float(np.max(w)), 0.0)
    wc = np.clip(w, eps, None)
    out = (v * wc) @ v.T
    return out, bool(np.any(wc != w))


def _extract_ul(mu_grid, chi2_grid, i_best):
    dchi2 = chi2_grid - chi2_grid[i_best]
    mx = mu_grid[i_best:]; dx = dchi2[i_best:]
    cr = np.where(dx >= DELTA_CHI2)[0]
    if cr.size == 0:
        return np.nan
    j = int(cr[0])
    if j == 0:
        return float(mx[0])
    x0, x1 = float(mx[j-1]), float(mx[j]); y0, y1 = float(dx[j-1]), float(dx[j])
    return float(x0 + (DELTA_CHI2 - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else x1


def _extract_interval_at_delta(x_grid, chi2_grid, i_best, delta=1.0):
    dchi2 = chi2_grid - chi2_grid[i_best]
    x_best = float(x_grid[i_best])

    # Lower crossing
    lo = np.nan
    for i in range(i_best - 1, -1, -1):
        if dchi2[i] >= delta:
            x0, x1 = float(x_grid[i]), float(x_grid[i + 1])
            y0, y1 = float(dchi2[i]), float(dchi2[i + 1])
            if y0 != y1:
                lo = x0 + (delta - y0) * (x1 - x0) / (y1 - y0)
            else:
                lo = x0
            break

    # Upper crossing
    hi = np.nan
    for i in range(i_best + 1, len(x_grid)):
        if dchi2[i] >= delta:
            x0, x1 = float(x_grid[i - 1]), float(x_grid[i])
            y0, y1 = float(dchi2[i - 1]), float(dchi2[i])
            if y0 != y1:
                hi = x0 + (delta - y0) * (x1 - x0) / (y1 - y0)
            else:
                hi = x1
            break

    err_lo = x_best - lo if np.isfinite(lo) else np.nan
    err_hi = hi - x_best if np.isfinite(hi) else np.nan
    return lo, hi, err_lo, err_hi


def _safe_tex(s):
    return str(s).replace("_", r"\_")


def _write_individual_results_tex(path, results_by_mode, loaded_meta):
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Mode & $\hat{\mu}$ & $\mu_{95\%}$ & $\hat{\mathcal{B}}[\%]$ & $\mathcal{B}_{95\%}[\%]$ \\",
        r"\hline",
    ]
    for meta in loaded_meta:
        mode = meta["mode"]
        res = results_by_mode.get(mode, {})
        lines.append(
            rf"{_safe_tex(mode)} & "
            rf"{res.get('mu_best', float('nan')):.3f} & "
            rf"{(res.get('mu_ul_95') if res.get('mu_ul_95') is not None else float('nan')):.3f} & "
            rf"{res.get('br_best_pct', float('nan')):.3f} & "
            rf"{(res.get('br_ul_95_pct') if res.get('br_ul_95_pct') is not None else float('nan')):.3f} \\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def _write_unconstrained_results_tex(path, unc_result):
    has_bayes = any("br_k_bayes_ul_95_pct" in e for e in unc_result["per_mode"].values())
    if has_bayes:
        lines = [
            r"\begin{tabular}{lrrr}",
            r"\hline",
            r"Mode & $\hat{\mathcal{B}}_k[\%]$ & $\mathcal{B}_{k,95\%}^{\rm prof}[\%]$ & $\mathcal{B}_{k,95\%}^{\rm Bayes}[\%]$ \\",
            r"\hline",
        ]
        for mode, entry in unc_result["per_mode"].items():
            ul_p = entry.get("br_k_ul_95_pct")
            ul_b = entry.get("br_k_bayes_ul_95_pct")
            ul_p_str = f"{ul_p:.3f}" if ul_p is not None else "---"
            ul_b_str = f"{ul_b:.3f}" if ul_b is not None else "---"
            lines.append(rf"{_safe_tex(mode)} & {entry['br_k_pct']:.3f} & {ul_p_str} & {ul_b_str} \\")
    else:
        lines = [
            r"\begin{tabular}{lrr}",
            r"\hline",
            r"Mode & $\hat{\mathcal{B}}_k[\%]$ & $\mathcal{B}_{k,95\%}[\%]$ \\",
            r"\hline",
        ]
        for mode, entry in unc_result["per_mode"].items():
            ul = entry.get("br_k_ul_95_pct")
            ul_str = f"{ul:.3f}" if ul is not None else "---"
            lines.append(rf"{_safe_tex(mode)} & {entry['br_k_pct']:.3f} & {ul_str} \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


# ── Load data ─────────────────────────────────────────────────────────────────

print("[1/5] Loading data …")
if MX_UPPER_CUT is not None:
    print(f"  applying event selection: M_X < {float(MX_UPPER_CUT):.3g} GeV")
avg_payload = json.loads((Path(cfg["paths"]["output"]) / "4" / "experimental_average.json").read_text())
avg_raw     = avg_payload.get("average_raw", avg_payload.get("average", {}))

y_list, e_list, pts, cuts_by_key = [], [], [], {}
for key in KEYS:
    if key not in avg_raw:
        continue
    cuts = np.asarray(avg_raw[key]["cuts"], dtype=float)
    vals = np.asarray(avg_raw[key]["values"], dtype=float)
    errs = np.asarray(avg_raw[key]["errors"], dtype=float)
    cuts_by_key[key] = cuts
    for i in range(len(cuts)):
        if np.isfinite(vals[i]) and np.isfinite(errs[i]) and errs[i] > 0:
            pts.append((key, i)); y_list.append(float(vals[i])); e_list.append(float(errs[i]))

y   = np.asarray(y_list, dtype=float)
err = np.asarray(e_list, dtype=float)

if len(y) == 0:
    raise RuntimeError("No valid data points in average JSON.")


def _point_key(k, cut):
    return f"{k}|{float(cut):.10g}"


def _cov_from_average_payload(payload):
    cov_block = payload.get("average_raw_cov", {})
    pt_defs = cov_block.get("points", [])
    cov_list = cov_block.get("cov", [])
    if not pt_defs or not cov_list:
        return None, None
    cov_full = np.asarray(cov_list, dtype=float)
    if cov_full.ndim != 2 or cov_full.shape[0] != cov_full.shape[1] or cov_full.shape[0] != len(pt_defs):
        return None, None
    idx_map = {
        _point_key(p.get("key"), p.get("cut")): i
        for i, p in enumerate(pt_defs)
    }
    want = []
    for key, i_cut in pts:
        cut = cuts_by_key[key][i_cut]
        k = _point_key(key, cut)
        if k not in idx_map:
            return None, None
        want.append(idx_map[k])
    want = np.asarray(want, dtype=int)
    rank_full = cov_block.get("rank")
    # Subsetting rows/cols of a matrix can only lower its rank, never raise it, so this
    # remains a valid (if occasionally loose) exact rank bound after selecting `want`.
    rank = min(int(rank_full), len(want)) if rank_full is not None else None
    return cov_full[np.ix_(want, want)], rank


def _rank_truncated_pinv(cov, rank):
    """Exact-rank pseudo-inverse: keep only the `rank` largest-magnitude eigenvalues
    and invert those, hard-zeroing everything else. Use this instead of a magnitude
    -based rcond when the true rank is known analytically (e.g. average_raw_cov from
    the joint polynomial average in 4_average.py, which is exactly rank-deficient by
    construction -- eigenvalues past the true rank are pure float64 roundoff, and a
    heuristic rcond threshold can't reliably separate that roundoff from genuinely
    small-but-real eigenvalues sitting nearby in magnitude)."""
    w, v = np.linalg.eigh(cov)
    order = np.argsort(np.abs(w))[::-1]
    keep = order[:rank]
    w_keep, v_keep = w[keep], v[:, keep]
    # A rank-truncated PSD covariance should keep only genuinely positive eigenvalues;
    # a non-positive value surviving the top-|w| cut would mean `rank` overcounts this
    # particular matrix's real positive eigenvalues -- drop it rather than invert a
    # near-zero/negative value into a spurious huge precision weight.
    pos = w_keep > 0.0
    w_keep, v_keep = w_keep[pos], v_keep[:, pos]
    cov_psd = (v_keep * w_keep) @ v_keep.T
    inv = (v_keep * (1.0 / w_keep)) @ v_keep.T
    return cov_psd, inv


cov, cov_rank = _cov_from_average_payload(avg_payload)
if cov is None:
    raise RuntimeError(
        "average_raw_cov missing or not aligned with fit points. "
        "Run 4_average.py first and ensure output/4/experimental_average.json contains aligned average_raw_cov."
    )

if cov_rank is not None and cov_rank < cov.shape[0]:
    cov, inv_cov = _rank_truncated_pinv(cov, cov_rank)
    print(f"  average_raw_cov: exact-rank truncation to rank={cov_rank} (of {cov.shape[0]} points)")
else:
    cov, _psd_fix = _nearest_psd(cov)
    # rcond relative to the eps used in _nearest_psd above -- must not be tighter than
    # that floor, or pinv will re-invert the same near-zero noise directions we just
    # clipped (see _nearest_psd docstring: the raw/central average covariance can be
    # exactly rank-deficient when produced by a low-rank fit like the joint polynomial
    # average, so a 1e-12 rcond is far too permissive and amplifies float64 roundoff
    # into artificial precision weights of O(1e10), inflating chi2 by orders of magnitude).
    inv_cov = np.linalg.pinv(cov, rcond=1e-8)

# ── Load SM cocktail ──────────────────────────────────────────────────────────

sm = pd.read_parquet(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet")
if "total_weight" not in sm.columns:
    sm["total_weight"] = sm["weight"] * sm["mc_weight"] * sm["ff_weight"]
if not INCLUDE_DSK_IN_SM:
    sm.loc[sm["category"] == "Ds(*) K", "total_weight"] = 0.0

ff_up_cols = sorted(c for c in sm.columns if c.startswith("ff_weight_up"))
ff_dn_cols = sorted(c for c in sm.columns if c.startswith("ff_weight_down"))
ff_ids = sorted(
    set(int(c.replace("ff_weight_up", "")) for c in ff_up_cols)
    & set(int(c.replace("ff_weight_down", "")) for c in ff_dn_cols)
)[:max(0, N_FF_MAX)]

mx2_sm = np.square(sm["Mx"].to_numpy(dtype=float))
el_sm  = sm["El_B"].to_numpy(dtype=float)
q2_sm  = sm["q2"].to_numpy(dtype=float)
w0_sm  = sm["total_weight"].to_numpy(dtype=float)
W0, N0 = _compute_WN(mx2_sm, el_sm, q2_sm, w0_sm, pts, cuts_by_key)

# Build nuisance templates
bf         = sm["bf"].to_numpy(dtype=float)
bf_unc     = sm["bf_unc"].to_numpy(dtype=float)
rel_bf     = np.divide(bf_unc, bf, out=np.zeros_like(bf_unc), where=bf > 0)
decay_arr  = sm["decay_name"].astype(str).to_numpy()
ff_c       = sm["ff_weight"].to_numpy(dtype=float)

# Select top-N BR nuisances by total weight contribution
agg = (pd.DataFrame({"d": decay_arr, "w": w0_sm, "r": rel_bf})
       .groupby("d", as_index=False)
       .agg(ws=("w", "sum"), r=("r", "first"))
       .query("ws > 0 and r > 0")
       .sort_values("ws", ascending=False))
br_names = agg["d"].head(N_BR_MAX).tolist()
br_rel_map = {str(r["d"]): float(r["r"]) for _, r in agg.iterrows()}

# If Ds(*)K is kept in the SM cocktail, ensure its BR nuisances are always present.
if INCLUDE_DSK_IN_SM:
    dsk_decay_names = sorted(
        set(
            sm.loc[(sm["category"] == "Ds(*) K") & (sm["total_weight"] > 0), "decay_name"]
            .astype(str)
            .tolist()
        )
    )
    for dn in dsk_decay_names:
        row = agg.loc[agg["d"] == dn]
        if not row.empty and dn not in br_names:
            br_names.append(dn)

nuisance_names, nuisance_types, dw_cols = [], [], []
br_theta_bounds = []
for dname in br_names:
    dw = w0_sm * rel_bf * (decay_arr == dname).astype(float)
    nuisance_names.append(dname)
    nuisance_types.append("br")
    dw_cols.append(dw)
    r = float(br_rel_map.get(dname, 0.0))
    # Enforce non-negative BR scaling: 1 + theta * r >= 0.
    lo = (-1.0 / r + 1e-8) if r > 0 else -10.0
    br_theta_bounds.append((max(lo, -10.0), 10.0))

FF_DECAYS = {
    "Bp_Denu", "Bp_Dstenu", "Bp_D1enu", "Bp_D0stenu", "Bp_Dp1enu", "Bp_D2stenu",
    "B0_Denu", "B0_Dstenu", "B0_D1enu", "B0_D0stenu", "B0_Dp1enu", "B0_D2stenu",
    "Bp_Dmunu", "Bp_Dstmunu", "Bp_D1munu", "Bp_D0stmunu", "Bp_Dp1munu", "Bp_D2stmunu",
    "B0_Dmunu", "B0_Dstmunu", "B0_D1munu", "B0_D0stmunu", "B0_Dp1munu", "B0_D2stmunu",
}
ff_added = 0
for dname in sorted(set(decay_arr) & FF_DECAYS):
    m_d = (decay_arr == dname)
    for j in ff_ids:
        if ff_added >= N_FF_MAX: break
        if f"ff_weight_up{j}" not in sm.columns: continue
        up = sm[f"ff_weight_up{j}"].to_numpy(dtype=float)
        dn = sm[f"ff_weight_down{j}"].to_numpy(dtype=float)
        slope = 0.5 * (np.divide(up - ff_c, ff_c, out=np.zeros_like(ff_c), where=np.abs(ff_c) > 0)
                      + np.divide(ff_c - dn, ff_c, out=np.zeros_like(ff_c), where=np.abs(ff_c) > 0))
        dw = w0_sm * slope * m_d.astype(float)
        if np.nanmax(np.abs(dw)) < 1e-10: continue
        nuisance_names.append(f"ff:{dname}:eig{j}")
        nuisance_types.append("ff")
        dw_cols.append(dw)
        ff_added += 1

if dw_cols:
    dw_mat = np.column_stack(dw_cols)
    dW, dN = _compute_WN_multi(mx2_sm, el_sm, q2_sm, dw_mat, pts, cuts_by_key)
    # Consistency spot-check: batched and scalar accumulation must agree.
    n_chk = min(3, dW.shape[1])
    for j in range(n_chk):
        dW_s, dN_s = _compute_WN(mx2_sm, el_sm, q2_sm, dw_mat[:, j], pts, cuts_by_key)
        if not (np.allclose(dW[:, j], dW_s, rtol=1e-10, atol=1e-12) and
                np.allclose(dN[:, j], dN_s, rtol=1e-10, atol=1e-12)):
            raise RuntimeError(f"Batched nuisance accumulation mismatch at column {j}")
else:
    dW = np.zeros((len(pts), 0), dtype=float)
    dN = np.zeros((len(pts), 0), dtype=float)
n_br = sum(1 for t in nuisance_types if t == "br")
M    = dW.shape[1]
print(f"  points={len(pts)}, nuisances={M} (BR={n_br}, FF={M-n_br})")

# ── Load gap modes ────────────────────────────────────────────────────────────

active_meta = [m for m in GAP_MODES if m["mode"] in ACTIVE_MODES]
Wg_list, Ng_list, loaded_meta, gap_arrays = [], [], [], []
for meta in active_meta:
    pq = Path(cfg["paths"]["output"]) / "3" / f"{meta['mode']}.parquet"
    if not pq.exists():
        print(f"  warning: {pq} not found — skipping {meta['mode']}"); continue
    gdf = pd.read_parquet(pq)
    if "total_weight" not in gdf.columns:
        gdf["total_weight"] = gdf["weight"] * gdf["mc_weight"] * gdf["ff_weight"]
    mx2_g = np.square(gdf["Mx"].to_numpy(dtype=float))
    el_g  = gdf["El_B"].to_numpy(dtype=float)
    q2_g  = gdf["q2"].to_numpy(dtype=float)
    w_g   = gdf["total_weight"].to_numpy(dtype=float)
    Wg, Ng = _compute_WN(mx2_g, el_g, q2_g, w_g, pts, cuts_by_key)
    Wg_list.append(Wg); Ng_list.append(Ng); loaded_meta.append(meta)
    gap_arrays.append((mx2_g, el_g, q2_g, w_g))

if not loaded_meta:
    raise RuntimeError("No gap mode parquet files found.")

K      = len(loaded_meta)
Wg_all = np.stack(Wg_list, axis=0)   # (K, P)
Ng_all = np.stack(Ng_list, axis=0)
theta_bounds = (br_theta_bounds + [(-10., 10.)] * (M - n_br))

# ── Run fits ──────────────────────────────────────────────────────────────────

print(f"[1b/4] Preparing fit (config mode={FIT_MODE}) …")

fit_results: dict = {"individual": {}}

_GRID_6 = [
    ("mx", ["mx_1", "mx_2", "mx_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("el", ["el_1", "el_2", "el_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("q2", ["q2_1", "q2_2", "q2_3"], r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$"),
]


def _postfit_plot(mu_vec, theta_vec, title, outfile, combine_gap=False, primary_meta=None):
    # Build a dense cut grid per family for smooth gap curves.
    _N_DENSE = 200
    dense_cuts_by_key: dict[str, np.ndarray] = {}
    for key, cuts in cuts_by_key.items():
        lo, hi = float(cuts.min()), float(cuts.max())
        dense_cuts_by_key[key] = np.linspace(lo, hi, _N_DENSE)

    dense_pts: list[tuple[str, int]] = [
        (key, i) for key in dense_cuts_by_key for i in range(_N_DENSE)
    ]

    k0_global = int(np.argmax(mu_vec))
    mx2_g, el_g, q2_g, w_g = gap_arrays[k0_global]
    Wg_d, Ng_d = _compute_WN(mx2_g, el_g, q2_g, w_g, dense_pts, dense_cuts_by_key)
    W0_d, N0_d = _compute_WN(mx2_sm, el_sm, q2_sm, w0_sm, dense_pts, dense_cuts_by_key)

    # For combine_gap: precompute dense WN for all modes once (not per axis).
    if combine_gap:
        _Wg_all_d = np.stack([
            _compute_WN(arr[0], arr[1], arr[2], arr[3], dense_pts, dense_cuts_by_key)[0]
            for arr in gap_arrays
        ])  # (K, N_DENSE * n_keys)
        _Ng_all_d = np.stack([
            _compute_WN(arr[0], arr[1], arr[2], arr[3], dense_pts, dense_cuts_by_key)[1]
            for arr in gap_arrays
        ])

    fig_pf, axes_pf = plt.subplots(3, 3, figsize=(18, 10), dpi=150)
    for r, (_, keys, xlabel) in enumerate(_GRID_6):
        for c, key in enumerate(keys):
            ax = axes_pf[r, c]
            if key not in cuts_by_key:
                ax.set_axis_off()
                continue
            pi = [i for i, (kk, _) in enumerate(pts) if kk == key]
            if not pi:
                ax.set_axis_off()
                continue
            ci = [pts[i][1] for i in pi]
            cuts_pt = cuts_by_key[key][ci]

            W0k = W0[pi]
            N0k = N0[pi]
            dWk = dW[pi]
            dNk = dN[pi]
            gap_Wk = np.sum(mu_vec[:, np.newaxis] * Wg_all[:, pi], axis=0)
            gap_Nk = np.sum(mu_vec[:, np.newaxis] * Ng_all[:, pi], axis=0)
            W_pf = W0k + (dWk @ theta_vec if dWk.size else 0.) + gap_Wk
            N_pf = N0k + (dNk @ theta_vec if dNk.size else 0.) + gap_Nk
            y_sm = np.divide(N0k, W0k, out=np.full_like(N0k, np.nan), where=W0k > 0)
            y_pf = np.divide(N_pf, W_pf, out=np.full_like(N_pf, np.nan), where=W_pf > 0)

            # Dense indices for smooth gap curve
            di = [i for i, (kk, _) in enumerate(dense_pts) if kk == key]
            cuts_dense = dense_cuts_by_key[key]
            W0_dk = W0_d[di]
            N0_dk = N0_d[di]
            Wg_dk = Wg_d[di]
            Ng_dk = Ng_d[di]

            if combine_gap:
                gap_Wk_d = np.sum(mu_vec[:, np.newaxis] * _Wg_all_d[:, di], axis=0)
                gap_Nk_d = np.sum(mu_vec[:, np.newaxis] * _Ng_all_d[:, di], axis=0)
                y_with_gap_d = np.divide(N0_dk + gap_Nk_d, W0_dk + gap_Wk_d,
                                         out=np.full_like(W0_dk, np.nan), where=(W0_dk + gap_Wk_d) > 0)
                y_sm_d = np.divide(N0_dk, W0_dk, out=np.full_like(W0_dk, np.nan), where=W0_dk > 0)
                y_gr_dense = y_with_gap_d - y_sm_d
                gr_label = r"Added fitted gap contribution $\Delta_{\mathrm{gap}}$"
                gr_color, gr_ls = "#9b59b6", "-"
            else:
                y_gr_dense = np.divide(Ng_dk, W0_dk + Wg_dk,
                                       out=np.full_like(W0_dk, np.nan), where=(W0_dk + Wg_dk) > 0)
                gr_label = rf"{primary_meta['label']} @ BR$_{{\rm ref}}$"
                gr_color, gr_ls = primary_meta["color"], primary_meta["ls"]

            vals = np.asarray(avg_raw[key]["values"], dtype=float)[ci]
            errs = np.asarray(avg_raw[key]["errors"], dtype=float)[ci]
            ax.fill_between(cuts_pt, vals - errs, vals + errs, color="black", alpha=0.12, zorder=10,
                            label=r"Exp avg $\pm1\sigma$" if (r == c == 0) else None)
            ax.errorbar(cuts_pt, vals, yerr=errs, ls="", marker="o", color="black",
                        mec="black", ecolor="black", lw=1, ms=3.8, zorder=11,
                        label="Experimental average" if (r == c == 0) else None)
            ax.plot(cuts_pt, y_sm, color="0.45", lw=2.2, zorder=5,
                    label="SM (nominal)" if (r == c == 0) else None)
            if combine_gap:
                ax.fill_between(cuts_dense, 0.0, y_gr_dense, color=gr_color, alpha=0.7, zorder=3,
                                label=gr_label if (r == c == 0) else None)
            else:
                ax.plot(cuts_dense, y_gr_dense, color=gr_color, lw=1.5, ls=gr_ls, zorder=6,
                        label=gr_label if (r == c == 0) else None)
            ax.plot(cuts_pt, y_pf, color="black", lw=1.8, zorder=7,
                    label="Postfit" if (r == c == 0) else None)
            ax.grid(alpha=0.2)
            ax.set_ylabel(YLABELS_RAW[key])
            ax.set_xlabel(xlabel)
            if r == 0:
                ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][c])
        axes_pf[r, 0].text(-0.38, 0.5, ROW_TITLES[r],
                           transform=axes_pf[r, 0].transAxes, rotation=90, va="center", ha="center")
    handles, labels = [], []
    seen: set[str] = set()
    for ax in fig_pf.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig_pf.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5),
                  frameon=False, bbox_to_anchor=(0.5, 0.998), fontsize=16)
    fig_pf.suptitle(title, y=1.04, fontsize=16)
    fig_pf.tight_layout(rect=[0.04, 0.02, 0.99, 0.92])
    fig_pf.savefig(fig6 / outfile, bbox_inches="tight")
    plt.close(fig_pf)
    print(f"  Wrote figures/6/{outfile}")


# Individual mode fits + individual postfit plots
if FIT_MODE in ("individual", "both"):
    print("[2/4] Running individual fits …")
    for k, meta in enumerate(loaded_meta):
        Wg = Wg_all[k][np.newaxis, :]
        Ng = Ng_all[k][np.newaxis, :]
        mu_grid_1d = np.linspace(0., MU_SCAN_MAX, max(50, N_SCAN))
        chi2_1d = np.full(len(mu_grid_1d), np.nan)
        theta_at_1d = np.full((len(mu_grid_1d), M), np.nan)
        theta_prev = np.zeros(M, dtype=float)

        for i, mu in enumerate(mu_grid_1d):
            mu_arr = np.array([mu], dtype=float)

            def _obj(th):
                x = np.concatenate([mu_arr, th])
                v, g = _eval_obj_and_grad(x, 1, y, inv_cov, W0, N0, Wg, Ng, dW, dN)
                return v, g[1:]

            res = minimize(_obj, theta_prev, method="L-BFGS-B", bounds=theta_bounds, jac=True)
            chi2_1d[i] = float(res.fun)
            theta_at_1d[i] = res.x
            theta_prev = res.x

        i_best = int(np.nanargmin(chi2_1d))
        mu_ul = _extract_ul(mu_grid_1d, chi2_1d, i_best)
        mu_b = float(mu_grid_1d[i_best])
        th_b = theta_at_1d[i_best]
        print(f"  {meta['mode']}: mu_best={mu_b:.3f}, UL95={mu_ul:.3f}, BR95={100*mu_ul*BR_REF:.3f}%")

        fit_results["individual"][meta["mode"]] = {
            "mu_best": mu_b,
            "mu_ul_95": float(mu_ul) if np.isfinite(mu_ul) else None,
            "br_best_pct": 100. * mu_b * BR_REF,
            "br_ul_95_pct": 100. * float(mu_ul) * BR_REF if np.isfinite(mu_ul) else None,
            "chi2_min": float(chi2_1d[i_best]),
            "theta_best": th_b.tolist(),
            "profile_scan": {
                "mu_grid": mu_grid_1d.tolist(),
                "chi2_grid": chi2_1d.tolist(),
            },
        }
        mu_vec = np.zeros(K, dtype=float)
        mu_vec[k] = mu_b
        _postfit_plot(
            mu_vec=mu_vec,
            theta_vec=th_b,
            title=rf"{meta['label']} (individual fit): $\mathcal{{B}}^{{\mathrm{{best\ fit}}}}_{{\mathrm{{gap}}}}={100. * mu_b * BR_REF:.3f}\%$, $\mathcal{{B}}^{{\mathrm{{UL95}}}}_{{\mathrm{{gap}}}}={100. * mu_ul * BR_REF:.3f}\%$",
            outfile=f"postfit_individual_{meta['mode']}.png",
            combine_gap=False,
            primary_meta=meta,
        )

# ── Unconstrained fit ────────────────────────────────────────────────────────
# All mu_k >= 0 float freely; profile each mode individually.
# Per-mode ULs are always >= individual single-mode ULs (shared minimum is lower).
if FIT_MODE in ("unconstrained", "both"):
    print("[3/4] Running unconstrained fit …")
    mu_hi_unc = MU_SCAN_MAX + 2.0
    unc_bounds = [(0., mu_hi_unc)] * K + list(theta_bounds)

    res_unc = minimize(
        lambda x: _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN),
        np.zeros(K + M), method="L-BFGS-B", jac=True, bounds=unc_bounds,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    mu_k_unc  = res_unc.x[:K].copy()
    theta_unc = res_unc.x[K:].copy()
    chi2_unc  = float(res_unc.fun)
    print(f"  chi2_min={chi2_unc:.4f}  mu_k={np.round(mu_k_unc, 3)}")

    n_prof_unc = max(80, N_SCAN // 2)
    mu_other_idx_list = [[j for j in range(K) if j != k] for k in range(K)]
    per_mode_unc = {}
    t0_unc = time.monotonic()
    for k_mode, meta in enumerate(loaded_meta):
        mu_star  = float(mu_k_unc[k_mode])
        mu_k_grid = np.linspace(0.0, mu_hi_unc, n_prof_unc)
        i_k_best  = int(np.argmin(np.abs(mu_k_grid - mu_star)))
        mu_k_grid[i_k_best] = mu_star
        mu_oi = mu_other_idx_list[k_mode]
        z_bnd = [(0., mu_hi_unc)] * (K - 1) + list(theta_bounds)
        chi2_k = np.full(len(mu_k_grid), np.nan)
        z_prev = np.concatenate([mu_k_unc[mu_oi], theta_unc])
        for ii, mu_fix in enumerate(mu_k_grid):
            def _obj_unc(z, _k=k_mode, _oi=mu_oi, _mf=mu_fix):
                mu_other = z[:K - 1]; theta_z = z[K - 1:]
                x = np.empty(K + M); x[_k] = _mf; x[_oi] = mu_other; x[K:] = theta_z
                val, grad = _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN)
                return val, np.concatenate([grad[_oi], grad[K:]])
            res_k = minimize(_obj_unc, z_prev, method="L-BFGS-B", jac=True, bounds=z_bnd,
                             options={"maxiter": 500})
            chi2_k[ii] = float(res_k.fun)
            z_prev = res_k.x
        i_prof_best = int(np.nanargmin(chi2_k))
        mu_ul_unc = _extract_ul(mu_k_grid, chi2_k, i_prof_best)
        lo, hi, err_lo, err_hi = _extract_interval_at_delta(mu_k_grid, chi2_k, i_prof_best, delta=1.0)
        br      = 100.0 * mu_star * BR_REF
        br_ul   = 100.0 * float(mu_ul_unc) * BR_REF if np.isfinite(mu_ul_unc) else None
        br_lo   = 100.0 * err_lo * BR_REF if np.isfinite(err_lo) else None
        br_hi   = 100.0 * err_hi * BR_REF if np.isfinite(err_hi) else None
        per_mode_unc[meta["mode"]] = {
            "mu_k": mu_star, "mu_k_ul_95": float(mu_ul_unc) if np.isfinite(mu_ul_unc) else None,
            "mu_k_err_lo_1sigma": float(err_lo) if np.isfinite(err_lo) else None,
            "mu_k_err_hi_1sigma": float(err_hi) if np.isfinite(err_hi) else None,
            "br_k_pct": br, "br_k_ul_95_pct": br_ul,
            "br_k_err_lo_1sigma_pct": float(br_lo) if br_lo is not None else None,
            "br_k_err_hi_1sigma_pct": float(br_hi) if br_hi is not None else None,
            "profile_scan_mu_k": {"mu_grid": mu_k_grid.tolist(), "chi2_grid": chi2_k.tolist()},
        }
        print(f"  {meta['mode']}: mu_best={mu_star:.3f}  UL95={mu_ul_unc:.3f}  BR_UL={100*mu_ul_unc*BR_REF:.3f}%")

    dt_unc = time.monotonic() - t0_unc
    print(f"  unconstrained profiling done in {dt_unc:.1f}s")

    # Bayesian ULs via MCMC (flat priors: mu_k >= 0, theta within bounds)
    bayes_uls: dict = {}
    if _EMCEE_AVAILABLE:
        ndim_mc = K + M
        nw = int(MCMC_N_WALKERS) if MCMC_N_WALKERS is not None else max(128, 4 * ndim_mc)
        if nw % 2 != 0:
            nw += 1
        th_lo_mc = np.array([b[0] for b in theta_bounds])
        th_hi_mc = np.array([b[1] for b in theta_bounds])
        def _log_prob(x):
            mu = x[:K]; th = x[K:]
            if np.any(mu < 0.) or np.any(mu > mu_hi_unc): return -np.inf
            if np.any(th < th_lo_mc) or np.any(th > th_hi_mc): return -np.inf
            chi2_mc, _ = _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN)
            return -np.inf if chi2_mc >= 1e29 else -0.5 * chi2_mc
        rng_mc = np.random.default_rng(42)
        p0_mc = np.concatenate([mu_k_unc, theta_unc]) + rng_mc.normal(0., 1e-3, size=(nw, ndim_mc))
        p0_mc[:, :K] = np.clip(p0_mc[:, :K], 0., mu_hi_unc)
        p0_mc[:, K:] = np.clip(p0_mc[:, K:], th_lo_mc, th_hi_mc)
        print(f"  MCMC: {nw} walkers, {MCMC_N_BURN} burn + {MCMC_N_STEPS} steps …", flush=True)
        t0_mc = time.monotonic()
        sampler_mc = _emcee.EnsembleSampler(nw, ndim_mc, _log_prob)
        sampler_mc.run_mcmc(p0_mc, MCMC_N_BURN + MCMC_N_STEPS, progress=False)
        flat_mc = sampler_mc.get_chain(discard=MCMC_N_BURN, flat=True)
        print(f"  MCMC done in {time.monotonic()-t0_mc:.1f}s  ({flat_mc.shape[0]} samples)", flush=True)
        mu_total_samples = flat_mc[:, :K].sum(axis=1)
        bayes_total_ul = float(np.percentile(mu_total_samples, 95.))
        print(f"  total: bayes_UL={bayes_total_ul:.3f}  BR_bayes={100*bayes_total_ul*BR_REF:.3f}%")
        for k_mode, meta in enumerate(loaded_meta):
            ul_b = float(np.percentile(flat_mc[:, k_mode], 95.))
            bayes_uls[meta["mode"]] = ul_b
            entry = per_mode_unc[meta["mode"]]
            entry["mu_k_bayes_ul_95"] = ul_b
            entry["br_k_bayes_ul_95_pct"] = 100.0 * ul_b * BR_REF
            prof_ul = entry["mu_k_ul_95"]
            prof_ul_str = f"{prof_ul:.3f}" if prof_ul is not None else "nan"
            print(f"  {meta['mode']}: prof_UL={prof_ul_str}  bayes_UL={ul_b:.3f}  "
                  f"BR_bayes={100*ul_b*BR_REF:.3f}%")
    else:
        bayes_total_ul = None
        print("  emcee not available, skipping Bayesian ULs (pip install emcee)")

    # Profile scan of mu_total = sum(mu_k) with all mu_k >= 0 free (SLSQP equality constraint)
    mu_total_best = float(np.sum(mu_k_unc))
    mu_total_hi = mu_total_best + mu_hi_unc
    mu_total_grid = np.linspace(0.0, mu_total_hi, max(100, N_SCAN))
    i_tot_best = int(np.argmin(np.abs(mu_total_grid - mu_total_best)))
    mu_total_grid[i_tot_best] = mu_total_best
    chi2_tot = np.full(len(mu_total_grid), np.nan)
    mu_k_prev = mu_k_unc.copy()
    theta_prev_tot = theta_unc.copy()
    t0_tot = time.monotonic()
    for ii, mu_T in enumerate(mu_total_grid):
        if mu_T == 0.:
            def _obj_tot0(th):
                vv, gg = _eval_obj_and_grad(np.concatenate([np.zeros(K), th]), K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN)
                return vv, gg[K:]
            res_t = minimize(_obj_tot0, theta_prev_tot, method="L-BFGS-B", bounds=theta_bounds, jac=True)
            chi2_tot[ii] = float(res_t.fun)
            theta_prev_tot = res_t.x
            mu_k_prev = np.zeros(K)
            continue
        s_prev = float(np.sum(mu_k_prev))
        mu_init = np.clip(mu_k_prev * mu_T / s_prev, 0., None) if s_prev > 0 else np.full(K, mu_T / K)
        x0 = np.concatenate([mu_init, theta_prev_tot])
        con = {"type": "eq", "fun": lambda x: float(np.sum(x[:K])) - mu_T,
               "jac": lambda x: np.concatenate([np.ones(K), np.zeros(M)])}
        res_t = minimize(
            lambda x: _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN),
            x0, method="SLSQP", jac=True, bounds=[(0., None)] * K + theta_bounds,
            constraints=[con], options={"ftol": 1e-10, "maxiter": 500},
        )
        chi2_tot[ii] = float(res_t.fun)
        mu_k_prev = res_t.x[:K].copy()
        theta_prev_tot = res_t.x[K:].copy()
    mu_total_ul = _extract_ul(mu_total_grid, chi2_tot, i_tot_best)
    print(f"  total: mu_best={mu_total_best:.3f}  UL95={mu_total_ul:.3f}  BR_UL={100*mu_total_ul*BR_REF:.3f}%  ({time.monotonic()-t0_tot:.1f}s)")

    fig_tot, ax_tot = plt.subplots(figsize=(8, 5), dpi=150)
    ax_tot.plot(mu_total_grid * BR_REF * 100, chi2_tot - chi2_unc, color="navy", lw=1.8)
    ax_tot.axhline(DELTA_CHI2, color="red", ls="--", lw=1.2, label=rf"$\Delta\chi^2={DELTA_CHI2}$")
    if np.isfinite(mu_total_ul):
        ax_tot.axvline(mu_total_ul * BR_REF * 100, color="red", ls=":", lw=1.2,
                       label=rf"UL95$={100*mu_total_ul*BR_REF:.3f}\%$")
    ax_tot.axvline(mu_total_best * BR_REF * 100, color="green", ls=":", lw=1.2,
                   label=rf"Best$={100*mu_total_best*BR_REF:.3f}\%$")
    ax_tot.set_xlabel(r"$\mathcal{B}_{\rm gap}^{\rm total}$ [%]")
    ax_tot.set_ylabel(r"$\Delta\chi^2$")
    ax_tot.set_ylim(bottom=-0.5)
    ax_tot.legend(fontsize=14, frameon=False)
    ax_tot.grid(alpha=0.25)
    fig_tot.suptitle("Profile scan: total gap-mode BR (unconstrained fit)", fontsize=14)
    fig_tot.tight_layout()
    fig_tot.savefig(fig6 / "profile_scan_total_unc.png")
    plt.close(fig_tot)
    print("  Wrote figures/6/profile_scan_total_unc.png")

    fit_results["unconstrained"] = {
        "chi2_min": chi2_unc,
        "mu_k_best": mu_k_unc.tolist(),
        "theta_best": theta_unc.tolist(),
        "per_mode": per_mode_unc,
        "total_profile_scan": {
            "mu_total_best": mu_total_best,
            "mu_total_ul_95": float(mu_total_ul) if np.isfinite(mu_total_ul) else None,
            "br_total_best_pct": 100.0 * mu_total_best * BR_REF,
            "br_total_ul_95_pct": 100.0 * float(mu_total_ul) * BR_REF if np.isfinite(mu_total_ul) else None,
            "mu_total_bayes_ul_95": bayes_total_ul,
            "br_total_bayes_ul_95_pct": 100.0 * bayes_total_ul * BR_REF if bayes_total_ul is not None else None,
            "mu_grid": mu_total_grid.tolist(),
            "chi2_grid": chi2_tot.tolist(),
        },
        "top_pulls": sorted(
            [{"name": nuisance_names[i], "theta": float(theta_unc[i])} for i in range(M)],
            key=lambda d: abs(d["theta"]), reverse=True,
        )[:10],
    }

    _postfit_plot(
        mu_vec=mu_k_unc, theta_vec=theta_unc,
        title="Unconstrained fit (all modes free)",
        outfile="postfit_unconstrained.png",
        combine_gap=True,
    )

# ── Save results ──────────────────────────────────────────────────────────────

print("[4/4] Writing output/6/fit_results.json …")
payload = {
    "settings": {
        "fit_mode_config": FIT_MODE, "active_modes": ACTIVE_MODES,
        "cov_source_used": "average_raw_cov_from_step4",
        "include_cocktail_dsk": INCLUDE_DSK_IN_SM,
        "el_cut_min": EL_PRECUT, "delta_chi2": DELTA_CHI2,
        "mu_scan_max": MU_SCAN_MAX, "n_scan": N_SCAN, "br_ref": BR_REF,
        "n_br_nuisance": n_br, "n_ff_nuisance": M - n_br,
        "nuisance_names": nuisance_names,
    },
    "results": fit_results,
}
(out6 / "fit_results.json").write_text(json.dumps(payload, indent=2))
if fit_results["individual"]:
    _write_individual_results_tex(out6 / "fit_results_individual.tex", fit_results["individual"], loaded_meta)
    print("  Wrote output/6/fit_results_individual.tex")
if "unconstrained" in fit_results:
    _write_unconstrained_results_tex(out6 / "fit_results_unconstrained.tex", fit_results["unconstrained"])
    print("  Wrote output/6/fit_results_unconstrained.tex")
print("Done.")
