#!/usr/bin/env python3
"""Step 6: Fit gap mode signal strengths to experimental raw-moment average.

Reads:  output/4/experimental_average.json  (data points + errors)
        output/3/cocktail.parquet            (SM templates + nuisance templates)
        output/3/{mode}.parquet              (gap mode templates, one per mode)

Outputs:
  output/6/fit_results.json
  output/6/fit_results_individual.tex
  output/6/fit_results_combined.tex
  output/6/fit_results_combined_top_pulls.tex
  figures/6/profile_scan.png
  figures/6/postfit_individual_{mode}.png  (one per gap mode)
  figures/6/postfit_combined.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import plothist

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

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "lib"))

from lib.gap_modes import GAP_MODES, GAP_MODES_BY_NAME

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
BR_REF          = float(fc.get("br_ref", 0.018))

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


def _nearest_psd(cov, eps=1e-12):
    w, v = np.linalg.eigh(cov); wc = np.clip(w, eps, None)
    out  = (v * wc) @ v.T; return out, bool(np.any(wc != w))


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


def _fmt_pm(lo, hi, nd=3):
    slo = f"{lo:.{nd}f}" if np.isfinite(lo) else "n/a"
    shi = f"{hi:.{nd}f}" if np.isfinite(hi) else "n/a"
    return f"-{slo}/+{shi}"


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


def _write_combined_results_tex(path, sim_result):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Quantity & Value & Notes \\",
        r"\hline",
        rf"$\hat{{\mu}}_{{\mathrm{{total}}}}$ & {sim_result['mu_total_best']:.3f} & best fit \\",
        rf"$\mu_{{95\%}}$ & {(sim_result['mu_total_ul_95'] if sim_result['mu_total_ul_95'] is not None else float('nan')):.3f} & profile UL \\",
        rf"$\hat{{\mathcal{{B}}}}_{{\mathrm{{total}}}}[\%]$ & {sim_result['br_total_best_pct']:.3f} & at best fit \\",
        rf"$\mathcal{{B}}_{{95\%}}[\%]$ & {(sim_result['br_total_ul_95_pct'] if sim_result['br_total_ul_95_pct'] is not None else float('nan')):.3f} & from $\mu_{{95\%}}$ \\",
        r"\hline",
        r"\end{tabular}",
        "",
        r"\vspace{0.5em}",
        "",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Mode & $\hat{\mu}_k$ & $\hat{\mathcal{B}}_k[\%]$ & $1\sigma$ on $\mu_k$ \\",
        r"\hline",
    ]
    for mode, entry in sim_result["per_mode"].items():
        elo = entry.get("mu_k_err_lo_1sigma")
        ehi = entry.get("mu_k_err_hi_1sigma")
        pm = _fmt_pm(elo if elo is not None else np.nan, ehi if ehi is not None else np.nan)
        lines.append(
            rf"{_safe_tex(mode)} & {entry['mu_k']:.3f} & {entry['br_k_pct']:.3f} & ${pm}$ \\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def _write_top_pulls_tex(path, top_pulls):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Nuisance & Pull $\theta$ & $|\theta|$ \\",
        r"\hline",
    ]
    for row in top_pulls:
        th = float(row["theta"])
        lines.append(rf"{_safe_tex(row['name'])} & {th:.3f} & {abs(th):.3f} \\")
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
        return None
    cov_full = np.asarray(cov_list, dtype=float)
    if cov_full.ndim != 2 or cov_full.shape[0] != cov_full.shape[1] or cov_full.shape[0] != len(pt_defs):
        return None
    idx_map = {
        _point_key(p.get("key"), p.get("cut")): i
        for i, p in enumerate(pt_defs)
    }
    want = []
    for key, i_cut in pts:
        cut = cuts_by_key[key][i_cut]
        k = _point_key(key, cut)
        if k not in idx_map:
            return None
        want.append(idx_map[k])
    want = np.asarray(want, dtype=int)
    return cov_full[np.ix_(want, want)]


cov = _cov_from_average_payload(avg_payload)
if cov is None:
    raise RuntimeError(
        "average_raw_cov missing or not aligned with fit points. "
        "Run 4_average.py first and ensure output/4/experimental_average.json contains aligned average_raw_cov."
    )

cov, _psd_fix = _nearest_psd(cov)
inv_cov = np.linalg.pinv(cov, rcond=1e-12)

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

nuisance_names, nuisance_types, dW_cols, dN_cols = [], [], [], []
dw_cols = []
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
Wg_list, Ng_list, loaded_meta = [], [], []
for meta in active_meta:
    pq = Path(cfg["paths"]["output"]) / "3" / f"{meta['mode']}.parquet"
    if not pq.exists():
        print(f"  warning: {pq} not found — skipping {meta['mode']}"); continue
    gdf = pd.read_parquet(pq)
    if "total_weight" not in gdf.columns:
        gdf["total_weight"] = gdf["weight"] * gdf["mc_weight"] * gdf["ff_weight"]
    mx2_g = np.square(gdf["Mx"].to_numpy(dtype=float))
    Wg, Ng = _compute_WN(mx2_g, gdf["El_B"].to_numpy(dtype=float),
                         gdf["q2"].to_numpy(dtype=float),
                         gdf["total_weight"].to_numpy(dtype=float), pts, cuts_by_key)
    Wg_list.append(Wg); Ng_list.append(Ng); loaded_meta.append(meta)

if not loaded_meta:
    raise RuntimeError("No gap mode parquet files found.")

K      = len(loaded_meta)
Wg_all = np.stack(Wg_list, axis=0)   # (K, P)
Ng_all = np.stack(Ng_list, axis=0)
theta_bounds = (br_theta_bounds + [(-10., 10.)] * (M - n_br))

# ── Run fits ──────────────────────────────────────────────────────────────────

print(f"[2/5] Running both fit modes (config mode={FIT_MODE}) …")
mu_grid = np.linspace(0., MU_SCAN_MAX, max(50, N_SCAN))

fit_results: dict = {"individual": {}, "simultaneous": {}}

_GRID_6 = [
    ("mx", ["mx_1", "mx_2", "mx_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("el", ["el_1", "el_2", "el_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("q2", ["q2_1", "q2_2", "q2_3"], r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$"),
]


def _postfit_plot(mu_vec, theta_vec, title, outfile, combine_gap=False, primary_meta=None):
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

            if combine_gap:
                # Additive contribution of the fitted gap in moment space.
                y_with_gap = np.divide(N0k + gap_Nk, W0k + gap_Wk, out=np.full_like(N0k, np.nan), where=(W0k + gap_Wk) > 0)
                y_gr = y_with_gap - y_sm
                gr_label = r"Added fitted gap contribution $\Delta_{\mathrm{gap}}$"
                gr_color, gr_ls = "#9b59b6", "-"
            else:
                k0 = int(np.argmax(mu_vec))
                W_ref = Wg_all[k0][pi]
                N_ref = Ng_all[k0][pi]
                y_gr = np.divide(N_ref, W0k + W_ref, out=np.full_like(N0k, np.nan), where=(W0k + W_ref) > 0)
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
                ax.fill_between(cuts_pt, 0.0, y_gr, color=gr_color, alpha=0.7, zorder=3,
                                label=gr_label if (r == c == 0) else None)
            else:
                ax.plot(cuts_pt, y_gr, color=gr_color, lw=1.5, ls=gr_ls, zorder=6,
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
print("[3/5] Running individual fits …")
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

# Simultaneous fit + combined postfit plot
print("[4/5] Running simultaneous fit and profiling per-mode uncertainties …")
chi2_prof = np.full(len(mu_grid), np.nan)
muk_at = np.full((len(mu_grid), K), np.nan)
theta_at = np.full((len(mu_grid), M), np.nan)
mu_prev = np.zeros(K, dtype=float)
theta_prev = np.zeros(M, dtype=float)
t0 = time.monotonic()

for i, mu_T in enumerate(mu_grid):
    if mu_T == 0.:
        def _obj0(th):
            vv, gg = _eval_obj_and_grad(np.concatenate([np.zeros(K), th]), K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN)
            return vv, gg[K:]
        res = minimize(_obj0, theta_prev, method="L-BFGS-B", bounds=theta_bounds, jac=True)
        chi2_prof[i] = float(res.fun)
        muk_at[i] = 0.
        theta_at[i] = res.x
        theta_prev = res.x
        continue

    mu_bounds = [(0., float(mu_T))] * K
    s_prev = float(np.sum(mu_prev))
    mu_init = np.clip(mu_prev * mu_T / s_prev, 0., mu_T) if s_prev > 0 else np.full(K, mu_T / K)
    x0_sl = np.concatenate([mu_init, theta_prev])
    constraint = {
        "type": "eq",
        "fun": lambda x: float(np.sum(x[:K])) - mu_T,
        "jac": lambda x: np.concatenate([np.ones(K), np.zeros(M)]),
    }
    res = minimize(lambda x: _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN),
                   x0_sl, method="SLSQP", jac=True, bounds=mu_bounds + theta_bounds,
                   constraints=[constraint], options={"ftol": 1e-10, "maxiter": 500})
    chi2_prof[i] = float(res.fun)
    muk_at[i] = res.x[:K]
    theta_at[i] = res.x[K:]
    mu_prev = res.x[:K].copy()
    theta_prev = res.x[K:].copy()
    if (i + 1) % max(1, len(mu_grid) // 10) == 0:
        dt = time.monotonic() - t0
        print(f"  scan {i+1}/{len(mu_grid)} mu_T={mu_T:.3f} ({dt:.1f}s)", flush=True)

i_best = int(np.nanargmin(chi2_prof))
mu_ul = _extract_ul(mu_grid, chi2_prof, i_best)
mu_k_best = muk_at[i_best]
theta_best = theta_at[i_best]
chi2_min = float(chi2_prof[i_best])

per_mode_out = {}
mu_hi = max(MU_SCAN_MAX, float(np.max(mu_k_best) + 2.0))
n_prof = max(80, N_SCAN // 2)
for k_mode, meta in enumerate(loaded_meta):
    mu_star = float(mu_k_best[k_mode])
    mu_k_grid = np.linspace(0.0, mu_hi, n_prof)
    i_k_best = int(np.argmin(np.abs(mu_k_grid - mu_star)))
    mu_k_grid[i_k_best] = mu_star
    mu_other_idx = [j for j in range(K) if j != k_mode]
    z0 = np.concatenate([mu_k_best[mu_other_idx], theta_best])
    z_bounds = [(0.0, mu_hi)] * (K - 1) + theta_bounds
    chi2_k = np.full(len(mu_k_grid), np.nan)
    z_prev = z0.copy()
    for ii, mu_fix in enumerate(mu_k_grid):
        def _obj_mode(z):
            mu_other = z[:K - 1]
            theta_z = z[K - 1:]
            x = np.empty(K + M, dtype=float)
            x[k_mode] = mu_fix
            x[mu_other_idx] = mu_other
            x[K:] = theta_z
            val, grad = _eval_obj_and_grad(x, K, y, inv_cov, W0, N0, Wg_all, Ng_all, dW, dN)
            g = np.concatenate([grad[mu_other_idx], grad[K:]])
            return val, g
        res_k = minimize(_obj_mode, z_prev, method="L-BFGS-B", jac=True, bounds=z_bounds, options={"maxiter": 500})
        chi2_k[ii] = float(res_k.fun)
        z_prev = res_k.x
    i_prof_best = int(np.nanargmin(chi2_k))
    lo, hi, err_lo, err_hi = _extract_interval_at_delta(mu_k_grid, chi2_k, i_prof_best, delta=1.0)
    br = 100.0 * mu_star * BR_REF
    br_lo = 100.0 * err_lo * BR_REF if np.isfinite(err_lo) else np.nan
    br_hi = 100.0 * err_hi * BR_REF if np.isfinite(err_hi) else np.nan
    per_mode_out[meta["mode"]] = {
        "mu_k": mu_star,
        "mu_k_err_lo_1sigma": float(err_lo) if np.isfinite(err_lo) else None,
        "mu_k_err_hi_1sigma": float(err_hi) if np.isfinite(err_hi) else None,
        "br_k_pct": br,
        "br_k_err_lo_1sigma_pct": float(br_lo) if np.isfinite(br_lo) else None,
        "br_k_err_hi_1sigma_pct": float(br_hi) if np.isfinite(br_hi) else None,
        "profile_scan_mu_k": {
            "mu_grid": mu_k_grid.tolist(),
            "chi2_grid": chi2_k.tolist(),
        },
    }

fit_results["simultaneous"] = {
    "mu_total_best": float(mu_grid[i_best]),
    "mu_total_ul_95": float(mu_ul) if np.isfinite(mu_ul) else None,
    "br_total_best_pct": 100. * float(mu_grid[i_best]) * BR_REF,
    "br_total_ul_95_pct": 100. * float(mu_ul) * BR_REF if np.isfinite(mu_ul) else None,
    "chi2_min": chi2_min,
    "mu_k_best": mu_k_best.tolist(),
    "theta_best": theta_best.tolist(),
    "profile_scan": {
        "mu_total_grid": mu_grid.tolist(),
        "chi2_grid": chi2_prof.tolist(),
    },
    "per_mode": per_mode_out,
    "nuisance_names": nuisance_names,
    "top_pulls": sorted(
        [{"name": nuisance_names[i], "theta": float(theta_best[i])} for i in range(M)],
        key=lambda d: abs(d["theta"]), reverse=True
    )[:10],
}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(mu_grid, chi2_prof - chi2_prof[i_best], color="navy", lw=1.8)
ax.axhline(DELTA_CHI2, color="red", ls="--", lw=1.2, label=rf"$\Delta\chi^2={DELTA_CHI2}$")
if np.isfinite(mu_ul):
    ax.axvline(mu_ul, color="red", ls=":", lw=1.2, label=f"UL95={mu_ul:.3f}")
ax.axvline(mu_grid[i_best], color="green", ls=":", lw=1.2, label=f"Best={mu_grid[i_best]:.3f}")
ax.set_xlabel(r"$\mu_{\rm total}$ (signal strength at $\mathcal{B}_{\rm ref}$)")
ax.set_ylabel(r"$\Delta\chi^2$")
ax.set_ylim(bottom=-0.5)
ax.legend(fontsize=16, frameon=False)
ax.grid(alpha=0.25)
fig.suptitle("Profile scan: simultaneous gap-mode fit", fontsize=16)
fig.tight_layout()
fig.savefig(fig6 / "profile_scan.png")
plt.close(fig)
print("  Wrote figures/6/profile_scan.png")

_postfit_plot(
    mu_vec=mu_k_best,
    theta_vec=theta_best,
    title=rf"Combined fit: $\mathcal{{B}}^{{\mathrm{{best\ fit}}}}_{{\mathrm{{gap}}}}={100. * float(mu_grid[i_best]) * BR_REF:.3f}\%$, $\mathcal{{B}}^{{\mathrm{{UL95}}}}_{{\mathrm{{gap}}}}={100. * mu_ul * BR_REF:.3f}\%$",
    outfile="postfit_combined.png",
    combine_gap=True,
)


# ── Save results ──────────────────────────────────────────────────────────────

print("[5/5] Writing output/6/fit_results.json …")
payload = {
    "settings": {
        "fit_mode_config": FIT_MODE, "fit_mode_executed": "both", "active_modes": ACTIVE_MODES,
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
_write_individual_results_tex(out6 / "fit_results_individual.tex", fit_results["individual"], loaded_meta)
_write_combined_results_tex(out6 / "fit_results_combined.tex", fit_results["simultaneous"])
_write_top_pulls_tex(out6 / "fit_results_combined_top_pulls.tex", fit_results["simultaneous"].get("top_pulls", []))
print("  Wrote output/6/fit_results_individual.tex")
print("  Wrote output/6/fit_results_combined.tex")
print("  Wrote output/6/fit_results_combined_top_pulls.tex")
print("Done.")
