#!/usr/bin/env python3
"""Step 4: GLS experimental average of raw and central moments.

Reads experimental data from inputs/sem.tar.gz, computes weighted averages
with configurable correlation assumptions, and optionally inflates errors when
chi2/dof > 1.

Outputs:
  output/4/experimental_average.json  (contains raw + central averages)
  figures/4/experimental_average_3x3.png
"""
import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import chi2 as chi2_dist
import plothist

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "lib"))

from lib.moments import EXP_STYLE, YLABELS, build_exp_moments_df, df_to_plot_dict, _jacobian

out4 = Path(cfg["paths"]["output"]) / "4"
fig4 = Path(cfg["paths"]["figures"]) / "4"
out4.mkdir(parents=True, exist_ok=True)
fig4.mkdir(parents=True, exist_ok=True)

av = cfg["average"]
corr_same_exp_same_var = float(av.get("corr_same_exp_same_var", 0.98))
corr_same_exp_diff_var = float(av.get("corr_same_exp_diff_var", 0.3))
corr_diff_exp          = float(av.get("corr_diff_exp", 0.1))
inflate                = bool(av.get("inflate_if_incompatible", True))
average_method         = str(av.get("method", "gls")).strip().lower()  # "gls" or "polynomial"
_pd = av.get("poly_degree", 3)  # int, or {mx:, el:, q2:} for a per-family degree
poly_degree            = {f: int(_pd[f]) for f in ("mx", "el", "q2")} if isinstance(_pd, dict) else int(_pd)
exclude_experiments_cfg = av.get("exclude_experiments", []) or []
exclude_experiments = {str(x).strip().lower() for x in exclude_experiments_cfg if str(x).strip()}
if av.get("exclude_sem", False):
    exclude_experiments |= {"cdf", "delphi"}

RAW_KEYS = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]
YLABELS_RAW = {
    "mx_1": r"$\langle M_X^2 \rangle$",   "mx_2": r"$\langle (M_X^2)^2 \rangle$",
    "mx_3": r"$\langle (M_X^2)^3 \rangle$","el_1": r"$\langle E_\ell \rangle$",
    "el_2": r"$\langle E_\ell^2 \rangle$", "el_3": r"$\langle E_\ell^3 \rangle$",
    "q2_1": r"$\langle q^2 \rangle$",      "q2_2": r"$\langle (q^2)^2 \rangle$",
    "q2_3": r"$\langle (q^2)^3 \rangle$",
}
ROW_TITLES = [r"$M_X^2$ moments", r"$E_\ell$ moments", r"$q^2$ moments"]

def _rounded(x: float) -> float:
    return float(np.round(float(x), 6))


def _parse_cov_h5(path: Path):
    """Parse one *cov*.h5 file into (tags, cov) with tags=[(obs,order,exp,cut), ...]
    restricted to obs in {mx,el,q2} and integer order, aligned with cov's rows/cols."""
    with h5py.File(path, "r") as f:
        g = f[list(f.keys())[0]]
        lvl0 = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in g["axis0_level0"][...]]
        lvl1 = np.asarray(g["axis0_level1"][...], dtype=float)
        lab0 = np.asarray(g["axis0_label0"][...], dtype=int)
        lab1 = np.asarray(g["axis0_label1"][...], dtype=int)
        raw_tags = [(lvl0[lab0[i]], float(lvl1[lab1[i]])) for i in range(len(lab0))]
        cov_full = np.asarray(g["block0_values"][...], dtype=float)
    keep_idx, tags = [], []
    for i, (ti, ci) in enumerate(raw_tags):
        pi = ti.split("_")
        if len(pi) < 3:
            continue
        obs_i, ord_i, exp_i = pi[0], pi[1], pi[2]
        if obs_i not in {"mx", "el", "q2"}:
            continue
        try:
            order_i = int(ord_i)
        except ValueError:
            continue
        if order_i not in (1, 2, 3):
            continue
        keep_idx.append(i)
        tags.append((obs_i, order_i, exp_i, ci))
    cov = cov_full[np.ix_(keep_idx, keep_idx)]
    return tags, cov


def _cov_to_rho_lookup(tags, cov) -> dict:
    """Keys are 6-tuples (exp_i, key_i, cut_i, exp_j, key_j, cut_j) so genuine
    cross-experiment entries (e.g. real Belle x Belle II correlation from their
    combined covariance file) are kept and unambiguous, rather than being dropped
    or collapsed onto a same-experiment-only 5-tuple key."""
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    den = np.outer(sig, sig)
    rho = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    out = {}
    for i, (obs_i, order_i, exp_i, ci) in enumerate(tags):
        key_i = f"{obs_i}_{order_i}"
        for j, (obs_j, order_j, exp_j, cj) in enumerate(tags):
            key_j = f"{obs_j}_{order_j}"
            out[(exp_i, key_i, _rounded(ci), exp_j, key_j, _rounded(cj))] = float(rho[i, j])
    return out


def _derive_raw_cov_from_central(tags, cov_central, exp_central: dict) -> tuple[list, np.ndarray]:
    """Propagate a central-moment covariance matrix to the raw-moment basis via the
    exact delta-method Jacobian (reusing lib.moments._jacobian/central_to_raw's
    convention), block-diagonal per (exp, family, cut) since raw_k at one cut only
    depends on central moments at that same cut. Groups whose orders aren't a contiguous 1..N (N<=3),
    or whose central *value* isn't available in exp_central, are dropped (not passed
    through as an identity/no-op, since that would silently mix untransformed central
    values into a nominally-raw lookup)."""
    from collections import defaultdict
    groups: dict = defaultdict(dict)  # (exp, family, cut) -> {order: idx}
    for i, (obs_i, order_i, exp_i, ci) in enumerate(tags):
        groups[(exp_i, obs_i, _rounded(ci))][order_i] = i

    n = len(tags)
    J_full = np.zeros((n, n))
    keep = np.zeros(n, dtype=bool)
    for (exp_i, obs_i, cut_i), by_order in groups.items():
        n_ord = len(by_order)
        if set(by_order) != set(range(1, n_ord + 1)):
            continue
        entry = exp_central.get(f"{obs_i}_1", {}).get(exp_i)
        if entry is None:
            continue
        cuts = [_rounded(c) for c in entry["cuts"]]
        if cut_i not in cuts:
            continue
        mus = []
        for k in range(1, n_ord + 1):
            e_k = exp_central[f"{obs_i}_{k}"][exp_i]
            mus.append(e_k["values"][[_rounded(c) for c in e_k["cuts"]].index(cut_i)])
        Jb = _jacobian(np.array(mus))
        idx = [by_order[k] for k in range(1, n_ord + 1)]
        for a in range(n_ord):
            keep[idx[a]] = True
            for b in range(n_ord):
                J_full[idx[a], idx[b]] = Jb[a, b]

    cov_raw = J_full @ cov_central @ J_full.T
    kept_tags = [t for t, k in zip(tags, keep) if k]
    kidx = np.flatnonzero(keep)
    cov_raw = cov_raw[np.ix_(kidx, kidx)]
    return kept_tags, cov_raw


def _load_sem_covariance_lookups(data_dir: Path, exp_central: dict) -> tuple[dict, dict]:
    """Load within-experiment correlation lookups from sem covariance h5 files.

    Files with "raw" in their name are genuine raw-moment covariance matrices and are
    used as-is. Files without it (babar/belle/cleo/delphi's own covariance files) were
    verified against the actual reported errors to be CENTRAL-moment covariances
    (their 2nd/3rd-order diagonal matches the central, not raw, errors exactly) --
    for those, derive the raw-moment correlation via the exact delta-method Jacobian
    (see _derive_raw_cov_from_central) instead of leaving those experiments with no
    raw correlation info at all (which silently falls back to the flat config default).

    Returns:
      central_lookup[(exp_i, key_i, cut_i, exp_j, key_j, cut_j)] = rho
      raw_lookup[(exp_i, key_i, cut_i, exp_j, key_j, cut_j)] = rho
    """
    central, raw = {}, {}
    for fp in sorted(data_dir.glob("*cov*.h5")):
        if not fp.exists():
            continue
        name = fp.name.lower()
        try:
            tags, cov = _parse_cov_h5(fp)
        except Exception as e:
            print(f"  warning: failed to parse covariance file {fp.name}: {e}")
            continue
        if "raw" in name:
            raw.update(_cov_to_rho_lookup(tags, cov))
        else:
            central.update(_cov_to_rho_lookup(tags, cov))
            raw_tags, raw_cov = _derive_raw_cov_from_central(tags, cov, exp_central)
            n_dropped = len(tags) - len(raw_tags)
            derived = _cov_to_rho_lookup(raw_tags, raw_cov)
            raw.update(derived)
            print(f"  {fp.name}: derived {len(derived)} raw-moment correlation entries "
                  f"from its central-moment covariance ({n_dropped} tag(s) dropped, "
                  f"no matching central value found)")
    print(f"  loaded sem covariance lookups: central={len(central)}, raw={len(raw)}")
    return central, raw

# ── Load experimental data ────────────────────────────────────────────────────

print("[1/3] Loading experimental data …")
tar_path = Path(cfg["plots"].get("exp_data_tar", "inputs/sem.tar.gz"))
if not tar_path.exists():
    raise FileNotFoundError(f"Experimental data tar not found: {tar_path}")

with tempfile.TemporaryDirectory(prefix="sem_avg_") as tmp:
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(tmp, filter="data")
        except TypeError:
            tf.extractall(tmp)
    exp_df = build_exp_moments_df(Path(tmp))
    exp_central = df_to_plot_dict(exp_df, "measured")
    cov_lookup_central, cov_lookup_raw = _load_sem_covariance_lookups(Path(tmp), exp_central)

exp_raw = df_to_plot_dict(exp_df, "calculated")

def _filter_experiments(exp_dict: dict, excluded: set[str]) -> dict:
    if not excluded:
        return exp_dict
    out: dict = {}
    for key, by_exp in exp_dict.items():
        kept = {exp: entry for exp, entry in by_exp.items() if exp.lower() not in excluded}
        if kept:
            out[key] = kept
    return out

if exclude_experiments:
    exp_raw = _filter_experiments(exp_raw, exclude_experiments)
    exp_central = _filter_experiments(exp_central, exclude_experiments)
    print(f"  excluded experiments: {sorted(exclude_experiments)}")

# Per-family experiment exclusion: drops only the named family's (mx/el/q2)
# measurements for that experiment, leaving its other families untouched --
# unlike exclude_experiments above, which drops an experiment everywhere.
# Motivated by the real, moderate (rho~0.15-0.3) Belle x Belle II q^2 cross-
# correlation not being enough to explain their persistent ~1sigma-per-point,
# highly-correlated offset (raw chi2/dof=999.0/141, p=0.000 with both kept);
# dropping Belle's own q^2 (keeping Belle II's, and keeping Belle's own Mx/El)
# gives chi2/dof=110.4/96, p=0.150, with no loss of MaxEnt convergence downstream
# and no side effects on El's own fit quality -- unlike inflating q^2 errors by
# hand, which "fixes" the p-value but visibly degrades El's own convergence via
# the shared joint-fit cross-family weighting.
exclude_family_experiments = {k: {e.lower() for e in v}
                               for k, v in av.get("exclude_family_experiments", {}).items()}
if exclude_family_experiments:
    for _fam, _excl in exclude_family_experiments.items():
        for _d in (exp_raw, exp_central):
            for _k in (f"{_fam}_1", f"{_fam}_2", f"{_fam}_3", f"{_fam}_4"):
                if _k in _d:
                    _d[_k] = {e: v for e, v in _d[_k].items() if e.lower() not in _excl}
        print(f"  excluded {_fam} entries for experiment(s): {sorted(_excl)}")

# ── Build measurement list and covariance ─────────────────────────────────────

def _build_joint_points_and_cov(
    exp_dict: dict,
    label: str,
    exp_cov_lookup: dict[tuple[str, str, float, str, float], float],
):
    """Collect every (experiment, key, cut) measurement into one joint vector and
    build its full n x n covariance (within- and cross-experiment, cross-key),
    using exp_cov_lookup overrides where available and the configured default
    correlation assumptions elsewhere. Shared by _gls_average and _poly_average
    so both methods start from the exact same input covariance."""
    pts: list[dict] = []
    truth_set: set[tuple[str, float]] = set()
    for key in RAW_KEYS:
        for exp, entry in exp_dict.get(key, {}).items():
            for c, v, e in zip(entry["cuts"], entry["values"], entry["errors"]):
                c_, v_, e_ = float(c), float(v), float(e)
                if not (np.isfinite(c_) and np.isfinite(v_) and np.isfinite(e_) and e_ > 0):
                    continue
                pts.append({"key": key, "family": key.split("_")[0], "exp": exp,
                            "cut": c_, "value": v_, "error": e_})
                truth_set.add((key, c_))
    if not pts:
        raise RuntimeError(f"No valid experimental {label} points found.")

    truth_pts = sorted(truth_set, key=lambda x: (RAW_KEYS.index(x[0]), x[1]))
    n = len(pts)
    s = np.array([p["error"] for p in pts], dtype=float)
    y = np.array([p["value"] for p in pts], dtype=float)
    cov = np.diag(s**2).astype(float)
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            if pi["exp"] == pj["exp"]:
                rho = corr_same_exp_same_var if pi["family"] == pj["family"] else corr_same_exp_diff_var
            else:
                rho = corr_diff_exp
            # Keyed by both experiment names now, so a genuine cross-experiment entry
            # (e.g. the real Belle x Belle II q^2 correlation from their combined
            # covariance file) can be found and used instead of the flat config default,
            # without risk of spuriously matching one side's own internal correlation.
            key_cov = (pi["exp"], pi["key"], _rounded(pi["cut"]), pj["exp"], pj["key"], _rounded(pj["cut"]))
            if key_cov in exp_cov_lookup:
                rho = float(exp_cov_lookup[key_cov])

            cov[i, j] = cov[j, i] = rho * s[i] * s[j]

    evals, evecs = np.linalg.eigh(cov)
    clipped = np.clip(evals, 1e-12, None)
    cov_psd = (evecs * clipped) @ evecs.T
    d = np.sqrt(np.maximum(np.diag(cov_psd), 1e-12))
    cov_psd = cov_psd / np.outer(d, d) * np.outer(s, s)

    return pts, truth_pts, y, s, cov_psd


def _gls_average(
    exp_dict: dict,
    label: str,
    exp_cov_lookup: dict[tuple[str, str, float, str, float], float],
):
    pts, truth_pts, y, s, cov_psd = _build_joint_points_and_cov(exp_dict, label, exp_cov_lookup)
    n, m = len(pts), len(truth_pts)
    idx  = {k: i for i, k in enumerate(truth_pts)}

    A = np.zeros((n, m), dtype=float)
    for i, p in enumerate(pts):
        A[i, idx[(p["key"], p["cut"])]] = 1.0

    cov_inv = np.linalg.pinv(cov_psd, rcond=1e-12)
    AtWi    = A.T @ cov_inv
    fisher  = AtWi @ A
    cov_t   = np.linalg.pinv(fisher, rcond=1e-12)
    t_hat   = cov_t @ (AtWi @ y)
    r       = y - A @ t_hat
    chi2_val = float(r @ cov_inv @ r)
    dof     = max(n - int(np.linalg.matrix_rank(A)), 0)
    pval    = float(1.0 - chi2_dist.cdf(chi2_val, dof)) if dof > 0 else float("nan")
    scale   = 1.0
    if inflate and dof > 0 and chi2_val / dof > 1.0:
        scale = float(np.sqrt(chi2_val / dof))
        cov_t = cov_t * scale**2
    print(f"  [{label}] chi2 = {chi2_val:.1f}, dof = {dof}, p = {pval:.3f}, scale = {scale:.3f}")

    avg_errs = np.sqrt(np.maximum(np.diag(cov_t), 0.0))
    avg: dict[str, dict] = {}
    for i, (key, cut) in enumerate(truth_pts):
        avg.setdefault(key, {"cuts": [], "values": [], "errors": []})
        avg[key]["cuts"].append(float(cut))
        avg[key]["values"].append(float(t_hat[i]))
        avg[key]["errors"].append(float(avg_errs[i]))
    points = [{"key": key, "cut": float(cut)} for key, cut in truth_pts]
    return avg, {"chi2": chi2_val, "dof": dof, "pval": pval, "scale": scale}, {
        "points": points,
        "cov": cov_t.tolist(),
        # One free parameter per (key, cut) truth point -- cov_t is generically full rank
        # (unlike _poly_average's, which is exactly rank-deficient by construction).
        "rank": m,
    }, {
        "truth_pts": truth_pts,
        "t_hat": t_hat,
        "cov_t": cov_t,
    }


def _poly_average(
    exp_dict: dict,
    label: str,
    degree,
    exp_cov_lookup: dict[tuple[str, str, float, str, float], float],
):
    """Joint GLS fit of one polynomial per observable key, all keys fit
    simultaneously against the full cross-experiment/cross-key input covariance
    (same _build_joint_points_and_cov as _gls_average). Unlike fitting each key
    independently against diagonal weights, this yields one joint Fisher matrix
    whose inverse gives the true output covariance across ALL keys and cuts --
    including nonzero cross-key blocks wherever the input data has real
    cross-key correlation, rather than a covariance forced block-diagonal by
    key. Return shape matches _gls_average so callers are agnostic to method."""
    pts, truth_pts, y, s, cov_psd = _build_joint_points_and_cov(exp_dict, label, exp_cov_lookup)
    n, m = len(pts), len(truth_pts)

    # Per-key degree, cut range, and column range within the joint block-diagonal
    # design matrix. Powers are taken of the cut value rescaled to [-1, 1] per key,
    # not the raw cut value: raw-domain Vandermonde matrices are badly conditioned
    # (cond ~1e5 already at degree 4 for a cut range of a few GeV, ~1e16 by degree
    # 12 -- past the float64 noise floor), and combined with the very large GLS
    # precision weights from highly-correlated nested-cut points, that numerical
    # noise gets amplified into spurious chi2 rather than a real shape mismatch.
    keys_present = [k for k in RAW_KEYS if any(p["key"] == k for p in pts)]
    key_deg: dict[str, int] = {}
    key_cols: dict[str, tuple[int, int]] = {}
    key_range: dict[str, tuple[float, float]] = {}
    col = 0
    for key in keys_present:
        x_key = [p["cut"] for p in pts if p["key"] == key]
        n_key = len(x_key)
        deg = min(degree[key.split("_")[0]] if isinstance(degree, dict) else degree, n_key - 1)
        key_deg[key] = deg
        key_cols[key] = (col, col + deg + 1)
        key_range[key] = (min(x_key), max(x_key))
        col += deg + 1
    p_total = col

    def _scaled(cut: float, key: str) -> float:
        lo, hi = key_range[key]
        return 2.0 * (cut - lo) / (hi - lo) - 1.0 if hi > lo else 0.0

    def _row(cut: float, key: str) -> np.ndarray:
        deg = key_deg[key]
        u = _scaled(cut, key)
        return np.array([u ** k for k in range(deg, -1, -1)], dtype=float)

    A = np.zeros((n, p_total), dtype=float)
    for i, p in enumerate(pts):
        c0, c1 = key_cols[p["key"]]
        A[i, c0:c1] = _row(p["cut"], p["key"])

    cov_inv    = np.linalg.pinv(cov_psd, rcond=1e-12)
    AtWi       = A.T @ cov_inv
    fisher     = AtWi @ A
    fisher_inv = np.linalg.pinv(fisher, rcond=1e-12)
    coeffs     = fisher_inv @ (AtWi @ y)

    r        = y - A @ coeffs
    chi2_val = float(r @ cov_inv @ r)
    dof      = max(n - int(np.linalg.matrix_rank(A)), 0)
    pval     = float(1.0 - chi2_dist.cdf(chi2_val, dof)) if dof > 0 else float("nan")
    scale    = 1.0
    if inflate and dof > 0 and chi2_val / dof > 1.0:
        scale = float(np.sqrt(chi2_val / dof))
    cov_coeffs = fisher_inv * scale**2
    print(f"  [{label}] poly deg={degree} chi2={chi2_val:.1f}, dof={dof}, p={pval:.3f}, scale={scale:.3f}")

    # Evaluate at each truth (key, cut) point, propagating the FULL joint
    # coefficient covariance -- this is what makes cross-key output
    # correlations come out correctly instead of being forced to zero.
    A_eval = np.zeros((m, p_total), dtype=float)
    for i, (key, cut) in enumerate(truth_pts):
        c0, c1 = key_cols[key]
        A_eval[i, c0:c1] = _row(cut, key)
    t_hat = A_eval @ coeffs
    cov_t = A_eval @ cov_coeffs @ A_eval.T

    # poly_coeffs stores coefficients in the *scaled* [-1, 1] basis (x_min, x_max
    # give the affine map back to raw cut units); _plot_average_grid rescales
    # before calling np.polyval so callers never see raw-domain coefficients.
    poly_coeffs: dict[str, tuple] = {}
    for key in keys_present:
        c0, c1 = key_cols[key]
        lo, hi = key_range[key]
        poly_coeffs[key] = (coeffs[c0:c1], key_deg[key], lo, hi)

    avg_errs = np.sqrt(np.maximum(np.diag(cov_t), 0.0))
    avg: dict[str, dict] = {}
    for i, (key, cut) in enumerate(truth_pts):
        avg.setdefault(key, {"cuts": [], "values": [], "errors": []})
        avg[key]["cuts"].append(float(cut))
        avg[key]["values"].append(float(t_hat[i]))
        avg[key]["errors"].append(float(avg_errs[i]))
    points = [{"key": key, "cut": float(cut)} for key, cut in truth_pts]
    return avg, {"chi2": chi2_val, "dof": dof, "pval": pval, "scale": scale}, {
        "points": points,
        "cov": cov_t.tolist(),
        # cov_t = A_eval @ cov_coeffs @ A_eval.T with cov_coeffs only p_total x p_total,
        # so cov_t is EXACTLY rank <= p_total regardless of how many output points m it
        # spans -- everything past the top p_total eigenvalues is float64 roundoff, not
        # small-but-real information. Consumers must invert against this exact rank
        # (e.g. a rank-truncated eigendecomposition), not a magnitude-based rcond, or
        # they will amplify that roundoff into huge spurious precision weights.
        "rank": p_total,
    }, {
        "truth_pts": truth_pts,
        "t_hat": t_hat,
        "cov_t": cov_t,
        "poly_coeffs": poly_coeffs,
    }


def _convert_central_to_raw(avg_central: dict, central_cov_payload: dict, central_model: dict):
    """Propagate averaged central moments to raw moments using Jacobian."""
    truth_pts = list(central_model["truth_pts"])
    t_hat = np.asarray(central_model["t_hat"], dtype=float)
    cov_t = np.asarray(central_model["cov_t"], dtype=float)
    idx = {(k, float(c)): i for i, (k, c) in enumerate(truth_pts)}

    fams = ("mx", "el", "q2")
    cuts_by_fam: dict[str, list[float]] = {}
    for fam in fams:
        c1 = set(avg_central.get(f"{fam}_1", {}).get("cuts", []))
        c2 = set(avg_central.get(f"{fam}_2", {}).get("cuts", []))
        c3 = set(avg_central.get(f"{fam}_3", {}).get("cuts", []))
        cuts_by_fam[fam] = sorted(float(c) for c in (c1 & c2 & c3))

    raw_points: list[tuple[str, float]] = []
    for key in RAW_KEYS:
        fam = key.split("_")[0]
        raw_points.extend((key, c) for c in cuts_by_fam[fam])

    n_raw = len(raw_points)
    n_cen = len(truth_pts)
    J = np.zeros((n_raw, n_cen), dtype=float)
    y_raw = np.zeros(n_raw, dtype=float)

    for r, (key, cut) in enumerate(raw_points):
        fam, order_s = key.split("_")
        order = int(order_s)
        i1 = idx.get((f"{fam}_1", float(cut)))
        i2 = idx.get((f"{fam}_2", float(cut)))
        i3 = idx.get((f"{fam}_3", float(cut)))
        if i1 is None or i2 is None or i3 is None:
            y_raw[r] = np.nan
            continue
        m1 = float(t_hat[i1]); m2 = float(t_hat[i2]); m3 = float(t_hat[i3])

        if order == 1:
            y_raw[r] = m1
            J[r, i1] = 1.0
        elif order == 2:
            y_raw[r] = m2 + m1 * m1
            J[r, i1] = 2.0 * m1
            J[r, i2] = 1.0
        elif order == 3:
            y_raw[r] = m3 + 3.0 * m1 * m2 + m1 ** 3
            J[r, i1] = 3.0 * m2 + 3.0 * m1 * m1
            J[r, i2] = 3.0 * m1
            J[r, i3] = 1.0
        else:
            y_raw[r] = np.nan

    cov_raw = J @ cov_t @ J.T
    err_raw = np.sqrt(np.clip(np.diag(cov_raw), 0.0, None))

    avg_raw: dict[str, dict] = {}
    for r, (key, cut) in enumerate(raw_points):
        avg_raw.setdefault(key, {"cuts": [], "values": [], "errors": []})
        avg_raw[key]["cuts"].append(float(cut))
        avg_raw[key]["values"].append(float(y_raw[r]))
        avg_raw[key]["errors"].append(float(err_raw[r]))

    points = [{"key": k, "cut": float(c)} for (k, c) in raw_points]
    return avg_raw, {"points": points, "cov": cov_raw.tolist()}


print(f"[2/3] Running {average_method} average …")
poly_coeffs_raw: dict = {}
poly_coeffs_central: dict = {}
if average_method == "polynomial":
    avg_central, stats_central, avg_central_cov, central_model = _poly_average(
        exp_central, "central", poly_degree, cov_lookup_central
    )
    poly_coeffs_central = central_model.get("poly_coeffs", {})
    avg_raw, stats_raw, avg_raw_cov, raw_model = _poly_average(
        exp_raw, "raw", poly_degree, cov_lookup_raw
    )
    poly_coeffs_raw = raw_model.get("poly_coeffs", {})
else:
    avg_central, stats_central, avg_central_cov, central_model = _gls_average(
        exp_central, "central", cov_lookup_central
    )
    avg_raw, avg_raw_cov = _convert_central_to_raw(avg_central, avg_central_cov, central_model)
    stats_raw = {
        "note": "propagated_from_central_gls",
        "chi2": stats_central["chi2"],
        "dof": stats_central["dof"],
        "pval": stats_central["pval"],
        "scale": stats_central["scale"],
    }

# ── Plot ──────────────────────────────────────────────────────────────────────

print("[3/3] Plotting …")
GRID_ROWS = [
    (["mx_1", "mx_2", "mx_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    (["el_1", "el_2", "el_3"], r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    (["q2_1", "q2_2", "q2_3"], r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$"),
]
def _plot_average_grid(avg_dict, exp_dict, stats, ylabels, outname, title_prefix, poly_coeffs=None):
    # Designed for a single-column placement (~3.4in final width): drawn at ~3.5x that size with
    # fonts scaled up to match, so text lands at ~12pt once LaTeX shrinks the raster back down to
    # columnwidth -- fonts tuned for a full-width figure* become illegible at this scale. Row/column
    # titles are placed with fig.text() in figure-fraction coordinates derived from each axes'
    # actual on-screen position (via get_position(), read out *after* the layout pass), not a fixed
    # axes-fraction offset -- a fixed offset collides with the tick labels once fonts get this large.
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), dpi=250)
    added: set[str] = set()

    for r, (keys, xlabel) in enumerate(GRID_ROWS):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            if key in avg_dict:
                cuts_a = np.asarray(avg_dict[key]["cuts"])
                vals_a = np.asarray(avg_dict[key]["values"])
                errs_a = np.asarray(avg_dict[key]["errors"])
                if poly_coeffs and key in poly_coeffs:
                    coeffs, deg, x_lo, x_hi = poly_coeffs[key]
                    xf = np.linspace(x_lo, x_hi, 300)
                    uf = 2.0 * (xf - x_lo) / (x_hi - x_lo) - 1.0 if x_hi > x_lo else np.zeros_like(xf)
                    yf = np.polyval(coeffs, uf)
                    ax.plot(xf, yf, color="black", lw=2.2, zorder=10,
                            label="Experimental average" if (r == c == 0) else None)
                    # band stays at the discrete truth points; the curve above is the only smoothed part
                    ax.fill_between(cuts_a, vals_a - errs_a, vals_a + errs_a,
                                    color="black", alpha=0.15, zorder=9,
                                    label=r"Average $\pm1\sigma$" if (r == c == 0) else None)
                else:
                    ax.plot(cuts_a, vals_a, color="black", lw=2.2, zorder=10,
                            label="Experimental average" if (r == c == 0) else None)
                    ax.fill_between(cuts_a, vals_a - errs_a, vals_a + errs_a,
                                    color="black", alpha=0.15, zorder=9,
                                    label=r"Average $\pm1\sigma$" if (r == c == 0) else None)
            for exp, entry in exp_dict.get(key, {}).items():
                st  = EXP_STYLE.get(exp, dict(color="grey", marker="o", label=exp))
                lbl = st["label"] if st["label"] not in added else None
                if lbl:
                    added.add(st["label"])
                ax.errorbar(entry["cuts"], entry["values"], yerr=entry["errors"],
                            ls="", marker=st["marker"], color=st["color"],
                            mec="black", ecolor=st["color"], lw=1.6, ms=7, capsize=2,
                            zorder=20, label=lbl)
            # Only the leftmost column carries the y-axis label -- the moment order is now
            # given in-panel (see below) rather than by a separate ylabel per column, so a
            # bare "n=1/2/3" label would otherwise be redundant across a row. Freeing the
            # middle/right columns of a rotated label lets wspace shrink, giving each panel
            # more room.
            if c == 0:
                ax.set_ylabel(ylabels[key], fontsize=30, labelpad=8)
            ax.set_xlabel(xlabel, fontsize=28, labelpad=8)
            ax.tick_params(axis="both", which="both", direction="in",
                            top=True, right=True, labelsize=22, pad=6)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            for spine in ax.spines.values():
                spine.set_visible(True)
            if r == 0:
                # Mx^2 row: extra vertical padding both above and below the data so the n=k
                # corner label clears the topmost BaBar point and the curve has breathing room.
                ax.margins(y=0.22)
            ax.text(0.06, 0.94, rf"$n={c + 1}$", transform=ax.transAxes,
                    fontsize=32, color="0.15", ha="left", va="top")

    seen3: set[str] = set()
    all_h, all_l = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen3:
                seen3.add(l); all_h.append(h); all_l.append(l)
    fig.legend(all_h, all_l, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=24)

    # No column-header row needed anymore (n=1/2/3 is now in-panel) and only column 0 carries
    # a ylabel, so both the top margin and wspace can shrink, giving each panel more room.
    fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.135, wspace=0.30, hspace=0.35)

    fig.savefig(fig4 / outname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote figures/4/{outname}")


_plot_average_grid(
    avg_raw, exp_raw, stats_raw, YLABELS_RAW,
    "experimental_average_3x3.pdf",
    "Experimental raw-moment average",
    poly_coeffs=poly_coeffs_raw or None,
)
_plot_average_grid(
    avg_central, exp_central, stats_central, YLABELS,
    "experimental_average_central_3x3.pdf",
    "Experimental central-moment average",
    poly_coeffs=poly_coeffs_central or None,
)

payload = {
    "settings": {
        "corr_same_exp_same_var": corr_same_exp_same_var,
        "corr_same_exp_diff_var": corr_same_exp_diff_var,
        "corr_diff_exp":          corr_diff_exp,
        "inflate":                inflate,
        "exclude_experiments":    sorted(exclude_experiments),
        "sem_cov_lookup_sizes": {"central": len(cov_lookup_central), "raw": len(cov_lookup_raw)},
        "raw_stats":              stats_raw,
        "central_stats":          stats_central,
    },
    "average_raw":  avg_raw,
    "average_central": avg_central,
    "average_raw_cov": avg_raw_cov,
    "average_central_cov": avg_central_cov,
    "exp_raw":  exp_raw,
    "exp_central": exp_central,
}
(out4 / "experimental_average.json").write_text(json.dumps(payload, indent=2))
print("  Wrote output/4/experimental_average.json")
print("Done.")
