"""Shared moment computation and experimental-data loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

EXP_STYLE: dict[str, dict] = {
    "babar":                      dict(color="#8c564b", marker="D",  label="BaBar"),
    "belle":                      dict(color="#1f77b4", marker="o",  label="Belle"),
    "belleII":                    dict(color="#ff7f0e", marker="s",  label="Belle II"),
    "cleo":                       dict(color="#2ca02c", marker="^",  label="CLEO"),
    "delphi":                     dict(color="#d62728", marker="P",  label="DELPHI"),
    "cdf":                        dict(color="#9467bd", marker="X",  label="CDF"),
    "avg_belle_belleII_combined": dict(color="#e377c2", marker="v",  label="Belle+BelleII comb."),
}

YLABELS = {
    "mx_1": r"$\langle M_X^2 \rangle$",
    "mx_2": r"$\langle (M_X^2-\mu)^2 \rangle$",
    "mx_3": r"$\langle (M_X^2-\mu)^3 \rangle$",
    "el_1": r"$\langle E_\ell \rangle$",
    "el_2": r"$\langle (E_\ell-\mu)^2 \rangle$",
    "el_3": r"$\langle (E_\ell-\mu)^3 \rangle$",
    "q2_1": r"$\langle q^2 \rangle$",
    "q2_2": r"$\langle (q^2-\mu)^2 \rangle$",
    "q2_3": r"$\langle (q^2-\mu)^3 \rangle$",
}

YLABELS_RAW = {
    "mx_1": r"$\langle M_X^2 \rangle$",
    "mx_2": r"$\langle (M_X^2)^2 \rangle$",
    "mx_3": r"$\langle (M_X^2)^3 \rangle$",
    "el_1": r"$\langle E_\ell \rangle$",
    "el_2": r"$\langle E_\ell^2 \rangle$",
    "el_3": r"$\langle E_\ell^3 \rangle$",
    "q2_1": r"$\langle q^2 \rangle$",
    "q2_2": r"$\langle (q^2)^2 \rangle$",
    "q2_3": r"$\langle (q^2)^3 \rangle$",
}

GRID = [
    ("mx", ["mx_1", "mx_2", "mx_3"], "el_cuts", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("el", ["el_1", "el_2", "el_3"], "el_cuts", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
    ("q2", ["q2_1", "q2_2", "q2_3"], "q2_cuts", r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$"),
]
ROW_TITLES = [r"$M_X^2$ moments", r"$E_\ell$ moments", r"$q^2$ moments"]

# ── Central → raw transformation ──────────────────────────────────────────────

def _jacobian(mus: np.ndarray) -> np.ndarray:
    n = len(mus)
    J = np.eye(n, dtype=float)
    m1 = mus[0]
    m2 = mus[1] if n > 1 else 0.0
    m3 = mus[2] if n > 2 else 0.0
    if n >= 2:
        J[1, 0] = 2 * m1
    if n >= 3:
        J[2, 0] = 3 * m2 + 3 * m1 ** 2
        J[2, 1] = 3 * m1
    if n >= 4:
        J[3, 0] = 4 * m3 + 12 * m1 * m2 + 4 * m1 ** 3
        J[3, 1] = 6 * m1 ** 2
        J[3, 2] = 4 * m1
    return J


def central_to_raw(mus: np.ndarray) -> np.ndarray:
    n = len(mus)
    r = np.empty(n, dtype=float)
    m1 = mus[0]
    r[0] = m1
    if n >= 2:
        r[1] = mus[1] + m1 ** 2
    if n >= 3:
        r[2] = mus[2] + 3 * m1 * mus[1] + m1 ** 3
    if n >= 4:
        r[3] = mus[3] + 4 * m1 * mus[2] + 6 * m1 ** 2 * mus[1] + m1 ** 4
    return r


# ── Experimental data loading ─────────────────────────────────────────────────

def _s(x: Any) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def _parse_central_h5(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with h5py.File(path) as f:
            g = f[list(f.keys())[0]]
            vals   = np.asarray(g["block0_values"][:, 0], dtype=float)
            errs   = np.asarray(g["block0_values"][:, 1], dtype=float)
            level0 = [_s(x) for x in g["axis1_level0"][...]]
            level1 = np.asarray(g["axis1_level1"][...], dtype=float)
            label0 = np.asarray(g["axis1_label0"][...], dtype=int)
            label1 = np.asarray(g["axis1_label1"][...], dtype=int)
            for i in range(len(vals)):
                tag   = level0[label0[i]]
                cut   = float(level1[label1[i]])
                parts = tag.split("_")
                if len(parts) < 3:
                    continue
                obs = parts[0].lower()
                if obs not in {"mx", "el", "q2"}:
                    continue
                try:
                    order = int(parts[1])
                except ValueError:
                    continue
                exp = "_".join(parts[2:])
                rows.append({"obs": obs, "order": order, "exp": exp,
                             "cut": cut, "value": float(vals[i]), "error": float(errs[i])})
    except Exception as exc:
        print(f"  Warning: could not parse {path.name}: {exc}")
    return rows


def build_exp_moments_df(data_dir: Path) -> pd.DataFrame:
    """Load *_central.h5 files; return DataFrame with measured + calculated rows."""
    central_files = sorted(
        p for p in data_dir.glob("*_central.h5")
        if "_raw" not in p.name and "cov" not in p.name.lower()
    )
    print(f"  Found {len(central_files)} central HDF5 files")
    raw_rows: list[dict] = []
    for h5_path in central_files:
        raw_rows.extend(_parse_central_h5(h5_path))
    if not raw_rows:
        return pd.DataFrame(columns=["obs", "order", "exp", "cut", "value", "error", "type"])
    measured = (pd.DataFrame(raw_rows)
                .drop_duplicates(subset=["obs", "order", "exp", "cut"], keep="first")
                .reset_index(drop=True))
    measured["type"] = "measured"

    calc_rows: list[dict] = []
    for (obs, exp, cut), grp in measured.groupby(["obs", "exp", "cut"], sort=False):
        grp     = grp.sort_values("order")
        orders  = grp["order"].tolist()
        if not orders or 1 not in orders or max(orders) < 2:
            continue
        consec = [o for o in range(1, max(orders) + 1) if o in orders]
        if len(consec) < 2:
            continue
        mus  = np.array([grp.loc[grp["order"] == o, "value"].values[0] for o in consec], dtype=float)
        errs = np.array([grp.loc[grp["order"] == o, "error"].values[0] for o in consec], dtype=float)
        raws     = central_to_raw(mus)
        J        = _jacobian(mus)
        raw_errs = np.sqrt(np.maximum(np.diag(J @ np.diag(errs ** 2) @ J.T), 0.0))
        for idx, o in enumerate(consec):
            calc_rows.append({"obs": obs, "order": o, "exp": exp, "cut": cut,
                              "value": float(raws[idx]), "error": float(raw_errs[idx]),
                              "type": "calculated"})

    if calc_rows:
        return pd.concat([measured, pd.DataFrame(calc_rows)], ignore_index=True)
    return measured


def df_to_plot_dict(df: pd.DataFrame, moment_type: str) -> dict:
    """Convert DataFrame subset to {moment_key: {exp: {cuts, values, errors}}}."""
    sub = df[df["type"] == moment_type].copy()
    result: dict = {}
    for (obs, order, exp), grp in sub.groupby(["obs", "order", "exp"]):
        grp = grp.sort_values("cut")
        key = f"{obs}_{order}"
        result.setdefault(key, {})[exp] = {
            "cuts":   grp["cut"].tolist(),
            "values": grp["value"].tolist(),
            "errors": grp["error"].tolist(),
        }
    return result


# ── Fast numpy moment computation (presorted suffix-sum) ─────────────────────
#
# compute_curves_np() re-sorts by el/q2 and recomputes mx2/el/q2 power arrays on every call,
# even though only the weight vector changes across repeated calls with the same events (e.g.
# toy loops). build_curve_context()/compute_curves_from_context() split the fixed (sort order +
# power arrays, O(N log N)) and weight-dependent (cumsum, O(N)) parts so a toy loop pays the
# sort cost once instead of once per toy -- ~4-8x faster per call at O(1e7)-row scale.

def build_curve_context(mx2: np.ndarray, el: np.ndarray, q2: np.ndarray) -> dict:
    el_order = np.argsort(el)
    el_s  = el[el_order];  mx2_s = mx2[el_order]
    q2_order = np.argsort(q2)
    q2_s  = q2[q2_order]
    return {
        "el_order": el_order, "el_s": el_s, "N_el": len(el_s),
        "mx2_s": mx2_s, "mx2_s2": mx2_s ** 2, "mx2_s3": mx2_s ** 3,
        "el_s2": el_s ** 2, "el_s3": el_s ** 3,
        "q2_order": q2_order, "q2_s": q2_s, "N_q2": len(q2_s),
        "q2_s2": q2_s ** 2, "q2_s3": q2_s ** 3,
    }


def compute_curves_from_context(ctx: dict, w: np.ndarray, el_cuts: np.ndarray,
                                 q2_cuts: np.ndarray) -> dict[str, np.ndarray]:
    el_order, el_s, N_el = ctx["el_order"], ctx["el_s"], ctx["N_el"]
    w_el = w[el_order]
    cw   = np.concatenate(([0.], np.cumsum(w_el)))
    cmx2 = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s"])))
    cmx4 = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s2"])))
    cmx6 = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s3"])))
    cel  = np.concatenate(([0.], np.cumsum(w_el * el_s)))
    cel2 = np.concatenate(([0.], np.cumsum(w_el * ctx["el_s2"])))
    cel3 = np.concatenate(([0.], np.cumsum(w_el * ctx["el_s3"])))

    q2_order, q2_s, N_q2 = ctx["q2_order"], ctx["q2_s"], ctx["N_q2"]
    w_q2 = w[q2_order]
    cw2  = np.concatenate(([0.], np.cumsum(w_q2)))
    cq2  = np.concatenate(([0.], np.cumsum(w_q2 * q2_s)))
    cq4  = np.concatenate(([0.], np.cumsum(w_q2 * ctx["q2_s2"])))
    cq6  = np.concatenate(([0.], np.cumsum(w_q2 * ctx["q2_s3"])))

    out: dict[str, list] = {k: [] for k in
        ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]}

    def sf(c: np.ndarray, k: int, N: int) -> float:
        return float(c[N] - c[k])

    for cut in el_cuts:
        k  = int(np.searchsorted(el_s, cut, side="right"))
        sw = sf(cw, k, N_el)
        if sw <= 0:
            for key in ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3"]:
                out[key].append(float("nan"))
            continue
        m1 = sf(cmx2, k, N_el) / sw; m2 = sf(cmx4, k, N_el) / sw; m3 = sf(cmx6, k, N_el) / sw
        e1 = sf(cel,  k, N_el) / sw; e2 = sf(cel2, k, N_el) / sw; e3 = sf(cel3, k, N_el) / sw
        out["mx_1"].append(m1)
        out["mx_2"].append(m2 - m1 ** 2)
        out["mx_3"].append(m3 - 3 * m2 * m1 + 2 * m1 ** 3)
        out["el_1"].append(e1)
        out["el_2"].append(e2 - e1 ** 2)
        out["el_3"].append(e3 - 3 * e2 * e1 + 2 * e1 ** 3)

    for cut in q2_cuts:
        k  = int(np.searchsorted(q2_s, cut, side="right"))
        sw = sf(cw2, k, N_q2)
        if sw <= 0:
            for key in ["q2_1", "q2_2", "q2_3"]:
                out[key].append(float("nan"))
            continue
        q1 = sf(cq2, k, N_q2) / sw; q2v = sf(cq4, k, N_q2) / sw; q3 = sf(cq6, k, N_q2) / sw
        out["q2_1"].append(q1)
        out["q2_2"].append(q2v - q1 ** 2)
        out["q2_3"].append(q3 - 3 * q2v * q1 + 2 * q1 ** 3)

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def compute_curves_np(
    mx2: np.ndarray, el: np.ndarray, q2: np.ndarray, w: np.ndarray,
    el_cuts: np.ndarray, q2_cuts: np.ndarray,
) -> dict[str, np.ndarray]:
    ctx = build_curve_context(mx2, el, q2)
    return compute_curves_from_context(ctx, w, el_cuts, q2_cuts)


def compute_raw_curves_np(
    mx2: np.ndarray, el: np.ndarray, q2: np.ndarray, w: np.ndarray,
    el_cuts: np.ndarray, q2_cuts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Like compute_curves_np, but returns RAW moments E[X^n] (n=1,2,3) above each cut instead
    of central moments -- what the Hausdorff/MaxEnt data pipeline (8_hausdorff_data.py) and its
    toy loop need directly (raw_to_mu01 consumes raw moments, not central ones)."""
    ctx = build_curve_context(mx2, el, q2)
    el_order, el_s, N_el = ctx["el_order"], ctx["el_s"], ctx["N_el"]
    w_el = w[el_order]
    cw    = np.concatenate(([0.], np.cumsum(w_el)))
    cmx2  = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s"])))
    cmx4  = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s2"])))
    cmx6  = np.concatenate(([0.], np.cumsum(w_el * ctx["mx2_s3"])))
    cel   = np.concatenate(([0.], np.cumsum(w_el * el_s)))
    cel2  = np.concatenate(([0.], np.cumsum(w_el * ctx["el_s2"])))
    cel3  = np.concatenate(([0.], np.cumsum(w_el * ctx["el_s3"])))

    q2_order, q2_s, N_q2 = ctx["q2_order"], ctx["q2_s"], ctx["N_q2"]
    w_q2 = w[q2_order]
    cw2   = np.concatenate(([0.], np.cumsum(w_q2)))
    cq2   = np.concatenate(([0.], np.cumsum(w_q2 * q2_s)))
    cq4   = np.concatenate(([0.], np.cumsum(w_q2 * ctx["q2_s2"])))
    cq6   = np.concatenate(([0.], np.cumsum(w_q2 * ctx["q2_s3"])))

    out: dict[str, list] = {k: [] for k in
        ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]}

    def sf(c: np.ndarray, k: int, N: int) -> float:
        return float(c[N] - c[k])

    for cut in el_cuts:
        k  = int(np.searchsorted(el_s, cut, side="right"))
        sw = sf(cw, k, N_el)
        if sw <= 0:
            for key in ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3"]:
                out[key].append(float("nan"))
            continue
        out["mx_1"].append(sf(cmx2, k, N_el) / sw)
        out["mx_2"].append(sf(cmx4, k, N_el) / sw)
        out["mx_3"].append(sf(cmx6, k, N_el) / sw)
        out["el_1"].append(sf(cel,  k, N_el) / sw)
        out["el_2"].append(sf(cel2, k, N_el) / sw)
        out["el_3"].append(sf(cel3, k, N_el) / sw)

    for cut in q2_cuts:
        k  = int(np.searchsorted(q2_s, cut, side="right"))
        sw = sf(cw2, k, N_q2)
        if sw <= 0:
            for key in ["q2_1", "q2_2", "q2_3"]:
                out[key].append(float("nan"))
            continue
        out["q2_1"].append(sf(cq2, k, N_q2) / sw)
        out["q2_2"].append(sf(cq4, k, N_q2) / sw)
        out["q2_3"].append(sf(cq6, k, N_q2) / sw)

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
