"""Shared moment computation and experimental-data loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

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


# ── kinematics of the step-1 ntuples ─────────────────────────────────────────

def cms_to_xyz(p, theta, phi):
    return (p * np.sin(theta) * np.cos(phi),
            p * np.sin(theta) * np.sin(phi),
            p * np.cos(theta))


def compute_kinematics(df):
    """Mx, q2 and the lepton energy in the B frame (El_B) from the truth-matched B daughters of a step-1 ntuple."""
    l_E = l_px = l_py = l_pz = pd.Series(np.nan, index=df.index)
    nu_E = nu_px = nu_py = nu_pz = pd.Series(np.nan, index=df.index)
    for i in range(5):
        pdg   = df[f"d{i}_mcPDG"].abs()
        is_l  = ((pdg == 11) | (pdg == 13)) & l_E.isna()
        is_nu = ((pdg == 12) | (pdg == 14)) & nu_E.isna()
        l_E   = l_E.where(~is_l,   df[f"d{i}_E"]);   l_px = l_px.where(~is_l,  df[f"d{i}_px"])
        l_py  = l_py.where(~is_l,  df[f"d{i}_py"]);  l_pz = l_pz.where(~is_l,  df[f"d{i}_pz"])
        nu_E  = nu_E.where(~is_nu, df[f"d{i}_E"]);   nu_px = nu_px.where(~is_nu, df[f"d{i}_px"])
        nu_py = nu_py.where(~is_nu, df[f"d{i}_py"]); nu_pz = nu_pz.where(~is_nu, df[f"d{i}_pz"])
    B_px, B_py, B_pz = cms_to_xyz(df["B_pCM"], df["B_thetaCM"], df["B_phiCM"])
    B_E = df["B_ECM"]
    X_E  = B_E - l_E - nu_E
    X_px = B_px - l_px - nu_px; X_py = B_py - l_py - nu_py; X_pz = B_pz - l_pz - nu_pz
    df["Mx"] = np.sqrt(np.clip(X_E**2 - (X_px**2 + X_py**2 + X_pz**2), 0, None))
    q_E  = l_E + nu_E; q_px = l_px + nu_px; q_py = l_py + nu_py; q_pz = l_pz + nu_pz
    df["q2"] = np.clip(q_E**2 - (q_px**2 + q_py**2 + q_pz**2), 0, None)
    B_p2       = B_px**2 + B_py**2 + B_pz**2
    B_gamma    = B_E / np.sqrt(np.clip(B_E**2 - B_p2, 1e-12, None))
    beta_dot_p = (B_px * l_px + B_py * l_py + B_pz * l_pz) / B_E
    df["El_B"] = B_gamma * (l_E - beta_dot_p)
    return df
