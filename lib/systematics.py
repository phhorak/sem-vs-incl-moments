"""Systematic-uncertainty ingredients for the step-6 toy loop.

  ff:           Hammer FF eigenvariations, grouped D / D* / D** (correlated within a group)
  bf_mode:      per-decay-mode BF Gaussian nuisance on the SEM template
  incl_moments: relative deviations of the inclusive raw moments at threshold 0 from the HQE-fit
                toys (Asimov only; mirrors the data-side threshold-0 inversion)
  bf_gap:       assumed gap budget B_gap, entering through c_true = B_incl / B_gap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FF_GROUPS = {
    "D":     {"Bp_Denu", "B0_Denu", "Bp_Dmunu", "B0_Dmunu"},
    "Dst":   {"Bp_Dstenu", "B0_Dstenu", "Bp_Dstmunu", "B0_Dstmunu"},
    "Dstst": {f"{b}_{m}{l}" for b in ("Bp", "B0") for m in ("D1", "D0st", "Dp1", "D2st")
              for l in ("enu", "munu")},
}


def read_parquet_downcast(path, columns, float32_cols=(), category_cols=("decay_name",),
                          batch_size=500_000):
    """pd.read_parquet, but streaming batches so float32/categorical columns never exist at full
    width in memory (the cocktail is ~24M rows; LSF slots are 4 GB). Categories are fixed up
    front, otherwise pd.concat falls back to object dtype."""
    float32_set = set(float32_cols)
    category_set = {c for c in category_cols if c in columns}
    if not float32_set and not category_set:
        return pd.read_parquet(path, columns=columns)
    cat_dtypes = {col: pd.CategoricalDtype(sorted(pc.unique(
        pq.read_table(path, columns=[col]).column(col)).to_pylist())) for col in category_set}
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
        tbl = pa.Table.from_batches([batch])
        schema = pa.schema([pa.field(f.name, pa.float32()) if f.name in float32_set else f
                            for f in tbl.schema])
        df_b = tbl.cast(schema).to_pandas()
        for col, dtype in cat_dtypes.items():
            df_b[col] = df_b[col].astype(dtype)
        parts.append(df_b)
    return pd.concat(parts, ignore_index=True)


# ── FF ───────────────────────────────────────────────────────────────────────

def build_ff_slope_matrix(df: pd.DataFrame, max_slots: int = 8) -> tuple[np.ndarray, list[str]]:
    """(n_events, n_nuisances) float32 matrix of linearized slopes
    0.5*[(up-c)/c + (c-down)/c], zero outside each group; only slots that actually vary."""
    dn = df["decay_name"].astype("category")
    codes, cats = dn.cat.codes.to_numpy(), dn.cat.categories
    c = df["ff_weight"].to_numpy(dtype=np.float32)
    names, cols = [], []
    for g, members in FF_GROUPS.items():
        mask = np.isin(codes, np.flatnonzero(cats.isin(list(members))))
        if not mask.any():
            continue
        cm = c[mask]
        for j in range(max_slots):
            up_col, dn_col = f"ff_weight_up{j}", f"ff_weight_down{j}"
            if up_col not in df.columns or dn_col not in df.columns:
                continue
            up = df[up_col].to_numpy(dtype=np.float32)[mask]
            if not np.nanmax(np.abs(up - cm)) > 1e-8:
                continue
            dn = df[dn_col].to_numpy(dtype=np.float32)[mask]
            s = np.zeros(len(df), dtype=np.float32)
            s[mask] = 0.5 * (np.divide(up - cm, cm, out=np.zeros_like(cm), where=cm != 0)
                             + np.divide(cm - dn, cm, out=np.zeros_like(cm), where=cm != 0))
            names.append(f"{g}:eig{j}")
            cols.append(s)
    matrix = np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=np.float32)
    return matrix, names


def sample_ff_multiplier(slope_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if slope_matrix.shape[1] == 0:
        return np.ones(slope_matrix.shape[0])
    # a few negligible-weight D** rows have |slope| >> 1; clip keeps weights physical
    return np.clip(1.0 + slope_matrix @ rng.standard_normal(slope_matrix.shape[1]), 0.0, None)


# ── BF per mode ──────────────────────────────────────────────────────────────

def bf_family(decay_name: str) -> str:
    """Physical mode shared by e/mu, B+/B0 and the charge splits of the non-resonant modes,
    which all come from one measurement (isospin, LFU): "Bp_Dmpipizenu" -> "Dpipi"."""
    core = decay_name[3:]
    core = core[:-4] if core.endswith("munu") else core[:-3]
    if "pi" in core:
        return ("Dst" if core.startswith("Dst") else "D") + ("pipi" if core.count("pi") >= 2 else "pi")
    return core


def bf_mode_setup(codes, names, bf, bf_unc) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode relative BF uncertainty and family index, for per-event mode `codes` into `names`."""
    codes = np.asarray(codes)
    present, first = np.unique(codes, return_index=True)
    rel = np.zeros(len(names))
    b, u = np.asarray(bf, float)[first], np.asarray(bf_unc, float)[first]
    rel[present] = np.divide(u, b, out=np.zeros(len(b)), where=b > 0)
    _, family = np.unique([bf_family(str(n)) for n in names], return_inverse=True)
    return rel, family


def sample_bf_multiplier(codes: np.ndarray, rel_unc: np.ndarray, family: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """One draw per family; each mode shifts by its own relative uncertainty."""
    z = rng.standard_normal(family.max() + 1)[family]
    return np.clip(1.0 + z[codes] * rel_unc[codes], 0.0, None)


# ── Inclusive moments ────────────────────────────────────────────────────────

def _nearest_psd_corr(corr: np.ndarray) -> np.ndarray:
    corr = 0.5 * (corr + corr.T)
    w, v = np.linalg.eigh(corr)
    psd = (v * np.clip(w, 0.0, None)) @ v.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-300, None))
    return psd / np.outer(d, d)


def load_hqe_raw(path) -> tuple[np.ndarray, np.ndarray]:
    """HQE-fit raw moments at threshold 0, orders 1-3 of (mx2, el, q2): central (9,), toys (n, 9)."""
    import h5py
    from lib.moments import central_to_raw
    with h5py.File(path, "r") as f:
        names = [x.decode() for x in f["central/axis0"][:]]
        central = f["central/block0_values"][0]
        toys = f["toys/block0_values"][:]
    cols = [[names.index(f"{fam}_{o}_cut0.0") for o in (1, 2, 3)] for fam in ("mx", "el", "q2")]
    raw_c = np.concatenate([central_to_raw(central[c]) for c in cols])
    raw_t = np.concatenate([np.array([central_to_raw(r) for r in toys[:, c]]) for c in cols], axis=1)
    return raw_c, raw_t


def hqe_incl_deviations(path, n: int, rng: np.random.Generator, max_order: int) -> np.ndarray:
    """(n, 3*max_order) relative deviations of the inclusive raw moments (mx2, el, q2 x orders
    1..max_order) from n HQE toys. The fit provides orders 1-3; higher orders are drawn
    conditionally on them, with the relative uncertainty continuing the power law in the order
    and rho(i,j) = rho_adj**|i-j| (rho_adj: mean measured adjacent-order correlation)."""
    raw_c, raw_t = load_hqe_raw(path)
    d3 = raw_t[rng.choice(len(raw_t), n, replace=n > len(raw_t))] / raw_c - 1.0
    K = max_order
    out = np.empty((n, 3 * K))
    for f in range(3):
        d = d3[:, 3 * f:3 * f + 3]
        sd3, c3 = d.std(0), np.corrcoef(d, rowvar=False)
        b, a = np.polyfit(np.log([1.0, 2.0, 3.0]), np.log(sd3), 1)
        sd = np.r_[sd3, np.exp(a + b * np.log(np.arange(4, K + 1)))]
        rho = 0.5 * (c3[0, 1] + c3[1, 2])
        corr = rho ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
        corr[:3, :3] = c3
        cov = _nearest_psd_corr(corr) * np.outer(sd, sd)
        A = cov[3:, :3] @ np.linalg.pinv(cov[:3, :3])
        out[:, K * f:K * f + 3] = d
        out[:, K * f + 3:K * (f + 1)] = d @ A.T + sample_mvn(cov[3:, 3:] - A @ cov[:3, 3:], rng, size=n)
    return out


def sample_mvn(cov: np.ndarray, rng: np.random.Generator, size=None) -> np.ndarray:
    """Zero-mean multivariate normal via eigen-decomposition (tolerates PSD covariances)."""
    w, v = np.linalg.eigh(np.asarray(cov, float))
    z = rng.standard_normal((() if size is None else (size,)) + (len(w),))
    return (z * np.sqrt(np.clip(w, 0.0, None))) @ v.T


# ── Gap budget ───────────────────────────────────────────────────────────────

def compute_bf_sem_gap(cocktail_df: pd.DataFrame, bf_incl: float) -> tuple[float, float]:
    """(B_SEM, B_gap) with B_SEM the sum of the per-mode BFs in the cocktail."""
    bf_sem = float(cocktail_df[["decay_name", "bf"]].drop_duplicates("decay_name")["bf"].sum())
    return bf_sem, bf_incl - bf_sem


def bf_gap_sigma(sysc: dict) -> float:
    """B_gap uncertainty of the summed 4-species budget. B0 values are isospin-derived from
    B+ and e/mu are equal by LFU, so every species moves with the same draw (linear sum)."""
    return float(sum(sysc["n_leptons"] * u for u in sysc["bf_gap_unc"].values()))


def c_true(sysc: dict, bf_gap: float, z=0.0):
    """c_true = B_incl / B_gap' with B_gap' = B_gap + z*sigma (c_sem = c_true - 1)."""
    return sysc["bf_incl"] / (bf_gap + np.asarray(z) * bf_gap_sigma(sysc))


def write_budget_tex(budget: dict, path, labels: dict, orders=(1, 2, 3)):
    """TeX table of a moment-level uncertainty budget: rows = sources, columns = (mx2, el, q2) x orders."""
    heads = [r"$M_X^2$", r"$E_\ell$", r"$q^2$"]
    n = len(orders)
    lines = [r"\begin{tabular}{l" + "|".join(["c" * n] * 3) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{h}}}" for h in heads) + r" \\",
             "Source [\\%] & " + " & ".join(rf"$m_{k}$" for _ in heads for k in orders) + r" \\", r"\midrule"]
    for key, vals in budget.items():
        if key == "total":
            lines.append(r"\midrule")
        fmt = lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"
        lines.append(labels.get(key, key) + " & " + " & ".join(fmt(v) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
