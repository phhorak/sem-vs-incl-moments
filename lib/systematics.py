"""Shared systematic-uncertainty sampling for the Hausdorff toy (7_) and data (8_/9_) pipelines.

Four systematic sources, each independently toggleable/isolatable:
  - incl_moments: correlated relative perturbation of raw moments, sized like the real
    experimental average (output/4/experimental_average.json).
  - ff: Hammer form-factor eigenvariation nuisances, grouped by D / D* / D** (each group
    internally correlated across its own eigenvariations, independent from the other groups).
  - bf_mode: per-decay-mode branching-fraction Gaussian nuisance (independent across modes).
  - bf_gap: uncertainty on the inclusive semileptonic branching fraction, propagated into the
    assumed gap-mode branching-fraction budget (bf_gap = bf_incl - sum of SEM exclusive BFs).

All sampling functions take an explicit `rng: np.random.Generator` and are pure/stateless so
7_hausdorff_toy.py and 8_hausdorff_data.py can call them identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW_KEYS = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]

# Physical limits for support variation (shared by 7_hausdorff_toy.py and 8_hausdorff_data.py).
MX2_LO_MIN, MX2_HI_MAX = 1.95**2, 5.35**2   # GeV²  (D* threshold² to B mass²)
EL_LO_MIN,  EL_HI_MAX  = 0.0, 3.0
Q2_LO_MIN,  Q2_HI_MAX  = 0.0, 13.0


def read_parquet_downcast(path, columns, float32_cols=(), category_cols=("decay_name",),
                           batch_size=500_000):
    """Like pd.read_parquet(path, columns=columns), but casts `float32_cols` to float32 and
    `category_cols` to a pandas category dtype while streaming row-group batches through
    pyarrow, so the full float64/string version of those columns is never materialized at once.
    Plain pd.read_parquet(...).astype(...) doesn't reduce *peak* memory -- the expensive read has
    already happened by the time you downcast -- which matters under LSF's ~4GB job cap for wide
    columns like the FF eigenvariations, and for `decay_name` once the full-isospin/lepton
    cocktail pushed row counts into the tens of millions (a plain object string column across
    30M+ rows with only ~90 distinct values dwarfs everything else in memory; categorical
    encoding collapses it to a handful of bytes per row).

    category_cols are converted with a *fixed*, pre-determined category list (a cheap
    single-column pass, done once up front) rather than letting each streamed batch pick up
    whatever values it happens to contain: pd.concat silently drops back to object dtype when
    concatenating Categoricals whose categories differ (even just in code order) across pieces,
    which would defeat the whole optimization.
    """
    float32_set = set(float32_cols)
    category_set = {c for c in category_cols if c in columns}
    if not float32_set and not category_set:
        return pd.read_parquet(path, columns=columns)
    cat_dtypes = {}
    for col in category_set:
        uniques = pc.unique(pq.read_table(path, columns=[col]).column(col)).to_pylist()
        cat_dtypes[col] = pd.CategoricalDtype(categories=sorted(uniques))
    pf = pq.ParquetFile(path)
    parts = []
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        tbl = pa.Table.from_batches([batch])
        new_fields = [pa.field(f.name, pa.float32()) if f.name in float32_set else f
                      for f in tbl.schema]
        df_b = tbl.cast(pa.schema(new_fields)).to_pandas()
        for col, dtype in cat_dtypes.items():
            df_b[col] = df_b[col].astype(dtype)
        parts.append(df_b)
    return pd.concat(parts, ignore_index=True)

# Mirrors _HAMMER_DECAYS in 3_reweight.py / FF_DECAYS in 6_fit.py, grouped into the three
# independent FF systematics the user asked for (D**: all 4 orbitally-excited states combined
# into one internally-correlated super-group).
FF_GROUPS = {
    "D":     {"Bp_Denu", "B0_Denu", "Bp_Dmunu", "B0_Dmunu"},
    "Dst":   {"Bp_Dstenu", "B0_Dstenu", "Bp_Dstmunu", "B0_Dstmunu"},
    "Dstst": {"Bp_D1enu", "Bp_D0stenu", "Bp_Dp1enu", "Bp_D2stenu",
              "B0_D1enu", "B0_D0stenu", "B0_Dp1enu", "B0_D2stenu",
              "Bp_D1munu", "Bp_D0stmunu", "Bp_Dp1munu", "Bp_D2stmunu",
              "B0_D1munu", "B0_D0stmunu", "B0_Dp1munu", "B0_D2stmunu"},
}

SYSTEMATIC_NAMES = ["incl_moments", "ff", "bf_mode", "bf_gap"]


# ── FF group nuisances ─────────────────────────────────────────────────────────

def ff_group_masks(decay_name: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per FF group from decay_name. Mutually exclusive by construction."""
    decay_name = np.asarray(decay_name)
    return {g: np.isin(decay_name, list(names)) for g, names in FF_GROUPS.items()}


def ff_group_eigvar_ids(df: pd.DataFrame, masks: dict[str, np.ndarray],
                         max_slots: int = 8) -> dict[str, list[int]]:
    """Detect which up{j}/down{j} slots actually vary (vs. Hammer's nominal-padding) for each
    group, by checking whether up_j differs from the central ff_weight within that group's rows.
    """
    if "ff_weight" not in df.columns:
        return {g: [] for g in masks}
    c = df["ff_weight"].to_numpy(dtype=float)
    ids: dict[str, list[int]] = {}
    for g, mask in masks.items():
        slots: list[int] = []
        if not mask.any():
            ids[g] = slots
            continue
        for j in range(max_slots):
            up_col, dn_col = f"ff_weight_up{j}", f"ff_weight_down{j}"
            if up_col not in df.columns or dn_col not in df.columns:
                continue
            up = df[up_col].to_numpy(dtype=float)
            diff = np.abs(up[mask] - c[mask])
            if np.isfinite(diff).any() and np.nanmax(diff) > 1e-8:
                slots.append(j)
        ids[g] = slots
    return ids


def ff_slopes(df: pd.DataFrame, masks: dict[str, np.ndarray],
              eigvar_ids: dict[str, list[int]]) -> dict[str, dict[int, np.ndarray]]:
    """Linearized per-slot slope, mirroring 6_fit.py's convention:
    slope_j = 0.5 * [(up_j - c)/c + (c - down_j)/c], masked to the group's rows only (zero
    elsewhere so this is directly summable into a single event-level multiplier).
    """
    # float32: this is a systematic-band estimate, not a precision-critical calculation, and
    # the slope matrix (n_events x n_nuisances) is the single largest object built per toy-job
    # chunk -- float32 halves its memory footprint, which matters under LSF's ~4GB job cap.
    n = len(df)
    c = df["ff_weight"].to_numpy(dtype=np.float32)
    slopes: dict[str, dict[int, np.ndarray]] = {}
    for g, mask in masks.items():
        slopes[g] = {}
        for j in eigvar_ids.get(g, []):
            up = df[f"ff_weight_up{j}"].to_numpy(dtype=np.float32)
            dn = df[f"ff_weight_down{j}"].to_numpy(dtype=np.float32)
            s = np.zeros(n, dtype=np.float32)
            cm = c[mask]
            s[mask] = 0.5 * (
                np.divide(up[mask] - cm, cm, out=np.zeros_like(cm), where=np.abs(cm) > 0)
                + np.divide(cm - dn[mask], cm, out=np.zeros_like(cm), where=np.abs(cm) > 0)
            )
            slopes[g][j] = s
    return slopes


def build_ff_slope_matrix(df: pd.DataFrame, max_slots: int = 8) -> tuple[np.ndarray, list[str]]:
    """Precompute-once primitive for toy loops: returns a (n_events, n_nuisances) matrix of
    per-(group,slot) slopes and the matching nuisance names, so a toy loop only needs
    `mult = clip(1 + slope_matrix @ theta, 0, None)` with a fresh `theta` each draw, instead of
    recomputing masks/eigvar-detection/slopes (an O(n_events) scan) on every toy.
    """
    decay_name = df["decay_name"].to_numpy()
    masks = ff_group_masks(decay_name)
    ids = ff_group_eigvar_ids(df, masks, max_slots=max_slots)
    slopes = ff_slopes(df, masks, ids)
    names: list[str] = []
    cols: list[np.ndarray] = []
    for g, per_slot in slopes.items():
        for j, s in per_slot.items():
            names.append(f"{g}:eig{j}")
            cols.append(s)
    matrix = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return matrix, names


def sample_ff_multiplier_from_matrix(slope_matrix: np.ndarray,
                                      rng: np.random.Generator) -> np.ndarray:
    """Draw one theta ~ N(0,1) per nuisance column of `slope_matrix` (built by
    build_ff_slope_matrix) and return the clipped per-event weight multiplier.
    """
    n = slope_matrix.shape[1]
    if n == 0:
        return np.ones(slope_matrix.shape[0], dtype=float)
    theta = rng.standard_normal(n)
    mult = 1.0 + slope_matrix @ theta
    # A small number of (negligible-weight) D** rows have poorly-constrained Hammer central
    # weights (see SYSTEMATICS_NOTES.md) that can drive |slope| >> 1; clip to keep weights
    # physical.
    return np.clip(mult, 0.0, None)


# ── Per-mode BF nuisance ────────────────────────────────────────────────────────

def bf_mode_setup(decay_name: np.ndarray, bf: np.ndarray,
                   bf_unc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute-once primitive for toy loops: factorize decay_name and build a per-mode
    relative-uncertainty vector, so a toy loop only needs
    `z = rng.standard_normal(len(rel_unc_per_mode)); mult = clip(1+z[mode_codes]*rel_unc_per_mode[mode_codes],0,None)`
    each draw (via sample_bf_multiplier_from_codes), instead of re-factorizing every toy.
    """
    decay_name = np.asarray(decay_name)
    bf = np.asarray(bf, dtype=float)
    bf_unc = np.asarray(bf_unc, dtype=float)
    codes, uniques = pd.factorize(decay_name, sort=True)
    # bf/bf_unc are constant within a mode -- take the first occurrence per code, vectorized.
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    first_pos_in_sorted = np.searchsorted(codes_sorted, np.arange(len(uniques)))
    first_idx = order[first_pos_in_sorted]
    rel_unc_per_mode = np.divide(bf_unc[first_idx], bf[first_idx],
                                  out=np.zeros(len(uniques)), where=bf[first_idx] > 0)
    return codes, rel_unc_per_mode


def sample_bf_multiplier_from_codes(mode_codes: np.ndarray, rel_unc_per_mode: np.ndarray,
                                     rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(len(rel_unc_per_mode))
    return np.clip(1.0 + z[mode_codes] * rel_unc_per_mode[mode_codes], 0.0, None)


# ── Inclusive-moment systematic (sized like the experimental average) ──────────

def _representative_cut_idx(entry: dict, cut_choice: str) -> int:
    """Pick which reported cut point represents a raw-moment key's uncertainty.

    "min_unc": the cut with the smallest relative uncertainty (used to size the toy's
    incl_moments systematic -- the toy itself always reconstructs from the no-cut/inclusive
    moments, this only controls how large a representative relative-uncertainty/correlation to
    assign; per-cut treatment is deferred to a later, cut-differential study).
    "lowest": the lowest (most inclusive) reported cut.
    """
    cuts = np.asarray(entry["cuts"], dtype=float)
    if cut_choice == "min_unc":
        vals = np.asarray(entry["values"], dtype=float)
        errs = np.asarray(entry["errors"], dtype=float)
        rel = np.abs(np.divide(errs, vals, out=np.full_like(errs, np.inf), where=vals != 0))
        return int(np.argmin(rel))
    if cut_choice == "lowest":
        return int(np.argmin(cuts))
    return 0


def incl_moment_relative_unc(exp_avg: dict, keys: list[str] = RAW_KEYS,
                              cut_choice: str = "min_unc") -> dict[str, float]:
    """Representative relative uncertainty per raw-moment key, taken from
    output/4/experimental_average.json's average_raw at the cut point selected by cut_choice
    (default: the cut with the smallest relative uncertainty for that key).
    """
    avg_raw = exp_avg.get("average_raw", {})
    rel: dict[str, float] = {}
    for k in keys:
        entry = avg_raw.get(k)
        if not entry or not entry.get("cuts"):
            rel[k] = 0.0
            continue
        idx = _representative_cut_idx(entry, cut_choice)
        vals = np.asarray(entry["values"], dtype=float)
        errs = np.asarray(entry["errors"], dtype=float)
        v, e = vals[idx], errs[idx]
        rel[k] = float(abs(e / v)) if v != 0 else 0.0
    return rel


def _nearest_psd_corr(corr: np.ndarray) -> np.ndarray:
    """Eigen-clip to PSD, then renormalize so the diagonal is exactly 1."""
    corr = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    psd = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-300, None))
    return psd / np.outer(d, d)


def incl_moment_corr_block(exp_avg: dict, keys: list[str] = RAW_KEYS,
                            cut_choice: str = "min_unc") -> np.ndarray:
    """Representative len(keys) x len(keys) correlation matrix, built from the point nearest
    each key's representative cut (selected the same way as incl_moment_relative_unc, so the
    correlation and the relative uncertainty always refer to the same cut) in average_raw_cov.
    Falls back to the identity for any key missing a covariance entry.
    """
    n = len(keys)
    cov_data = exp_avg.get("average_raw_cov")
    if not cov_data:
        return np.eye(n)
    points = cov_data["points"]
    cov = np.asarray(cov_data["cov"], dtype=float)
    avg_raw = exp_avg.get("average_raw", {})

    idxs: list[int | None] = []
    for k in keys:
        entry = avg_raw.get(k, {})
        cuts = entry.get("cuts", [])
        if not cuts:
            idxs.append(None)
            continue
        target_cut = float(np.asarray(cuts, dtype=float)[_representative_cut_idx(entry, cut_choice)])
        cand = [i for i, p in enumerate(points) if p["key"] == k]
        if not cand:
            idxs.append(None)
            continue
        idxs.append(min(cand, key=lambda i: abs(points[i]["cut"] - target_cut)))

    corr = np.eye(n)
    sig = np.zeros(n)
    valid = [a for a, ix in enumerate(idxs) if ix is not None]
    for a in valid:
        sig[a] = np.sqrt(max(cov[idxs[a], idxs[a]], 0.0))
    for a in valid:
        for b in valid:
            if sig[a] > 0 and sig[b] > 0:
                corr[a, b] = cov[idxs[a], idxs[b]] / (sig[a] * sig[b])
    return _nearest_psd_corr(corr)


def extrapolate_incl_moment_order4(exp_avg: dict, families: tuple[str, ...] = ("mx", "el", "q2"),
                                    cut_choice: str = "min_unc") -> tuple[list[str], dict[str, float], np.ndarray]:
    """Ad-hoc extension of the incl_moments systematic to order 4, which the experimental
    average does not measure (only orders 1-3 are reported). Without this, order 4 is held
    completely fixed while orders 1-3 move together under a strong (rho ~ 0.85-0.99) measured
    correlation -- an unperturbed anchor fighting a coherent shift in the rest of the moment
    vector, which the Hausdorff/MaxEnt inversion is very sensitive to (relevant for
    n_moments=5, which uses orders 0-4).

    rel_unc(4) is extrapolated via a power-law fit (log rel_unc = a + b*log(order)) through the
    measured orders 1-3; correlation to/among order 4 is extrapolated via an AR(1)-like
    geometric decay in order-distance, using the mean measured adjacent-order correlation
    (rho(1,2), rho(2,3)) as the decay ratio: rho(i,4) = rho_adj**(4-i). Both are explicit
    approximations, not measurements -- flagged as a caveat wherever this is used.
    """
    rel = incl_moment_relative_unc(exp_avg, keys=RAW_KEYS, cut_choice=cut_choice)
    corr = incl_moment_corr_block(exp_avg, keys=RAW_KEYS, cut_choice=cut_choice)
    orders3 = np.array([1, 2, 3], dtype=float)

    keys_ext: list[str] = []
    rel_ext: dict[str, float] = {}
    blocks = []
    for fam in families:
        ks3 = [f"{fam}_{i}" for i in (1, 2, 3)]
        r3 = np.array([rel[k] for k in ks3])
        idx3 = [RAW_KEYS.index(k) for k in ks3]
        c3 = corr[np.ix_(idx3, idx3)]

        b, a = np.polyfit(np.log(orders3), np.log(np.clip(r3, 1e-12, None)), 1)
        r4 = float(np.exp(a + b * np.log(4.0)))

        rho_adj = 0.5 * (c3[0, 1] + c3[1, 2])
        c4 = np.eye(4)
        c4[:3, :3] = c3
        for i in range(3):
            c4[i, 3] = c4[3, i] = rho_adj ** (4 - (i + 1))
        c4 = _nearest_psd_corr(c4)

        ks4 = ks3 + [f"{fam}_4"]
        for i, k in enumerate(ks4):
            rel_ext[k] = float(r4) if k.endswith("_4") else float(r3[i])
        keys_ext.extend(ks4)
        blocks.append(c4)

    n_tot = len(keys_ext)
    corr_ext = np.eye(n_tot)
    off = 0
    for c4 in blocks:
        corr_ext[off:off + 4, off:off + 4] = c4
        off += 4
    return keys_ext, rel_ext, corr_ext


def sample_from_cov(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SVD/eigen-clipped multivariate-normal draw (shared primitive, also used to replace the
    inline sampling in 8_hausdorff_data.py's --toy-job for the experimental average_raw_cov).
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    z = rng.standard_normal(len(mean))
    return mean + (z * np.sqrt(eigvals)) @ eigvecs.T


def sample_incl_moment_perturbation(mu: dict[str, float], keys: list[str],
                                     rel_unc: dict[str, float], corr: np.ndarray,
                                     rng: np.random.Generator) -> dict[str, float]:
    """Correlated relative perturbation of raw moments: mu_k * (1 + z_k * rel_unc_k), with
    z ~ N(0, corr) (unit-variance marginals, so rel_unc alone sets the scale).
    """
    z = sample_from_cov(np.zeros(len(keys)), corr, rng)
    return {k: mu[k] * (1.0 + z[i] * rel_unc.get(k, 0.0)) for i, k in enumerate(keys)}


# ── bf_gap propagated from the inclusive branching ratio ───────────────────────

def compute_bf_gap(cocktail_df: pd.DataFrame, bf_incl: float,
                    bf_incl_unc: float) -> tuple[float, float, float]:
    """bf_gap = bf_incl - sum(per-mode BFs actually in the cocktail), with
    sigma_bf_gap = sqrt(bf_incl_unc^2 + sum(bf_unc_i^2)) (independent per-mode uncertainties).
    Computed fresh from the parquet at runtime -- not a hardcoded constant.

    Returns (bf_sem_computed, bf_gap_central, sigma_bf_gap), all in the same units as bf_incl.
    """
    g = cocktail_df[["decay_name", "bf", "bf_unc"]].drop_duplicates(subset="decay_name")
    bf_sem_computed = float(g["bf"].sum())
    sigma_bf_sem = float(np.sqrt((g["bf_unc"].to_numpy(dtype=float) ** 2).sum()))
    bf_gap_central = bf_incl - bf_sem_computed
    sigma_bf_gap = float(np.sqrt(bf_incl_unc ** 2 + sigma_bf_sem ** 2))
    return bf_sem_computed, bf_gap_central, sigma_bf_gap


def sample_bf_gap(bf_gap_central: float, sigma_bf_gap: float, rng: np.random.Generator) -> float:
    return float(max(bf_gap_central + rng.standard_normal() * sigma_bf_gap, 1e-9))


# ── Run-label / toggle registry ─────────────────────────────────────────────────

def active_systematics(cfg_block: dict) -> dict[str, list[str]]:
    """Parse the config's `systematics:` block into {run_label: [active systematic names]}.

    Always includes "base" (no systematics -- stat/support-only baseline). Adds "solo_<name>"
    for each enabled systematic, and "all" (every enabled systematic together) if
    run_combined is true. Callers that don't want a "base" run (e.g. --submit dispatch when
    systematics.run_base is false) can simply drop that key from the returned dict.
    """
    cfg_block = cfg_block or {}
    enable = cfg_block.get("enable", {})
    names = [n for n in SYSTEMATIC_NAMES if enable.get(n, True)]
    runs: dict[str, list[str]] = {"base": []}
    for n in names:
        runs[f"solo_{n}"] = [n]
    if cfg_block.get("run_combined", True) and names:
        runs["all"] = list(names)
    return runs
