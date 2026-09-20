#!/usr/bin/env python3
"""
5_residual_covariance.py

Step 5: residual raw-moment covariance for El, Q2, Mx at each observable's own full threshold
grid (not the coarser 5-threshold grid in config.yaml's `joint_maxent` block -- an earlier check
on this branch showed that grid is too ill-conditioned to fit reliably). Toy loop combines:
  - exp measurement covariance (vectorized batch draw from output/4's average_raw_cov)
  - SEM/cocktail stat bootstrap (always on)
  - FF reweighting systematic (on/off per source variant)
  - BR/bf_mode systematic (on/off per source variant)
into the residual raw_gap = r_exp*exp - r_sem*sem at each threshold, order 1..3, split into
three source variants (stat_ff, stat_bfmode, stat_ff_bfmode) so a downstream fit blowing up can
be backtracked to a specific systematic source. This step does NOT fit anything -- it only
produces the mean + covariance. The joint MaxEnt fit (Asimov check, source-split fit, band) is
step 6.

Usage:
  Build the toy-based source-split covariance (dispatches bsub chunk jobs):
    python3 5_residual_covariance.py --submit

  Merge chunks once jobs finish:
    python3 5_residual_covariance.py --merge

  Single chunk (used by bsub jobs):
    python3 5_residual_covariance.py --toy-job --obs El --source stat_ff --chunk 0 --n-chunks 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lib.moments import build_curve_context, compute_raw_curves_from_context
from lib.systematics import (
    read_parquet_downcast, build_ff_slope_matrix, sample_ff_multiplier_from_matrix,
    bf_mode_setup, sample_bf_multiplier_from_codes, compute_bf_gap,
)

SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode"]
SOURCE_USE = {"stat_ff": (True, False), "stat_bfmode": (False, True), "stat_ff_bfmode": (True, True)}
OBS_LIST = ["Mx", "El", "Q2"]


def load_common(cfg):
    hd = cfg["hausdorff_data"]
    out_root = Path(cfg["paths"]["output"])
    exp_avg = json.load(open(out_root / "4" / "experimental_average.json"))
    sem_raw = json.load(open(out_root / "5" / "sem_moments_raw.json"))
    avg_raw = exp_avg["average_raw"]
    sem_nom_raw = sem_raw["sem_nominal_raw"]
    sem_el_cuts = np.array(sem_raw["cuts"]["el"])
    sem_q2_cuts = np.array(sem_raw["cuts"]["q2"])

    sysc = cfg.get("systematics", {})
    bf_incl = float(sysc.get("bf_incl", 0.1105))
    bf_incl_unc = float(sysc.get("bf_incl_unc", 0.0016))
    cocktail_path = out_root / "3" / "cocktail.parquet"
    df_bf = pd.read_parquet(cocktail_path, columns=["decay_name", "bf", "bf_unc"])
    _, bf_gap_central, _ = compute_bf_gap(df_bf, bf_incl, bf_incl_unc)
    del df_bf
    r_exp = bf_incl / bf_gap_central
    r_sem = (bf_incl - bf_gap_central) / bf_gap_central

    mx2_lo, mx2_hi = float(hd["mx_support"][0]), float(hd["mx_support"][1])
    el_lo, el_hi = float(hd["el_support"][0]), float(hd["el_support"][1])
    q2_lo, q2_hi = float(hd["q2_support"][0]), float(hd["q2_support"][1])

    OBS_CFG = {
        # Mx's own "cuts" are El-cut values (its reported moments are El-selected Mx2 sub-
        # population moments, not a mask on Mx2 itself) -- thr_lo/hi must therefore be El's
        # domain, not Mx2's (the bug this fixes: filtering El-valued thresholds against the
        # Mx2 domain silently dropped every one of them).
        "Mx": dict(keys=["mx_1", "mx_2", "mx_3"], cut_ref=sem_el_cuts, thr_lo=el_lo, thr_hi=el_hi),
        "El": dict(keys=["el_1", "el_2", "el_3"], cut_ref=sem_el_cuts, thr_lo=el_lo, thr_hi=el_hi),
        "Q2": dict(keys=["q2_1", "q2_2", "q2_3"], cut_ref=sem_q2_cuts, thr_lo=q2_lo, thr_hi=q2_hi),
    }
    return dict(avg_raw=avg_raw, sem_nom_raw=sem_nom_raw, r_exp=r_exp, r_sem=r_sem,
                OBS_CFG=OBS_CFG, exp_avg=exp_avg)


def thresholds_for(obs, C):
    cfg_o = C["OBS_CFG"][obs]
    keys = cfg_o["keys"]
    all_thr = sorted(float(c) for c in C["avg_raw"][keys[0]]["cuts"])
    thr = [t for t in all_thr if cfg_o["thr_lo"] <= t < cfg_o["thr_hi"]]
    dropped = len(all_thr) - len(thr)
    if dropped:
        print(f"[{obs}] thr domain=[{cfg_o['thr_lo']},{cfg_o['thr_hi']}]: dropped {dropped} "
              f"threshold(s) outside it")
    return thr


def nominal_mean(obs, C, thresholds):
    """The measured raw_gap = r_exp*exp - r_sem*sem at each threshold, order 1..3 -- the
    (noise-free) central value the toy covariance is centered on."""
    cfg_o = C["OBS_CFG"][obs]
    keys, cut_ref = cfg_o["keys"], cfg_o["cut_ref"]
    out = []
    for thr in thresholds:
        for key in keys:
            kc = np.array(C["avg_raw"][key]["cuts"]); kv = np.array(C["avg_raw"][key]["values"])
            exp_val = np.interp(thr, kc, kv)
            sem_val = float(np.interp(thr, cut_ref, np.array(C["sem_nom_raw"][key])))
            out.append(C["r_exp"] * exp_val - C["r_sem"] * sem_val)
    return np.array(out)


# ── Toy chunk (bsub) ─────────────────────────────────────────────────────────────────────────
def run_toy_job(cfg, obs, source, chunk, n_chunks, n_toys_total, seed_base, toys_root):
    C = load_common(cfg)
    thresholds = thresholds_for(obs, C)
    n_thr = len(thresholds)
    cfg_o = C["OBS_CFG"][obs]
    keys, cut_ref = cfg_o["keys"], cfg_o["cut_ref"]
    use_ff, use_bfmode = SOURCE_USE[source]

    n_per = (n_toys_total + n_chunks - 1) // n_chunks
    i_start, i_end = chunk * n_per, min((chunk + 1) * n_per, n_toys_total)
    n_local = i_end - i_start
    out_dir = Path(toys_root) / obs / source
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_local <= 0:
        print(f"Chunk {chunk}: nothing to do"); return

    # exp side: batch-sampled once, vectorized (validated: matches the analytic exp covariance
    # to <1% on both diagonal and off-diagonal for El's full threshold grid).
    cov_data = C["exp_avg"]["average_raw_cov"]
    cov_points = cov_data["points"]
    cov_matrix = np.array(cov_data["cov"])
    cov_mean = np.array([np.interp(p["cut"], C["avg_raw"][p["key"]]["cuts"], C["avg_raw"][p["key"]]["values"])
                          for p in cov_points])
    idx_lookup = {(p["key"], round(float(p["cut"]), 6)): i for i, p in enumerate(cov_points)}
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    rng_exp = np.random.default_rng(seed_base + chunk * 1_000_000)
    Z = rng_exp.standard_normal((n_local, len(cov_mean)))
    exp_samples = cov_mean[None, :] + (Z * np.sqrt(eigvals)[None, :]) @ eigvecs.T

    exp_vals_toys = np.empty((n_local, n_thr, 3))
    for a, key in enumerate(keys):
        node_cuts = sorted(float(p["cut"]) for p in cov_points if p["key"] == key)
        node_idx = [idx_lookup[(key, round(c, 6))] for c in node_cuts]
        node_cuts = np.array(node_cuts)
        W = np.zeros((n_thr, len(node_cuts)))
        for ti, thr in enumerate(thresholds):
            k = np.searchsorted(node_cuts, thr)
            if k == 0: W[ti, 0] = 1.0
            elif k >= len(node_cuts): W[ti, -1] = 1.0
            else:
                x0, x1 = node_cuts[k-1], node_cuts[k]; f = (thr - x0) / (x1 - x0)
                W[ti, k-1] = 1 - f; W[ti, k] = f
        exp_vals_toys[:, :, a] = exp_samples[:, node_idx] @ W.T

    # SEM/cocktail side: load + sort once (both El- and Q2-sort context, shared across Mx/El/Q2).
    t0 = time.time()
    base_cols = ["Mx", "El_B", "q2", "total_weight", "decay_name", "bf", "bf_unc"]
    cocktail_path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    schema_cols = pq.ParquetFile(cocktail_path).schema_arrow.names
    ff_cols = (["ff_weight"] + [c for c in schema_cols
               if c.startswith("ff_weight_up") or c.startswith("ff_weight_down")]) if use_ff else []
    df = read_parquet_downcast(cocktail_path, base_cols + ff_cols, float32_cols=ff_cols)
    ok = (np.isfinite(df["Mx"].to_numpy()) & np.isfinite(df["El_B"].to_numpy())
          & np.isfinite(df["q2"].to_numpy()) & np.isfinite(df["total_weight"].to_numpy())
          & (df["total_weight"].to_numpy() > 0))
    df = df.loc[ok].reset_index(drop=True)
    mx2 = df["Mx"].to_numpy(float) ** 2
    el = df["El_B"].to_numpy(float)
    q2 = df["q2"].to_numpy(float)
    w_all = df["total_weight"].to_numpy(float)
    N_c = len(df)

    ff_slope_matrix, _ = build_ff_slope_matrix(df) if use_ff else (np.zeros((N_c, 0)), [])
    bf_mode_codes, bf_rel_unc_per_mode = bf_mode_setup(
        df["decay_name"].to_numpy(), df["bf"].to_numpy(), df["bf_unc"].to_numpy())
    del df

    ctx = build_curve_context(mx2, el, q2)
    print(f"Chunk {chunk}/{n_chunks} [{obs}/{source}]: setup {time.time()-t0:.1f}s  N_c={N_c:,}  "
          f"toys {i_start}-{i_end-1} ({n_local})")

    el_thr_core = np.array(thresholds) if obs in ("Mx", "El") else np.array([0.0])
    q2_thr_core = np.array(thresholds) if obs == "Q2" else np.array([0.0])
    key_out = {"Mx": ["mx_1", "mx_2", "mx_3"], "El": ["el_1", "el_2", "el_3"],
               "Q2": ["q2_1", "q2_2", "q2_3"]}[obs]

    # Always also evaluate at el=0/q2=0 (a cheap extra point on top of the O(N) cumsum context,
    # regardless of whether 0.0 is already in this obs's own measured threshold grid -- El/Q2/Mx's
    # measured cuts start at 0.4/1.5/0.7, never 0.0). This gives, for free, a per-toy fully-
    # inclusive (thr=0) SEM raw moment for ALL NINE mx/el/q2 x 1/2/3 keys at once (curves always
    # contains all nine regardless of `obs`) -- consumed by 6b_kolya_vs_joint.py's "Kolya" toy
    # loop, which needs SEM's OWN thr=0 stat+FF+BR-mode toy variation (no exp side) to pair
    # against Markus's HQE-fit toys. Kept in every obs job (not just one) since it costs nothing
    # extra here and lets 6b sanity-check the three obs jobs agree (same rng2 stream up to this
    # point since obs doesn't affect the use_ff/use_bfmode branching that consumes it).
    el_thr = np.union1d(el_thr_core, [0.0])
    q2_thr = np.union1d(q2_thr_core, [0.0])
    el_core_idx = np.searchsorted(el_thr, el_thr_core)
    q2_core_idx = np.searchsorted(q2_thr, q2_thr_core)
    core_idx = el_core_idx if obs in ("Mx", "El") else q2_core_idx
    el0_idx = int(np.searchsorted(el_thr, 0.0))
    q20_idx = int(np.searchsorted(q2_thr, 0.0))

    rng2 = np.random.default_rng(seed_base + chunk * 2_000_000 + 777)
    dim = n_thr * 3
    # Per-toy vectors persisted directly (restructured from the old running sum/outer-product
    # accumulators): lets downstream code both recompute mean+cov (as before) AND refit the
    # joint chi^2 model per toy for a genuine toy-based uncertainty band instead of the
    # delta-method linear propagation -- see 6_joint_fit.py / 6b_kolya_vs_joint.py.
    raw_gap_all = np.empty((n_local, dim))
    sem0_all = np.empty((n_local, 9))  # [mx_1,mx_2,mx_3, el_1,el_2,el_3, q2_1,q2_2,q2_3] at thr=0
    t0 = time.time()
    for it in range(n_local):
        idx = rng2.integers(0, N_c, size=N_c)
        counts = np.bincount(idx, minlength=N_c).astype(float)
        w = w_all * counts
        if use_ff:
            w = w * sample_ff_multiplier_from_matrix(ff_slope_matrix, rng2)
        if use_bfmode:
            w = w * sample_bf_multiplier_from_codes(bf_mode_codes, bf_rel_unc_per_mode, rng2)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.)

        curves = compute_raw_curves_from_context(ctx, w, el_thr, q2_thr)
        sem_vals = np.stack([curves[key_out[0]][core_idx], curves[key_out[1]][core_idx],
                              curves[key_out[2]][core_idx]], axis=1)
        raw_gap_all[it] = (C["r_exp"] * exp_vals_toys[it] - C["r_sem"] * sem_vals).reshape(-1)
        sem0_all[it] = [curves["mx_1"][el0_idx], curves["mx_2"][el0_idx], curves["mx_3"][el0_idx],
                         curves["el_1"][el0_idx], curves["el_2"][el0_idx], curves["el_3"][el0_idx],
                         curves["q2_1"][q20_idx], curves["q2_2"][q20_idx], curves["q2_3"][q20_idx]]
    elapsed = time.time() - t0
    print(f"Chunk {chunk} done: {n_local} toys in {elapsed:.0f}s ({elapsed/max(n_local,1):.2f}s/toy)")

    out = out_dir / f"chunk_{chunk:04d}.npz"
    np.savez(out, raw_gap=raw_gap_all, sem0=sem0_all, n_toys=n_local, thresholds=thresholds)
    print(f"-> {out}")


# ── Submit ───────────────────────────────────────────────────────────────────────────────────
def run_submit(cfg, n_chunks, n_toys_total, dry_run):
    import subprocess
    here = Path(__file__).parent.resolve()
    queue = cfg["generation"]["queue"]
    toys_root = here / "output" / "5" / "toys"
    log_dir = here / "logs" / "5"
    toys_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for obs in OBS_LIST:
        for source in SOURCES:
            for chunk in range(n_chunks):
                log_file = log_dir / f"{obs}_{source}_{chunk}.log"
                job_cmd = (f"cd {here} && python3 5_residual_covariance.py --toy-job --obs {obs} "
                           f"--source {source} --chunk {chunk} --n-chunks {n_chunks} "
                           f"--n-toys-total {n_toys_total}")
                bsub_cmd = f'bsub -q {queue} -env all -n "4" -oo "{log_file}" "{job_cmd}"'
                print(bsub_cmd)
                if not dry_run:
                    subprocess.run(bsub_cmd, shell=True, check=True)
    print("(dry run — no jobs submitted)" if dry_run else "Submitted.")


# ── Merge ────────────────────────────────────────────────────────────────────────────────────
def run_merge(cfg):
    C = load_common(cfg)
    out_dir = Path(cfg["paths"]["output"]) / "5"
    out_dir.mkdir(parents=True, exist_ok=True)
    toys_root = out_dir / "toys"

    sem0_ref = None  # cross-check: sem0 should match bit-for-bit across obs for the same source
    for obs in OBS_LIST:
        thresholds = thresholds_for(obs, C)
        n_thr = len(thresholds)
        dim = n_thr * 3
        points = [{"obs": obs, "thr": thr, "order": k} for thr in thresholds for k in (1, 2, 3)]
        nominal = nominal_mean(obs, C, thresholds)

        for source in SOURCES:
            files = sorted((toys_root / obs / source).glob("chunk_*.npz"))
            if not files:
                print(f"[{obs}/{source}] no chunks found, skipping")
                continue
            raw_gap_parts, sem0_parts = [], []
            for f in files:
                d = np.load(f)
                raw_gap_parts.append(d["raw_gap"]); sem0_parts.append(d["sem0"])
            raw_gap = np.concatenate(raw_gap_parts, axis=0)  # (n_total, dim)
            sem0 = np.concatenate(sem0_parts, axis=0)        # (n_total, 9)
            n_total = raw_gap.shape[0]
            assert raw_gap.shape[1] == dim, f"[{obs}/{source}] dim mismatch: {raw_gap.shape[1]} vs {dim}"

            mean = raw_gap.mean(axis=0)
            cov = np.cov(raw_gap, rowvar=False)

            w_eig = np.linalg.eigvalsh(cov)
            payload = dict(
                points=points, mean=mean.tolist(), cov=cov.tolist(), nominal=nominal.tolist(),
                n_toys=n_total, n_chunks=len(files), source=source,
                diagnostics=dict(min_eig=float(w_eig.min()), max_eig=float(w_eig.max()),
                                  condition_number=float(w_eig.max() / max(w_eig.min(), 1e-300))),
            )
            out_path = out_dir / f"residual_covariance_{obs}_{source}.json"
            with open(out_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[{obs}/{source}] merged {n_total} toys from {len(files)} chunks -> {out_path}")

            # Full toy ensemble (raw_gap per toy, not just mean+cov): needed to refit the joint
            # chi^2 model per toy for a genuine toy-based band (6_joint_fit.py / 6b_...).
            np.savez(out_dir / f"toy_ensemble_{obs}_{source}.npz",
                     raw_gap=raw_gap, thresholds=np.array(thresholds), n_toys=n_total)

            if source == "stat_ff_bfmode":
                if sem0_ref is None:
                    sem0_ref = sem0
                    np.savez(out_dir / "sem0_stat_ff_bfmode.npz", sem0=sem0,
                             columns=np.array(["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3",
                                                "q2_1", "q2_2", "q2_3"]))
                    print(f"  -> {out_dir / 'sem0_stat_ff_bfmode.npz'} ({n_total} toys, thr=0, all 9 obs)")
                elif sem0_ref.shape == sem0.shape:
                    max_rel = float(np.max(np.abs(sem0 - sem0_ref) / (np.abs(sem0_ref) + 1e-12)))
                    print(f"  [{obs}] sem0 cross-check vs first obs: max rel diff = {max_rel:.2e}"
                          + ("  (OK, same rng stream)" if max_rel < 1e-6 else "  !! MISMATCH, investigate"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--toy-job", action="store_true")
    p.add_argument("--obs", choices=OBS_LIST)
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=20)
    p.add_argument("--n-toys-total", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.toy_job:
        toys_root = Path(cfg["paths"]["output"]) / "5" / "toys"
        run_toy_job(cfg, args.obs, args.source, args.chunk, args.n_chunks, args.n_toys_total,
                    args.seed_base, toys_root)
    elif args.submit:
        run_submit(cfg, args.n_chunks, args.n_toys_total, args.dry_run)
    elif args.merge:
        run_merge(cfg)
    else:
        print("Nothing to do -- pass --submit, --merge, or --toy-job.")


if __name__ == "__main__":
    main()
