#!/usr/bin/env python3
"""Step 5: toy ensembles shared by step 6 (Asimov closure) and step 7 (data).

Every toy draws, independently:
  SEM template      cocktail bootstrap A, weights x {1, FF, BF-mode, FF*BF-mode}
                    -> raw moments at the data thresholds (orders 1-3) and at thr=0 (orders 1-MAX_ORDER)
  pseudo-inclusive  cocktail bootstrap B and a bootstrap of each Asimov gap truth (thr=0)
  experiment        draw from the averaged raw-moment covariance (data thresholds)
  bf_gap            z for the gap budget: B_gap' = B_gap + z*sigma, c_true = B_incl / B_gap'
  SEM stack         binned SEM densities (FF*BF-mode variant) for the step-7 stack band

Outputs (output/5/):
  toys.npz                                  all per-toy components + nominal values
  residual_covariance_{obs}_{source}.json   data-side residual mean/cov per systematic source
  toy_ensemble_{obs}_{source}.npz           the corresponding per-toy residuals

Usage:
  python3 5_residual_covariance.py --submit [--dry-run]   # toy chunks + dependent merge job
  python3 5_residual_covariance.py --toy-job --chunk 0
  python3 5_residual_covariance.py --merge
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER, power_matrix, truth_groups, support
from lib.moments import build_curve_context, compute_raw_curves_from_context
from lib.systematics import (
    read_parquet_downcast, build_ff_slope_matrix, sample_ff_multiplier, bf_mode_setup,
    sample_bf_multiplier, sample_mvn, compute_bf_sem_gap, bf_gap_sigma, c_true,
)

OBS = ["Mx", "El", "Q2"]
OBS_KEYS = {"Mx": ["mx_1", "mx_2", "mx_3"], "El": ["el_1", "el_2", "el_3"],
            "Q2": ["q2_1", "q2_2", "q2_3"]}
VARIANTS = {"stat": (False, False), "stat_ff": (True, False),
            "stat_bfmode": (False, True), "stat_ff_bfmode": (True, True)}
DATA_SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode", "all"]   # "all" adds bf_gap
BASE_COLS = ["Mx", "El_B", "q2", "total_weight", "decay_name"]


def out_dir(cfg):
    return Path(cfg["paths"]["output"]) / "5"


def thresholds(cfg, exp_avg):
    """Measured thresholds inside the support. Mx moments are reported at El cuts."""
    sup = cfg["data"]["support"]
    dom = {"Mx": sup["el"], "El": sup["el"], "Q2": sup["q2"]}
    return {o: [float(c) for c in sorted(exp_avg["average_raw"][OBS_KEYS[o][0]]["cuts"])
                if dom[o][0] <= c < dom[o][1]] for o in OBS}


def stack_edges(cfg):
    """Bins of the step-7 SEM stack: Mx [GeV], El [GeV], q2 [GeV^2] over the data support."""
    sup = cfg["data"]["support"]
    return {"Mx": np.linspace(*np.sqrt(sup["mx2"]), 80), "El": np.linspace(*sup["el"], 80),
            "Q2": np.linspace(*sup["q2"], 80)}


def bin_index(x, edges):
    """Bin index per event, len(edges)-1 for out of range (dropped after bincount)."""
    i = np.digitize(x, edges) - 1
    i[(i < 0) | (i >= len(edges) - 1)] = len(edges) - 1
    return i.astype(np.int16)


def gap_truth_moments(groups, counts_B, rng):
    """Raw moments (3*MAX_ORDER,) of the gap truth; cocktail-derived groups reuse bootstrap B."""
    m = np.zeros(3 * MAX_ORDER)
    for g in groups:
        if counts_B is None:
            c = 1.0
        elif g["src"] == "cocktail":
            c = counts_B[g["rows"]]
        else:
            c = np.bincount(rng.integers(0, len(g["w"]), len(g["w"])), minlength=len(g["w"]))
        wc = g["w"] * c
        m += g["w"].sum() * (g["pw"] @ wc) / wc.sum()
    return m


def exp_draws(exp_avg, thr, n, rng):
    """Per obs: (n, n_thr, 3) draws of the averaged raw moments, linearly interpolated from the
    covariance nodes to the thresholds."""
    cd = exp_avg["average_raw_cov"]
    pts, cov = cd["points"], np.array(cd["cov"])
    avg = exp_avg["average_raw"]
    mean = np.array([np.interp(p["cut"], avg[p["key"]]["cuts"], avg[p["key"]]["values"]) for p in pts])
    samples = mean + sample_mvn(cov, rng, size=n)
    out = {}
    for o in OBS:
        vals = np.empty((n, len(thr[o]), 3))
        for a, key in enumerate(OBS_KEYS[o]):
            idx = sorted((i for i, p in enumerate(pts) if p["key"] == key), key=lambda i: pts[i]["cut"])
            cuts = np.array([pts[i]["cut"] for i in idx])
            W = np.array([np.interp(t, cuts, np.eye(len(cuts))[j]) for t in thr[o]
                          for j in range(len(cuts))]).reshape(len(thr[o]), len(cuts))
            vals[:, :, a] = samples[:, idx] @ W.T
        out[o] = vals
    return out


def sem_curves(ctx, w, thr, el_cuts, q2_cuts):
    curves = compute_raw_curves_from_context(ctx, w, el_cuts, q2_cuts)
    out = {}
    for o in OBS:
        cuts = q2_cuts if o == "Q2" else el_cuts
        idx = np.searchsorted(cuts, thr[o])
        out[o] = np.stack([curves[k][idx] for k in OBS_KEYS[o]], axis=1)
    return out


# ── Toy chunk ────────────────────────────────────────────────────────────────

def run_toy_job(cfg, chunk):
    tc = cfg["toys"]
    n_per = -(-tc["n_toys"] // tc["n_chunks"])
    n = min(n_per, tc["n_toys"] - chunk * n_per)
    if n <= 0:
        return
    rng = np.random.default_rng([tc["seed"], chunk])
    exp_avg = json.load(open(Path(cfg["paths"]["output"]) / "4" / "experimental_average.json"))
    thr = thresholds(cfg, exp_avg)
    el_cuts = np.array(sorted(set(thr["Mx"]) | set(thr["El"])))
    q2_cuts = np.array(thr["Q2"])

    t0 = time.time()
    path = Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet"
    ff_cols = ["ff_weight"] + [c for c in pq.ParquetFile(path).schema_arrow.names
                               if c.startswith(("ff_weight_up", "ff_weight_down"))]
    df = read_parquet_downcast(path, BASE_COLS + ["bf", "bf_unc"] + ff_cols, float32_cols=ff_cols)
    df = df[np.isfinite(df[BASE_COLS[:4]].to_numpy()).all(axis=1) & (df["total_weight"] > 0)]
    df = df.reset_index(drop=True)
    bf_sem, bf_gap = compute_bf_sem_gap(df, cfg["systematics"]["bf_incl"])
    mx2, el, q2 = df["Mx"].to_numpy(float) ** 2, df["El_B"].to_numpy(float), df["q2"].to_numpy(float)
    w_all = df["total_weight"].to_numpy(float)
    codes = df["decay_name"].cat.codes.to_numpy()
    cats = np.asarray(df["decay_name"].cat.categories, dtype=str)
    ff_slopes, _ = build_ff_slope_matrix(df)
    bf_rel, bf_fam = bf_mode_setup(codes, cats, df["bf"].to_numpy(), df["bf_unc"].to_numpy())
    del df
    ctx = build_curve_context(mx2, el, q2)
    edges = stack_edges(cfg)
    sidx = {o: bin_index(x, edges[o]) for o, x in zip(OBS, (np.sqrt(mx2), el, q2))}
    P = power_matrix(mx2, el, q2)
    ck = dict(mx2=mx2, el=el, q2=q2, w=w_all, codes=codes, cats=cats)
    gap = {}
    for s in cfg["asimov"]["scenarios"]:
        gap[s] = truth_groups(cfg, s, ck)
        for g in gap[s]:
            g["pw"] = P[:, g["rows"]] if g["src"] == "cocktail" else power_matrix(g["mx2"], g["el"], g["q2"])
    sup = {f"support_{s}": np.array([support(g)[o] for o in ("mx2", "el", "q2")]) for s, g in gap.items()}
    del mx2, el, q2, ck
    N = len(w_all)
    print(f"chunk {chunk}: setup {time.time() - t0:.0f}s, N={N:,}, {n} toys, "
          f"gap fraction B_gap/B_incl={bf_gap / cfg['systematics']['bf_incl']:.4f}", flush=True)

    rec = {f"sem_{o}_{v}": np.empty((n, len(thr[o]), 3)) for o in OBS for v in VARIANTS}
    rec.update({f"sem0_{v}": np.empty((n, 3 * MAX_ORDER)) for v in VARIANTS})
    rec.update({f"gap_{s}": np.empty((n, 3 * MAX_ORDER)) for s in gap})
    rec["ck_B"] = np.empty((n, 3 * MAX_ORDER))
    rec.update({f"stack_{o}": np.empty((n, len(e) - 1)) for o, e in edges.items()})
    rec.update({f"exp_{o}": v for o, v in exp_draws(exp_avg, thr, n, rng).items()})
    rec["z_gap"] = rng.standard_normal(n)

    t0 = time.time()
    for i in range(n):
        base = w_all * np.bincount(rng.integers(0, N, N), minlength=N)
        ff = sample_ff_multiplier(ff_slopes, rng)
        bfm = sample_bf_multiplier(codes, bf_rel, bf_fam, rng)
        for v, (use_ff, use_bf) in VARIANTS.items():
            w = base * (ff if use_ff else 1.0) * (bfm if use_bf else 1.0)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            rec[f"sem0_{v}"][i] = P @ w / w.sum()
            for o, vals in sem_curves(ctx, w, thr, el_cuts, q2_cuts).items():
                rec[f"sem_{o}_{v}"][i] = vals
            if v == "stat_ff_bfmode":
                for o, e in edges.items():
                    rec[f"stack_{o}"][i] = np.bincount(sidx[o], w, len(e))[:-1] / (w.sum() * np.diff(e))
        counts_B = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        wB = w_all * counts_B
        rec["ck_B"][i] = P @ wB / wB.sum()
        for s, groups in gap.items():
            rec[f"gap_{s}"][i] = gap_truth_moments(groups, counts_B, rng)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}  {(time.time() - t0) / (i + 1):.1f}s/toy", flush=True)

    nom = {"nom_sem0": P @ w_all / w_all.sum()}
    nom.update({f"nom_sem_{o}": v for o, v in sem_curves(ctx, w_all, thr, el_cuts, q2_cuts).items()})
    nom.update({f"nom_gap_{s}": gap_truth_moments(g, None, rng) for s, g in gap.items()})
    d = out_dir(cfg) / "toys"
    d.mkdir(parents=True, exist_ok=True)
    nom.update({f"stackedges_{o}": e for o, e in edges.items()})
    np.savez(d / f"chunk_{chunk:04d}.npz", bf_sem=bf_sem, bf_gap=bf_gap, **rec, **nom, **sup,
             **{f"thr_{o}": np.array(thr[o]) for o in OBS})
    print(f"chunk {chunk}: {n} toys in {time.time() - t0:.0f}s", flush=True)


# ── Merge ────────────────────────────────────────────────────────────────────

def run_merge(cfg):
    od = out_dir(cfg)
    files = sorted((od / "toys").glob("chunk_*.npz"))
    parts = [dict(np.load(f)) for f in files]
    keys = [k for k in parts[0] if not k.startswith(("nom_", "thr_", "bf_", "support_", "stackedges_"))]
    T = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    T.update({k: v for k, v in parts[0].items() if k.startswith(("nom_", "thr_", "bf_", "support_", "stackedges_"))})
    n = len(T["z_gap"])
    bf_gap, sysc = float(T["bf_gap"]), cfg["systematics"]
    print(f"merged {n} toys from {len(files)}/{cfg['toys']['n_chunks']} chunks, gap fraction "
          f"B_gap/B_incl = {bf_gap / sysc['bf_incl']:.4f} (rel. unc. {bf_gap_sigma(sysc) / bf_gap:.3f}), "
          f"c_true = {float(c_true(sysc, bf_gap)):.3f}")
    np.savez(od / "toys.npz", **T)

    c_nom = float(c_true(cfg["systematics"], float(T["bf_gap"])))
    c_toy = c_true(cfg["systematics"], float(T["bf_gap"]), T["z_gap"])[:, None, None]
    exp_avg = json.load(open(Path(cfg["paths"]["output"]) / "4" / "experimental_average.json"))
    for o in OBS:
        thr = T[f"thr_{o}"]
        exp_nom = np.stack([np.interp(thr, exp_avg["average_raw"][k]["cuts"],
                                      exp_avg["average_raw"][k]["values"]) for k in OBS_KEYS[o]], 1)
        nominal = (c_nom * exp_nom - (c_nom - 1) * T[f"nom_sem_{o}"]).reshape(-1)
        points = [{"obs": o, "thr": float(t), "order": k} for t in thr for k in (1, 2, 3)]
        for src in DATA_SOURCES:
            sem = T[f"sem_{o}_{'stat_ff_bfmode' if src == 'all' else src}"]
            c = c_toy if src == "all" else c_nom
            raw_gap = (c * T[f"exp_{o}"] - (c - 1) * sem).reshape(n, -1)
            cov = np.cov(raw_gap, rowvar=False)
            json.dump(dict(points=points, mean=raw_gap.mean(0).tolist(), cov=cov.tolist(),
                           nominal=nominal.tolist(), n_toys=n, source=src),
                      open(od / f"residual_covariance_{o}_{src}.json", "w"))
            np.savez(od / f"toy_ensemble_{o}_{src}.npz", raw_gap=raw_gap, thresholds=thr)
        print(f"  {o}: {len(thr)} thresholds, sources {DATA_SOURCES}")


# ── Submit ───────────────────────────────────────────────────────────────────

def run_submit(cfg, config_path, dry_run):
    logs = HERE / "logs" / "5"
    logs.mkdir(parents=True, exist_ok=True)
    queue, tag = cfg["generation"]["queue"], "s5toy"
    py = f"cd {HERE} && python3 5_residual_covariance.py --config {config_path}"
    cmds = [f'bsub -q {queue} -env all -J {tag}{c} -n 6 -oo {logs}/chunk_{c:04d}.log "{py} --toy-job --chunk {c}"'
            for c in range(cfg["toys"]["n_chunks"])]
    cmds.append(f'bsub -q {queue} -env all -J {tag}_merge -w "ended({tag}*)" -n 2 -oo {logs}/merge.log "{py} --merge"')
    for cmd in cmds:
        print(cmd)
        if not dry_run:
            subprocess.run(cmd, shell=True, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--toy-job", action="store_true")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.toy_job:
        run_toy_job(cfg, args.chunk)
    elif args.merge:
        run_merge(cfg)
    elif args.submit:
        run_submit(cfg, args.config, args.dry_run)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
