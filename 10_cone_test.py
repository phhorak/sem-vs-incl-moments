#!/usr/bin/env python3
"""Step 10: admissibility of the threshold-0 residual moments as a hypothesis test (claims C6, C5).

  d^2   Mahalanobis distance (toy covariance) from the residual moments m1..m3, scaled to the data
        support, to the moment space of positive densities; 0 when admissible. For three moments on
        [0, 1] its boundary is the set of two-point measures w delta_e + (1-w) delta_t with e in {0, 1},
        so the projection is a scan in t with the optimal w in closed form. Jointly (correlated
        observables): block coordinate descent over the three per-observable projections.
  p     parametric bootstrap: toy fluctuations re-centred on the closest admissible point, p = P(d^2* >= d^2).
  p_chi2  P(chi2_k >= d^2), k = number of moments, the Gaussian-toy approximation (chi-bar-square <= chi2_k).
        The toys are not Gaussian (c_true = B_incl/B_gap has a heavy tail): for one observable p_chi2 comes out
        smaller than the bootstrap p, i.e. it is not conservative there; the bootstrap p is the result. The
        component scans use the joint p_chi2 (k = 9), which was larger than the bootstrap p where both were checked.

Component scans (C7): each SEM category normalisation scaled by s (one at a time, data/sem_category_moments.json),
which changes the SEM moments and B_SEM, hence c_true; allowed range = s with joint p_chi2 > 0.05.

Per observable and jointly (full 9x9 covariance). SEM variants: nominal (resonant NR D(*)pipi) and
"phsp", the nominal SEM with the NR D(*)pipi component moments replaced by those of the 5-body PHSP
lineshape (data/nr_dpipi_moments.json; moments are linear in the component weights), and "split_Dp1_f" with
the D1':D0* weight fraction of NR D(*)pipi set to f (parent moments from the same file).

Usage:
  python3 10_cone_test.py [--bundle FILE] [--n-toys 2000]
  python3 10_cone_test.py --submit [--after JOB]
  python3 10_cone_test.py --nr-moments resonant=COCKTAIL phsp=COCKTAIL   # rebuild data/nr_dpipi_moments.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.bundle import Bundle, default_path, output_root
from lib.maxent import hausdorff_check, raw_to_mu01
from lib.systematics import c_true

OBS = ["mx2", "el", "q2"]
NR_JSON = HERE / "data" / "nr_dpipi_moments.json"
CAT_JSON = HERE / "data" / "sem_category_moments.json"
SCALES = np.round(np.arange(0.0, 3.001, 0.05), 2)
TG = np.linspace(0, 1, 4001)
G = np.stack([TG, TG ** 2, TG ** 3], 1)


def project(target, P):
    """Closest point (metric P) to target in the moment space of probability measures on [0, 1]."""
    if hausdorff_check(np.r_[1.0, target])[0]:
        return target
    L = np.linalg.cholesky(P).T
    best = (np.inf, None)
    for e in (0.0, 1.0):
        D = np.array([e, e ** 2, e ** 3]) - G                    # direction to the endpoint atom
        R, LD = (L @ (target - G).T).T, (L @ D.T).T
        w = np.clip(np.sum(R * LD, 1) / np.maximum(np.sum(LD * LD, 1), 1e-300), 0, 1)
        d2 = np.sum((R - w[:, None] * LD) ** 2, 1)
        i = int(np.argmin(d2))
        if d2[i] < best[0]:
            best = (d2[i], G[i] + w[i] * D[i])
    return best[1]


def distance(mu, P, nb, iters=50):
    """d^2 from mu (scaled moments 1..3 per block) to the product of moment spaces, metric P."""
    nu = mu.copy()
    for _ in range(iters if nb > 1 else 1):
        prev = nu.copy()
        for i in range(nb):
            s, o = slice(3 * i, 3 * i + 3), np.r_[0:3 * i, 3 * i + 3:3 * nb]
            Pii = P[s, s]
            target = mu[s] + np.linalg.solve(Pii, P[s][:, o] @ (mu[o] - nu[o])) if nb > 1 else mu[s]
            nu[s] = project(target, Pii)
        if np.max(np.abs(nu - prev)) < 1e-10:
            break
    r = mu - nu
    return float(r @ P @ r), nu


def test(mu_nom, mu_toys, blocks):
    P = np.linalg.inv(np.cov(mu_toys, rowvar=False))
    d2, nu = distance(mu_nom, P, len(blocks))
    if d2 < 1e-9:
        return dict(d2=0.0, p=1.0, z=0.0, p_chi2=1.0, z_chi2=0.0)
    boot = np.array([distance(nu + dm, P, len(blocks))[0] for dm in mu_toys - mu_toys.mean(0)])
    p = float(np.mean(boot >= d2))
    pc = float(chi2.sf(d2, len(mu_nom)))
    return dict(d2=d2, p=p, p_floor=1 / len(boot), z=float(norm.isf(max(p, 0.5 / len(boot)))), p_chi2=pc, z_chi2=float(norm.isf(pc)))


def nr_moments(specs):
    import pyarrow.parquet as pq
    out = {"_doc": NR_JSON.name + ": raw moments 1..3 (Mx2, El, q2) of the NR D(*)pipi category and its SEM weight share"}
    for spec in specs:
        name, path = spec.split("=", 1)
        s = {k: np.zeros(10) for k in ("nr", "all")}
        for b in pq.ParquetFile(path).iter_batches(columns=["category", "total_weight", "Mx", "El_B", "q2"], batch_size=2_000_000):
            d = b.to_pandas()
            w = d.total_weight.to_numpy(float)
            x = np.stack([d.Mx.to_numpy(float) ** 2, d.El_B.to_numpy(float), d.q2.to_numpy(float)])
            ok = np.isfinite(w) & (w > 0) & np.isfinite(x).all(0)
            for k, m in (("nr", ok & (d.category.astype(str) == "D(*) pi pi").to_numpy()), ("all", ok)):
                s[k] += np.r_[w[m].sum(), [(w[m] * x[i, m] ** p).sum() for i in range(3) for p in (1, 2, 3)]]
        out[name] = dict(share=s["nr"][0] / s["all"][0], moments=(s["nr"][1:] / s["nr"][0]).tolist(), source=path)
    json.dump(out, open(NR_JSON, "w"), indent=1)


def run(B, n_toys, seed, support=()):
    cfg, T = B.cfg, B.toys
    for spec in support:
        o, v = spec.split("=")
        cfg["data"]["support"][o] = [float(x) for x in v.split(",")]
    sup, sysc = cfg["data"]["support"], cfg["systematics"]
    bf_gap = float(T["bf_gap"])
    c0 = float(c_true(sysc, bf_gap))
    rng = np.random.default_rng(seed)
    n = min(n_toys, len(T["z_gap"]))
    ih = rng.choice(len(T["hqe_toys"]), n, replace=False)
    c = c_true(sysc, bf_gap, T["z_gap"][:n])[:, None]
    idx = np.r_[[MAX_ORDER * k + j for k in range(3) for j in range(3)]]
    sem_c, sem_t = T["nom_sem0"][idx], T["sem0_stat_ff_bfmode"][:n][:, idx]
    nr = json.load(open(NR_JSON))
    res_m = np.array(nr["resonant"]["moments"])
    shift = {"nominal": np.zeros(9), "phsp": nr["resonant"]["share"] * (np.array(nr["phsp"]["moments"]) - res_m)}
    for f in (0.0, 0.25, 0.75, 1.0):   # D1' weight fraction of NR D(*)pipi (nominal 0.48 after weights)
        m = f * np.array(nr["parent_Dp1"]["moments"]) + (1 - f) * np.array(nr["parent_D0st"]["moments"])
        shift[f"split_Dp1_{f:.2f}"] = nr["resonant"]["share"] * (m - res_m)
    scale = lambda raw: np.concatenate([raw_to_mu01(np.c_[np.ones(len(raw)), raw[:, 3 * k:3 * k + 3]].T, *sup[o])[1:].T
                                        for k, o in enumerate(OBS)], axis=1)
    summary = {}
    for variant, dm in shift.items():
        r_nom = c0 * T["hqe_central"] - (c0 - 1) * (sem_c + dm)
        r_toy = c * T["hqe_toys"][ih] - (c - 1) * (sem_t + dm)
        mu, mt = scale(r_nom[None])[0], scale(r_toy)
        res = {o: test(mu[3 * k:3 * k + 3], mt[:, 3 * k:3 * k + 3], [o]) for k, o in enumerate(OBS)}
        res["joint"] = test(mu, mt, OBS)
        summary[variant] = res
        print(f"[{variant}] " + "; ".join(f"{k}: d2 {v['d2']:.2f}, p {v['p']:.4f} (Z {v['z']:.1f}), p_chi2 {v['p_chi2']:.1e}"
                                       for k, v in res.items()),
              flush=True)
    cats, b_incl = json.load(open(CAT_JSON)), sysc["bf_incl"]
    scan = {}
    for cat, v in cats.items():
        rows = []
        for sc in SCALES:
            k = 1 + (sc - 1) * v["share"]
            gap = b_incl - (b_incl - bf_gap) * k
            if gap <= 0:
                rows.append(dict(s=float(sc), valid=False))
                continue
            cs = b_incl / gap
            ct = cs / c0 * c                                    # keep the relative B_gap draws of the toys
            sem = lambda m: (m + (sc - 1) * v["share"] * np.array(v["moments"])) / k
            mu = scale((cs * T["hqe_central"] - (cs - 1) * sem(sem_c))[None])[0]
            mt = scale(ct * T["hqe_toys"][ih] - (ct - 1) * sem(sem_t))
            P = np.linalg.inv(np.cov(mt, rowvar=False))
            d2 = distance(mu, P, 3)[0]
            rows.append(dict(s=float(sc), valid=True, c_true=float(cs), d2=d2, p_chi2=float(chi2.sf(d2, 9)),
                             admissible={o: float(np.mean([hausdorff_check(np.r_[1.0, m])[0] for m in mt[:, 3 * i:3 * i + 3]]))
                                         for i, o in enumerate(OBS)}))
        ok = [r["s"] for r in rows if r["valid"] and r["p_chi2"] > 0.05]
        scan[cat] = dict(share=v["share"], allowed=[min(ok), max(ok)] if ok else None, rows=rows)
        print(f"[scan {cat}] share {v['share']:.3f}: allowed scale (joint p_chi2 > 0.05) {scan[cat]['allowed']}", flush=True)
    summary["component_scan"] = scan

    od = output_root(cfg, HERE) / "10"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_toys=n, c_true=c0, nr_shift=shift["phsp"].tolist(), **summary), open(od / "summary.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt
    fd = HERE / "figures" / "10"
    fd.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
    for cat, v in scan.items():
        r = [x for x in v["rows"] if x["valid"]]
        ax.plot([x["s"] for x in r], [max(x["p_chi2"], 1e-12) for x in r], lw=2.6, label=f"{cat} ({v['share']:.1%})")
    ax.axhline(0.05, color="0.5", ls="--", lw=1.2)
    ax.axvline(1, color="0.5", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 2)
    ax.set_xlim(SCALES[0], SCALES[-1])
    ax.set_xlabel("scale of the SEM component normalisation", fontsize=24)
    ax.set_ylabel(r"joint $p$ ($\chi^2_9$ approximation)", fontsize=24)
    ax.legend(fontsize=17, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fd / "component_scan.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {od}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle")
    p.add_argument("--n-toys", type=int, default=2000)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after")
    p.add_argument("--nr-moments", nargs="+", metavar="NAME=COCKTAIL")
    p.add_argument("--support", nargs="+", default=(), metavar="OBS=LO,HI", help="override the data support")
    args = p.parse_args()
    if args.nr_moments:
        nr_moments(args.nr_moments)
        return
    bundle = Path(args.bundle) if args.bundle else default_path(HERE)
    if args.submit:
        w = f' -w "done({args.after})"' if args.after else ""
        (HERE / "logs" / "10").mkdir(parents=True, exist_ok=True)
        cmd = (f'bsub -q h -env all -J s10cone{w} -n 2 -oo {HERE}/logs/10/run.log '
               f'"cd {HERE} && python3 10_cone_test.py --bundle {bundle.resolve()} --n-toys {args.n_toys}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        return
    run(Bundle(bundle), args.n_toys, args.seed, args.support)


if __name__ == "__main__":
    main()
