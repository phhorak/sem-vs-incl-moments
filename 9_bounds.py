#!/usr/bin/env python3
"""Step 9: model-independent statements on the threshold-0 gap (HQE route, moments m1..m3).

  CDF bounds      min/max of P(x <= X) over all positive densities on the data support with the
                  residual moments (linear programs), nominal and per toy (claim C8)
  Support tests   minimal gap fraction outside the kinematic range of a single final-state class
                  (Lambda_c pbar, D_s K), per observable; > 0 means the class alone cannot be the gap (C9)
  Gap-fraction    admissible fraction of toys vs the assumed B_gap/B_incl, and the bootstrap p of the
                  admissibility test (lib/cone.py) with its 95% CL lower limit on B_gap/B_incl

Toys pair the step-6 SEM toys (stat+FF+BF per mode) and gap-normalisation draws with HQE-fit toys.
Reads the step-6 bundle only. Writes output/9/summary.json and figures/9/.

Usage:
  python3 9_bounds.py [--bundle FILE] [--n-toys 2000]
  python3 9_bounds.py --submit [--after JOB]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.bundle import Bundle, default_path, output_root
from lib.cone import test
from lib.maxent import hausdorff_check, raw_to_mu01
from lib.systematics import bf_gap_sigma, c_true

OBS = ["mx2", "el", "q2"]
M_B = 5.27934
# single-class hypotheses: lightest final state; the class lives at M_X^2 >= m^2, q^2 <= (M_B-m)^2, E_l <= (M_B^2-m^2)/(2 M_B)
CLASSES = {"LcP": 2.28646 + 0.93827, "DsK": 1.96835 + 0.49368}
QUERIES = {"el": [0.5, 0.8, 1.0, 1.5], "q2": [2.0, 4.0, 6.0, 8.0], "mx2": [6.0, 8.0, 10.0]}
TG = np.linspace(0, 1, 401)
A = np.stack([TG ** k for k in range(4)])


def lp(c, mu):
    r = linprog(c, A_eq=A, b_eq=mu, bounds=(0, None), method="highs")
    return r.fun if r.status == 0 else np.nan


def p_bounds(mu, t):
    """[min, max] of P(scaled variable <= t) for each t."""
    ind = (TG[None, :] <= np.asarray(t)[:, None]).astype(float)
    return np.array([[lp(i, mu), -lp(-i, mu)] for i in ind])


def run(B, n_toys, seed):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt

    cfg, T = B.cfg, B.toys
    sup, sysc = cfg["data"]["support"], cfg["systematics"]
    bf_gap = float(T["bf_gap"])
    f_nom, c0 = bf_gap / sysc["bf_incl"], float(c_true(sysc, bf_gap))
    rng = np.random.default_rng(seed)
    n = min(n_toys, len(T["z_gap"]))
    ih = rng.choice(len(T["hqe_toys"]), n, replace=False)
    c = c_true(sysc, bf_gap, T["z_gap"][:n])[:, None]
    incl = {o: (T["hqe_central"][3 * k:3 * k + 3], T["hqe_toys"][ih, 3 * k:3 * k + 3]) for k, o in enumerate(OBS)}
    sem = {o: (T["nom_sem0"][MAX_ORDER * k:MAX_ORDER * k + 3], T["sem0_stat_ff_bfmode"][:n, MAX_ORDER * k:MAX_ORDER * k + 3])
           for k, o in enumerate(OBS)}
    mu01 = lambda r, o: raw_to_mu01(np.r_[1.0, r], *sup[o])
    nom = {o: mu01(c0 * incl[o][0] - (c0 - 1) * sem[o][0], o) for o in OBS}
    toys = {o: np.array([mu01(r, o) for r in c * incl[o][1] - (c - 1) * sem[o][1]]) for o in OBS}
    adm = {o: np.array([hausdorff_check(m)[0] for m in toys[o]]) for o in OBS}
    scale = lambda x, o: (np.asarray(x, float) - sup[o][0]) / (sup[o][1] - sup[o][0])
    summary = dict(f_nom=f_nom, c_true=c0, n_toys=n, admissible={o: float(adm[o].mean()) for o in OBS})

    # ── CDF bounds ──
    xt = np.linspace(0, 1, 41)
    cdf = {}
    for o in OBS:
        b_nom = p_bounds(nom[o], xt) if hausdorff_check(nom[o])[0] else np.full((len(xt), 2), np.nan)
        bt = np.array([p_bounds(m, xt) for m in toys[o][adm[o]]])
        q = {x: p_bounds(nom[o], scale([x], o))[0].tolist() for x in QUERIES[o]}
        qt = {x: np.nanpercentile(np.array([p_bounds(m, scale([x], o))[0] for m in toys[o][adm[o]]]), [16, 84], 0).tolist()
              for x in QUERIES[o]}
        cdf[o] = dict(x=sup[o][0] + xt * (sup[o][1] - sup[o][0]), nom=b_nom, lo68=np.nanpercentile(bt[:, :, 0], 16, 0),
                      hi68=np.nanpercentile(bt[:, :, 1], 84, 0))
        summary.setdefault("cdf", {})[o] = {f"P(x<={x})": dict(nominal=q[x], toys68_min_max=[qt[x][0][0], qt[x][1][1]])
                                            for x in QUERIES[o]}
        print(f"[cdf {o}] " + "; ".join(f"P(<={x}) in [{q[x][0]:.3f}, {q[x][1]:.3f}]" for x in QUERIES[o]), flush=True)

    # ── support tests: minimal gap mass outside each class's kinematic range ──
    for cls, m in CLASSES.items():
        edge = {"mx2": (m ** 2, "above"), "el": ((M_B ** 2 - m ** 2) / (2 * M_B), "below"), "q2": ((M_B - m) ** 2, "below")}
        res = {}
        for o in OBS:
            x, side = edge[o]
            t = scale([x], o)
            # class region in scaled variable: t >= edge (mx2) or t <= edge (el, q2); outside mass = 1 - max P(region)
            out = lambda mu: 1 - (-lp(-(TG >= t[0]).astype(float), mu) if side == "above" else p_bounds(mu, t)[0][1])
            o_nom = out(nom[o]) if hausdorff_check(nom[o])[0] else np.nan
            o_toy = np.array([out(mu) for mu in toys[o][adm[o]]])
            res[o] = dict(edge=float(x), outside_min_nominal=float(o_nom),
                          outside_min_toys68=np.nanpercentile(o_toy, [16, 50, 84]).tolist(),
                          frac_toys_class_alone_allowed=float(np.mean(o_toy < 1e-6)))
            print(f"[support {cls} {o}] edge {x:.3f}: gap outside >= {o_nom:.3f} (toys 68%: "
                  f"{res[o]['outside_min_toys68'][0]:.3f}-{res[o]['outside_min_toys68'][2]:.3f}), "
                  f"class alone allowed in {res[o]['frac_toys_class_alone_allowed']:.1%}", flush=True)
        summary.setdefault("support", {})[cls] = res

    # ── admissibility vs assumed gap fraction ──
    fs = np.linspace(0.04, 0.40, 73)
    scan = {o: [float(np.mean([hausdorff_check(mu01(r, o))[0] for r in (1 / f) * incl[o][1] - (1 / f - 1) * sem[o][1]]))
                for f in fs] for o in OBS}
    summary["gap_fraction_scan"] = dict(f=fs.tolist(), **scan,
                                        f_min_50pct={o: float(fs[np.argmax(np.array(scan[o]) >= 0.5)]) if max(scan[o]) >= 0.5
                                                     else None for o in OBS})
    print("[scan] f at 50% admissible: " + ", ".join(f"{o} {v}" for o, v in summary["gap_fraction_scan"]["f_min_50pct"].items()))
    fp = np.round(np.arange(0.03, 0.2501, 0.005), 3)
    pscan = {o: [test(mu01((1 / f) * incl[o][0] - (1 / f - 1) * sem[o][0], o)[1:],
                      np.array([mu01(r, o)[1:] for r in (1 / f) * incl[o][1] - (1 / f - 1) * sem[o][1]]))["p"] for f in fp]
             for o in OBS}
    lim = {o: next((float(f) for f, p in zip(fp, pscan[o]) if p >= 0.05), None) for o in OBS}
    summary["gap_fraction_test"] = dict(f=fp.tolist(), p=pscan, limit_95=lim)
    print("[scan] 95% CL lower limit on B_gap/B_incl: " + ", ".join(f"{o} {v}" for o, v in lim.items()), flush=True)

    od, fd = output_root(cfg, HERE) / "9", HERE / "figures" / "9"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)

    XL = {"mx2": r"$M_X^2\ [\mathrm{GeV}^2]$", "el": r"$E_\ell^B\ [\mathrm{GeV}]$", "q2": r"$q^2\ [\mathrm{GeV}^2]$"}
    COL = {"mx2": "#1f77b4", "el": "#ff7f0e", "q2": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.2), dpi=200)
    for ax, o in zip(axes, OBS):
        b = cdf[o]
        ax.fill_between(b["x"], b["lo68"], b["hi68"], color=COL[o], alpha=0.18, lw=0)
        ax.fill_between(b["x"], b["nom"][:, 0], b["nom"][:, 1], color=COL[o], alpha=0.5, lw=0)
        for cls, m in CLASSES.items():
            x = {"mx2": m ** 2, "el": (M_B ** 2 - m ** 2) / (2 * M_B), "q2": (M_B - m) ** 2}[o]
            ax.axvline(x, color="0.3", ls="--" if cls == "LcP" else ":", lw=1.6)
            ax.text(x, 1.04, {"LcP": r"$\Lambda_c\bar p$", "DsK": r"$D_sK$"}[cls], ha="center", fontsize=15, color="0.3")
        ax.set_xlim(*sup[o])
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(XL[o], fontsize=24)
        ax.set_ylabel(r"$P(\mathrm{gap} \leq x)$", fontsize=24)
    fig.tight_layout()
    fig.savefig(fd / "cdf_bounds.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
    rel = bf_gap_sigma(sysc) / bf_gap
    ax.axvspan(f_nom * (1 - rel), f_nom * (1 + rel), color="0.88", lw=0)
    ax.axvline(f_nom, color="0.5", lw=1.2, ls="--")
    for o in OBS:
        ax.plot(fs, scan[o], color=COL[o], lw=2.8, label=XL[o].split(r"\ ")[0] + "$")
    ax.set_xlim(fs[0], fs[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"assumed $\mathcal{B}_\mathrm{gap}/\mathcal{B}_\mathrm{incl}$", fontsize=24)
    ax.set_ylabel("admissible fraction of toys", fontsize=24)
    ax.legend(fontsize=20, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fd / "gap_fraction_scan.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {fd}, {od}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle")
    p.add_argument("--n-toys", type=int, default=2000)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after")
    args = p.parse_args()
    bundle = Path(args.bundle) if args.bundle else default_path(HERE)
    if args.submit:
        w = f' -w "done({args.after})"' if args.after else ""
        (HERE / "logs" / "9").mkdir(parents=True, exist_ok=True)
        cmd = (f'bsub -q h -env all -J s9bounds{w} -n 2 -oo {HERE}/logs/9/run.log '
               f'"cd {HERE} && python3 9_bounds.py --bundle {bundle.resolve()} --n-toys {args.n_toys}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        return
    run(Bundle(bundle), args.n_toys, args.seed)


if __name__ == "__main__":
    main()
