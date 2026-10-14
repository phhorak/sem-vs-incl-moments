#!/usr/bin/env python3
"""Step 7: tests on the threshold-0 moments, Results 1-3 of the paper.

Result 1  the gap is not more of the known modes. Inclusive moments vs a model, d^2 with the toy covariance and p
          from the re-centred toys, no c_true amplification:
    shape_test       gap with the SEM shape (all SEM BFs low by a common factor), for every SEM variant
    mode_shape_test  gap with the shape of one SEM mode X: inclusive vs (1-f) SEM + f X, at the nominal gap fraction
                     f (toys vary it with B_gap) or with f free in [0, 1] (closed-form fit, bootstrap refit)
Result 2  where the gap can be and how big it must be:
    cdf              min/max of P(x <= X) over all positive densities on the data support with the gap moments
                     (linear programs), nominal and per toy
    support          minimal gap fraction outside the kinematic range of a single final-state class (Lambda_c pbar,
                     D_s K), per observable; > 0 means the class alone cannot be the gap
    gap_fraction_*   admissible fraction of toys vs the assumed B_gap/B_incl, and the bootstrap p of the
                     admissibility test with its 95% CL lower limit on B_gap/B_incl
Result 3  the M_X^2 moments check SEM lineshapes:
    variants         admissibility (lib/cone.py) of the gap moments for each SEM variant (lib/variants.py), and for
                     phsp_Dpipi, bare_evtgen_broad with sigma(B_incl) doubled: sigma(B_gap)^2 + 3 sigma(B_incl)^2
    supports         nominal admissibility for alternative data supports; q2 <= 6 is deliberately too narrow

Toys: the shared pairing of lib/gap.py. Reads the step-4 bundle. Writes output/7/summary.json and figures/7/.

Usage:
  python3 7_tests.py [--bundle FILE]
  python3 7_tests.py --submit [--after JOB]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linprog
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import variants as V
from lib.bundle import Bundle, default_path, output_root
from lib.cone import test
from lib.gap import SL, Gap
from lib.maxent import hausdorff_check, raw_to_mu01
from lib.systematics import bf_gap_sigma

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


SUPPORTS = {"el_2.26": {"el": [0.0, 2.26]}, "el_2.40": {"el": [0.0, 2.4]}, "mx2_12": {"mx2": [3.2, 12.0]},
            "mx2_20": {"mx2": [3.2, 20.0]}, "q2_12.5": {"q2": [0.0, 12.5]}, "q2_6_wrong": {"q2": [0.0, 6.0]}}


def compare(dl, dt):
    """d^2 of dl with the covariance of the toys dt, and p from the re-centred toys."""
    out = {}
    for key, sl in SL.items():
        P = np.linalg.inv(np.cov(dt[:, sl], rowvar=False))
        fl = dt[:, sl] - dt[:, sl].mean(0)
        d2, d2t = float(dl[sl] @ P @ dl[sl]), np.einsum("ij,jk,ik->i", fl, P, fl)
        p = float(np.mean(d2t >= d2))
        out[key] = dict(d2=d2, p=p, z=float(norm.isf(max(p, 0.5 / len(d2t)))))
    return out


def shape_tests(G, C):
    summary = dict(shape_test={}, mode_shape_test={})
    for name, (changes, mx) in V.variants(C).items():
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        st = compare(G.scale(G.inc_c[None])[0] - G.scale(((G.sem_c + add) / k)[None])[0],
                     G.scale(G.inc(mx)) - G.scale((G.sem_t + add) / k))
        summary["shape_test"][name] = st
        print(f"[{name}] gap = SEM shape: " + "; ".join(f"{k}: d2 {v['d2']:.1f} p {v['p']:.4f}" for k, v in st.items()), flush=True)
    # S is linear on measures, so the model (1-f) SEM + f X maps to S(SEM) + f (S(X) - S(SEM))
    ft = 1 / G.c()
    s_inc, s_sem, s_inct, s_semt = G.scale(G.inc_c[None])[0], G.scale(G.sem_c[None])[0], G.scale(G.inc_t), G.scale(G.sem_t)
    for name, mx in V.modes(C).items():
        s_x = G.scale(mx[None])[0]
        fixed = compare(s_inc - s_sem - G.f0 * (s_x - s_sem), s_inct - s_semt - ft * (s_x - s_semt))
        free = {}
        for key, sl in SL.items():
            a, b, dt = s_inc[sl] - s_sem[sl], (s_x - s_sem)[sl], (s_inct - s_semt)[:, sl]
            P = np.linalg.inv(np.cov(dt, rowvar=False))
            fit = lambda t: (lambda f: (float((t - f * b) @ P @ (t - f * b)), f))(float(np.clip(t @ P @ b / (b @ P @ b), 0, 1)))
            d2, f = fit(a)
            boot = np.array([fit(f * b + e)[0] for e in dt - dt.mean(0)])
            free[key] = dict(d2=d2, f=f, p=float(np.mean(boot >= d2)))
        summary["mode_shape_test"][name] = dict(fixed=fixed, free=free)
        print(f"[mode {name}] fixed f: " + "; ".join(f"{k}: d2 {v['d2']:.1f} p {v['p']:.4f}" for k, v in fixed.items())
              + " | free f: " + "; ".join(f"{k}: d2 {v['d2']:.1f} f {v['f']:.3f} p {v['p']:.4f}" for k, v in free.items()), flush=True)
    return summary


def bounds(G, plt, fd):
    sup, sysc, bf_gap = G.sup, G.sysc, G.bf_gap
    f_nom, c0, c, n = G.f0, G.c0, G.c(), G.n
    incl = {o: (G.inc_c[SL[o]], G.inc_t[:, SL[o]]) for o in OBS}
    sem = {o: (G.sem_c[SL[o]], G.sem_t[:, SL[o]]) for o in OBS}
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
    return summary


def lineshapes(G, C):
    s_incl = G.sysc["n_leptons"] * sum(G.sysc["bf_incl_unc"].values())
    g = float(np.sqrt(bf_gap_sigma(G.sysc) ** 2 + 3 * s_incl ** 2) / bf_gap_sigma(G.sysc))
    var = V.variants(C)
    runs = {k: (ch, mx, 1.0) for k, (ch, mx) in var.items()} \
        | {f"{k}_incl_unc_x2": (var[k][0], 1.0, g) for k in ("phsp_Dpipi", "bare_evtgen_broad")}
    summary = dict(c_true=G.c0, g_incl_unc_x2=g, variants={}, supports={})
    for name, (changes, mx, zs) in runs.items():
        mu, mt, cs = G.residual(changes, mx, zs)
        res = {o: test(mu[sl], mt[:, sl], power=name != "nominal") for o, sl in SL.items()}
        summary["variants"][name] = dict(c_true=cs, **res)
        print(f"[{name}] c_true {cs:.2f}; " + "; ".join(f"{k}: d2 {v['d2']:.2f} p {v['p']:.4f}" for k, v in res.items()), flush=True)
    for name, alt in SUPPORTS.items():
        mu, mt, _ = G.residual(sup=G.sup | alt)
        summary["supports"][name] = {o: test(mu[sl], mt[:, sl]) for o, sl in SL.items()}
        print(f"[support {name}] " + "; ".join(f"{k}: d2 {v['d2']:.2f} p {v['p']:.4f}" for k, v in summary["supports"][name].items()), flush=True)
    return summary


def run(B):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt

    G, C = Gap(B), V.load()
    od, fd = output_root(yaml.safe_load(open(HERE / "config.yaml")), HERE) / "7", HERE / "figures" / "7"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    summary = shape_tests(G, C) | bounds(G, plt, fd) | lineshapes(G, C)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
    print(f"-> {od}, {fd}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after")
    args = p.parse_args()
    bundle = Path(args.bundle) if args.bundle else default_path(HERE)
    if args.submit:
        w = f' -w "done({args.after})"' if args.after else ""
        (HERE / "logs" / "7").mkdir(parents=True, exist_ok=True)
        cmd = (f'bsub -q {yaml.safe_load(open(HERE / "config.yaml"))["batch"]["queue"]} -env all -J s7tests{w} -n 2 -oo {HERE}/logs/7/run.log '
               f'"cd {HERE} && python3 7_tests.py --bundle {bundle.resolve()}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        return
    run(Bundle(bundle))


if __name__ == "__main__":
    main()
