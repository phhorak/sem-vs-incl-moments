#!/usr/bin/env python3
"""Step 10: admissibility tests of the threshold-0 residual moments and SEM variations (claims C5-C7, C12).

  d^2     Mahalanobis distance (toy covariance) from the residual moments m1..m3, scaled to the data support,
          to the moment space of probability measures; 0 when admissible. For three moments on [0, 1] its
          boundary is the set of two-point measures w delta_e + (1-w) delta_t, e in {0, 1}: exact projection
          by a scan in t with w in closed form. Jointly: block coordinate descent with the full 9x9 covariance.
  p       parametric bootstrap: toy fluctuations re-centred on the closest admissible point. p_chi2 =
          P(chi2_k >= d^2) is the Gaussian approximation; the toys are not Gaussian (c_true = B_incl/B_gap has a
          heavy tail), and for one observable p_chi2 is smaller than the bootstrap p, so the bootstrap p is the result.
  power   for SEM variants: distribution of the toy-by-toy p (toys fluctuate around the observed moments).

SEM variants change components with fixed BFs (shape swaps) or with changed shares; the SEM moments are linear in
the component weights, B_SEM and hence c_true follow the shares. Components and their moment variants come from
data/sem_component_moments.json (built with --build-moments from the cocktail, the retired PHSP cocktail and the
LHCb-style samples in output/1/lhcb):
  phsp_Dpipi        NR D(*)pipi with the 5-body PHSP lineshape
  split_Dp1_f       D1':D0* weight fraction of NR D(*)pipi set to f
  noFF_broad        no Hammer weight for the broad states (D0*, D1', NR D(*)pipi), MCAmbulance kept
  bare_evtgen_broad neither Hammer nor MCAmbulance for the broad states (EvtGen lineshape as generated)
  hqe_mx2_x2        HQE M_X^2 moment fluctuations doubled
  lhcb_*            LHCb-style cocktail: D** fractions (11874430/12873450), ISGW2 D** shapes, higher D** (D(2S),
                    D*(2S), D(2750), D(3000) -> D* pi pi) in place of NR D(*)pipi, no D_s K; and all together
Shape test ("common shortfall"): if all SEM BFs were low by a common factor the gap would have the SEM shape, i.e.
the inclusive and SEM moments would agree; d^2 between them with the toy covariance, bootstrap p (no c_true).
Scans (C7): one component scaled by s, and pairs scaled by (s1, s2); joint p_chi2 (k = 9) on a grid.

Usage:
  python3 10_cone_test.py [--bundle FILE] [--n-toys 2000] [--support OBS=LO,HI ...]
  python3 10_cone_test.py --submit [--after JOB]
  python3 10_cone_test.py --build-moments [--phsp-cocktail FILE]
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

OBS = ["mx2", "el", "q2"]
COMP_JSON = HERE / "data" / "sem_component_moments.json"
SCALES = np.round(np.arange(0.0, 3.001, 0.05), 2)
GRID2 = np.round(np.arange(0.0, 2.501, 0.1), 2)
PAIRS = [("D*", "D"), ("D*", "D**"), ("D", "D**"), ("D**", "D(*) pi pi")]
DSS = ("D1", "D0st", "Dp1", "D2st")
LHCB_DSS = {"D0st": (0.2941 + 0.2970) / 2, "D1": (0.2941 + 0.2772) / 2, "Dp1": (0.2647 + 0.3069) / 2, "D2st": (0.1471 + 0.1188) / 2}
LHCB_HIGHER = {"D2S": 0.45, "Dst2S": 0.05, "D2750": 0.45, "D3000": 0.05}
TG = np.linspace(0, 1, 4001)
G = np.stack([TG, TG ** 2, TG ** 3], 1)


def project(target, P):
    """Closest point (metric P) to target in the moment space of probability measures on [0, 1]."""
    if hausdorff_check(np.r_[1.0, target])[0]:
        return target
    L = np.linalg.cholesky(P).T
    best = (np.inf, None)
    for e in (0.0, 1.0):
        D = np.array([e, e ** 2, e ** 3]) - G
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
            target = mu[s] + np.linalg.solve(P[s, s], P[s][:, o] @ (mu[o] - nu[o])) if nb > 1 else mu[s]
            nu[s] = project(target, P[s, s])
        if np.max(np.abs(nu - prev)) < 1e-10:
            break
    r = mu - nu
    return float(r @ P @ r), nu


def test(mu_nom, mu_toys, power=False):
    nb = len(mu_nom) // 3
    P = np.linalg.inv(np.cov(mu_toys, rowvar=False))
    d2, nu = distance(mu_nom, P, nb)
    pc = float(chi2.sf(d2, len(mu_nom)))
    out = dict(d2=d2, p_chi2=pc)
    if d2 < 1e-9:
        return out | dict(d2=0.0, p=1.0, z=0.0)
    boot = np.sort([distance(nu + dm, P, nb)[0] for dm in mu_toys - mu_toys.mean(0)])
    p = float(np.mean(boot >= d2))
    out |= dict(p=p, z=float(norm.isf(max(p, 0.5 / len(boot)))))
    if power:
        pt = 1 - np.searchsorted(boot, [distance(m, P, nb)[0] for m in mu_toys[:500]], side="left") / len(boot)
        out |= dict(p_toys_median=float(np.median(pt)), frac_toys_p_below_0p05=float(np.mean(pt < 0.05)))
    return out


def build_moments(phsp_cocktail):
    """Component moments (raw m1..m3 of Mx2, El, q2) and SEM weight shares, with weight variants."""
    import pandas as pd
    import pyarrow.parquet as pq
    import uproot
    from lib.moments import compute_kinematics

    def acc(store, key, w, x):
        store[key] = store.get(key, 0) + np.r_[w.sum(), [(w * x[i] ** p).sum() for i in range(3) for p in (1, 2, 3)]]

    tot, s = {}, {}
    cols = ["category", "decay_name", "total_weight", "mc_weight", "ff_weight", "Mx", "El_B", "q2", "d0_mcPDG"]
    for b in pq.ParquetFile(output_root(load_cfg(), HERE) / "3" / "cocktail.parquet").iter_batches(columns=cols, batch_size=2_000_000):
        d = b.to_pandas()
        w, mc, ff = (d[c].to_numpy(float) for c in ("total_weight", "mc_weight", "ff_weight"))
        x = np.stack([d.Mx.to_numpy(float) ** 2, d.El_B.to_numpy(float), d.q2.to_numpy(float)])
        ok = np.isfinite(w) & (w > 0) & np.isfinite(x).all(0) & (mc > 0) & (ff > 0)
        cat = d.category.astype(str).to_numpy()
        stem = d.decay_name.astype(str).str.replace(r"^(Bp|B0)_", "", regex=True).str.replace(r"(e|mu)nu$", "", regex=True).to_numpy()
        par = np.where(np.isin(np.abs(d.d0_mcPDG.to_numpy()), [20413, 20423]), "Dp1", "D0st")
        key = np.where(cat == "D**", "D**:" + stem, cat)
        for k in np.unique(key[ok]):
            m = ok & (key == k)
            for tag, ww in (("", w), ("noFF", w / ff), ("noMCA_noFF", w / mc / ff)):
                acc(s, (k, tag), ww[m], x[:, m])
            if k == "D(*) pi pi":
                for pk in ("Dp1", "D0st"):
                    acc(s, (k, "parent_" + pk), w[m & (par == pk)], x[:, m & (par == pk)])
        acc(tot, "all", w[ok], x[:, ok])
    out = {k: {} for k, _ in s}
    for (k, tag), v in s.items():
        out[k]["moments" + ("_" + tag if tag else "")] = (v[1:] / v[0]).tolist()
        if not tag:
            out[k]["share"] = v[0] / tot["all"][0]
    if phsp_cocktail:
        p = {}
        for b in pq.ParquetFile(phsp_cocktail).iter_batches(columns=["category", "total_weight", "Mx", "El_B", "q2"], batch_size=2_000_000):
            d = b.to_pandas()
            w = d.total_weight.to_numpy(float)
            x = np.stack([d.Mx.to_numpy(float) ** 2, d.El_B.to_numpy(float), d.q2.to_numpy(float)])
            m = np.isfinite(w) & (w > 0) & np.isfinite(x).all(0) & (d.category.astype(str) == "D(*) pi pi").to_numpy()
            acc(p, "nr", w[m], x[:, m])
        out["D(*) pi pi"]["moments_phsp"] = (p["nr"][1:] / p["nr"][0]).tolist()
        out["D(*) pi pi"]["phsp_source"] = str(phsp_cocktail)
    # LHCb-style samples: ISGW2 D** (MCAmbulance ISGW2 corrections for D0*, D1') and higher D**; B+ and B0 by f+-, f00
    from mcambulance import MCAmbulance
    lh = {}
    for f in sorted((output_root(load_cfg(), HERE) / "1" / "lhcb").glob("*/*/0000.root")):
        charge, state = f.parts[-3], f.parts[-2]
        d = compute_kinematics(uproot.open(f)["reco"].arrays(library="pd"))
        x = np.stack([d.Mx.to_numpy(float) ** 2, d.El_B.to_numpy(float), d.q2.to_numpy(float)])
        ok = np.isfinite(x).all(0)
        w = np.where(ok, 1.0, 0.0)
        ch = {"D0st": "d0_d_pi_isgw2", "Dp1": "d1p_dstar_pi_isgw2"}.get(state)
        if ch:
            lep = np.where(d[[f"d{i}_mcPDG" for i in range(5)]].abs().isin([13]).any(axis=1), "mu", "e")
            for l in ("e", "mu"):
                sel = ok & (lep == l)
                w[sel] = MCAmbulance("bp" if charge == "Bplus" else "b0", ch, l).CorrectionWeight(d.Mx.to_numpy(float)[sel])
        w = w / w.sum() * (0.514 if charge == "Bplus" else 0.486)
        acc(lh, state, w[ok], x[:, ok])
    out["lhcb"] = {k: (v[1:] / v[0]).tolist() for k, v in lh.items()}
    json.dump(out, open(COMP_JSON, "w"), indent=1)
    for k, v in out.items():
        if "share" in v:
            print(f"{k:12s} share {v['share']:.4f}  <Mx2> {v['moments'][0]:.3f}")
    print("lhcb states:", {k: round(v[0], 3) for k, v in out["lhcb"].items()})


def load_cfg():
    import yaml
    return yaml.safe_load(open(HERE / "config.yaml"))


def run(B, n_toys, seed, support=()):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt

    cfg, T = B.cfg, B.toys
    for spec in support:
        o, v = spec.split("=")
        cfg["data"]["support"][o] = [float(x) for x in v.split(",")]
    sup, sysc = cfg["data"]["support"], cfg["systematics"]
    b_incl, bf_gap = sysc["bf_incl"], float(T["bf_gap"])
    from lib.systematics import c_true
    c0 = float(c_true(sysc, bf_gap))
    rng = np.random.default_rng(seed)
    n = min(n_toys, len(T["z_gap"]))
    ih = rng.choice(len(T["hqe_toys"]), n, replace=False)
    c = c_true(sysc, bf_gap, T["z_gap"][:n])[:, None]
    idx = np.r_[[MAX_ORDER * k + j for k in range(3) for j in range(3)]]
    sem_c, sem_t = T["nom_sem0"][idx], T["sem0_stat_ff_bfmode"][:n][:, idx]
    inc_c, inc_t = T["hqe_central"], T["hqe_toys"][ih]
    scale = lambda raw: np.concatenate([raw_to_mu01(np.c_[np.ones(len(raw)), raw[:, 3 * k:3 * k + 3]].T, *sup[o])[1:].T
                                        for k, o in enumerate(OBS)], axis=1)
    C = json.load(open(COMP_JSON))
    m = lambda k, tag="": np.array(C[k]["moments" + ("_" + tag if tag else "")])
    sh = lambda k: C[k]["share"]
    cats = {"D": (sh("D"), m("D")), "D*": (sh("D*"), m("D*")), "D(*) pi pi": (sh("D(*) pi pi"), m("D(*) pi pi")),
            "Ds(*) K": (sh("Ds(*) K"), m("Ds(*) K"))}
    s_dss = sum(sh("D**:" + k) for k in DSS)
    cats["D**"] = (s_dss, sum(sh("D**:" + k) * m("D**:" + k) for k in DSS) / s_dss)

    def residuals(changes, inc_mx2_scale=1.0):
        """Residual moments (nominal, toys) for changes = [(share, moments), ...] added to the nominal SEM
        (negative share removes); None if the SEM then exceeds the inclusive rate."""
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        gap = b_incl - (b_incl - bf_gap) * k
        if gap <= 0:
            return None
        cs = b_incl / gap
        ct = cs / c0 * c
        it = inc_t.copy()
        it[:, :3] = inc_c[:3] + inc_mx2_scale * (inc_t[:, :3] - inc_c[:3])
        return (scale((cs * inc_c - (cs - 1) * (sem_c + add) / k)[None])[0],
                scale(ct * it - (ct - 1) * (sem_t + add) / k), cs)

    swap = lambda comps, tag: [x for k in comps for x in ((-sh(k), m(k)), (sh(k), m(k, tag)))]
    s_nr = sh("D(*) pi pi")
    lhcb = C.get("lhcb", {})
    lhcb_dss_frac = [(-sh("D**:" + k), m("D**:" + k)) for k in DSS] + [(s_dss * f, m("D**:" + k)) for k, f in LHCB_DSS.items()]
    lhcb_isgw2 = [(-sh("D**:" + k), m("D**:" + k)) for k in DSS] + [(sh("D**:" + k), np.array(lhcb[k])) for k in DSS if k in lhcb]
    lhcb_higher = [(-s_nr, m("D(*) pi pi"))] + [(s_nr * f, np.array(lhcb[k])) for k, f in LHCB_HIGHER.items() if k in lhcb]
    lhcb_nodsk = [(-sh("Ds(*) K"), m("Ds(*) K"))]
    variants = {"nominal": ([], 1.0),
                "phsp_Dpipi": ([(-s_nr, m("D(*) pi pi")), (s_nr, m("D(*) pi pi", "phsp"))], 1.0),
                "noFF_broad": (swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noFF"), 1.0),
                "bare_evtgen_broad": (swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noMCA_noFF"), 1.0),
                "hqe_mx2_x2": ([], 2.0)}
    for f in (0.0, 0.25, 0.75, 1.0):
        variants[f"split_Dp1_{f:.2f}"] = ([(-s_nr, m("D(*) pi pi")),
                                           (s_nr, f * m("D(*) pi pi", "parent_Dp1") + (1 - f) * m("D(*) pi pi", "parent_D0st"))], 1.0)
    if len(lhcb) == 8:
        variants |= {"lhcb_dss_fractions": (lhcb_dss_frac, 1.0), "lhcb_isgw2_dss": (lhcb_isgw2, 1.0),
                     "lhcb_higher_dss": (lhcb_higher, 1.0), "lhcb_no_dsk": (lhcb_nodsk, 1.0),
                     "lhcb_all": ([(-sh("D**:" + k), m("D**:" + k)) for k in DSS]
                                  + [(s_dss * f, np.array(lhcb[k])) for k, f in LHCB_DSS.items()] + lhcb_higher + lhcb_nodsk, 1.0)}
    summary = dict(n_toys=n, c_true=c0, variants={}, shape_test={})
    for name, (changes, infl) in variants.items():
        mu, mt, cs = residuals(changes, infl)
        res = {o: test(mu[3 * i:3 * i + 3], mt[:, 3 * i:3 * i + 3], power=name != "nominal") for i, o in enumerate(OBS)}
        res["joint"] = test(mu, mt, power=name != "nominal")
        summary["variants"][name] = dict(c_true=cs, **res)
        print(f"[{name}] c_true {cs:.2f}; " + "; ".join(f"{k}: d2 {v['d2']:.2f} p {v['p']:.4f}" for k, v in res.items()), flush=True)
        # shape test: inclusive vs SEM moments directly (gap with the SEM shape)
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        it = inc_t.copy()
        it[:, :3] = inc_c[:3] + infl * (inc_t[:, :3] - inc_c[:3])
        dl, dt = scale(inc_c[None])[0] - scale(((sem_c + add) / k)[None])[0], scale(it) - scale((sem_t + add) / k)
        st = {}
        for key, sl in [(o, slice(3 * i, 3 * i + 3)) for i, o in enumerate(OBS)] + [("joint", slice(0, 9))]:
            P = np.linalg.inv(np.cov(dt[:, sl], rowvar=False))
            fl = dt[:, sl] - dt[:, sl].mean(0)
            d2, d2t = float(dl[sl] @ P @ dl[sl]), np.einsum("ij,jk,ik->i", fl, P, fl)
            p = float(np.mean(d2t >= d2))
            st[key] = dict(d2=d2, p=p, z=float(norm.isf(max(p, 0.5 / len(d2t)))))
        summary["shape_test"][name] = st
        print(f"   shape test (gap = SEM shape): " + "; ".join(f"{k}: d2 {v['d2']:.1f} p {v['p']:.4f}" for k, v in st.items()), flush=True)

    def joint_p(changes):
        r = residuals(changes)
        if r is None:
            return np.nan
        mu, mt, _ = r
        return float(chi2.sf(distance(mu, np.linalg.inv(np.cov(mt, rowvar=False)), 3)[0], 9))

    scan = {}
    for cat, (s0, m0) in cats.items():
        ps = [joint_p([((sc - 1) * s0, m0)]) for sc in SCALES]
        ok = [sc for sc, p in zip(SCALES, ps) if np.isfinite(p) and p > 0.05]
        closes = 1 + (bf_gap / b_incl) / ((1 - bf_gap / b_incl) * s0)
        scan[cat] = dict(share=s0, allowed=[float(min(ok)), float(max(ok))] if ok else None, closes_gap_at=closes,
                         s=SCALES.tolist(), p=[None if not np.isfinite(p) else p for p in ps])
        print(f"[scan {cat}] allowed {scan[cat]['allowed']}, closes the gap alone at {closes:.2f}", flush=True)
    pairs = {}
    for a, b in PAIRS:
        grid = np.array([[joint_p([((s1 - 1) * cats[a][0], cats[a][1]), ((s2 - 1) * cats[b][0], cats[b][1])])
                          for s2 in GRID2] for s1 in GRID2])
        pairs[f"{a}|{b}"] = dict(s=GRID2.tolist(), p=np.where(np.isfinite(grid), grid, -1).tolist())
        print(f"[pair {a} x {b}] allowed fraction of grid {np.mean(np.nan_to_num(grid, nan=0) > 0.05):.2f}", flush=True)
    summary |= dict(component_scan=scan, pair_scan=pairs)

    od, fd = output_root(cfg, HERE) / "10", HERE / "figures" / "10"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
    for cat, v in scan.items():
        x = [s for s, p in zip(v["s"], v["p"]) if p is not None]
        ax.plot(x, [max(p, 1e-12) for p in v["p"] if p is not None], lw=2.6, label=f"{cat} ({v['share']:.1%})")
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

    fig, axes = plt.subplots(1, len(PAIRS), figsize=(7 * len(PAIRS), 6.5), dpi=200)
    for ax, (a, b) in zip(axes, PAIRS):
        g = np.array(pairs[f"{a}|{b}"]["p"])
        ax.contourf(GRID2, GRID2, np.where(g < 0, np.nan, g).T, levels=[0, 0.003, 0.05, 1.01], colors=["#d62728", "#ff9896", "#c7e9c0"])
        ax.contour(GRID2, GRID2, np.where(g < 0, 0, g).T, levels=[0.05], colors="k", linewidths=1.5)
        ax.plot(1, 1, "k*", ms=14)
        ax.set_xlabel(f"scale {a}", fontsize=20)
        ax.set_ylabel(f"scale {b}", fontsize=20)
    fig.tight_layout()
    fig.savefig(fd / "pair_scan.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {od}, {fd}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle")
    p.add_argument("--n-toys", type=int, default=2000)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after")
    p.add_argument("--support", nargs="+", default=(), metavar="OBS=LO,HI", help="override the data support")
    p.add_argument("--build-moments", action="store_true")
    p.add_argument("--phsp-cocktail", help="cocktail parquet with the 5-body PHSP NR D(*)pipi (for --build-moments)")
    args = p.parse_args()
    if args.build_moments:
        sys.path.insert(0, str(HERE / "MCAmbulance"))
        build_moments(args.phsp_cocktail)
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
