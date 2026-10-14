#!/usr/bin/env python3
"""Step 10: admissibility tests of the threshold-0 residual moments, SEM lineshape variants and shape tests.

  Admissibility  d^2 to the moment space of probability measures and its bootstrap p (lib/cone.py), per observable
                 and jointly, for the nominal SEM and its variants.
  Variants       change components with fixed BFs (shape swaps) or with changed shares; the SEM moments are linear
                 in the component weights, B_SEM and hence c_true follow the shares. Components and their moment
                 variants come from data/sem_component_moments.json (built with --build-moments from the cocktail,
                 the retired PHSP cocktail and the LHCb-style samples in output/1/lhcb):
    phsp_Dpipi        D(*)pipi with the 5-body PHSP lineshape
    split_Dp1_f       D1':D0* weight fraction of D(*)pipi set to f
    noFF_broad        no Hammer weight for the broad states (D0*, D1', D(*)pipi), MCAmbulance kept
    bare_evtgen_broad neither Hammer nor MCAmbulance for the broad states (EvtGen lineshape as generated)
    hqe_mx2_x2        HQE M_X^2 moment fluctuations doubled
    *_incl_unc_x2     B_gap fluctuations scaled as if the B_incl uncertainty were doubled
    lhcb_*            LHCb-style cocktail: D** fractions (11874430/12873450), ISGW2 D** shapes, higher D** (D(2S),
                      D*(2S), D(2750), D(3000) -> D* pi pi) in place of D(*)pipi, no D_s K; and all together
  Supports       nominal admissibility for the alternative data supports in SUPPORTS
  Shape tests    inclusive moments vs a model, d^2 with the toy covariance, no c_true amplification:
    shape_test        gap with the SEM shape (all SEM BFs low by a common factor): inclusive vs SEM moments
    mode_shape_test   gap with the shape of one SEM mode X: inclusive vs (1-f) SEM + f X, at the nominal gap
                      fraction f (toys vary it with B_gap) or with f free in [0, 1] (closed-form fit, bootstrap refit)

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
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.bundle import Bundle, default_path, output_root
from lib.cone import test
from lib.maxent import raw_to_mu01
from lib.systematics import bf_gap_sigma, c_true

OBS = ["mx2", "el", "q2"]
COMP_JSON = HERE / "data" / "sem_component_moments.json"
DSS = ("D1", "D0st", "Dp1", "D2st")
LHCB_DSS = {"D0st": (0.2941 + 0.2970) / 2, "D1": (0.2941 + 0.2772) / 2, "Dp1": (0.2647 + 0.3069) / 2, "D2st": (0.1471 + 0.1188) / 2}
LHCB_HIGHER = {"D2S": 0.45, "Dst2S": 0.05, "D2750": 0.45, "D3000": 0.05}
# alternative data supports for the nominal admissibility test; q2 <= 6 is deliberately too narrow
SUPPORTS = {"el_2.26": {"el": [0.0, 2.26]}, "el_2.40": {"el": [0.0, 2.4]}, "mx2_12": {"mx2": [3.2, 12.0]},
            "mx2_20": {"mx2": [3.2, 20.0]}, "q2_12.5": {"q2": [0.0, 12.5]}, "q2_6_wrong": {"q2": [0.0, 6.0]}}
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
    cfg, T = B.cfg, B.toys
    for spec in support:
        o, v = spec.split("=")
        cfg["data"]["support"][o] = [float(x) for x in v.split(",")]
    sup, sysc = cfg["data"]["support"], cfg["systematics"]
    b_incl, bf_gap = sysc["bf_incl"], float(T["bf_gap"])
    c0 = float(c_true(sysc, bf_gap))
    rng = np.random.default_rng(seed)
    n = min(n_toys, len(T["z_gap"]))
    ih = rng.choice(len(T["hqe_toys"]), n, replace=False)
    z = T["z_gap"][:n]
    idx = np.r_[[MAX_ORDER * k + j for k in range(3) for j in range(3)]]
    sem_c, sem_t = T["nom_sem0"][idx], T["sem0_stat_ff_bfmode"][:n][:, idx]
    inc_c, inc_t = T["hqe_central"], T["hqe_toys"][ih]
    scale = lambda raw, s=sup: np.concatenate([raw_to_mu01(np.c_[np.ones(len(raw)), raw[:, 3 * k:3 * k + 3]].T, *s[o])[1:].T
                                        for k, o in enumerate(OBS)], axis=1)
    C = json.load(open(COMP_JSON))
    m = lambda k, tag="": np.array(C[k]["moments" + ("_" + tag if tag else "")])
    sh = lambda k: C[k]["share"]
    s_dss = sum(sh("D**:" + k) for k in DSS)
    modes = {k: m(k) for k in ("D", "D*", "D(*) pi pi", "Ds(*) K")} | {"D**": sum(sh("D**:" + k) * m("D**:" + k) for k in DSS) / s_dss}
    # doubling sigma(B_incl): sigma(B_gap)^2 = sigma(B_SEM)^2 + sigma(B_incl)^2 -> + 3 sigma(B_incl)^2
    s_gap, s_incl = bf_gap_sigma(sysc), sysc["n_leptons"] * sum(load_cfg()["systematics"]["bf_incl_unc"].values())
    g_incl = float(np.sqrt(s_gap ** 2 + 3 * s_incl ** 2) / s_gap)

    def residuals(changes, inc_mx2_scale=1.0, zscale=1.0, s=sup):
        """Residual moments (nominal, toys) for changes = [(share, moments), ...] added to the nominal SEM
        (negative share removes)."""
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        cs = b_incl / (b_incl - (b_incl - bf_gap) * k)
        ct = cs / c0 * c_true(sysc, bf_gap, zscale * z)[:, None]
        it = inc_t.copy()
        it[:, :3] = inc_c[:3] + inc_mx2_scale * (inc_t[:, :3] - inc_c[:3])
        return (scale((cs * inc_c - (cs - 1) * (sem_c + add) / k)[None], s)[0],
                scale(ct * it - (ct - 1) * (sem_t + add) / k, s), cs)

    swap = lambda comps, tag: [x for k in comps for x in ((-sh(k), m(k)), (sh(k), m(k, tag)))]
    s_nr = sh("D(*) pi pi")
    lhcb = C.get("lhcb", {})
    lhcb_dss_frac = [(-sh("D**:" + k), m("D**:" + k)) for k in DSS] + [(s_dss * f, m("D**:" + k)) for k, f in LHCB_DSS.items()]
    lhcb_isgw2 = [(-sh("D**:" + k), m("D**:" + k)) for k in DSS] + [(sh("D**:" + k), np.array(lhcb[k])) for k in DSS if k in lhcb]
    lhcb_higher = [(-s_nr, m("D(*) pi pi"))] + [(s_nr * f, np.array(lhcb[k])) for k, f in LHCB_HIGHER.items() if k in lhcb]
    lhcb_nodsk = [(-sh("Ds(*) K"), m("Ds(*) K"))]
    phsp = [(-s_nr, m("D(*) pi pi")), (s_nr, m("D(*) pi pi", "phsp"))]
    bare = swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noMCA_noFF")
    variants = {"nominal": ([], 1.0, 1.0), "phsp_Dpipi": (phsp, 1.0, 1.0),
                "noFF_broad": (swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noFF"), 1.0, 1.0),
                "bare_evtgen_broad": (bare, 1.0, 1.0), "hqe_mx2_x2": ([], 2.0, 1.0),
                "phsp_Dpipi_incl_unc_x2": (phsp, 1.0, g_incl), "bare_evtgen_broad_incl_unc_x2": (bare, 1.0, g_incl)}
    for f in (0.0, 0.25, 0.75, 1.0):
        variants[f"split_Dp1_{f:.2f}"] = ([(-s_nr, m("D(*) pi pi")),
                                           (s_nr, f * m("D(*) pi pi", "parent_Dp1") + (1 - f) * m("D(*) pi pi", "parent_D0st"))], 1.0, 1.0)
    if len(lhcb) == 8:
        variants |= {"lhcb_dss_fractions": (lhcb_dss_frac, 1.0, 1.0), "lhcb_isgw2_dss": (lhcb_isgw2, 1.0, 1.0),
                     "lhcb_higher_dss": (lhcb_higher, 1.0, 1.0), "lhcb_no_dsk": (lhcb_nodsk, 1.0, 1.0),
                     "lhcb_all": ([(-sh("D**:" + k), m("D**:" + k)) for k in DSS]
                                  + [(s_dss * f, np.array(lhcb[k])) for k, f in LHCB_DSS.items()] + lhcb_higher + lhcb_nodsk, 1.0, 1.0)}
    SL = [(o, slice(3 * i, 3 * i + 3)) for i, o in enumerate(OBS)] + [("joint", slice(0, 9))]

    def compare(dl, dt):
        """d^2 of dl with the covariance of the toys dt, and p from the re-centred toys."""
        out = {}
        for key, sl in SL:
            P = np.linalg.inv(np.cov(dt[:, sl], rowvar=False))
            fl = dt[:, sl] - dt[:, sl].mean(0)
            d2, d2t = float(dl[sl] @ P @ dl[sl]), np.einsum("ij,jk,ik->i", fl, P, fl)
            p = float(np.mean(d2t >= d2))
            out[key] = dict(d2=d2, p=p, z=float(norm.isf(max(p, 0.5 / len(d2t)))))
        return out

    summary = dict(n_toys=n, c_true=c0, g_incl_unc_x2=g_incl, variants={}, supports={}, shape_test={}, mode_shape_test={})
    for name, alt in SUPPORTS.items():
        mu, mt, _ = residuals([], s=sup | alt)
        summary["supports"][name] = {o: test(mu[sl], mt[:, sl]) for o, sl in SL}
        print(f"[support {name}] " + "; ".join(f"{k}: d2 {v['d2']:.2f} p {v['p']:.4f}" for k, v in summary["supports"][name].items()), flush=True)
    for name, (changes, infl, zs) in variants.items():
        mu, mt, cs = residuals(changes, infl, zs)
        res = {o: test(mu[sl], mt[:, sl], power=name != "nominal") for o, sl in SL}
        summary["variants"][name] = dict(c_true=cs, **res)
        print(f"[{name}] c_true {cs:.2f}; " + "; ".join(f"{k}: d2 {v['d2']:.2f} p {v['p']:.4f}" for k, v in res.items()), flush=True)
        if zs != 1.0:
            continue
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        it = inc_t.copy()
        it[:, :3] = inc_c[:3] + infl * (inc_t[:, :3] - inc_c[:3])
        st = compare(scale(inc_c[None])[0] - scale(((sem_c + add) / k)[None])[0], scale(it) - scale((sem_t + add) / k))
        summary["shape_test"][name] = st
        print("   shape test (gap = SEM shape): " + "; ".join(f"{k}: d2 {v['d2']:.1f} p {v['p']:.4f}" for k, v in st.items()), flush=True)

    # gap with the shape of one SEM mode X; S is linear on measures, so model = S(SEM) + f (S(X) - S(SEM))
    f0, ft = bf_gap / b_incl, 1 / c_true(sysc, bf_gap, z)[:, None]
    s_inc, s_sem, s_inct, s_semt = scale(inc_c[None])[0], scale(sem_c[None])[0], scale(inc_t), scale(sem_t)
    for name, mx in modes.items():
        s_x = scale(mx[None])[0]
        fixed = compare(s_inc - s_sem - f0 * (s_x - s_sem), s_inct - s_semt - ft * (s_x - s_semt))
        free = {}
        for key, sl in SL:
            a, b, dt = s_inc[sl] - s_sem[sl], (s_x - s_sem)[sl], (s_inct - s_semt)[:, sl]
            P = np.linalg.inv(np.cov(dt, rowvar=False))
            fit = lambda t: (lambda f: (float((t - f * b) @ P @ (t - f * b)), f))(float(np.clip(t @ P @ b / (b @ P @ b), 0, 1)))
            d2, f = fit(a)
            boot = np.array([fit(f * b + e)[0] for e in dt - dt.mean(0)])
            free[key] = dict(d2=d2, f=f, p=float(np.mean(boot >= d2)))
        summary["mode_shape_test"][name] = dict(fixed=fixed, free=free)
        print(f"[mode {name}] fixed f: " + "; ".join(f"{k}: d2 {v['d2']:.1f} p {v['p']:.4f}" for k, v in fixed.items())
              + " | free f: " + "; ".join(f"{k}: d2 {v['d2']:.1f} f {v['f']:.3f} p {v['p']:.4f}" for k, v in free.items()), flush=True)

    od = output_root(cfg, HERE) / "10"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
    print(f"-> {od}")


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
