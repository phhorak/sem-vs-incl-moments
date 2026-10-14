#!/usr/bin/env python3
"""Step 3: raw moments m1..m3 of (M_X^2, E_l, q^2) per SEM mode and their lineshape variants.

From the step-2 cocktail (shares and moments per mode and per D** state; without Hammer weights for, and without
MCAmbulance on, the broad states; D(*)pipi per D1'/D0* parent), the retired 5-body PHSP cocktail (D(*)pipi with the
PHSP lineshape) and the LHCb-style samples in output/1/lhcb (ISGW2 D** with the MCAmbulance ISGW2 correction, higher
D**). Writes data/sem_component_moments.json, read by lib/variants.py.

Usage:
  python3 3_sem_modes.py [--phsp-cocktail FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.bundle import output_root
from lib.variants import PATH as COMP_JSON


def build_moments(cfg, phsp_cocktail):
    """Component moments (raw m1..m3 of Mx2, El, q2) and SEM weight shares, with weight variants."""
    import pyarrow.parquet as pq
    import uproot
    from lib.moments import compute_kinematics

    def acc(store, key, w, x):
        store[key] = store.get(key, 0) + np.r_[w.sum(), [(w * x[i] ** p).sum() for i in range(3) for p in (1, 2, 3)]]

    tot, s = {}, {}
    cols = ["category", "decay_name", "total_weight", "mc_weight", "ff_weight", "Mx", "El_B", "q2", "d0_mcPDG"]
    for b in pq.ParquetFile(output_root(cfg, HERE) / "2" / "cocktail.parquet").iter_batches(columns=cols, batch_size=2_000_000):
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
    for f in sorted((output_root(cfg, HERE) / "1" / "lhcb").glob("*/*/0000.root")):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phsp-cocktail", help="cocktail parquet with the 5-body PHSP D(*)pipi")
    args = p.parse_args()
    sys.path.insert(0, str(HERE / "MCAmbulance"))
    build_moments(yaml.safe_load(open(HERE / "config.yaml")), args.phsp_cocktail)


if __name__ == "__main__":
    main()
