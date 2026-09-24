"""The step-6 bundle: one file holding everything steps 7 and 8 need, so they run without the
MC samples or any other input (e.g. on a collaborator's machine).

Contents (numpy .npz, compressed):
  pseudo-experiments  the arrays of toys.npz (moments per toy, SEM variants, z draws, nominal
                      values, supports, stack binning)
  residuals           per observable (Mx, El, Q2) and source: thresholds, nominal residual
                      moments, covariance, per-toy residuals
  HQE                 zero-threshold raw moments of the HQE fit, central and likelihood toys
  truth histograms    per Asimov scenario and variable: component densities on the display
                      binning, component weight/mean/width in M_X, a 200-bin density for the
                      distance to truth, and the M_X display range
  SEM stack           nominal per-category densities on the stack binning
  config              the config.yaml text the toys were made with
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

NAME = "gap_toys_bundle.npz"
RES_OBS = ["Mx", "El", "Q2"]
RES_SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode", "all"]
VARS = ["mx2", "el", "q2"]
DISPLAY_BINS = {"mx2": 1200, "el": 500, "q2": 500}
L1_BINS = 200


def output_root(cfg: dict, here: Path) -> Path:
    """The configured output directory where it exists, else <repo>/output."""
    p = Path(cfg["paths"]["output"])
    return p if p.is_dir() else here / "output"


def default_path(here: Path) -> Path:
    """This machine's step-6 output if built here, else the copy shipped in data/."""
    p = output_root(yaml.safe_load(open(here / "config.yaml")), here) / "6" / NAME
    return p if p.exists() else here / "data" / NAME


def display_range(groups, q=(0.001, 0.995), pad=0.1):
    """M_X display range [GeV] covering the truth's weighted quantiles q, padded."""
    mx = np.sqrt(np.concatenate([g["mx2"] for g in groups]))
    w = np.concatenate([g["w"] for g in groups])
    o = np.argsort(mx)
    cw = np.cumsum(w[o]) / w.sum()
    lo, hi = np.interp(q, cw, mx[o])
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def truth_arrays(cfg, scen, groups, support):
    """Histograms of one scenario's gap truth, as step 7 draws them."""
    sc = cfg["asimov"]["scenarios"][scen]
    ncomp = len(sc["components"])
    comp = np.concatenate([np.full(len(g["w"]), g["comp"]) for g in groups])
    wt = np.concatenate([g["w"] for g in groups])
    out = {}
    mx = np.sqrt(np.concatenate([g["mx2"] for g in groups]))
    out["stats"] = np.array([[wt[comp == i].sum(), np.average(mx[comp == i], weights=wt[comp == i]),
                              np.sqrt(np.average((mx[comp == i] - np.average(mx[comp == i], weights=wt[comp == i])) ** 2,
                                                 weights=wt[comp == i]))] for i in range(ncomp)])
    out["mxrange"] = np.array(sc.get("mx_display") or display_range(groups))
    for v in VARS:
        lo, hi = support[v]
        x = np.concatenate([g[v] for g in groups])
        xd, (d0, d1) = (np.sqrt(x), (np.sqrt(lo), np.sqrt(hi))) if v == "mx2" else (x, (lo, hi))
        edges = np.linspace(d0, d1, DISPLAY_BINS[v])
        db = edges[1] - edges[0]
        out[f"{v}_edges"] = edges
        out[f"{v}_dens"] = np.stack([np.histogram(xd, edges, weights=np.where(comp == i, wt, 0.0))[0] / db
                                     for i in range(ncomp)])
        h, e = np.histogram(x, L1_BINS, (lo, hi), weights=wt)
        out[f"{v}_l1"] = h / (wt.sum() * np.diff(e))
    return out


def build(cfg: dict, config_text: str, here: Path) -> Path:
    """Write <output>/6/NAME from the merged step-6 outputs, the HQE toys and the MC samples."""
    from lib.asimov import truth_groups
    from lib.systematics import load_hqe_raw, read_parquet_downcast
    out6 = output_root(cfg, here) / "6"
    arrays = dict(np.load(out6 / "toys.npz"))
    for o in RES_OBS:
        for s in RES_SOURCES:
            d = json.load(open(out6 / f"residual_covariance_{o}_{s}.json"))
            e = np.load(out6 / f"toy_ensemble_{o}_{s}.npz")
            arrays[f"res_{o}_{s}_nominal"] = np.array(d["nominal"])
            arrays[f"res_{o}_{s}_cov"] = np.array(d["cov"])
            arrays[f"res_{o}_{s}_toys"] = e["raw_gap"]
        arrays[f"res_{o}_thr"] = np.array(sorted({p["thr"] for p in d["points"]}))
    arrays["hqe_central"], arrays["hqe_toys"] = load_hqe_raw(here / "inputs" / "hqe_likelihood_toys" / "likelihood_toys.h5")

    df = read_parquet_downcast(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet",
                               ["Mx", "El_B", "q2", "total_weight", "decay_name", "category"],
                               category_cols=("decay_name", "category"))
    df = df[np.isfinite(df[["Mx", "El_B", "q2", "total_weight"]].to_numpy(float)).all(axis=1) & (df["total_weight"] > 0)]
    w = df["total_weight"].to_numpy(float)
    ck = dict(mx2=df["Mx"].to_numpy(float) ** 2, el=df["El_B"].to_numpy(float), q2=df["q2"].to_numpy(float), w=w,
              codes=df["decay_name"].cat.codes.to_numpy(), cats=np.asarray(df["decay_name"].cat.categories, dtype=str))
    for scen in cfg["asimov"]["scenarios"]:
        sup = {v: tuple(arrays[f"support_{scen}"][i]) for i, v in enumerate(VARS)}
        for k, v in truth_arrays(cfg, scen, truth_groups(cfg, scen, ck), sup).items():
            arrays[f"truth_{scen}_{k}"] = v

    cat = df["category"].astype(str).to_numpy()
    names = sorted(np.unique(cat))
    arrays["stack_categories"] = np.array(names)
    xs = {"Mx": df["Mx"].to_numpy(float), "El": ck["el"], "Q2": ck["q2"]}
    for o, x in xs.items():
        e = arrays[f"stackedges_{o}"]
        arrays[f"stack_nominal_{o}"] = np.stack([np.histogram(x[cat == c], e, weights=w[cat == c])[0] for c in names]) \
            / (w.sum() * np.diff(e))
    arrays["config_yaml"] = np.array(config_text)
    path = out6 / NAME
    np.savez_compressed(path, **arrays)
    return path


class Bundle:
    """Read-only view of a bundle file."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.a = {k: z[k] for k in z.files}
        self.cfg = yaml.safe_load(str(self.a["config_yaml"]))

    @property
    def toys(self) -> dict:
        return self.a

    def residual(self, obs: str, src: str) -> dict:
        return dict(nominal=self.a[f"res_{obs}_{src}_nominal"], cov=self.a[f"res_{obs}_{src}_cov"],
                    toys=self.a[f"res_{obs}_{src}_toys"], thr=list(self.a[f"res_{obs}_thr"]))

    @property
    def hqe(self):
        return self.a["hqe_central"], self.a["hqe_toys"]

    def truth(self, scen: str) -> dict:
        p = f"truth_{scen}_"
        return {k[len(p):]: v for k, v in self.a.items() if k.startswith(p)}

    def stack(self, obs: str):
        return [str(c) for c in self.a["stack_categories"]], self.a[f"stack_nominal_{obs}"], self.a[f"stackedges_{obs}"]
