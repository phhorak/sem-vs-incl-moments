#!/usr/bin/env python3
"""Step 3: Reweight cocktail and gap-mode ntuples; produce plots.

Reads ROOT files from output/1/, writes parquets to output/3/.
Produces figures:
  figures/3/mcambulance_impact.png   — pre/post MCAmbulance
  figures/3/ff_reweight_{mode}.png   — pre/post FF for Hammer-reweighted modes
  figures/3/uncertainty_bands.png    — stat/BF/FF uncertainty breakdown
  figures/3/gap_reweight_{mode}.png  — pre/post digitized-shape reweight (DsKenu, LcPenu)
"""
import argparse
import glob
import os
import subprocess
import sys
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pandas as pd
import uproot
import yaml
import matplotlib.pyplot as plt
import plothist


warnings.filterwarnings("ignore", category=FutureWarning, module="uproot")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--submit-gap", action="store_true", help="submit gap-mode reweighting jobs to bsub")
parser.add_argument("--submit-cocktail", action="store_true", help="submit cocktail/Hammer chunk jobs to bsub")
parser.add_argument("--cocktail-job-index", type=int, default=None, help="worker chunk index for cocktail processing")
parser.add_argument("--cocktail-job-count", type=int, default=None, help="total chunk count for cocktail processing")
parser.add_argument("--merge-cocktail", action="store_true", help="merge cocktail chunk parquets into output/3/cocktail.parquet")
parser.add_argument("--mode", default=None, help="single gap mode to process in section B")
parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
parser.add_argument("--skip-cocktail", action="store_true", help="skip section A (cocktail)")
parser.add_argument("--skip-gap", action="store_true", help="skip section B (gap modes)")
parser.add_argument("--skip-plots", action="store_true", help="skip section C (plots)")
args = parser.parse_args()

with open(args.config) as _f:
    cfg = yaml.safe_load(_f)

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
_LOCAL_PYTHON_PATHS = [
    _ROOT / "MCAmbulance",
    _ROOT / "lib",
    _REPO_ROOT / "references",
    _REPO_ROOT / "references" / "SysVar" / "src",
]
for _path in reversed(_LOCAL_PYTHON_PATHS):
    _path_str = str(_path)
    if _path.is_dir() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

_SYSVAR_SRC = _REPO_ROOT / "references" / "SysVar" / "src"
if _SYSVAR_SRC.is_dir():
    _sysvar_src_str = str(_SYSVAR_SRC)
    if _sysvar_src_str in sys.path:
        sys.path.remove(_sysvar_src_str)
    sys.path.insert(0, _sysvar_src_str)

_loaded_sysvar = sys.modules.get("sysvar")
if _loaded_sysvar is not None and not hasattr(_loaded_sysvar, "__path__"):
    sys.modules.pop("sysvar", None)

from lib.gap_modes import GAP_MODES as _GAP_MODES_META
from lib.moments import compute_curves_np, YLABELS, GRID, ROW_TITLES

out3  = Path(cfg["paths"]["output"]) / "3"
fig3  = Path(cfg["paths"]["figures"]) / "3"
out1  = Path(cfg["paths"]["output"]) / "1"
out2  = Path(cfg["paths"]["output"]) / "2"
out3.mkdir(parents=True, exist_ok=True)
fig3.mkdir(parents=True, exist_ok=True)

rw = cfg["reweight"]

# Optional batch submission for section-B gap processing.
if args.submit_gap:
    active_gap = cfg["generation"]["gap_modes"]
    if args.mode:
        active_gap = [m for m in active_gap if m == args.mode]
        if not active_gap:
            print(f"error: mode {args.mode!r} not found in generation.gap_modes")
            sys.exit(1)
    queue = rw.get("queue", cfg.get("generation", {}).get("queue", "s"))
    log_base = _ROOT / "logs" / "3" / "gap_modes"
    log_base.mkdir(parents=True, exist_ok=True)
    print(f"Submitting {len(active_gap)} gap-mode reweight job(s) to queue '{queue}'")
    for mode in active_gap:
        log_file = log_base / f"{mode}.log"
        job_cmd = (
            f'python3 "{_ROOT / "3_reweight.py"}" '
            f'--config "{args.config}" --mode "{mode}" --skip-cocktail --skip-plots'
        )
        bsub_cmd = f'bsub -q {queue} -oo "{log_file}" {job_cmd}'
        print(bsub_cmd)
        if not args.dry_run:
            subprocess.run(bsub_cmd, shell=True, check=True)
    if args.dry_run:
        print("(dry run — no jobs submitted)")
    else:
        print("all gap-mode jobs submitted")
    sys.exit(0)

if args.submit_cocktail:
    root_files = sorted(glob.glob(str((Path(cfg["paths"]["output"]) / "1") / "**" / "*.root"), recursive=True))
    root_files = [p for p in root_files if "gap_modes" not in p]
    if not root_files:
        print("error: no cocktail ROOT files found under output/1")
        sys.exit(1)
    queue = rw.get("queue", cfg.get("generation", {}).get("queue", "s"))
    n_jobs = int(rw.get("cocktail_n_jobs", min(32, max(1, len(root_files)))))
    log_base = _ROOT / "logs" / "3" / "cocktail"
    log_base.mkdir(parents=True, exist_ok=True)
    print(f"Submitting {n_jobs} cocktail/Hammer chunk job(s) to queue '{queue}'")
    for j in range(n_jobs):
        log_file = log_base / f"{j:04d}.log"
        job_cmd = (
            f'python3 "{_ROOT / "3_reweight.py"}" '
            f'--config "{args.config}" '
            f'--cocktail-job-index {j} --cocktail-job-count {n_jobs} '
            f'--skip-gap --skip-plots'
        )
        bsub_cmd = f'bsub -q {queue} -oo "{log_file}" {job_cmd}'
        print(bsub_cmd)
        if not args.dry_run:
            subprocess.run(bsub_cmd, shell=True, check=True)
    if args.dry_run:
        print("(dry run — no jobs submitted)")
    else:
        print("all cocktail chunk jobs submitted")
    sys.exit(0)

# ── MCAmbulance ───────────────────────────────────────────────────────────────

from mcambulance import MCAmbulance  

_MCA = {
    "Bp_D0stenu": MCAmbulance("bp", "d0_d_pi",    "e"),
    "B0_D0stenu": MCAmbulance("b0", "d0_d_pi",    "e"),
    "Bp_Dp1enu":  MCAmbulance("bp", "d1p_dstar_pi", "e"),
    "B0_Dp1enu":  MCAmbulance("b0", "d1p_dstar_pi", "e"),
}

# ── Hammer ────────────────────────────────────────────────────────────────────

try:
    from hammer_wrapper.ff_reweighting import HammerStudy
    from hammer_wrapper.ff_models import BGL, BLR

    _hs = HammerStudy(mode="B2", default_names=False)
    _hs._generations["B"]                  = "B"
    _hs._generations["B_daughters"]        = ["d0", "d1", "d2"]
    _hs._generations["Xc_daughters"]       = ["d1d0", "d1d1"]
    _hs._generations["Xc_grand_daughters"] = ["d1d0d0", "d1d0d1"]
    _hs._pdg_var            = "mcPDG"
    _hs._four_momentum_vars = ["mcE", "mcPX", "mcPY", "mcPZ"]
    _hs.set_ff_input_scheme(verbose=False)
    _FF_SCHEME = "ff_weight"
    _hs.add_ff_scheme([
        BGL.BtoDEllNu_dec_updated(),
        BGL.BtoDstEllNu_new(),
        BLR.BtoD1EllNu_new(),
        BLR.BtoD0stEllNu_new(),
        BLR.BtoD1prEllNu_new(),
        BLR.BtoD2stEllNu_new(),
    ], scheme_name=_FF_SCHEME, verbose=False)
    _HAMMER_OK = True
    print("Hammer initialised successfully.")
except ImportError as e:
    _HAMMER_OK = False
    _FF_SCHEME = "ff_weight"
    print(f"Hammer not available ({e}): ff_weight will be NaN.")

# ── PDG → decay name mapping ──────────────────────────────────────────────────

PDG_NAME = {
    (521, frozenset([11, 12, 421])):          "Bp_Denu",
    (521, frozenset([11, 12, 423])):          "Bp_Dstenu",
    (521, frozenset([11, 12, 10421])):        "Bp_D0stenu",
    (521, frozenset([11, 12, 10423])):        "Bp_D1enu",
    (521, frozenset([11, 12, 20423])):        "Bp_Dp1enu",
    (521, frozenset([11, 12, 425])):          "Bp_D2stenu",
    (521, frozenset([11, 12, 411, 211])):     "Bp_Dmpienu",
    (521, frozenset([11, 12, 413, 211])):     "Bp_Dstmpienu",
    (521, frozenset([11, 12, 421, 111])):     "Bp_D0pi0enu",
    (521, frozenset([11, 12, 423, 111])):     "Bp_Dst0pi0enu",
    (521, frozenset([11, 12, 421, 211, 0])): "Bp_D0pipipenu",
    (521, frozenset([11, 12, 411, 211, 111])):"Bp_Dmpipizenu",
    (521, frozenset([11, 12, 421, 111, 0])): "Bp_D0pizpizenu",
    (521, frozenset([11, 12, 423, 211, 0])): "Bp_Dst0pipipenu",
    (521, frozenset([11, 12, 413, 211, 111])):"Bp_Dstmpipizenu",
    (521, frozenset([11, 12, 423, 111, 0])): "Bp_Dst0pizpizenu",
    (521, frozenset([11, 12, 431, 321])):    "Bp_DsKenu",
    (521, frozenset([11, 12, 433, 321])):    "Bp_DsstkKenu",
    (521, frozenset([11, 12, 421, 221])):    "Bp_D0etaenu",
    (521, frozenset([11, 12, 423, 221])):    "Bp_Dst0etaenu",
    (511, frozenset([11, 12, 421])):         "B0_Denu",
    (511, frozenset([11, 12, 423])):         "B0_Dstenu",
    (511, frozenset([11, 12, 10421])):       "B0_D0stenu",
    (511, frozenset([11, 12, 10423])):       "B0_D1enu",
    (511, frozenset([11, 12, 20423])):       "B0_Dp1enu",
    (511, frozenset([11, 12, 425])):         "B0_D2stenu",
    (511, frozenset([11, 12, 421, 211])):    "B0_D0pienu",
    (511, frozenset([11, 12, 423, 111])):    "B0_Dstpi0enu",
    (511, frozenset([11, 12, 411, 111])):    "B0_Dpi0enu",
    (511, frozenset([11, 12, 413, 211])):    "B0_Dst0pienu",
    (511, frozenset([11, 12, 411, 211, 0])): "B0_Dpipipenu",
    (511, frozenset([11, 12, 421, 211, 111])):"B0_D0pipipzenu",
    (511, frozenset([11, 12, 411, 111, 0])): "B0_Dpizpizenu",
    (511, frozenset([11, 12, 413, 211, 0])): "B0_Dstpipipenu",
    (511, frozenset([11, 12, 423, 211, 111])):"B0_Dst0pipipzenu",
    (511, frozenset([11, 12, 413, 111, 0])): "B0_Dstpizpizenu",
    (511, frozenset([11, 12, 411, 221])):    "B0_Detaenu",
    (511, frozenset([11, 12, 413, 221])):    "B0_Dstetaenu",
}

_CATS = {
    "Denu": "D", "Dstenu": "D*",
    "D1enu": "D**", "D0stenu": "D**", "Dp1enu": "D**", "D2stenu": "D**",
    "Dmpienu": "D(*) pi", "Dstmpienu": "D(*) pi", "D0pi0enu": "D(*) pi",
    "Dst0pi0enu": "D(*) pi", "D0pienu": "D(*) pi", "Dpi0enu": "D(*) pi",
    "Dst0pienu": "D(*) pi", "Dstpi0enu": "D(*) pi",
    "D0pipipenu": "D(*) pi pi", "Dmpipizenu": "D(*) pi pi",
    "D0pizpizenu": "D(*) pi pi", "Dst0pipipenu": "D(*) pi pi",
    "Dstmpipizenu": "D(*) pi pi", "Dst0pizpizenu": "D(*) pi pi",
    "Dpipipenu": "D(*) pi pi", "D0pipipzenu": "D(*) pi pi",
    "Dpizpizenu": "D(*) pi pi", "Dstpipipenu": "D(*) pi pi",
    "Dst0pipipzenu": "D(*) pi pi", "Dstpizpizenu": "D(*) pi pi",
    "DsKenu": "Ds(*) K", "DsstkKenu": "Ds(*) K",
    "D0etaenu": "gap", "Dst0etaenu": "gap", "Detaenu": "gap", "Dstetaenu": "gap",
}

CATEGORY = {}
for name in PDG_NAME.values():
    n = name.split("_", 1)[1]
    CATEGORY[name] = _CATS.get(n, "unknown")

_BF_TABLE = {
    "Bp_Denu":         (2.27e-2, 0.06e-2),
    "Bp_Dstenu":       (5.27e-2, 0.12e-2),
    "Bp_D1enu":        (6.4e-3,  1.0e-3),
    "Bp_D0stenu":      (1.3e-3,  0.3e-3),
    "Bp_Dp1enu":       (2.8e-3,  0.4e-3),
    "Bp_D2stenu":      (3.2e-3,  0.3e-3),
    "Bp_Dmpienu":      (0.0,     0.0),
    "Bp_D0pi0enu":     (0.0,     0.0),
    "Bp_Dstmpienu":    (0.0,     0.0),
    "Bp_Dst0pi0enu":   (0.0,     0.0),
    "Bp_D0pipipenu":   (0.7e-3 / 3, 0.9e-3 / 3),
    "Bp_Dmpipizenu":   (0.7e-3 / 3, 0.9e-3 / 3),
    "Bp_D0pizpizenu":  (0.7e-3 / 3, 0.9e-3 / 3),
    "Bp_Dst0pipipenu": (2.2e-3 / 3, 1.0e-3 / 3),
    "Bp_Dstmpipizenu": (2.2e-3 / 3, 1.0e-3 / 3),
    "Bp_Dst0pizpizenu":(2.2e-3 / 3, 1.0e-3 / 3),
    "Bp_DsKenu":       (0.30e-3, 0.13e-3),
    "Bp_DsstkKenu":    (0.29e-3, 0.19e-3),
    "Bp_D0etaenu":     (0.0,     0.0),
    "Bp_Dst0etaenu":   (0.0,     0.0),
    "B0_Denu":         (2.11e-2, 0.05e-2),
    "B0_Dstenu":       (4.90e-2, 0.11e-2),
    "B0_D1enu":        (5.9e-3,  1.0e-3),
    "B0_D0stenu":      (1.2e-3,  0.3e-3),
    "B0_Dp1enu":       (2.6e-3,  0.4e-3),
    "B0_D2stenu":      (3.0e-3,  0.3e-3),
    "B0_D0pienu":      (1.2e-3 * 2/3, 0.3e-3 * 2/3),
    "B0_Dpi0enu":      (1.2e-3 / 3,   0.3e-3 / 3),
    "B0_Dst0pienu":    (2.6e-3 * 2/3, 0.4e-3 * 2/3),
    "B0_Dstpi0enu":    (2.6e-3 / 3,   0.4e-3 / 3),
    "B0_Dpipipenu":    (0.7e-3 / 3,   0.8e-3 / 3),
    "B0_D0pipipzenu":  (0.7e-3 / 3,   0.8e-3 / 3),
    "B0_Dpizpizenu":   (0.7e-3 / 3,   0.8e-3 / 3),
    "B0_Dstpipipenu":  (2.0e-3 / 3,   1.0e-3 / 3),
    "B0_Dst0pipipzenu":(2.0e-3 / 3,   1.0e-3 / 3),
    "B0_Dstpizpizenu": (2.0e-3 / 3,   1.0e-3 / 3),
    "B0_Detaenu":      (0.0,     0.0),
    "B0_Dstetaenu":    (0.0,     0.0),
}
BF     = {k: v[0] for k, v in _BF_TABLE.items()}
BF_UNC = {k: v[1] for k, v in _BF_TABLE.items()}

_HAMMER_DECAYS = {
    "Bp_Denu", "Bp_Dstenu", "Bp_D1enu", "Bp_D0stenu", "Bp_Dp1enu", "Bp_D2stenu",
    "B0_Denu", "B0_Dstenu", "B0_D1enu", "B0_D0stenu", "B0_Dp1enu", "B0_D2stenu",
}


# ── Physics computation ───────────────────────────────────────────────────────

def _cms_to_xyz(p, theta, phi):
    return (p * np.sin(theta) * np.cos(phi),
            p * np.sin(theta) * np.sin(phi),
            p * np.cos(theta))


def _compute_vars(df):
    l_E = l_px = l_py = l_pz = pd.Series(np.nan, index=df.index)
    nu_E = nu_px = nu_py = nu_pz = pd.Series(np.nan, index=df.index)
    for i in range(5):
        pdg   = df[f"d{i}_mcPDG"].abs()
        is_l  = (pdg == 11) & l_E.isna()
        is_nu = (pdg == 12) & nu_E.isna()
        l_E   = l_E.where(~is_l,   df[f"d{i}_E"]);   l_px = l_px.where(~is_l,  df[f"d{i}_px"])
        l_py  = l_py.where(~is_l,  df[f"d{i}_py"]);  l_pz = l_pz.where(~is_l,  df[f"d{i}_pz"])
        nu_E  = nu_E.where(~is_nu, df[f"d{i}_E"]);   nu_px = nu_px.where(~is_nu, df[f"d{i}_px"])
        nu_py = nu_py.where(~is_nu, df[f"d{i}_py"]); nu_pz = nu_pz.where(~is_nu, df[f"d{i}_pz"])
    B_px, B_py, B_pz = _cms_to_xyz(df["B_pCM"], df["B_thetaCM"], df["B_phiCM"])
    B_E = df["B_ECM"]
    X_E  = B_E - l_E - nu_E
    X_px = B_px - l_px - nu_px; X_py = B_py - l_py - nu_py; X_pz = B_pz - l_pz - nu_pz
    df["Mx"] = np.sqrt(np.clip(X_E**2 - (X_px**2 + X_py**2 + X_pz**2), 0, None))
    q_E  = l_E + nu_E; q_px = l_px + nu_px; q_py = l_py + nu_py; q_pz = l_pz + nu_pz
    df["q2"] = np.clip(q_E**2 - (q_px**2 + q_py**2 + q_pz**2), 0, None)
    B_p2       = B_px**2 + B_py**2 + B_pz**2
    B_gamma    = B_E / np.sqrt(np.clip(B_E**2 - B_p2, 1e-12, None))
    beta_dot_p = (B_px * l_px + B_py * l_py + B_pz * l_pz) / B_E
    df["El_B"] = B_gamma * (l_E - beta_dot_p)
    return df


def _pdg_sig(row):
    b = int(abs(row["mcPDG"]))
    raw = [int(abs(row[f"d{i}_mcPDG"])) for i in range(5) if not np.isnan(row[f"d{i}_mcPDG"])]
    sig = set(raw)
    if len(raw) != len(sig):
        sig.add(0)
    return (b, frozenset(sig))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: Cocktail processing
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== SECTION A: cocktail reweighting ===")
root_files = sorted(glob.glob(str(out1 / "**" / "*.root"), recursive=True))
# Exclude gap_modes subtree from cocktail
root_files = [p for p in root_files if "gap_modes" not in p]

if not args.skip_cocktail and not root_files:
    print(f"No ROOT files found under {out1}. Run 1_generate.py first.")
    sys.exit(1)

cocktail_path = out3 / "cocktail.parquet"
chunk_dir = out3 / "cocktail_chunks"
chunk_dir.mkdir(parents=True, exist_ok=True)

if args.merge_cocktail:
    chunks = sorted(chunk_dir.glob("cocktail_*.parquet"))
    if not chunks:
        print(f"error: no chunk files found in {chunk_dir}")
        sys.exit(1)
    print(f"  merging {len(chunks)} cocktail chunk(s)")
    merged = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    merged.to_parquet(cocktail_path, index=False)
    print(f"  merged cocktail saved → {cocktail_path}")
    args.skip_cocktail = True

if not args.skip_cocktail:
    if (args.cocktail_job_index is None) ^ (args.cocktail_job_count is None):
        print("error: set both --cocktail-job-index and --cocktail-job-count (or neither)")
        sys.exit(1)
    if args.cocktail_job_index is not None:
        if args.cocktail_job_count <= 0 or args.cocktail_job_index < 0 or args.cocktail_job_index >= args.cocktail_job_count:
            print("error: invalid cocktail chunk index/count")
            sys.exit(1)
        root_files = [p for i, p in enumerate(root_files) if (i % args.cocktail_job_count) == args.cocktail_job_index]
        print(f"  cocktail chunk {args.cocktail_job_index+1}/{args.cocktail_job_count}: {len(root_files)} file(s)")
        if not root_files:
            print("  [warn] no files assigned to this chunk; writing empty chunk parquet")
            pd.DataFrame().to_parquet(chunk_dir / f"cocktail_{args.cocktail_job_index:04d}.parquet", index=False)
            sys.exit(0)

    frames = []
    for path in root_files:
        print(f"  reading {path}")
        f = uproot.open(path)
        if "reco" not in f:
            print("    no 'reco' tree — skipping")
            continue
        df = f["reco"].arrays(library="pd")
        df = _compute_vars(df)
        # Force a 1D Series of tuple signatures; without reduce, pandas may expand
        # tuple-like outputs into a temporary DataFrame for some inputs/versions.
        sigs = df.apply(_pdg_sig, axis=1, result_type="reduce")
        df["decay_name"] = sigs.map(lambda s: PDG_NAME.get(s, f"unknown_{s[1]}"))
        df["category"]   = df["decay_name"].map(CATEGORY).fillna("unknown")
        df["_n_reco"]    = len(df)
        frames.append(df)

    out_df = pd.concat(frames, ignore_index=True)

    out_df["bf"]     = out_df["decay_name"].map(BF).fillna(0.0)
    out_df["bf_unc"] = out_df["decay_name"].map(BF_UNC).fillna(0.0)
    out_df["weight"] = (out_df["bf"] / out_df["_n_reco"].replace(0, np.nan)).fillna(0.0)
    out_df = out_df.drop(columns=["_n_reco"])

    out_df["mc_weight"] = 1.0
    for dname, corrector in _MCA.items():
        mask = out_df["decay_name"] == dname
        if mask.any():
            out_df.loc[mask, "mc_weight"] = corrector.CorrectionWeight(out_df.loc[mask, "Mx"].to_numpy())

    if _HAMMER_OK:
        out_df["B_mcPDG"] = out_df["mcPDG"]
        out_df["B_mcE"]   = out_df["mcE"]
        ff_cols = _hs._get_new_df_columnnames(_FF_SCHEME)
        for col in ff_cols:
            out_df[col] = -1 if col.startswith(("hammer_found_rates_", "hammer_reweighted_")) else 1.0
        hmask = out_df["decay_name"].isin(_HAMMER_DECAYS)
        if hmask.any():
            tmp = out_df.loc[hmask].copy()
            _hs.process_events(tmp, _FF_SCHEME, verbose=False)
            out_df.loc[hmask, ff_cols] = tmp[ff_cols].to_numpy()
    else:
        out_df[_FF_SCHEME] = np.nan

    out_df["total_weight"] = out_df["weight"] * out_df["mc_weight"] * out_df["ff_weight"]

    out_df = pd.DataFrame({c: np.asarray(out_df[c]) for c in out_df.columns})
    if args.cocktail_job_index is None:
        out_df.to_parquet(cocktail_path, index=False)
        print(f"  saved {len(out_df):,} events → {cocktail_path}")
    else:
        chunk_path = chunk_dir / f"cocktail_{args.cocktail_job_index:04d}.parquet"
        out_df.to_parquet(chunk_path, index=False)
        print(f"  saved {len(out_df):,} events → {chunk_path}")
else:
    print("  [skip] cocktail processing disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: Gap mode processing
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== SECTION B: gap mode processing ===")

GAP_MODES_PROC = {
    "D2stDeta":  {"decay_name": "Bp_D2stDeta",   "category": r"gap: $D_2^*{\to}D\eta$"},
    "DummyDeta": {"decay_name": "Bp_DummyDeta",  "category": r"gap: $D\eta$ (non-res.)"},
    "Dp1DstEta": {"decay_name": "Bp_Dp1DstEta",  "category": r"gap: $D_1'{\to}D^*\eta$"},
    "Dp1Deta":   {"decay_name": "Bp_Dp1Deta",    "category": r"gap: $D_1'{\to}D\eta$"},
    "DsKenu":    {"decay_name": "Bp_DsKenu_gap", "category": r"gap: $D_s K$ (dummy-res.)"},
    "LcPenu":    {"decay_name": "Bp_LcPenu",     "category": r"gap: $\Lambda_c\bar{p}$ (dummy-res.)"},
}

_DIGITIZED_MODES = {
    "DsKenu": ("dsk",       "d1d0", "d1d1"),
    "LcPenu": ("lambdac_p", "d1d0", "d1d1"),
}

bf_gap   = rw["bf_gap"]
mass_max = rw["lcp_mass_max"]
n_bins   = rw["lcp_nbins"]
max_r    = rw["lcp_max_ratio"]

active_gap = cfg["generation"]["gap_modes"]
if args.mode:
    active_gap = [m for m in active_gap if m == args.mode]
    if not active_gap:
        print(f"error: mode {args.mode!r} not found in generation.gap_modes")
        sys.exit(1)

if not args.skip_gap:
    for mode in active_gap:
        info    = GAP_MODES_PROC.get(mode)
        if info is None:
            print(f"  [warn] unknown gap mode {mode!r} — skipping")
            continue
        in_dir  = out1 / "gap_modes" / mode
        rf      = sorted(glob.glob(str(in_dir / "**" / "*.root"), recursive=True))
        if not rf:
            print(f"  [warn] no ROOT files for gap mode {mode} under {in_dir}")
            continue

        gframes = []
        for path in rf:
            print(f"  reading {path}")
            f = uproot.open(path)
            tn = "reco" if "reco" in f else list(f.keys())[0].split(";")[0]
            if tn not in f:
                continue
            df = f[tn].arrays(library="pd")
            df = _compute_vars(df)
            gframes.append(df)

        if not gframes:
            continue

        gdf = pd.concat(gframes, ignore_index=True)
        n   = len(gdf)
        gdf["decay_name"]    = info["decay_name"]
        gdf["category"]      = info["category"]
        gdf["bf"]            = bf_gap
        gdf["bf_unc"]        = 0.0
        gdf["weight"]        = bf_gap / n if n > 0 else 0.0
        gdf["mc_weight"]     = 1.0
        gdf["ff_weight"]     = 1.0
        gdf["total_weight"]  = gdf["weight"]
        gdf["spectrum_weight"] = 1.0

        if mode in _DIGITIZED_MODES:
            key, d1, d2 = _DIGITIZED_MODES[mode]
            csv_path = out2 / f"{key}_digitized.csv"
            needed   = [f"{d1}_mcE", f"{d1}_mcPX", f"{d1}_mcPY", f"{d1}_mcPZ",
                        f"{d2}_mcE", f"{d2}_mcPX", f"{d2}_mcPY", f"{d2}_mcPZ"]
            if not csv_path.exists():
                print(f"  [warn] digitized CSV not found for {mode}: {csv_path} — skipping shape reweight")
            elif not all(c in gdf.columns for c in needed):
                missing = [c for c in needed if c not in gdf.columns]
                print(f"  [warn] missing columns for {mode} shape reweight: {missing} — skipping")
            else:
                pts   = pd.read_csv(csv_path)
                if "x" in pts.columns:
                    x_d = pts["x"].to_numpy(float)
                else:
                    x_d = pts["x_norm"].to_numpy(float)
                y_d   = np.clip(pts["y_norm"].to_numpy(float), 1e-8, None)
                ok_d  = np.isfinite(x_d) & np.isfinite(y_d)
                x_d, y_d = x_d[ok_d], y_d[ok_d]
                order = np.argsort(x_d)
                x_d, y_d = x_d[order], y_d[order]

                e1  = gdf[f"{d1}_mcE"].to_numpy(float);  px1 = gdf[f"{d1}_mcPX"].to_numpy(float)
                py1 = gdf[f"{d1}_mcPY"].to_numpy(float); pz1 = gdf[f"{d1}_mcPZ"].to_numpy(float)
                e2  = gdf[f"{d2}_mcE"].to_numpy(float);  px2 = gdf[f"{d2}_mcPX"].to_numpy(float)
                py2 = gdf[f"{d2}_mcPY"].to_numpy(float); pz2 = gdf[f"{d2}_mcPZ"].to_numpy(float)
                m2  = (e1+e2)**2 - ((px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2)
                m   = np.sqrt(np.clip(m2, 0, None))

                m_thr = float(np.nanmin(m))
                m_pts = x_d

                win = min(5, max(3, len(y_d)))
                y_sm = np.convolve(y_d, np.ones(win) / win, mode="same")
                y_sm = np.clip(y_sm, 1e-8, None)

                bins    = np.linspace(m_thr, mass_max, n_bins + 1)
                centers = 0.5 * (bins[:-1] + bins[1:])
                w0      = gdf["total_weight"].to_numpy(float)
                mc_h, _ = np.histogram(m, bins=bins, weights=w0)
                mc_pdf  = mc_h / max(float(mc_h.sum()), 1e-12)
                tgt_y   = np.interp(centers, m_pts, y_sm, left=np.nan, right=np.nan)
                in_sup  = np.isfinite(tgt_y)
                tgt_pdf = np.zeros_like(centers)
                if in_sup.any():
                    tgt_pdf[in_sup] = tgt_y[in_sup] / max(float(tgt_y[in_sup].sum()), 1e-12)
                ratio   = np.ones_like(centers)
                good    = in_sup & (mc_pdf > 0)
                ratio[good] = np.clip(tgt_pdf[good] / np.clip(mc_pdf[good], 1e-12, None), 0, max_r)
                ib      = np.clip(np.digitize(m, bins) - 1, 0, len(ratio) - 1)
                rw_arr  = ratio[ib]
                wsum    = float(w0.sum())
                if wsum > 0:
                    mean_rw = float((w0 * rw_arr).sum() / wsum)
                    if mean_rw > 0:
                        rw_arr /= mean_rw
                gdf["spectrum_weight"] = rw_arr
                gdf["total_weight"]    = gdf["total_weight"] * rw_arr
                print(f"  [{mode}] applied {key} shape reweight")

                # Plot before/after
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
                x_label  = r"$m(\Lambda_c\bar{p})$" if mode == "LcPenu" else r"$m(D_s K)$"
                m_range  = (float(m_thr), float(mass_max))
                _bins_p  = np.linspace(*m_range, 40)
                axes[0].hist(m, bins=_bins_p, weights=gdf["weight"].to_numpy(float), density=True,
                             histtype="step", color="steelblue", label="MC (before)")
                axes[1].hist(m, bins=_bins_p, weights=gdf["total_weight"].to_numpy(float), density=True,
                             histtype="step", color="darkorange", label="MC (after)")
                for ax in axes:
                    ax.set_xlabel(f"{x_label} [GeV]"); ax.set_ylabel("Density"); ax.legend(frameon=False)
                axes[0].set_title(f"{mode}: before shape reweight")
                axes[1].set_title(f"{mode}: after shape reweight")
                fig.tight_layout()
                fig.savefig(fig3 / f"gap_reweight_{mode}.png", bbox_inches="tight")
                plt.close(fig)
                print(f"  wrote {fig3 / f'gap_reweight_{mode}.png'}")

        out_path = out3 / f"{mode}.parquet"
        gdf = pd.DataFrame({c: np.asarray(gdf[c]) for c in gdf.columns})
        gdf.to_parquet(out_path, index=False)
        print(f"  saved {n:,} events → {out_path}")
else:
    print("  [skip] gap mode processing disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: Reweighting plots
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== SECTION C: reweighting plots ===")
if args.skip_plots:
    print("  [skip] reweighting plots disabled")
    print("Step 3 complete.")
    sys.exit(0)

df = pd.read_parquet(cocktail_path)
if "total_weight" not in df.columns:
    df["total_weight"] = df["weight"] * df["mc_weight"] * df["ff_weight"]

pcfg     = cfg["plots"]
el_cuts  = np.linspace(pcfg["el_cuts_min"], pcfg["el_cuts_max"], pcfg["el_cuts_n"])
q2_cuts  = np.linspace(pcfg["q2_cuts_min"], pcfg["q2_cuts_max"], pcfg["q2_cuts_n"])
n_boot   = pcfg["n_bootstrap"]
seed     = pcfg["toy_seed"]

# --- MCAmbulance impact ---
print("  computing MCAmbulance impact curves ...")
if _HAMMER_OK:
    df_mca = df[df["ff_weight"] <= 10].copy()
else:
    df_mca = df.copy()
ok = (np.isfinite(df_mca["Mx"]) & np.isfinite(df_mca["El_B"]) & np.isfinite(df_mca["q2"])
      & np.isfinite(df_mca["total_weight"]) & (df_mca["total_weight"] > 0)).to_numpy()
nom_with    = compute_curves_np(np.square(df_mca.loc[ok, "Mx"].to_numpy(float)),
                                df_mca.loc[ok, "El_B"].to_numpy(float),
                                df_mca.loc[ok, "q2"].to_numpy(float),
                                df_mca.loc[ok, "total_weight"].to_numpy(float),
                                el_cuts, q2_cuts)
tw_nomc     = df_mca["weight"].to_numpy(float) * df_mca["ff_weight"].to_numpy(float)
nom_without = compute_curves_np(np.square(df_mca.loc[ok, "Mx"].to_numpy(float)),
                                df_mca.loc[ok, "El_B"].to_numpy(float),
                                df_mca.loc[ok, "q2"].to_numpy(float),
                                tw_nomc[ok], el_cuts, q2_cuts)

cuts_map = {"el_cuts": el_cuts, "q2_cuts": q2_cuts}
fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
for r, (_, keys, ck, xlabel) in enumerate(GRID):
    cuts = cuts_map[ck]
    for c, key in enumerate(keys):
        ax = axes[r, c]
        lw  = "With MCAmbulance"    if (r == 0 and c == 0) else None
        lwt = "Without MCAmbulance" if (r == 0 and c == 0) else None
        ax.plot(cuts, nom_with[key],    lw=1.8, color="steelblue",   label=lw)
        ax.plot(cuts, nom_without[key], lw=1.8, color="darkorange", ls="--", label=lwt)
        ax.set_ylabel(YLABELS[key]); ax.set_xlabel(xlabel)
        ax.set_xlim(float(cuts[0]), float(cuts[-1])); ax.grid(alpha=0.2)
        if r == 0:
            ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][c])
    axes[r, 0].text(-0.38, 0.5, ROW_TITLES[r], transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="center")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.tight_layout(rect=[0.04, 0.02, 0.99, 0.95])
fig.savefig(fig3 / "mcambulance_impact.png", bbox_inches="tight"); plt.close(fig)
print(f"  wrote {fig3 / 'mcambulance_impact.png'}")

# --- FF before/after (per Hammer-reweighted mode: q² distributions) ---
print("  producing FF reweight plots ...")
ff_modes_present = [m for m in _HAMMER_DECAYS if (df["decay_name"] == m).any()]
q2_bins = np.linspace(0, 12, 50)

def _mode_ff_band(sub_df: pd.DataFrame, bins: np.ndarray, n_toys: int, mode_seed: int):
    ff_ups = [c for c in sub_df.columns if c.startswith("ff_weight_up")]
    ff_dns = [c for c in sub_df.columns if c.startswith("ff_weight_down")]
    var_ids = sorted(set(int(c.replace("ff_weight_up", "")) for c in ff_ups)
                     & set(int(c.replace("ff_weight_down", "")) for c in ff_dns))
    if not var_ids:
        return None

    q2 = sub_df["q2"].to_numpy(float)
    base_w = (sub_df["weight"] * sub_df["mc_weight"]).to_numpy(float)
    ff_c = sub_df["ff_weight"].to_numpy(float)
    ok = np.isfinite(q2) & np.isfinite(base_w) & np.isfinite(ff_c) & (ff_c > 0)
    if not ok.any():
        return None

    q2 = q2[ok]
    base_w = base_w[ok]
    ff_c = ff_c[ok]
    ff_ud = [
        (
            sub_df[f"ff_weight_up{i}"].to_numpy(float)[ok] - ff_c,
            ff_c - sub_df[f"ff_weight_down{i}"].to_numpy(float)[ok],
        )
        for i in var_ids
    ]
    if not ff_ud:
        return None

    decay_codes = np.zeros(len(q2), dtype=np.int32)
    rng_mode = np.random.default_rng(mode_seed)
    toy_density = []
    for _ in range(n_toys):
        z = rng_mode.standard_normal((1, len(ff_ud)))
        ff_sm = ff_c.copy()
        for j, (u, d) in enumerate(ff_ud):
            zj = z[decay_codes, j]
            ff_sm += np.where(zj >= 0, zj * u, zj * d)
        w_toy = base_w * ff_sm
        w_toy = np.where(np.isfinite(w_toy), w_toy, 0.0)
        h, _ = np.histogram(q2, bins=bins, weights=w_toy, density=True)
        toy_density.append(h)
    arr = np.asarray(toy_density, dtype=float)
    return np.nanstd(arr, axis=0, ddof=1)

for mode_name in ff_modes_present:
    mask = (df["decay_name"] == mode_name).to_numpy()
    if _HAMMER_OK:
        mask &= (df["ff_weight"].to_numpy(float) <= 10)
    if not mask.any():
        continue
    sub   = df.loc[mask]
    w_no  = sub["weight"].to_numpy(float) * sub["mc_weight"].to_numpy(float)
    w_ff  = sub["total_weight"].to_numpy(float)
    q2    = sub["q2"].to_numpy(float)
    valid = np.isfinite(q2) & np.isfinite(w_no) & np.isfinite(w_ff)
    q2 = q2[valid]
    w_no = w_no[valid]
    w_ff = w_ff[valid]
    if len(q2) == 0:
        continue

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.hist(q2, bins=q2_bins, weights=w_no, density=True, histtype="step",
            color="steelblue", lw=1.5, label="Before FF reweight")
    ax.hist(q2, bins=q2_bins, weights=w_ff, density=True, histtype="step",
            color="darkorange", lw=1.5, ls="--", label="After FF reweight")
    if _HAMMER_OK:
        ff_std = _mode_ff_band(sub.loc[valid], q2_bins, n_boot, seed + (abs(hash(mode_name)) % 10_000))
        if ff_std is not None and np.any(np.isfinite(ff_std)):
            h_nom, _ = np.histogram(q2, bins=q2_bins, weights=w_ff, density=True)
            if np.any(np.isfinite(h_nom)):
                centers = 0.5 * (q2_bins[:-1] + q2_bins[1:])
                lo = np.clip(h_nom - ff_std, 0.0, None)
                hi = h_nom + ff_std
                ax.fill_between(
                    centers,
                    lo,
                    hi,
                    color="forestgreen",
                    alpha=0.25,
                    step="mid",
                    label="FF toy unc.",
                )
    ax.set_xlabel(r"$q^2\,[\mathrm{GeV}^2]$"); ax.set_ylabel("Density")
    ax.set_title(mode_name.replace("_", r"\_")); ax.legend(frameon=False); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig3 / f"ff_reweight_{mode_name}.png", bbox_inches="tight"); plt.close(fig)
print(f"  wrote FF plots for {len(ff_modes_present)} modes")

# --- Uncertainty breakdown (stat / BF / FF toys) ---
print(f"  running {n_boot} toys per uncertainty source ...")

if _HAMMER_OK:
    df_unc = df[df["ff_weight"] <= 10].copy()
else:
    # If Hammer is unavailable, keep stat/BF toys meaningful by using unit FF.
    df_unc = df.copy()
    df_unc["ff_weight"] = 1.0
    df_unc["total_weight"] = df_unc["weight"] * df_unc["mc_weight"]
rng    = np.random.default_rng(seed)
KEY_ORDER = ["mx_1", "mx_2", "mx_3", "el_1", "el_2", "el_3", "q2_1", "q2_2", "q2_3"]

def _run_toys(source: str) -> dict[str, np.ndarray]:
    decay_codes, _ = pd.factorize(df_unc["decay_name"], sort=True)
    n_dec   = int(decay_codes.max() + 1) if len(decay_codes) else 0
    decay_codes = decay_codes.astype(np.int32)
    base_w  = (df_unc["weight"] * df_unc["mc_weight"]).to_numpy(float)
    bf_v    = df_unc["bf"].to_numpy(float)
    bf_u    = df_unc["bf_unc"].to_numpy(float)
    rel_bf  = np.divide(bf_u, bf_v, out=np.zeros_like(bf_u), where=bf_v > 0)
    ff_c    = df_unc["ff_weight"].to_numpy(float)
    ff_ups  = [c for c in df_unc.columns if c.startswith("ff_weight_up")]
    ff_dns  = [c for c in df_unc.columns if c.startswith("ff_weight_down")]
    var_ids = sorted(set(int(c.replace("ff_weight_up","")) for c in ff_ups)
                   & set(int(c.replace("ff_weight_down","")) for c in ff_dns))
    ff_ud   = [(df_unc[f"ff_weight_up{i}"].to_numpy(float) - ff_c,
                ff_c - df_unc[f"ff_weight_down{i}"].to_numpy(float)) for i in var_ids]

    ok  = (np.isfinite(df_unc["Mx"]) & np.isfinite(df_unc["El_B"])
           & np.isfinite(df_unc["q2"]) & np.isfinite(base_w)).to_numpy()
    N_ok   = int(ok.sum())
    if N_ok == 0:
        empty = {
            "mx_1": np.full_like(el_cuts, np.nan, dtype=float),
            "mx_2": np.full_like(el_cuts, np.nan, dtype=float),
            "mx_3": np.full_like(el_cuts, np.nan, dtype=float),
            "el_1": np.full_like(el_cuts, np.nan, dtype=float),
            "el_2": np.full_like(el_cuts, np.nan, dtype=float),
            "el_3": np.full_like(el_cuts, np.nan, dtype=float),
            "q2_1": np.full_like(q2_cuts, np.nan, dtype=float),
            "q2_2": np.full_like(q2_cuts, np.nan, dtype=float),
            "q2_3": np.full_like(q2_cuts, np.nan, dtype=float),
        }
        return empty
    mx2_ok = np.square(df_unc["Mx"].to_numpy(float)[ok])
    el_ok  = df_unc["El_B"].to_numpy(float)[ok]
    q2_ok  = df_unc["q2"].to_numpy(float)[ok]
    bw_ok  = base_w[ok]; dc_ok = decay_codes[ok]; rb_ok = rel_bf[ok]; fc_ok = ff_c[ok]
    fud_ok = [(u[ok], d[ok]) for u, d in ff_ud]

    storage: dict[str, list] = {k: [] for k in KEY_ORDER}
    for _ in range(n_boot):
        counts = rng.integers(0, N_ok, size=N_ok) if source == "stat" else np.arange(N_ok)
        c_arr  = np.bincount(counts, minlength=N_ok).astype(float) if source == "stat" else np.ones(N_ok)
        br_fac = (1.0 + rng.standard_normal(n_dec)[dc_ok] * rb_ok) if source == "bf" else 1.0
        ff_sm  = fc_ok.copy()
        if source == "ff" and fud_ok:
            z = rng.standard_normal((n_dec, len(fud_ok)))
            for j, (u, d) in enumerate(fud_ok):
                zj = z[dc_ok, j]
                ff_sm += np.where(zj >= 0, zj * u, zj * d)
        w_toy = bw_ok * br_fac * ff_sm * c_arr
        w_toy = np.where(np.isfinite(w_toy), w_toy, 0.0)
        cur   = compute_curves_np(mx2_ok, el_ok, q2_ok, w_toy, el_cuts, q2_cuts)
        for k in KEY_ORDER:
            storage[k].append(cur[k])
    return {k: np.nanstd(np.stack(storage[k]), axis=0, ddof=1) for k in KEY_ORDER}

ok_unc = (np.isfinite(df_unc["Mx"]) & np.isfinite(df_unc["El_B"])
          & np.isfinite(df_unc["q2"]) & np.isfinite(df_unc["total_weight"])
          & (df_unc["total_weight"] > 0)).to_numpy()
nom = compute_curves_np(np.square(df_unc.loc[ok_unc, "Mx"].to_numpy(float)),
                        df_unc.loc[ok_unc, "El_B"].to_numpy(float),
                        df_unc.loc[ok_unc, "q2"].to_numpy(float),
                        df_unc.loc[ok_unc, "total_weight"].to_numpy(float),
                        el_cuts, q2_cuts)

t0 = time.monotonic()
std_stat = _run_toys("stat"); print(f"  stat done ({time.monotonic()-t0:.0f}s)")
std_bf   = _run_toys("bf");   print(f"  BF done ({time.monotonic()-t0:.0f}s)")
std_ff   = _run_toys("ff");   print(f"  FF done ({time.monotonic()-t0:.0f}s)")

band_cfg = [("stat", std_stat, "steelblue", 0.45),
            ("BF",   std_bf,   "darkorange", 0.45),
            ("FF",   std_ff,   "forestgreen", 0.45)]

fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
for r, (_, keys, ck, xlabel) in enumerate(GRID):
    cuts = cuts_map[ck]
    for c, key in enumerate(keys):
        ax = axes[r, c]
        ax.plot(cuts, nom[key], color="black", lw=1.5, zorder=5)
        for label, std, color, alpha in band_cfg:
            if not np.any(np.isfinite(std[key])):
                continue
            lbl = label if (r == 0 and c == 0) else None
            ax.fill_between(cuts, nom[key] - std[key], nom[key] + std[key],
                            alpha=alpha, lw=0, color=color, label=lbl)
        ax.set_ylabel(YLABELS[key]); ax.set_xlabel(xlabel)
        ax.set_xlim(float(cuts[0]), float(cuts[-1])); ax.grid(alpha=0.2)
        if r == 0:
            ax.set_title([r"$n=1$", r"$n=2$", r"$n=3$"][c])
    axes[r, 0].text(-0.38, 0.5, ROW_TITLES[r], transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="center")
from matplotlib.lines import Line2D
h_nom = Line2D([0], [0], color="black", lw=1.5, label="Nominal")
h, l  = axes[0, 0].get_legend_handles_labels()
fig.legend([h_nom] + h, ["Nominal"] + l, loc="upper center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.tight_layout(rect=[0.04, 0.02, 0.99, 0.95])
fig.savefig(fig3 / "uncertainty_bands.png", bbox_inches="tight"); plt.close(fig)
print(f"  wrote {fig3 / 'uncertainty_bands.png'}")
print("Step 3 complete.")
