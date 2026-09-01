#!/usr/bin/env python3
"""Step 3: Reweight cocktail and gap-mode ntuples.

Reads ROOT files from output/1/, writes parquets to output/3/.
Produces figures:
  figures/3/gap_reweight_{mode}.png  — pre/post digitized-shape reweight (DsKenu, LcPenu)
"""
import argparse
import glob
import os
import subprocess
import sys
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


out3  = Path(cfg["paths"]["output"]) / "3"
fig3  = Path(cfg["paths"]["figures"]) / "3"
out1  = Path(cfg["paths"]["output"]) / "1"
out2  = Path(cfg["paths"]["output"]) / "2"
out3.mkdir(parents=True, exist_ok=True)
fig3.mkdir(parents=True, exist_ok=True)

rw       = cfg["reweight"]
mass_max = rw["lcp_mass_max"]
n_bins   = rw["lcp_nbins"]
max_r    = rw["lcp_max_ratio"]

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
            f'--config "{args.config}" --mode "{mode}" --skip-cocktail'
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
            f'--skip-gap'
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
    # Cocktail (Section A): keyed directly by decay_name, which already encodes charge+lepton.
    "Bp_D0stenu":      MCAmbulance("bp", "d0_d_pi",      "e"),
    "Bp_D0stmunu":     MCAmbulance("bp", "d0_d_pi",      "mu"),
    "B0_D0stenu":      MCAmbulance("b0", "d0_d_pi",      "e"),
    "B0_D0stmunu":     MCAmbulance("b0", "d0_d_pi",      "mu"),
    "Bp_Dp1enu":       MCAmbulance("bp", "d1p_dstar_pi", "e"),
    "Bp_Dp1munu":      MCAmbulance("bp", "d1p_dstar_pi", "mu"),
    "B0_Dp1enu":       MCAmbulance("b0", "d1p_dstar_pi", "e"),
    "B0_Dp1munu":      MCAmbulance("b0", "d1p_dstar_pi", "mu"),
    # Gap modes (Section B): keyed by "{decay_name}_{lepton}" since one gdf mixes both lepton
    # flavors (decay_name itself carries only charge, not lepton, for gap modes -- see
    # GAP_MODES_PROC). decay_name here is the Bp_/B0_-prefixed value from GAP_MODES_PROC, not
    # the bare config mode name.
    "Bp_Dp1DstEta_e":  MCAmbulance("bp", "d1p_dstar_eta", "e"),
    "Bp_Dp1DstEta_mu": MCAmbulance("bp", "d1p_dstar_eta", "mu"),
    "B0_Dp1DstEta_e":  MCAmbulance("b0", "d1p_dstar_eta", "e"),
    "B0_Dp1DstEta_mu": MCAmbulance("b0", "d1p_dstar_eta", "mu"),
    "Bp_D0stDeta_e":   MCAmbulance("bp", "d0_d_eta",      "e"),
    "Bp_D0stDeta_mu":  MCAmbulance("bp", "d0_d_eta",      "mu"),
    "B0_D0stDeta_e":   MCAmbulance("b0", "d0_d_eta",      "e"),
    "B0_D0stDeta_mu":  MCAmbulance("b0", "d0_d_eta",      "mu"),
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
    # NOTE: the six single-Xc entries below (Denu/Dstenu/D1enu/D0stenu/Dp1enu/D2stenu) and
    # Dst0pienu/Dstpi0enu were fixed 2026-08-31: the original entries used the NEUTRAL D**
    # PDG code (e.g. 421=D0, 423=D*0, 10421=D_0*0) where the actual B0/*.dec decfiles decay to
    # the CHARGED partner (411=D+, 413=D*+, 10411=D_0*+, etc, per `anti-B0sig -> D_0*+ e- ...`)
    # -- a latent bug invisible until B0 modes were actually generated (they were always
    # commented out in config.yaml before the full-isospin generalization). Verified against
    # every B0 decfile's actual daughter list; Dst0pienu/Dstpi0enu instead had the charge/
    # neutral pairing backwards relative to each other. Bplus entries were cross-checked and
    # found already correct (0/20 mismatches) -- this bug was specific to the B0 block.
    (511, frozenset([11, 12, 411])):         "B0_Denu",
    (511, frozenset([11, 12, 413])):         "B0_Dstenu",
    (511, frozenset([11, 12, 10411])):       "B0_D0stenu",
    (511, frozenset([11, 12, 10413])):       "B0_D1enu",
    (511, frozenset([11, 12, 20413])):       "B0_Dp1enu",
    (511, frozenset([11, 12, 415])):         "B0_D2stenu",
    (511, frozenset([11, 12, 421, 211])):    "B0_D0pienu",
    (511, frozenset([11, 12, 413, 111])):    "B0_Dstpi0enu",
    (511, frozenset([11, 12, 411, 111])):    "B0_Dpi0enu",
    (511, frozenset([11, 12, 423, 211])):    "B0_Dst0pienu",
    (511, frozenset([11, 12, 411, 211, 0])): "B0_Dpipipenu",
    (511, frozenset([11, 12, 421, 211, 111])):"B0_D0pipipzenu",
    (511, frozenset([11, 12, 411, 111, 0])): "B0_Dpizpizenu",
    (511, frozenset([11, 12, 413, 211, 0])): "B0_Dstpipipenu",
    (511, frozenset([11, 12, 423, 211, 111])):"B0_Dst0pipipzenu",
    (511, frozenset([11, 12, 413, 111, 0])): "B0_Dstpizpizenu",
    (511, frozenset([11, 12, 411, 221])):    "B0_Detaenu",
    (511, frozenset([11, 12, 413, 221])):    "B0_Dstetaenu",
    # Muon mirrors (full lepton-flavor generalization; PDG 11/12 -> 13/14, "enu" -> "munu")
    (521, frozenset([13, 14, 421])): "Bp_Dmunu",
    (521, frozenset([13, 14, 423])): "Bp_Dstmunu",
    (521, frozenset([13, 14, 10421])): "Bp_D0stmunu",
    (521, frozenset([13, 14, 10423])): "Bp_D1munu",
    (521, frozenset([13, 14, 20423])): "Bp_Dp1munu",
    (521, frozenset([13, 14, 425])): "Bp_D2stmunu",
    (521, frozenset([13, 14, 211, 411])): "Bp_Dmpimunu",
    (521, frozenset([13, 14, 211, 413])): "Bp_Dstmpimunu",
    (521, frozenset([13, 14, 111, 421])): "Bp_D0pi0munu",
    (521, frozenset([13, 14, 111, 423])): "Bp_Dst0pi0munu",
    (521, frozenset([0, 13, 14, 211, 421])): "Bp_D0pipipmunu",
    (521, frozenset([13, 14, 111, 211, 411])): "Bp_Dmpipizmunu",
    (521, frozenset([0, 13, 14, 111, 421])): "Bp_D0pizpizmunu",
    (521, frozenset([0, 13, 14, 211, 423])): "Bp_Dst0pipipmunu",
    (521, frozenset([13, 14, 111, 211, 413])): "Bp_Dstmpipizmunu",
    (521, frozenset([0, 13, 14, 111, 423])): "Bp_Dst0pizpizmunu",
    (521, frozenset([13, 14, 321, 431])): "Bp_DsKmunu",
    (521, frozenset([13, 14, 321, 433])): "Bp_DsstkKmunu",
    (521, frozenset([13, 14, 221, 421])): "Bp_D0etamunu",
    (521, frozenset([13, 14, 221, 423])): "Bp_Dst0etamunu",
    (511, frozenset([13, 14, 411])): "B0_Dmunu",
    (511, frozenset([13, 14, 413])): "B0_Dstmunu",
    (511, frozenset([13, 14, 10411])): "B0_D0stmunu",
    (511, frozenset([13, 14, 10413])): "B0_D1munu",
    (511, frozenset([13, 14, 20413])): "B0_Dp1munu",
    (511, frozenset([13, 14, 415])): "B0_D2stmunu",
    (511, frozenset([13, 14, 211, 421])): "B0_D0pimunu",
    (511, frozenset([13, 14, 111, 413])): "B0_Dstpi0munu",
    (511, frozenset([13, 14, 111, 411])): "B0_Dpi0munu",
    (511, frozenset([13, 14, 211, 423])): "B0_Dst0pimunu",
    (511, frozenset([0, 13, 14, 211, 411])): "B0_Dpipipmunu",
    (511, frozenset([13, 14, 111, 211, 421])): "B0_D0pipipzmunu",
    (511, frozenset([0, 13, 14, 111, 411])): "B0_Dpizpizmunu",
    (511, frozenset([0, 13, 14, 211, 413])): "B0_Dstpipipmunu",
    (511, frozenset([13, 14, 111, 211, 423])): "B0_Dst0pipipzmunu",
    (511, frozenset([0, 13, 14, 111, 413])): "B0_Dstpizpizmunu",
    (511, frozenset([13, 14, 221, 411])): "B0_Detamunu",
    (511, frozenset([13, 14, 221, 413])): "B0_Dstetamunu",
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
    if n.endswith("munu"):
        n = n[:-4] + "enu"  # _CATS is keyed by the canonical electron suffix only (LFU: same category)
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
# Muon rows (LFU: same branching fraction as the electron channel, confirmed with the analysis
# author) -- derived from the electron table above rather than duplicated by hand, one row per
# muon decay_name that PDG_NAME actually produces.
for _name in PDG_NAME.values():
    if _name.endswith("munu"):
        _e_key = _name[:-4] + "enu"
        if _e_key in _BF_TABLE:
            _BF_TABLE[_name] = _BF_TABLE[_e_key]

BF     = {k: v[0] for k, v in _BF_TABLE.items()}
BF_UNC = {k: v[1] for k, v in _BF_TABLE.items()}

_HAMMER_DECAYS = {
    "Bp_Denu", "Bp_Dstenu", "Bp_D1enu", "Bp_D0stenu", "Bp_Dp1enu", "Bp_D2stenu",
    "B0_Denu", "B0_Dstenu", "B0_D1enu", "B0_D0stenu", "B0_Dp1enu", "B0_D2stenu",
    "Bp_Dmunu", "Bp_Dstmunu", "Bp_D1munu", "Bp_D0stmunu", "Bp_Dp1munu", "Bp_D2stmunu",
    "B0_Dmunu", "B0_Dstmunu", "B0_D1munu", "B0_D0stmunu", "B0_Dp1munu", "B0_D2stmunu",
}


# ── DsK spectrum reweight helper ─────────────────────────────────────────────

_DSK_DECAY_NAMES = {"Bp_DsKenu", "Bp_DsstkKenu", "Bp_DsKmunu", "Bp_DsstkKmunu"}

def _load_digitized_curve(csv_path, x_cols=("x",), y_cols=("y_fit",)):
    """Load a digitized (x, y_fit) curve and return a smoothed, sorted (x, y) pair, or None if
    the CSV is missing the expected columns."""
    pts = pd.read_csv(csv_path)
    x_col = next((c for c in x_cols if c in pts.columns), None)
    y_col = next((c for c in y_cols if c in pts.columns), None)
    if x_col is None or y_col is None:
        return None
    x_d = pts[x_col].to_numpy(float)
    y_d = np.clip(pts[y_col].to_numpy(float), 0, None)
    y_d = np.where(np.isfinite(y_d), y_d, 0.0)
    y_d = np.clip(y_d, 1e-8, None)
    ok = np.isfinite(x_d) & np.isfinite(y_d)
    x_d, y_d = x_d[ok], y_d[ok]
    order = np.argsort(x_d); x_d, y_d = x_d[order], y_d[order]
    win  = min(5, max(3, len(y_d)))
    y_sm = np.convolve(y_d, np.ones(win) / win, mode="same")
    return x_d, np.clip(y_sm, 1e-8, None)


def _spectrum_reweight_ratio(m, w0, x_d, y_sm):
    """Per-event reweight factor that reshapes the weighted MC mass histogram `m`/`w0` onto the
    digitized target density (x_d, y_sm), normalized to preserve the total weighted yield."""
    m_thr   = float(np.nanmin(m))
    bins    = np.linspace(m_thr, mass_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mc_h, _ = np.histogram(m, bins=bins, weights=w0)
    mc_pdf  = mc_h / max(float(mc_h.sum()), 1e-12)
    tgt_y   = np.interp(centers, x_d, y_sm, left=np.nan, right=np.nan)
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
    return rw_arr


def _apply_dsk_spectrum_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Apply digitized y_fit spectrum reweight to Bp_Ds(st)K{e,mu}nu rows (LFU: the digitized
    hadronic-side mass spectrum is lepton-flavor-independent, so the same correction applies to
    both)."""
    # Ensure the column always exists (even with no digitized CSV / no matching rows) so a
    # concat of per-chunk frames doesn't produce spurious NaN for chunks that never touch this
    # function's mask.
    if "spectrum_weight" not in df.columns:
        df = df.copy(); df["spectrum_weight"] = 1.0

    csv_path = out2 / "dsk_digitized.csv"
    if not csv_path.exists():
        return df
    curve = _load_digitized_curve(csv_path)
    if curve is None:
        return df
    x_d, y_sm = curve

    mask = df["decay_name"].isin(_DSK_DECAY_NAMES).to_numpy()
    if not mask.any():
        return df

    sub  = df.loc[mask]
    e    = (sub["d0_mcE"]  + sub["d1_mcE"]).to_numpy(float)
    px   = (sub["d0_mcPX"] + sub["d1_mcPX"]).to_numpy(float)
    py   = (sub["d0_mcPY"] + sub["d1_mcPY"]).to_numpy(float)
    pz   = (sub["d0_mcPZ"] + sub["d1_mcPZ"]).to_numpy(float)
    m    = np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))
    w0   = sub["weight"].to_numpy(float)

    rw_arr = _spectrum_reweight_ratio(m, w0, x_d, y_sm)
    df.loc[mask, "spectrum_weight"] = rw_arr
    df.loc[mask, "total_weight"]    = (
        sub["weight"] * sub["mc_weight"] * sub["ff_weight"] * rw_arr
    ).to_numpy()
    print(f"  [DsK] spectrum_weight [{rw_arr.min():.4f}, {rw_arr.max():.4f}] applied to {mask.sum():,} rows")
    return df


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
        is_l  = ((pdg == 11) | (pdg == 13)) & l_E.isna()
        is_nu = ((pdg == 12) | (pdg == 14)) & nu_E.isna()
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


# Upsilon(4S) -> B+B- / B0-B0bar production fractions (HFLAV). Each decay_name's BF/n_reco
# weight is already an absolute per-B-meson branching fraction, normalized within its own
# species; to combine Bp_*/B0_* rows into one Upsilon(4S)-admixture prediction -- matching the
# charge-untagged experimental moments this cocktail is compared against, see the isospin-
# generalization plan/memory -- each row must additionally be scaled by its species' physical
# production fraction, not just concatenated 1:1.
F_PLUS = 0.514
F_ZERO = 0.486


def _apply_isospin_weight(df, decay_name_col="decay_name", weight_col="total_weight"):
    """Scale total_weight by the Upsilon(4S) production fraction (F_PLUS for Bp_*, F_ZERO for
    B0_*) implied by the decay_name prefix. Rows whose decay_name doesn't start with Bp_/B0_
    (e.g. unknown_*) are left unscaled."""
    is_bp = df[decay_name_col].astype(str).str.startswith("Bp_")
    is_b0 = df[decay_name_col].astype(str).str.startswith("B0_")
    factor = np.ones(len(df))
    factor[is_bp.to_numpy()] = F_PLUS
    factor[is_b0.to_numpy()] = F_ZERO
    df[weight_col] = df[weight_col].to_numpy() * factor
    return df


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
    merged = _apply_dsk_spectrum_weight(merged)
    merged = _apply_isospin_weight(merged)
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
    out_df = _apply_dsk_spectrum_weight(out_df)

    out_df = pd.DataFrame({c: np.asarray(out_df[c]) for c in out_df.columns})
    if args.cocktail_job_index is None:
        # Single-shot (non-chunked) run: this is the final full sample, so apply the
        # isospin (f+/f00) weighting now -- same "full sample only" rule as the merge path
        # above, since a chunk-local subset can't see the global Bp_/B0_ mix.
        out_df = _apply_isospin_weight(out_df)
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

def _gap_info(name_e, category):
    """One GAP_MODES_PROC entry, per charge. decay_name carries charge only (not lepton) --
    both e and mu rows share one entry and are distinguished at the row level via _lepton
    (from extraInfo(decayModeID)) instead, since gap categories stay lepton-inclusive."""
    return {
        "Bplus": {"decay_name": f"Bp_{name_e}", "category": category},
        "B0":    {"decay_name": f"B0_{name_e}", "category": category},
    }


GAP_MODES_PROC = {
    "D2stDeta":       _gap_info("D2stDeta",       r"gap: $D_2^*{\to}D\eta$"),
    "DummyDeta":      _gap_info("DummyDeta",      r"gap: $D\eta$ (non-res.)"),
    "Dp1DstEta":      _gap_info("Dp1DstEta",      r"gap: $D_1'{\to}D^*\eta$"),
    "Dp1Deta":        _gap_info("Dp1Deta",        r"gap: $D_1'{\to}D\eta$"),
    "DsKenu":         _gap_info("DsKenu_gap",     r"gap: $D_s K$ (dummy-res.)"),
    "LcPenu":         _gap_info("LcPenu",         r"gap: $\Lambda_c\bar{p}$ (dummy-res.)"),
    "D0stDeta":       _gap_info("D0stDeta",       r"gap: $D_0^*{\to}D\eta$"),
    "D1_2550Deta":    _gap_info("D1_2550Deta",    r"gap: $D_1(2550){\to}D\eta$"),
    "D1Dgamma":       _gap_info("D1Dgamma",       r"gap: $D_1{\to}D\gamma$"),
    "D0pipipipenu":   _gap_info("D0pipipipenu",   r"gap: $D_1{\to}D\pi\pi\pi$"),
    "Dst0pipipipenu": _gap_info("Dst0pipipipenu", r"gap: $D_2^*{\to}D^*\pi\pi\pi$"),
}

_DIGITIZED_MODES = {
    "DsKenu": ("dsk",       "d1d0", "d1d1"),
    "LcPenu": ("lambdac_p", "d1d0", "d1d1"),
}

bf_gap   = rw["bf_gap"]

active_gap = cfg["generation"]["gap_modes"]
if args.mode:
    active_gap = [m for m in active_gap if m == args.mode]
    if not active_gap:
        print(f"error: mode {args.mode!r} not found in generation.gap_modes")
        sys.exit(1)

if not args.skip_gap:
    for mode in active_gap:
        mode_info = GAP_MODES_PROC.get(mode)
        if mode_info is None:
            print(f"  [warn] unknown gap mode {mode!r} — skipping")
            continue

        # One (charge, mode) pair per findMCDecay-truth-matched sample; both lepton flavors
        # come out of the SAME sample (see _generate_and_reco.py's two-pass truth matching),
        # distinguished at the row level via extraInfo(decayModeID) rather than a separate
        # config/path axis.
        charge_frames = []
        for charge in ("Bplus", "B0"):
            info   = mode_info[charge]
            in_dir = out1 / "gap_modes" / charge / mode
            rf     = sorted(glob.glob(str(in_dir / "**" / "*.root"), recursive=True))
            if not rf:
                print(f"  [warn] no ROOT files for gap mode {charge}/{mode} under {in_dir}")
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

            cdf = pd.concat(gframes, ignore_index=True)
            cdf["decay_name"] = info["decay_name"]
            cdf["category"]   = info["category"]
            dmid_col = "extraInfo__bodecayModeID__bc"
            if dmid_col in cdf.columns:
                cdf["_lepton"] = cdf[dmid_col].map({0: "e", 1: "mu"}).fillna("e")
            else:
                cdf["_lepton"] = "e"
            charge_frames.append(cdf)

        if not charge_frames:
            continue

        gdf = pd.concat(charge_frames, ignore_index=True)
        n   = len(gdf)
        gdf["bf"]              = bf_gap
        gdf["bf_unc"]          = 0.0
        gdf["weight"]          = bf_gap / n if n > 0 else 0.0
        gdf["mc_weight"]       = 1.0
        gdf["ff_weight"]       = 1.0
        gdf["total_weight"]    = gdf["weight"]
        gdf["spectrum_weight"] = 1.0

        # MCAmbulance correction, applied per (decay_name, lepton) subset -- the correction
        # genuinely differs between e and mu (and bp/b0), unlike a single shared decay_name.
        mca_cols = ["d1d0_mcE", "d1d0_mcPX", "d1d0_mcPY", "d1d0_mcPZ",
                    "d1d1_mcE", "d1d1_mcPX", "d1d1_mcPY", "d1d1_mcPZ"]
        if all(c in gdf.columns for c in mca_cols):
            any_corrected = False
            for dname in gdf["decay_name"].unique():
                for lep in ("e", "mu"):
                    mca_key = f"{dname}_{lep}"
                    if mca_key not in _MCA:
                        continue
                    m = ((gdf["decay_name"] == dname) & (gdf["_lepton"] == lep)).to_numpy()
                    if not m.any():
                        continue
                    e  = (gdf.loc[m, "d1d0_mcE"]  + gdf.loc[m, "d1d1_mcE"]).to_numpy(float)
                    px = (gdf.loc[m, "d1d0_mcPX"] + gdf.loc[m, "d1d1_mcPX"]).to_numpy(float)
                    py = (gdf.loc[m, "d1d0_mcPY"] + gdf.loc[m, "d1d1_mcPY"]).to_numpy(float)
                    pz = (gdf.loc[m, "d1d0_mcPZ"] + gdf.loc[m, "d1d1_mcPZ"]).to_numpy(float)
                    m_res = np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))
                    gdf.loc[m, "mc_weight"] = _MCA[mca_key].CorrectionWeight(m_res)
                    any_corrected = True
            if any_corrected:
                gdf["total_weight"] = gdf["weight"] * gdf["mc_weight"]
                print(f"  [{mode}] MCAmbulance mc_weight [{gdf['mc_weight'].min():.4f}, {gdf['mc_weight'].max():.4f}]")
        else:
            missing = [c for c in mca_cols if c not in gdf.columns]
            print(f"  [warn] MCAmbulance skipped for {mode}: missing columns {missing}")

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
                curve = _load_digitized_curve(
                    csv_path, x_cols=("x", "x_norm"), y_cols=("y_fit", "y", "y_norm")
                )
                if curve is None:
                    print(f"  [warn] digitized CSV {csv_path} missing expected x/y columns — skipping shape reweight")
                else:
                    x_d, y_sm = curve
                    e1  = gdf[f"{d1}_mcE"].to_numpy(float);  px1 = gdf[f"{d1}_mcPX"].to_numpy(float)
                    py1 = gdf[f"{d1}_mcPY"].to_numpy(float); pz1 = gdf[f"{d1}_mcPZ"].to_numpy(float)
                    e2  = gdf[f"{d2}_mcE"].to_numpy(float);  px2 = gdf[f"{d2}_mcPX"].to_numpy(float)
                    py2 = gdf[f"{d2}_mcPY"].to_numpy(float); pz2 = gdf[f"{d2}_mcPZ"].to_numpy(float)
                    m2  = (e1+e2)**2 - ((px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2)
                    m   = np.sqrt(np.clip(m2, 0, None))

                    m_thr  = float(np.nanmin(m))
                    w0     = gdf["total_weight"].to_numpy(float)
                    rw_arr = _spectrum_reweight_ratio(m, w0, x_d, y_sm)
                    gdf["spectrum_weight"] = rw_arr
                    gdf["total_weight"]    = gdf["total_weight"] * rw_arr
                    print(f"  [{mode}] applied {key} shape reweight")

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

        # Isospin (f+/f00) weighting, applied last -- same rule as Section A: combine the
        # Bp_/B0_ rows into one Upsilon(4S)-admixture prediction rather than a 1:1 sum.
        gdf = _apply_isospin_weight(gdf)
        gdf = gdf.drop(columns=["_lepton"])

        out_path = out3 / f"{mode}.parquet"
        gdf = pd.DataFrame({c: np.asarray(gdf[c]) for c in gdf.columns})
        gdf.to_parquet(out_path, index=False)
        print(f"  saved {n:,} events → {out_path}")
else:
    print("  [skip] gap mode processing disabled")

print("Step 3 complete.")
