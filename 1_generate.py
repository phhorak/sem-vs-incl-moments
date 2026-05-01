#!/usr/bin/env python3
"""Step 1: Generate and reconstruct signal MC.

Without --submit: runs a single mode locally (requires basf2 environment).
With --submit:    submits all configured modes to bsub.

Output: output/1/{mode}/reco.root
"""
import argparse
import glob
import os
import subprocess
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config",   default="config.yaml")
parser.add_argument("--submit",   action="store_true", help="submit all modes to bsub")
parser.add_argument("--mode",     default=None, help="single mode to run/submit (overrides config list)")
parser.add_argument("--dry-run",  action="store_true", help="print commands without executing")
parser.add_argument("--clear",    action="store_true", help="delete existing .root files before submitting")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

here     = os.path.dirname(os.path.abspath(__file__))
gen_script = os.path.join(here, "_generate_and_reco.py")
out_root = os.path.join(cfg["paths"]["output"], "1")
dec_cocktail = cfg["paths"]["decfiles_cocktail"]
dec_gap      = cfg["paths"]["decfiles_gap_modes"]
g = cfg["generation"]

cocktail_modes = g["cocktail_modes"]
gap_modes_keys = [f"gap_modes/Bplus/{m}" for m in g["gap_modes"]]
all_modes = cocktail_modes + gap_modes_keys

if args.mode:
    all_modes = [m for m in all_modes if m == args.mode or os.path.basename(m) == args.mode]
    if not all_modes:
        print(f"error: mode {args.mode!r} not found in config")
        sys.exit(1)

def _dec_file(mode: str) -> str:
    if mode.startswith("gap_modes/"):
        bare = mode[len("gap_modes/Bplus/"):]
        return os.path.join(dec_gap, "Bplus", bare, "decay.dec")
    return os.path.join(dec_cocktail, mode, "decay.dec")

if not args.submit:
    # Local single-mode run
    if len(all_modes) != 1:
        print("error: specify --mode when running locally (no --submit)")
        sys.exit(1)
    mode = all_modes[0]
    dec  = _dec_file(mode)
    out  = os.path.join(out_root, mode.replace("gap_modes/Bplus/", "gap_modes/"), "reco.root")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = f"basf2 {gen_script} -- --dec_file {dec} --output {out} --mode {mode} --nevents {g['n_events_per_job']}"
    print(cmd)
    if not args.dry_run:
        subprocess.run(cmd, shell=True, check=True)
    sys.exit(0)

# Batch submission
queue   = g["queue"]
n_jobs  = g["n_jobs"]
log_dir = os.path.join(here, "logs", "1")

for mode in all_modes:
    dec = _dec_file(mode)
    if not os.path.exists(dec):
        print(f"warning: {dec} not found — skipping {mode}")
        continue
    mode_key = mode.replace("gap_modes/Bplus/", "gap_modes/")
    out_dir  = os.path.join(out_root, mode_key)
    mlog_dir = os.path.join(log_dir, mode_key)
    os.makedirs(mlog_dir, exist_ok=True)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        if args.clear:
            for f in glob.glob(os.path.join(out_dir, "*.root")):
                os.remove(f)

    print(f"\n{mode}: submitting {n_jobs} job(s)")
    for job_idx in range(n_jobs):
        out_file = os.path.join(out_dir, f"{job_idx:04d}.root")
        log_file = os.path.join(mlog_dir, f"{job_idx:04d}.log")
        job_cmd  = (f"basf2 {gen_script} -- "
                    f"--dec_file {dec} --output {out_file} "
                    f"--mode {mode} --nevents {g['n_events_per_job']}")
        bsub_cmd = f'bsub -q {queue} -oo {log_file} "{job_cmd}"'
        print(bsub_cmd)
        if not args.dry_run:
            subprocess.run(bsub_cmd, shell=True, check=True)

if args.dry_run:
    print("\n(dry run — no jobs submitted)")
else:
    print("\nall modes submitted")
