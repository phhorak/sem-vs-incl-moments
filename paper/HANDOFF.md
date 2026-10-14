# Handoff (state of 2026-10-15)

Read first: `paper/CLAIMS.md` (plan of record: decisions D1–D8, claims C1–C13, open inputs I1–I7) and the
narrative for collaborators `paper/narrative.pdf` (build: `cd paper && latexmk -pdf narrative.tex`). Every number in
it comes from `paper/numbers.tex`, every figure from `figures/7`–`11`.

## Where things stand
- Pipeline rerun on 2026-10-14 with the Belle II analysis cocktail (D(*)ππ via D1'/D0*, LLSW→BLR, MCAmbulance),
  BLPRXP for D/D*, and fixed Hammer inputs. Steps 1–4, 6–11 current; step 5 is not (see below, not needed for the paper).
- 2026-10-15: test code in `lib/cone.py`. Step 9 adds the 95 % CL gap-size limit; step 10 drops the component/pair
  scans and adds single-mode shape tests (C7), support variants and σ(B_incl) × 2 variants; new `11_paper.py` writes
  the paper figures (HQE moments, closure N = 3, moment planes) and `paper/numbers.tex`. `claims_summary.tex` and the
  migrated prototypes in `scratch/interpretation/` are removed (in git history).
- Results (narrative): (1) the gap is not more of the known modes: neither a common BF factor (joint d² 171) nor any
  single SEM mode (196–246; 153–171 with free size); (2) Λc p̄ alone excluded, gap ≥ 6.5 % of inclusive (95 % CL);
  (3) M_X² rejects PHSP D(*)ππ (p 0.002) and bare EvtGen broad lineshapes (p 0.009); the cocktail passes.
- Not pushed: commits since `ccf53c8` are local only; push only with the user's OK.

## Open work (priority order; details in CLAIMS.md)
1. Send `paper/narrative.pdf` to Markus and Danny for sign-off (I1) and inputs I4–I6. Paper 1 vs 2 scope of the
   single-mode shape test: the ledger puts C7 in Paper 1; the positive-mix fit was dropped (closer to Paper 2's likelihood).
2. Restructure the pipeline (M3b): one shared HQE/SEM toy pairing (steps 9, 10, 11 use seeds 9, 10, 11), inputs vs
   results by function, retire steps 4, 5 and the threshold paths.
3. C5 caveats: MC calibration of the measured M_X moments (Markus's suspect); LHCb broad states *without*
   MCAmbulance (likely what LHCb MC does) — one more variant in `10_cone_test.py` (`lhcb` moments are built with the
   ISGW2 MCAmbulance correction; add an uncorrected set in `build_moments`).
4. C9 coverage and power study, incl. the reverse test (reject "light states only" for an injected heavy gap).
5. Step 5 runs out of memory even with 12 slots (48 GB): it builds per-event bands over the 40M-row cocktail;
   needs streaming or subsampling. Only its first plots (FF reweight, MCAmbulance, bands) are current.
6. Update the clone bundle `data/gap_toys_bundle.npz` (still the old one) if clones should reproduce today's results.
7. Paper skeleton (M4) once I1 and I7 are settled.

## How to run (KEKCC)
- Environments: generation (step 1) **must** use `b2setup release-09-00-11` (the cocktail's release; Γ(D1') = 314 MeV
  as MCAmbulance assumes; 08-02-05 segfaults on B0 with VSS_BMIX). Everything else: `b2setup light-2509-fornax`.
  The setup sometimes leaves `/usr/bin/python3` on PATH: check `command -v python3` is under `/cvmfs/...externals`
  and retry (`scratch/nr_test/b2run.sh CMD` does that). Don't use `python3 -I` (drops the user site with pyarrow).
- Batch: queue `h` is far less loaded than `s` (memory 4 GB/slot, max 12 slots; ask for few slots, multi-slot jobs
  wait for a host with that many free cores). Steps 6–10 have `--submit`; step 3 `--submit-cocktail`, then
  `--merge-cocktail` (streamed, fits one slot).
- Typical chain after a cocktail change: step 1 (affected modes) → step 3 chunks + merge → step 6 → steps 7, 8 →
  `10_cone_test.py --build-moments --phsp-cocktail output/_retired_2026-10-14/3/cocktail_bgl_cln-input_phsp.parquet`
  → steps 9, 10 → step 11 (needs 7–10) → `cd paper && latexmk -pdf narrative.tex`.

## Traps hit today
- `/ghi` is tape-backed (HSM). `ghils FILE` shows H (tape only), B (disk+tape), G (disk only). Reading an H file or
  **overwriting a B file** blocks the job forever (frozen CPU, low memory). Before rerunning a step, move its old
  outputs in state B/H aside (`mv` is safe) — e.g. `output/_retired_2026-10-14/`. Recalls took > 2.5 h without
  returning; regenerating was faster.
- Jobs stuck that way can't be killed normally: `bkill -r`. They then linger as ZOMBI and match job-name wildcards,
  so use job-ID dependencies (`-w "done(ID)"`), not `ended(name*)`.
- LSF writes `-oo` logs only at job end and sometimes late: a log can be from the previous run. Check output file
  timestamps instead.
- Don't run a step locally while its batch job writes the same `output/N` (a local test once deleted a batch result).
- Hammer wrapper: the input scheme dict is keyed by hadronic transition only, so τ inputs overwrote light-lepton ones
  (fixed: light leptons only). EvtGen's B→D BGL has Blaschke factors = 1; `BGL.BtoDEllNu_evtgen()` reproduces it in
  Hammer's BGL. D1' Hammer weights reach 1e5 far off-shell; only MCAmbulance (≈ 0 there) suppresses them.
- The toys are non-Gaussian (c_true has a heavy tail): χ² p-values for the cone test are not conservative; use the
  bootstrap p.
- Steps 7–11 read the config stored in the step-6 bundle (`B.cfg`), not `config.yaml`. A key added to `config.yaml`
  after step 6 (e.g. `systematics.bf_incl_unc`) must be read with `load_cfg()`.

## Data layout
- `output/` → `/ghi/.../clean/output` (not in git). Retired inputs/outputs of 2026-10-14 in `output/_retired_2026-10-14/`
  (old cocktail with PHSP NR and buggy FF weights, old gap-mode samples, old step-6/7 outputs).
- `data/sem_component_moments.json`: component moments for step 10 (rebuild with `--build-moments`).
- LHCb-style samples: `output/1/lhcb/{Bplus,B0}/<state>/0000.root`, decfiles in `decfiles/lhcb/`.
- Outputs replaced on 2026-10-15 (steps 9, 10) are in `output/_retired_2026-10-15/`.
