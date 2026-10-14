# Handoff (state of 2026-10-15)

Read first: `paper/CLAIMS.md` (plan of record: decisions D1–D8, claims C1–C13, open inputs I1–I7) and the
narrative for collaborators `paper/narrative.pdf` (build: `cd paper && latexmk -pdf narrative.tex`). Every number in
it comes from `paper/numbers.tex`, every figure from `figures/7`–`11`.

## Where things stand
- Pipeline restructured on 2026-10-15 into eight steps by function (table below) and rerun from step 3 (per-mode
  moments, reproduced exactly) through step 8. Steps 1–2 (generation, reweighting) unchanged since 2026-10-14.
- Every number in the narrative comes from `paper/numbers.tex` (step 8). The restructure removed the threshold draws
  from the toy random stream and pairs all result steps with one HQE/SEM toy pairing, so numbers moved at the level of
  toy statistics (38 of 105, e.g. PHSP p 0.002 → 0.003, bare EvtGen 0.009 → 0.013; a p near 0.01 from 2000 toys has
  ±0.0025); no conclusion changed.
- Results: (1) the gap is not more of the known modes: neither a common BF factor (joint d² 168) nor any single SEM
  mode (194–250; 150–168 with free size); (2) Λc p̄ alone excluded, gap ≥ 6.5 % of inclusive (95 % CL);
  (3) M_X² rejects PHSP D(*)ππ (p 0.003) and bare EvtGen broad lineshapes (p 0.013); the cocktail passes.
- Not pushed: commits since `ccf53c8` are local only; push only with the user's OK.

## Open work (priority order; details in CLAIMS.md)
1. Send `paper/narrative.pdf` to Markus and Danny for sign-off (I1) and inputs I4–I6. Paper 1 vs 2 scope of the
   single-mode shape test: the ledger puts C7 in Paper 1; the positive-mix fit was dropped (closer to Paper 2's likelihood).
2. C5 caveats: MC calibration of the measured M_X moments (Markus's suspect); LHCb broad states *without*
   MCAmbulance (likely what LHCb MC does) — one more variant: add an uncorrected `lhcb` set in `3_sem_modes.py` and a
   variant in `lib/variants.py`. p-values near 0.01 need more toys (or more bootstrap) to be quoted to two digits.
3. C9 coverage and power study, incl. the reverse test (reject "light states only" for an injected heavy gap).
4. Update the clone bundle `data/gap_toys_bundle.npz` (still from September) if clones should reproduce the results.
5. Paper skeleton (M4) once I1 and I7 are settled.

## How to run (KEKCC)
- Environments: generation (step 1) **must** use `b2setup release-09-00-11` (the cocktail's release; Γ(D1') = 314 MeV
  as MCAmbulance assumes; 08-02-05 segfaults on B0 with VSS_BMIX). Everything else: `b2setup light-2509-fornax`.
  The setup sometimes leaves `/usr/bin/python3` on PATH: check `command -v python3` is under `/cvmfs/...externals`
  and retry (`scratch/nr_test/b2run.sh CMD` does that). Don't use `python3 -I` (drops the user site with pyarrow).
- Pipeline (restructured 2026-10-15, by function):
  | Step | Script | Does |
  |------|--------|------|
  | 1 | `1_generate` | signal MC (EvtGen, basf2) |
  | 2 | `2_reweight` | cocktail and gap-mode reweighting (Hammer, MCAmbulance, BFs; D_sK, Λc p̄ spectra from `inputs/digitized/`) |
  | 3 | `3_sem_modes` | per-mode moments and lineshape variants → `data/sem_component_moments.json` |
  | 4 | `4_toys` | threshold-0 toys and the bundle `output/4/gap_toys_bundle.npz` |
  | 5 | `5_closure` | Asimov closure (MaxEnt bands per source) |
  | 6 | `6_data_inversion` | inclusive and gap MaxEnt spectra, SEM stack, uncertainty budget |
  | 7 | `7_tests` | Results 1–3: shape tests, CDF bounds and support tests, minimum gap size, lineshape variants |
  | 8 | `8_paper` | paper figures and `paper/numbers.tex` |

  `inputs/digitize.py` digitized the D_sK and Λc p̄ spectra once (May 2026); its curves are committed inputs.
  Steps 6–8 use one pairing of HQE-fit toys to the step-4 toys (`lib/gap.py`); SEM variants are in `lib/variants.py`,
  the admissibility test in `lib/cone.py`. Retired: the experimental average, the SEM-vs-data diagnostics, the
  threshold toys, joint fit and threshold closure (in git history before the restructure commit).
- Batch: every `--submit` uses `batch.queue` in `config.yaml`. `h` has 4 GB/slot and max 12 slots; `s` is larger
  but often has many pending jobs; which is faster changes during the day (`bqueues h s`). Multi-slot jobs wait for a
  host with that many free cores; `bswitch` and `bmod -n` change a pending job. Steps 4–8 have `--submit`; step 2 `--submit-cocktail`, then
  `--merge-cocktail` (streamed, fits one slot).
- Typical chain after a cocktail change: step 1 (affected modes) → step 2 chunks + merge →
  `3_sem_modes.py --phsp-cocktail output/_retired_2026-10-14/3/cocktail_bgl_cln-input_phsp.parquet` → step 4 →
  steps 5–7 → step 8 → `cd paper && latexmk -pdf narrative.tex`.

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
- Steps 5–8 read the config stored in the step-4 bundle (`B.cfg`), not `config.yaml`. A key added to `config.yaml`
  needs a rerun of step 4 (bundle) before the later steps see it.

## Data layout
- `output/` → `/ghi/.../clean/output` (not in git). Retired inputs/outputs of 2026-10-14 in `output/_retired_2026-10-14/`
  (old cocktail with PHSP NR and buggy FF weights, old gap-mode samples, old step-6/7 outputs).
- `data/sem_component_moments.json`: per-mode moments (step 3), read by `lib/variants.py`.
- LHCb-style samples: `output/1/lhcb/{Bplus,B0}/<state>/0000.root`, decfiles in `decfiles/lhcb/`.
- Outputs before the restructure are in `output/_retired_2026-10-15/` (`old_numbering/`: old step numbers). The
  cocktail and gap-mode parquets moved from `output/3` to `output/2`; the old `output/2` (digitizer) is retired, its
  curves are in `inputs/digitized/`.
