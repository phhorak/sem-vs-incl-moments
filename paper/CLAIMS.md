# Paper 1 claims ledger

Paper 1: model-independent constraints on the semileptonic gap (Markus's working note, Sep 2026).
Every statement in the paper maps to a row here; every row maps to a pipeline output. Numbers reach
the paper only through `paper/numbers.tex` (written by step 8); figures from `figures/5`–`8`. The
current write-up for collaborators is `paper/narrative.tex` (Summary, Method, Results 1–3). Prose is
written only for rows marked **done**. Frozen starting point: tag `slac-seminar-2026-10`.

Status: **done** (pipeline step, reproducible) · **todo** · **input** (blocked on input, see bottom).

## Decisions

| ID | Decision | State |
|----|----------|-------|
| D1 | Primary inclusive input: zero-threshold HQE moments m1–m3 + likelihood toys (Fael, Milutin, Prim, Vos, 2609.28752) | adopted |
| D2 | Constant c_true above thresholds dropped. Thresholds only via partial BFs per cut (cross-check, C9) | adopted (joint fit off by default, `config.yaml data.joint_fit`) |
| D3 | Polynomial average (old step 4), per-threshold table, joint fit → removed from the pipeline (2026-10-15) | adopted |
| D4 | MaxEnt densities are illustration; quantitative statements are tests and bounds | adopted (Markus) |
| D5 | Every statement is conditional on the SEM normalisations; c_true ≈ 6 stated up front | adopted (Markus) |
| D6 | Nominal NR D(*)ππ: Belle II analysis cocktail, B → D1'/D0*(→ D(*)ππ PHSP) ℓν, LLSW → BLR, MCAmbulance; 5-body PHSP kept as alternative | adopted 2026-10-14 (split D1':D0* = 50:50, `reweight.nr_parent_split`, I2) |
| D7 | Nominal D/D* FF: BLPRXP (one fit, D and D* eigenvariations shared) | adopted 2026-10-14 |
| D8 | Explicit candidate χ² ("no single mode fits") → Paper 2; Paper 1 keeps only kinematic-class support tests | proposed |

## Claims

Narrative section in brackets.

| ID | Claim | Produced by | Status |
|----|-------|-------------|--------|
| C1 | SEM from HFLAV: B_SEM = 9.26 / 8.55 %, gap 16.5 ± 2.5 % of inclusive, c_true = 6.07; shares D 24.6, D* 57.2, D** 14.8, D(*)ππ 3.1, D_s(*)K 0.3 % [1.2] | step 4 bundle, step 3 `data/sem_component_moments.json` | done |
| C2 | Inclusive input: zero-threshold HQE moments m1–m3 with 12 000 likelihood toys; measured moments vs threshold [1.1, Fig. 1] | step 8 `hqe_moments.pdf` | done |
| C3 | Inclusive inversion without SEM subtraction vs the SEM stack; allowed in all toys, no c_true needed [2.1] | step 6 | done |
| C4 | Closure at N = 3 for D1'→D*η and 50 % Dℓν + 50 % Λc p̄ℓν; L1 0.02–0.08 in E_ℓ, q²; M_X² outside the allowed region in 21–66 % of toys; leading uncertainty B_gap [1.6, Fig. 2] | step 5, step 8 `closure_n3.pdf` | done |
| C5 | M_X² gap moments allowed with the Belle II cocktail; PHSP D(*)ππ rejected (d² 20.4, p 0.003), bare EvtGen broad states (10.1, 0.013); robust against no Hammer for broad states, HQE M_X² unc. × 2, D1':D0* split, supports, LHCb-style cocktail. σ(B_incl) × 2 weakens to p 0.020 / 0.038. Open caveat: MC calibration of the measured M_X moments [2.5] | step 7 `variants`, `supports` | done |
| C6 | Admissibility test: d² to the moment space (exact two-atom boundary), bootstrap p [1.5] | `lib/cone.py` | done |
| C7 | Gap with the shape of one SEM mode (D, D*, D**, D(*)ππ, D_s(*)K) excluded: joint d² 194–250 at the nominal gap size (toys vary B_gap), 150–168 with the size free; D** fails only in q² (d² 32). Replaces the component and pair scans; the positive-mix fit is not in Paper 1 (D8) [2.3, Tab. 2] | step 7 `mode_shape_test` | done |
| C8 | CDF bounds (LP): P(M_X² ≤ 10) ≥ 0.93; P(E_ℓ ≤ 0.5) ≤ 0.16, ≥ 60 % above 0.8 GeV; P(q² ≤ 4) ∈ [0.19, 0.79] [2.4, Fig. 7] | step 7 `cdf` | done |
| C9 | Support tests: Λc p̄ alone excluded by M_X² (≥ 95 % below threshold) and q² (≥ 17 % above endpoint); D_sK alone only disfavoured (8 % of toys) [2.4] | step 7 `support` | done (coverage, power study incl. the reverse test todo) |
| C10 | Minimum gap size: B_gap/B_incl > 0.065 at 95 % CL (M_X²; E_ℓ 0.035); means and variances alone give ≥ 0.11; admissible fraction of toys 79 % at nominal, 50 % at 0.13 [2.4, Fig. 8] | step 7 `gap_fraction_test`, `gap_fraction_scan` | done |
| C11 | Gap mean/variance vs SEM, inclusive and SEM modes; same plane with candidate modes for reference; gap density peaks near M_X ≈ 2.4 GeV [2.2, 2.3, Figs. 4–6] | step 6 `gap_density.pdf`, step 8 `moment_plane_*.pdf` | done |
| C12 | Robustness: LHCb-style cocktail (D** fractions 11874430/12873450, ISGW2 D**, higher D** instead of D(*)ππ, no D_sK) and alternative supports leave C5, C6, C13 unchanged; a too-narrow q² ≤ 6 support is rejected (d² 28.7). Open: LHCb absolute normalisation (I6), partial-BF cross-check (I5) | step 7 | done (I5, I6 open) |
| C13 | A common BF shortfall is excluded: inclusive vs SEM d² 41.9 (M_X²), 13.9 (E_ℓ), 5.7 (q², p 0.12), joint 168, p < 0.001; for every SEM variant [2.3, Tab. 2] | step 7 `shape_test` | done |

## Pipeline fixes (2026-10-14)

- Hammer input scheme: the τ CLN inputs overwrote the light-lepton ones (dict keyed by "BD"/"BD*"),
  so every D/D* FF weight so far was target / CLN(τ decfile parameters). Effect at fixed BGL target:
  D ≤ 2 % per event; D* 0.54–1.83 per event, q² shape ±5 %, ⟨q²⟩ +0.08 GeV².
- B → D input: EvtGen BGL has Blaschke = 1 and the `BGLB2` Hammer class does not exist in the
  externals; replaced by Hammer-BGL coefficients reproducing EvtGen to 3e-5 (`BGL.BtoDEllNu_evtgen`).
- B → D* input: decfile coefficients (`BGL.BtoDstEllNu_dec`), not Table V of 2008.09341.
- FF slope matrix read only 8 eigenvariation slots (BLPRXP has 9).
- B0 NR decfiles without mixing (findMCDecay segfault with VSS_BMIX on 3-body D** decays).

## Retired (appendix at most)

Polynomial average of the measurements (old step 4, draft Sec. 5.1), per-threshold admissibility
table, joint χ² fit and per-E_ℓ-cut M_X fits, threshold Asimov closure with constant c_true; SEM-vs-data
diagnostics (old step 5). Removed from the pipeline on 2026-10-15 (in git history).
Review rounds 1–2 (claude.ai artifacts) refer to the old outline and are not carried over.

## Milestones

- **M1** ~~C6~~ (done); ~~C2 → pipeline~~ (step 8).
- **M2** ~~D6, C5 comparison, C7~~ (done; C7 is now the single-mode shape test, step 7).
- **M3** ~~C8, C10, C9 nominal~~ (done, step 7); C9 coverage and power study from the gap-mode samples.
- **M3b** ~~Restructure the pipeline by function~~ (2026-10-15): one shared HQE/SEM toy pairing, steps 1–8 as in
  HANDOFF.md; experimental average, SEM-vs-data diagnostics and the threshold paths retired; digitizer moved to inputs/.
- **M4** ~~C12~~; new draft skeleton, reusing SEM construction, endpoint asymptotics,
  THMP appendix and introduction from `references/draft.tex`.

## Cocktail definitions for C12

Belle II (basf2 `decfiles/dec`, public): D(*)ℓν generic; D** BLR; NR D(*)ππℓν via
D1'/D0* (LLSW, e.g. 1196700001: B0 → D1'(→ Dππ)ℓν); D(*)ηℓν via D1'/D0*; D_s(*)Kℓν generic (PHSP).

LHCb (DecFiles Sim10, public): D*μν, Dμν CLN; D** cocktail ISGW2 with D1', D0*, D1, D2*
(11874430: 0.307 / 0.297 / 0.277 / 0.119; 12873450: 0.294 / 0.294 / 0.265 / 0.147, "HFAG PDG10");
"higher D**" D*ππ via D*(2S), D(2S), D(2750), D(3000) at 0.05 / 0.45 / 0.45 / 0.05
(11676012, 12675402). No NR D(*)ππ, no D_sK. Absolute BFs not in the DecFiles.

## Open inputs

| # | Needed | From | Unblocks |
|---|--------|------|----------|
| I1 | Sign-off on `paper/narrative.pdf`, this ledger, D3, D8 | Markus, Danny | everything |
| I2 | D6: BF split of each NR D(*)ππ final state over D1' and D0* (50:50 assumed; C5/C6 shown insensitive) | Markus / you | systematic only |
| I3 | ~~D7~~ settled (BLPRXP) | — | — |
| I4 | Confirm `inputs/hqe_likelihood_toys/likelihood_toys.h5` is the 2609.28752 release; is m4 (q²) available at zero threshold? | Markus | C2, C6, C8 |
| I5 | Partial BFs ΔB(E_ℓ > cut), ΔB(q² > cut) with correlations to the moments, from the HQE fit | Markus | C12 cross-check |
| I6 | LHCb cocktail normalisation: which BFs (HFLAV as ours, or LHCb fit values) and whether the "higher D**" cocktail counts as SEM or as gap | note author / Markus | C12 |
| I7 | Where the paper source lives (Overleaf git URL) and OK to generate figures, tables and numbers into it | you, co-authors | M4 |
