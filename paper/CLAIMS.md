# Paper 1 claims ledger

Paper 1: model-independent constraints on the semileptonic gap (Markus's working note, Sep 2026).
Every statement in the paper maps to a row here; every row maps to a pipeline output. Prose is
written only for rows marked **done**. Frozen starting point: tag `slac-seminar-2026-10`.

Status: **done** (pipeline step, reproducible) · **proto** (script in `scratch/interpretation/`,
numbers exist, not in the pipeline) · **todo** · **input** (blocked on input, see bottom).

## Decisions

| ID | Decision | State |
|----|----------|-------|
| D1 | Primary inclusive input: zero-threshold HQE moments m1–m3 + likelihood toys (Fael, Milutin, Prim, Vos, 2609.28752) | adopted |
| D2 | Constant c_true above thresholds dropped. Thresholds only via partial BFs per cut (cross-check, C9) | adopted (joint fit off by default, `config.yaml data.joint_fit`) |
| D3 | Polynomial average (step 4), per-threshold table, joint fit → appendix or removed | proposed |
| D4 | MaxEnt densities are illustration; quantitative statements are tests and bounds | adopted (Markus) |
| D5 | Every statement is conditional on the SEM normalisations; c_true ≈ 6 stated up front | adopted (Markus) |
| D6 | Nominal NR D(*)ππ: Belle II analysis cocktail, B → D1'/D0*(→ D(*)ππ PHSP) ℓν, LLSW → BLR, MCAmbulance; 5-body PHSP kept as alternative | adopted 2026-10-14 (split D1':D0* = 50:50, `reweight.nr_parent_split`, I2) |
| D7 | Nominal D/D* FF: BLPRXP (one fit, D and D* eigenvariations shared) | adopted 2026-10-14 |
| D8 | Explicit candidate χ² ("no single mode fits") → Paper 2; Paper 1 keeps only kinematic-class support tests | proposed |

## Claims

| ID | Claim | Evidence | Produced by | Status |
|----|-------|----------|-------------|--------|
| C1 | SEM from HFLAV: B_SEM = 9.26 / 8.55 %, gap 1.79 / 1.72 % (16.2 / 16.7 % of inclusive), c_true = 6.07 | Tab. SEM BFs | step 5, `lib/systematics.py` | done (revisit after D6, D7) |
| C2 | Inclusive full-phase-space moments with full theory covariance | Fig. measurements + HQE | `scratch/interpretation/measurements_hqe.py` | proto |
| C3 | Inclusive spectrum inverted without SEM subtraction agrees with the SEM shape in E_ℓ; no c_true needed | `figures/8/hqe_vs_sem_stack.pdf` | step 8 | done |
| C4 | Closure: the zero-threshold procedure recovers known gap truths (D1'→D*η, 50 % Dℓν + 50 % Λc p̄ℓν); M_X² cannot represent lines with 3 moments | `figures/7/*/convergence_all3_band.pdf`, `output/7/summary.json` | step 7 | done (zero threshold only) |
| C5 | M_X²: residual admissible with the Belle II cocktail (SEM Var(M_X²) 4.06 → 1.22 GeV⁴, inclusive 1.62 ± 0.08; gap ⟨M_X²⟩ ≈ 5.8 GeV², Var ≈ 1.7 GeV⁴). The PHSP NR D(*)ππ lineshape makes it inadmissible: cone distance d² = 20.9, bootstrap p = 0.002 (2.9σ); E_ℓ, q² admissible either way. Independent of the D1':D0* split and of the support | `output/10/summary.json` | step 10 | done |
| C6 | Admissibility as a test: Mahalanobis distance to the moment space (exact two-atom boundary), bootstrap p. Nominal: d² = 0 for M_X², E_ℓ, q² and jointly (p = 1). χ² approximation not conservative (non-Gaussian toys) | `output/10/summary.json` | step 10 | done |
| C7 | (reframed: nominal is admissible, so the minimal shift is 0) Positivity-allowed scale of each SEM component, one at a time, incl. the B_SEM/c_true change: D* ≤ 1.30, D ≤ 1.65, D** ≤ 2.25; D(*)ππ, D_sK unconstrained up to 3. Scale that would close the gap alone: 1.35, 1.8, 2.3 → no single component can fill the gap | `figures/10/component_scan.pdf` | step 10 | done (joint χ²₉ approximation; bootstrap scan todo) |
| C8 | Model-independent CDF bounds (LP, 3 moments, nominal / 68 % toys): P(E_ℓ ≤ 0.5) ≤ 0.16 (0.17); P(E_ℓ ≤ 1.0) ∈ [0.02, 0.56]; P(q² ≤ 4) ∈ [0.19, 0.79]; P(q² ≤ 6) ≥ 0.40 (0.36); P(M_X² ≤ 10) ≥ 0.93 (0.88) | `figures/9/cdf_bounds.pdf` | step 9 | done |
| C9 | Support tests: Λc p̄ alone excluded — ≥ 94 % of the gap below its M_X² threshold (10.4 GeV²) and ≥ 17 % above its q² endpoint (4.22 GeV²), allowed in 0 % of toys. D_sK alone disfavoured only by M_X² (≥ 12 % below 6.06 GeV², allowed in 6 % of toys); E_ℓ, q² allow it | `output/9/summary.json` | step 9 | done (coverage, power study todo) |
| C10 | Positivity lower-bounds the gap size: M_X² admissible in ≥ 50 % of toys only for B_gap/B_incl ≥ 0.13 (nominal 0.165: 78 %); E_ℓ ≥ 0.055, q² ≥ 0.04 | `figures/9/gap_fraction_scan.pdf` | step 9 | done |
| C11 | Gap location vs SEM and candidates (mean–variance plane), new cocktail: E_ℓ gap ⟨E_ℓ⟩ ≈ 1.30 GeV, Dπππ inside 68 %; q² gap on top of SEM/inclusive (⟨q²⟩ ≈ 4.3, Var ≈ 8), all candidates ≤ 3.2 → no single candidate, driven by q². Seminar statements "no low-E_ℓ gap" and "gap hard in E_ℓ" no longer hold | `moment_plane_clean.pdf`, `gap_density_hqe.pdf` | `moment_plane_clean.py`, `gap_density_hqe.py` | proto, rerun 2026-10-14 |
| C12 | Robustness: D1':D0* split 0–1 and supports (E_ℓ ≤ 2.26 / 2.4, M_X² ≤ 12 / 20, q² ≤ 12.5) leave C5, C6 unchanged (step 10 `--support`). Open: LHCb cocktail (I6), β = 2 vs 7/2 (MaxEnt only), partial-BF cross-check (I5) | — | step 10 | partly done, **input** |

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

Polynomial average of the measurements (step 4, draft Sec. 5.1), per-threshold admissibility
table, joint χ² fit and per-E_ℓ-cut M_X fits, threshold Asimov closure with constant c_true.
Review rounds 1–2 (claude.ai artifacts) refer to the old outline and are not carried over.

## Milestones

- **M1** ~~C6~~ (done, step 10); C2 → pipeline.
- **M2** ~~D6, C5 comparison, C7~~ (done, step 10); bootstrap-based component scan.
- **M3** ~~C8, C10, C9 nominal~~ (done, step 9); C9 coverage and power study from the gap-mode samples.
- **M4** C12; then new draft skeleton, reusing SEM construction, endpoint asymptotics,
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
| I1 | Sign-off on this ledger, D3, D8 | Markus, Danny | everything |
| I2 | D6: BF split of each NR D(*)ππ final state over D1' and D0* (50:50 assumed; C5/C6 shown insensitive) | Markus / you | systematic only |
| I3 | ~~D7~~ settled (BLPRXP) | — | — |
| I4 | Confirm `inputs/hqe_likelihood_toys/likelihood_toys.h5` is the 2609.28752 release; is m4 (q²) available at zero threshold? | Markus | C2, C6, C8 |
| I5 | Partial BFs ΔB(E_ℓ > cut), ΔB(q² > cut) with correlations to the moments, from the HQE fit | Markus | C12 cross-check |
| I6 | LHCb cocktail normalisation: which BFs (HFLAV as ours, or LHCb fit values) and whether the "higher D**" cocktail counts as SEM or as gap | note author / Markus | C12 |
| I7 | Where the paper source lives (Overleaf git URL) and OK to generate figures, tables and numbers into it | you, co-authors | M4 |
