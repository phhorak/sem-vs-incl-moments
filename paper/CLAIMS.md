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
| D6 | Nominal NR D(*)ππ lineshape: Belle II resonant model (D1'/D0* → D(*)ππ, LLSW→BLR) vs current 5-body PHSP | **open, decides C5** |
| D7 | Nominal D/D* FF: BGL (current) vs BLPRXP (Belle II convention) | open |
| D8 | Explicit candidate χ² ("no single mode fits") → Paper 2; Paper 1 keeps only kinematic-class support tests | proposed |

## Claims

| ID | Claim | Evidence | Produced by | Status |
|----|-------|----------|-------------|--------|
| C1 | SEM from HFLAV: B_SEM = 9.26 / 8.55 %, gap 1.79 / 1.72 % (16.2 / 16.7 % of inclusive), c_true = 6.07 | Tab. SEM BFs | step 5, `lib/systematics.py` | done (revisit after D6, D7) |
| C2 | Inclusive full-phase-space moments with full theory covariance | Fig. measurements + HQE | `scratch/interpretation/measurements_hqe.py` | proto |
| C3 | Inclusive spectrum inverted without SEM subtraction agrees with the SEM shape in E_ℓ; no c_true needed | `figures/8/hqe_vs_sem_stack.pdf` | step 8 | done |
| C4 | Closure: the zero-threshold procedure recovers known gap truths (D1'→D*η, 50 % Dℓν + 50 % Λc p̄ℓν); M_X² cannot represent lines with 3 moments | `figures/7/*/convergence_all3_band.pdf`, `output/7/summary.json` | step 7 | done (zero threshold only) |
| C5 | M_X²: residual has Var < 0 (SEM broader than inclusive), 0.4 % of toys admissible. Var_incl = 1.62 ± 0.08 GeV⁴ vs Var_SEM = 4.06 ± 1.28. Driver: NR D(*)ππ (3.1 % of SEM, ⟨M_X²⟩ = 13.8 GeV² from PHSP). Scaling it to 0.33× restores admissibility. Estimate with a D1'-like lineshape: Var_SEM ≈ 0.97 < 1.94 needed → likely admissible | moment plane, budget | `sem_mx_components.py`, `mx_variance.py` | proto, **depends on D6** |
| C6 | Admissibility as a test: cone-distance statistic and p-value per observable (replaces "fraction admissible": E_ℓ 98.2 %, q² 99.7 %, M_X² 0.4 %) | table | — | todo (M1) |
| C7 | Minimal SEM shift that restores admissibility: convex fit of BF / B_incl / FF / lineshape pulls under Hankel positivity of the unnormalised gap moments; pulls and binding modes | table | — | todo (M2), plan in `scratch/interpretation/NOTES.md` |
| C8 | Model-independent bounds on gap fractions per region (LP, 3 moments): P(E_ℓ ≤ 0.5) ≤ 0.10 (0.14 with unc.), P(E_ℓ ≤ 1.0) ≤ 0.51, P(q² ≤ 6) ≥ 0.55 (0.49) | `cdf_bounds.pdf` | `interp.py` (reads stale `output/5/toys.npz`) | proto → M3 |
| C9 | Support tests for kinematic classes: Λc p̄ requires q² ≤ 4.22 GeV², E_ℓ ≤ 1.66 GeV; D_sK q² ≤ 7.94, E_ℓ ≤ 2.07. Bound on the gap mass beyond these → "light states only" / "no-D" tests, with power study (inject Λc p̄) and coverage | table + fig | — | todo (M3) |
| C10 | Positivity lower-bounds the gap: E_ℓ, q² admissible only for B_gap / B_incl ≳ 0.10–0.11 | `admissibility_vs_gap.pdf` | `interp.py` | proto |
| C11 | Gap location vs SEM and candidates (mean–variance plane): gap hard in E_ℓ, high in q²; candidates softer / lower | `moment_plane_clean.pdf` | `moment_plane_clean.py` | proto (illustration, see D8) |
| C12 | Robustness: Belle II vs LHCb cocktail, β = 2 vs 7/2 for E_ℓ, support choices, partial-BF-per-threshold cross-check | table | — | todo (M4), **input** |

## Retired (appendix at most)

Polynomial average of the measurements (step 4, draft Sec. 5.1), per-threshold admissibility
table, joint χ² fit and per-E_ℓ-cut M_X fits, threshold Asimov closure with constant c_true.
Review rounds 1–2 (claude.ai artifacts) refer to the old outline and are not carried over.

## Milestones

- **M1** C2 → pipeline; C6 test statistic and p-values on the HQE toys.
- **M2** D6 settled; C5 with both lineshapes; C7 minimal-shift fit (B_gap included).
- **M3** C8, C9 bounds with toys, coverage and power from the step-3 gap-mode samples; C10.
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
| I2 | D6: nominal NR D(*)ππ lineshape and its BF split over D1'/D0* (or a measured M(D(*)ππ) spectrum) | Markus / you | C1, C5, C7 |
| I3 | D7: BGL or BLPRXP as nominal D/D* FF | Markus | C1 |
| I4 | Confirm `inputs/hqe_likelihood_toys/likelihood_toys.h5` is the 2609.28752 release; is m4 (q²) available at zero threshold? | Markus | C2, C6, C8 |
| I5 | Partial BFs ΔB(E_ℓ > cut), ΔB(q² > cut) with correlations to the moments, from the HQE fit | Markus | C12 cross-check |
| I6 | LHCb cocktail normalisation: which BFs (HFLAV as ours, or LHCb fit values) and whether the "higher D**" cocktail counts as SEM or as gap | note author / Markus | C12 |
| I7 | Where the paper source lives (Overleaf git URL) and OK to generate figures, tables and numbers into it | you, co-authors | M4 |
