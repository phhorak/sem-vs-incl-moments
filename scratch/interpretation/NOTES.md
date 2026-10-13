# Interpretation of the data gap — status and plan (2026-09-24)

Prototypes in this directory (not part of the pipeline yet):
`interp.py` (moment plane, CDF bounds, admissibility vs gap fraction, candidate chi2),
`mx_variance.py`, `sem_mx_components.py`, and their logs, PDFs and `interpretation.json`.
All use the data residual at threshold 0 (HQE route, moments m1..m3) with the step-5 toys.

## Findings
- No single candidate gap mode fits the data gap: best chi2 over 9 moments ~130 (D1->Dgamma,
  Dpipipi), then D1'->D*eta 156. In E_l and q2 the gap looks SEM-like (hard lepton, high q2);
  all candidates are softer.
- M_X^2 residual is inadmissible because Var_incl(HQE) = 1.62 +- 0.08 GeV^4 < (1-f) Var_SEM,
  Var_SEM = 4.06 +- 1.28 GeV^4; admissibility needs f >~ 0.60 (Table 1: 0.165).
- Driver: the non-resonant D(*)pipi modes (3% of SEM, <M_X^2> = 13.8 GeV^2). Without them
  Var_SEM = 0.81. Scaling their BF to ~0.33 suffices. Their decfiles use pure 5-body PHSP,
  so the M_X shape is a model artefact (likely the real cause, more than the rate).
- Model-independent CDF bounds (LP, 3 moments): P(E_l<=0.5) <= 0.10 (0.14 with unc.),
  P(E_l<=1.0) <= 0.51, P(q2<=6) >= 0.55 (0.49).
- Positivity lower-bounds the gap: E_l, q2 admissible only for B_gap/B_incl >~ 0.10-0.11.

## Plan (agreed direction, not started)
1. Store SEM moments per BF family in step 5 (moments are linear in BFs; no new toys).
2. Shape nuisance for NR D(*)pipi M_X (reweight PHSP toward a measured M(D(*)pipi) spectrum;
   ask user which reference, e.g. BaBar).
3. Positivity-constrained SEM fit: minimise prior chi2 of BF/incl/FF/shape pulls subject to
   Hankel positivity of the UNNORMALISED gap moments M_k = B_incl m_k^incl - sum_i B_i mu_ik
   (linear in B_i -> LMI, convex). All three observables at thr 0, then all thresholds.
   Report chi2_min (consistency test), pulls, binding constraints; refit gap with profiled SEM.
4. Discriminate: SEM rates vs SEM shapes vs HQE M_X^2 extrapolation vs positivity (interference).
5. Then: house-style versions of the three plots into step 7; a "physics message" section in the
   draft-review dashboard (https://claude.ai/artifact/8iGkZycinKXCynsAAyZ7kG).
