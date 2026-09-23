# HQE global-fit likelihood toys (Markus Prim)

`likelihood_toys.h5`: fully-inclusive (no El/q2/Mx cut, i.e. "cut0.0") moment predictions
from the HQE (kinetic-scheme) global fit to inclusive B->Xc l nu moments, sent by Markus Prim
via Discord on 2026-09-18 (preliminary). Two HDF5 groups:

  - `central`: 1 row, 73 columns -- best-fit central values.
  - `toys`: 12000 rows, same 73 columns -- likelihood toys (used instead of the fit covariance
    because the errors are non-Gaussian, per Markus: "Kovarianz geht nicht weil die Fehler
    non-Gaussian sind. Aber ich kann dir Toys geben.").

Relevant columns (of 73): `{mx,el,q2}_{1,2,3}_cut0.0` -- 1st/2nd/3rd **central** moments
(mean, then central 2nd/3rd -- NOT raw moments; convert with `lib.moments.central_to_raw`)
of M_X^2, E_l, q^2 over the full phase space. The `_cut{X}` columns at X>0 are HQE
predictions at nonzero thresholds, included for cross-checks but not used by the
"Kolya" pipeline step (which only needs the fully-inclusive cut0.0 values).

`Paper - Likelihood Toys.ipynb`: Markus's own notebook showing what's what.

Context: originally proposed by Markus (2026-09-14/15) as a way to extrapolate to the full
phase space via the HQE fit and solve the Hausdorff moment problem there, instead of (or in
addition to) the gap-region SEM-subtraction approach. Used by `7_data_results.py` (HQE inversion)
and `6_asimov_closure.py` (inclusive-moment systematic of the Asimov closure test).
