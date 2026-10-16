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
addition to) the gap-region SEM-subtraction approach. Read by the step-4 bundle (`data.hqe_toys` in
config.yaml) and from there by steps 5-8.

`likelihood_toys_2026-10-16_belle_belle2.h5` (nominal since 2026-10-16): same 73 columns, 10800 toys,
from the fit to the corrected experimental input (`inputs/inclusive_moments_data_2026-10-15.tar.gz`:
Belle E_l and CLEO M_X covariances were diagonal before, Belle M_X errors reduced), with the combined
Belle + Belle II q^2 moments. `likelihood_toys.h5` is the previous version (2026-09-18).
