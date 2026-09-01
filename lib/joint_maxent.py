"""Joint multi-threshold MaxEnt inversion with soft (Gaussian-relaxed) constraints and
value/slope threshold-continuity penalties.

Replaces the older per-threshold MaxEnt inversion + post-hoc curve stitching (multiple
independent single-threshold fits, spliced together with a chain of renormalization ratios --
see 9_hausdorff_bands.py's original stitching code) with a SINGLE fit that uses every passing
threshold's moments as simultaneous constraints on one global-support density. Each threshold's
raw moment is genuinely a constraint on the same density (integral over its own tail), so this
needs no post-hoc normalization matching and produces no stitching-seam artifacts.

Three ingredients, all validated on the D2stDeta Asimov/toy closure test (see
scratch/joint_slack_test.py and scratch/joint_slack_toy_test.py):

  1. Soft constraints: minimize logZ(lam) - lam.targets + 0.5*sum((sigma_j*lam_j)^2) instead of
     an exact equality dual. sigma_j = sigma_rel * |target_j| is slack, not a measured
     uncertainty -- but it's also what keeps the dual bounded when a toy's (noisy) moments fall
     just outside the achievable moment polytope, where the un-regularized ("hard") dual has NO
     minimum and L-BFGS runs the multipliers to +-infinity instead of failing gracefully.

  2. Value(m=0) + slope(m=1) continuity penalties at each threshold: each threshold's own
     constraint block only "turns on" for t >= t_thr, so the m-th derivative jump of the
     exponent there is a closed-form LINEAR function of that block's own lambda's --
        Jump_i^(m)(lam) = sum_{k=m}^{N-1} lam_{i,k} * [k!/(k-m)!] * span^m * thr_i^(k-m)
     Each row is normalized to unit L1 norm before applying a single scalar `gamma` so one
     gamma value means the same thing regardless of threshold/span/moment order (unnormalized,
     this factor can differ by 3-4 orders of magnitude between e.g. El and Q2 -- see
     scratch/joint_slack_toy_test.py's debugging notes). Curvature (m=2) was tested and found to
     over-penalize (not enough moments left per threshold to spare), so only m=0,1 are used.

  3. Per-toy Hausdorff threshold filtering: a threshold whose own (toy-noisy) residual moments
     fail the Hausdorff consistency check is dropped from the constraint set for that toy,
     rather than forced in. This mirrors 8_hausdorff_data.py's existing haus_results pass/fail
     gating, just recomputed fresh per toy instead of once on the nominal residual -- a
     threshold probing a region with no real signal is just noise around zero, which generically
     fails Hausdorff (can't be the tail of any positive density).

Closure-test finding: El (broad spectrum) is stable with N=5 raw moments/threshold. Q2 (steeply
falling spectrum, so raw high-order moments are dominated by rare tail events) needed N=3 --
even fairly inclusive Q2 cuts failed Hausdorff ~30% of the time at N=5, dropping to a solidly
usable ~70% converged at N=3. Tune N per observable via config, not a global constant.
"""
from __future__ import annotations

from math import comb, factorial

import numpy as np
from scipy.optimize import minimize


def hausdorff_moment_check(mu, tol: float = 1e-8) -> tuple[bool, float]:
    mu = np.asarray(mu, dtype=float)
    N = len(mu) - 1
    worst = np.inf
    for n in range(N + 1):
        row = mu[n:N + 1].copy()
        worst = min(worst, row[0])
        for k in range(1, N - n + 1):
            row = row[1:] - row[:-1]
            worst = min(worst, ((-1) ** k) * row[0])
    return (worst >= -tol), float(worst)


def raw_to_mu01(raw, lo: float, hi: float) -> np.ndarray:
    span = hi - lo
    nmax = len(raw)
    mu = np.zeros(nmax)
    for k in range(nmax):
        mu[k] = sum(comb(k, j) * (-lo) ** (k - j) * raw[j] for j in range(k + 1)) / span ** k
    return mu


def passing_thresholds_from_conditional_raw(raw_by_thr: dict[float, np.ndarray],
                                             xhi: float) -> list[float]:
    """Given, for each candidate threshold, the CONDITIONAL raw moments above it (raw[0] must
    be 1, i.e. already normalized to that threshold's own surviving yield), return the sorted
    subset that passes the Hausdorff check. Threshold-representation-agnostic: callers can
    build raw_by_thr either from event-level sums (7_hausdorff_toy.py) or from interpolated
    per-cut moment tables (8_hausdorff_data.py).
    """
    passing = []
    for thr in sorted(raw_by_thr):
        raw = raw_by_thr[thr]
        if raw is None:
            continue
        mu01 = raw_to_mu01(raw, thr, xhi)
        ok, _ = hausdorff_moment_check(mu01)
        if ok:
            passing.append(thr)
    return passing


def passing_thresholds_events(x_arr, wi, we, xlo: float, xhi: float, cuts_sorted, N: int) -> list[float]:
    """Event-level convenience wrapper (7_hausdorff_toy.py: per-event x_arr + toy weight
    vectors wi/we). For the moment-table representation (8_hausdorff_data.py: per-cut
    interpolated raw moments, no per-event arrays), build raw_by_thr directly and call
    passing_thresholds_from_conditional_raw instead.
    """
    w_gap_arr = wi - we
    raw_by_thr = {}
    for thr in cuts_sorted:
        mask = x_arr >= thr
        yld = w_gap_arr[mask].sum()
        if yld <= 0:
            raw_by_thr[float(thr)] = None
            continue
        raw_by_thr[float(thr)] = np.array(
            [(w_gap_arr * np.where(mask, x_arr ** k, 0.)).sum() / yld for k in range(N)])
    return passing_thresholds_from_conditional_raw(raw_by_thr, xhi)


def compute_targets_events(x_arr, wi, we, xlo: float, xhi: float, cuts_sorted, N: int):
    """Event-level target builder matching build_basis's threshold-major/order-minor ordering.
    Targets are normalized to the OVERALL yield above xlo (not per-threshold), matching what
    the joint fit's basis functions integrate against -- NOT the same normalization as the
    per-threshold conditional moments used for the Hausdorff check above.
    """
    w_gap_arr = wi - we
    norm_total = w_gap_arr[x_arr >= xlo].sum()
    if norm_total <= 0:
        return None
    targets = []
    for thr in cuts_sorted:
        mask_phys = x_arr >= thr
        for k in range(N):
            targets.append((w_gap_arr * np.where(mask_phys, x_arr ** k, 0.)).sum() / norm_total)
    return np.array(targets)


def fit_joint_events(x_arr, wi, we, xlo: float, xhi: float, cuts_candidates, N: int,
                      t_grid: np.ndarray, prior: np.ndarray, sigma_rel: float,
                      gammas: tuple[float, ...]):
    """End-to-end: Hausdorff-filter the candidate thresholds using THIS toy's own residual, then
    fit the joint soft+continuity-penalty MaxEnt using only the passing subset. Returns
    (f_on_t_grid, converged_bool, mom_err, passing_thresholds); f/converged/mom_err are all
    None/False/inf if fewer than 2 thresholds pass (a joint fit needs at least 2 to mean
    anything -- with 1 it degenerates to a plain single-threshold fit).
    """
    cuts_sorted = sorted(float(c) for c in cuts_candidates)
    passing = passing_thresholds_events(x_arr, wi, we, xlo, xhi, cuts_sorted, N)
    if len(passing) < 2:
        return None, False, np.inf, passing
    span = xhi - xlo
    basis, _ = build_basis(t_grid, xlo, xhi, passing, N)
    targets = compute_targets_events(x_arr, wi, we, xlo, xhi, passing, N)
    if targets is None:
        return None, False, np.inf, passing
    try:
        f, ok, err = maxent_joint_soft(basis, targets, passing, N, span, t_grid, prior,
                                        sigma_rel=sigma_rel, gammas=gammas)
    except Exception:
        return None, False, np.inf, passing
    if err > 0.2 or not sane_density(f):
        return None, False, err, passing
    return f, True, err, passing


def build_basis(t_grid: np.ndarray, xlo: float, xhi: float, cuts_sorted, N: int):
    """basis[j] = mask(t>=t_thr) * x(t)^k on t_grid, for each (threshold, order) pair, in the
    SAME nested order (threshold-major, order-minor) as compute_targets_events returns its
    target list -- callers must keep the two in sync (fit_joint_events does).
    """
    span = xhi - xlo
    xgrid = xlo + t_grid * span
    basis = []
    for thr in cuts_sorted:
        t_thr = (float(thr) - xlo) / span
        mask_t = (t_grid >= t_thr).astype(float)
        for k in range(N):
            basis.append(mask_t * xgrid ** k)
    return basis, xgrid


def maxent_joint_soft(basis, targets: np.ndarray, cuts_sorted, N: int, span: float,
                       t_grid: np.ndarray, prior: np.ndarray, sigma_rel: float = 0.03,
                       gammas: tuple[float, ...] = (30.0, 30.0)):
    """Soft-constrained joint MaxEnt with value(m=0)+slope(m=1)[+curvature(m=2) if a 3rd gamma
    is given] threshold-continuity penalties. Returns (f_on_t_grid, converged_bool, mom_err).
    `converged` reflects physical sanity + mom_err, not L-BFGS's internal gtol/ftol flag (which
    can stay "not converged" for a perfectly good regularized solution -- see closure-test notes
    in scratch/joint_slack_toy_test.py).
    """
    n = len(basis)
    sigma = sigma_rel * np.maximum(np.abs(targets), 1e-12)
    n_thr = len(cuts_sorted)
    Jmats = []
    for m in range(len(gammas)):
        Jm = np.zeros((n_thr, n))
        for i, thr in enumerate(cuts_sorted):
            row = np.zeros(N)
            for k in range(m, N):
                row[k] = factorial(k) / factorial(k - m) * span ** m * float(thr) ** (k - m)
            row_scale = np.sum(np.abs(row))
            if row_scale > 0:
                row = row / row_scale
            Jm[i, i * N:(i + 1) * N] = row
        Jmats.append(Jm)

    def _g(lam):
        lf = sum(lam[j] * basis[j] for j in range(n)); lf -= lf.max()
        return prior * np.exp(lf)

    def _pen(lam):
        pen, grad_pen = 0.0, np.zeros(n)
        for gamma, Jm in zip(gammas, Jmats):
            if gamma == 0:
                continue
            jump = Jm @ lam
            pen += 0.5 * gamma * np.sum(jump ** 2)
            grad_pen += gamma * (Jm.T @ jump)
        return pen, grad_pen

    def dual(lam):
        lf = sum(lam[j] * basis[j] for j in range(n)); lf_m = lf.max()
        g = prior * np.exp(lf - lf_m); Z = np.trapz(g, t_grid)
        pen, _ = _pen(lam)
        return np.log(Z) + lf_m - lam @ targets + 0.5 * np.sum((sigma * lam) ** 2) + pen

    def grad(lam):
        g = _g(lam); Z = np.trapz(g, t_grid); gn = g / Z
        _, grad_pen = _pen(lam)
        return (np.array([np.trapz(basis[j] * gn, t_grid) for j in range(n)]) - targets
                + (sigma ** 2) * lam + grad_pen)

    res = minimize(dual, np.zeros(n), jac=grad, method="L-BFGS-B",
                    options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-10})
    g = _g(res.x); Z = np.trapz(g, t_grid); fn = g / Z
    mom_err = max(abs(np.trapz(basis[j] * fn, t_grid) - targets[j]) for j in range(n))
    return fn, bool(res.success), float(mom_err)


def sane_density(f, max_peak: float = 1e6) -> bool:
    return (f is not None and np.all(np.isfinite(f)) and np.all(f >= -1e-6)
            and np.max(f) < max_peak)


# ── Conditional-moment formulation (for 8_hausdorff_data.py: real-data moment tables report
# only CONDITIONAL shape moments per threshold, mu_0=1 by construction of the r_exp/r_sem
# identity -- no cross-threshold yield/survival-fraction information, unlike 7_hausdorff_toy.py's
# event-level MC truth which can compute genuine survival fractions directly). ─────────────────
#
# The naive fix (borrow the SEM cocktail's own yield curve as a survival-fraction proxy) was
# tried and found to badly distort the fit (~5-15x worse mom_err than the validated MC case) --
# SEM and the true gap signal are different populations with no reason to share a survival
# curve. The CORRECT fix needs no proxy at all: the measured constraint
#     E[X^k | X >= thr] = m_k(thr)
# is, by definition, a RATIO of two integrals over [thr, hi]: (integral x^k f dx) / (integral f
# dx). Multiplying through by the (unknown) denominator and rearranging:
#     integral_{thr}^{hi} (x^k - m_k(thr)) f(x) dx = 0
# This is LINEAR in f and the survival fraction cancels out algebraically -- it never appears
# anywhere in the constraint. Valid whenever the threshold has a nonzero measured yield (true by
# construction: if m_k(thr) was measured, thr had surviving statistics).

def build_basis_conditional(t_grid: np.ndarray, xlo: float, xhi: float, cuts_sorted,
                             N: int, m_by_thr: dict[float, np.ndarray]):
    """Basis for the conditional-moment (no-proxy) formulation. m_by_thr[thr] must hold the
    N-1 measured conditional raw moments (orders 1..N-1, in order) for that threshold. Returns
    (basis, xgrid); every target is identically 0 (see module docstring above), so there is no
    separate target list -- maxent_joint_soft_conditional builds it internally.
    """
    span = xhi - xlo
    xgrid = xlo + t_grid * span
    basis = []
    for thr in cuts_sorted:
        t_thr = (float(thr) - xlo) / span
        mask_t = (t_grid >= t_thr).astype(float)
        m_vals = m_by_thr[thr]
        for k in range(1, N):
            basis.append(mask_t * (xgrid ** k - m_vals[k - 1]))
    return basis, xgrid


def maxent_joint_soft_conditional(basis, cuts_sorted, N: int, span: float, t_grid: np.ndarray,
                                   prior: np.ndarray, m_by_thr: dict[float, np.ndarray],
                                   sigma_rel: float = 0.03, gammas: tuple[float, ...] = (30.0, 30.0)):
    """Soft-constrained joint MaxEnt for the conditional-moment (no-proxy) formulation.
    Targets are identically 0 (see build_basis_conditional), so slack for constraint (thr, k) is
    sigma_rel * |m_k(thr)| -- the measured moment itself is the natural "yardstick" here, in
    place of sigma_rel*|target| (which would trivially vanish since targets are 0).

    Continuity penalties: each threshold's block now has N-1 entries (orders 1..N-1, no k=0
    term -- k=0 would just be the trivial identity mu_0=1, already implicit). The SLOPE (m=1)
    row is UNCHANGED from the plain x^k formulation (d/dt of a threshold-constant shift is 0),
    but the VALUE (m=0) row must reflect this basis's actual jump at t=thr, i.e.
    (thr^k - m_k(thr)), not raw thr^k.
    """
    n = len(basis)
    n_thr = len(cuts_sorted)
    n_per_thr = N - 1
    targets = np.zeros(n)
    sigma = np.array([sigma_rel * max(abs(m_by_thr[thr][k - 1]), 1e-12)
                       for thr in cuts_sorted for k in range(1, N)])

    Jmats = []
    for m in range(len(gammas)):
        Jm = np.zeros((n_thr, n))
        for i, thr in enumerate(cuts_sorted):
            thr = float(thr)
            m_vals = m_by_thr[thr]
            row = np.zeros(n_per_thr)
            for k in range(1, N):
                idx = k - 1
                if k < m:
                    continue
                if m == 0:
                    row[idx] = thr ** k - m_vals[idx]
                else:
                    row[idx] = factorial(k) / factorial(k - m) * span ** m * thr ** (k - m)
            row_scale = np.sum(np.abs(row))
            if row_scale > 0:
                row = row / row_scale
            Jm[i, i * n_per_thr:(i + 1) * n_per_thr] = row
        Jmats.append(Jm)

    def _g(lam):
        lf = sum(lam[j] * basis[j] for j in range(n)); lf -= lf.max()
        return prior * np.exp(lf)

    def _pen(lam):
        pen, grad_pen = 0.0, np.zeros(n)
        for gamma, Jm in zip(gammas, Jmats):
            if gamma == 0:
                continue
            jump = Jm @ lam
            pen += 0.5 * gamma * np.sum(jump ** 2)
            grad_pen += gamma * (Jm.T @ jump)
        return pen, grad_pen

    def dual(lam):
        lf = sum(lam[j] * basis[j] for j in range(n)); lf_m = lf.max()
        g = prior * np.exp(lf - lf_m); Z = np.trapz(g, t_grid)
        pen, _ = _pen(lam)
        return np.log(Z) + lf_m - lam @ targets + 0.5 * np.sum((sigma * lam) ** 2) + pen

    def grad(lam):
        g = _g(lam); Z = np.trapz(g, t_grid); gn = g / Z
        _, grad_pen = _pen(lam)
        return (np.array([np.trapz(basis[j] * gn, t_grid) for j in range(n)]) - targets
                + (sigma ** 2) * lam + grad_pen)

    res = minimize(dual, np.zeros(n), jac=grad, method="L-BFGS-B",
                    options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-10})
    g = _g(res.x); Z = np.trapz(g, t_grid); fn = g / Z
    mom_err = max(abs(np.trapz(basis[j] * fn, t_grid) - targets[j]) for j in range(n))
    return fn, bool(res.success), float(mom_err)

