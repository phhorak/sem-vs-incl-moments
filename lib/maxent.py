"""Shared Hausdorff-feasibility check + MaxEnt inversion on [0,1] (steps 6 and 7).

Replaces the four diverging copies that lived in the retired 6_joint_fit / 6b_kolya_vs_joint /
7_hausdorff_toy / 8_hausdorff_data scripts. The algorithm (L-BFGS-B on the convex dual, same
options) is unchanged from the one behind figures/7/maxent_convergence_El_overlay_2x2.pdf.

Conventions:
  * Moment arrays include m_0 = 1: using moments up to order N means mu01[:N + 1].
  * Convergence is judged ONLY by moment closure, max_k |m_k(f) - mu01_k| < mom_tol.
    scipy's res.success is recorded but not used: it reports ABNORMAL_TERMINATION on
    solutions that close to 1e-12, and success=True on diverged ones near the boundary of
    the moment space (both seen in this project).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import minimize

DEFAULT_MOM_TOL = 1e-6
_OPT = {"maxiter": 30000, "ftol": 1e-15, "gtol": 1e-12}


def raw_to_mu01(raw, lo: float, hi: float) -> np.ndarray:
    """Raw moments [m_0=1, m_1, ...] of x on [lo, hi] -> raw moments of t=(x-lo)/(hi-lo)."""
    raw = np.asarray(raw, dtype=float)
    span = hi - lo
    return np.array([sum(comb(k, j) * (-lo) ** (k - j) * raw[j] for j in range(k + 1)) / span ** k
                     for k in range(len(raw))])


def raw_to_mu01_jacobian(lo: float, hi: float, nmax: int) -> np.ndarray:
    span = hi - lo
    J = np.zeros((nmax, nmax))
    for k in range(nmax):
        for j in range(k + 1):
            J[k, j] = comb(k, j) * (-lo) ** (k - j) / span ** k
    return J


def hausdorff_check(mu01, tol: float = 1e-12) -> tuple[bool, float]:
    """Exact truncated Hausdorff test on [0,1]: m_0..m_N lie strictly inside the moment space
    (where a MaxEnt density exists) iff the Hankel and localizing matrices are positive definite:
      N = 2k:   [m_{i+j}]_{0..k}    and [m_{i+j+1} - m_{i+j+2}]_{0..k-1}
      N = 2k+1: [m_{i+j+1}]_{0..k}  and [m_{i+j} - m_{i+j+1}]_{0..k}
    Returns (feasible, smallest eigenvalue relative to its matrix's largest)."""
    m = np.asarray(mu01, dtype=float)
    if m.ndim != 1 or len(m) < 2 or not np.all(np.isfinite(m)):
        return False, -np.inf
    N, k = len(m) - 1, (len(m) - 1) // 2
    H = lambda f, n: np.array([[f(i + j) for j in range(n)] for i in range(n)])
    if N % 2 == 0:
        mats = [H(lambda s: m[s], k + 1), H(lambda s: m[s + 1] - m[s + 2], k)]
    else:
        mats = [H(lambda s: m[s + 1], k + 1), H(lambda s: m[s] - m[s + 1], k + 1)]
    worst = min(np.linalg.eigvalsh(X).min() / max(np.abs(np.linalg.eigvalsh(X)).max(), 1e-300)
                for X in mats if X.size)
    return bool(worst > tol), float(worst)


@dataclass
class MaxEntResult:
    f: np.ndarray          # density on the t grid, normalized to unit area on [0,1]
    lam: np.ndarray
    mom_err: float
    converged: bool
    opt_success: bool


class MaxEnt:
    """MaxEnt solver bound to one t grid (powers precomputed once, reused across toys)."""

    def __init__(self, t: np.ndarray, max_order: int = 10, mom_tol: float = DEFAULT_MOM_TOL):
        self.t = np.asarray(t, dtype=float)
        self.tp = np.stack([self.t ** k for k in range(max_order + 1)])
        self.mom_tol = mom_tol
        self._priors: dict[tuple[float, float], np.ndarray] = {}

    def prior(self, alpha: float, beta: float) -> np.ndarray:
        key = (float(alpha), float(beta))
        if key not in self._priors:
            eps = 1e-300
            self._priors[key] = (np.maximum(self.t, eps) ** alpha) * (np.maximum(1 - self.t, eps) ** beta)
        return self._priors[key]

    def solve(self, mu01, alpha: float = 0.0, beta: float = 0.0, lam0=None) -> MaxEntResult:
        """rho(t) ~ t^alpha (1-t)^beta exp(sum_k lam_k t^k), matching mu01[0:N]."""
        mu = np.asarray(mu01, dtype=float)
        n = len(mu)
        t, tp, prior = self.t, self.tp[:n], self.prior(alpha, beta)

        def _g(lam):
            lf = lam @ tp
            return prior * np.exp(lf - lf.max())

        def dual(lam):
            lf = lam @ tp
            m = lf.max()
            return np.log(np.trapz(prior * np.exp(lf - m), t)) + m - lam @ mu

        def grad(lam):
            g = _g(lam)
            gn = g / np.trapz(g, t)
            return np.trapz(tp * gn, t, axis=1) - mu

        x0 = np.zeros(n) if lam0 is None else np.asarray(lam0, dtype=float).copy()
        res = minimize(dual, x0, jac=grad, method="L-BFGS-B", options=_OPT)
        f = _g(res.x)
        Z = np.trapz(f, t)
        f = f / Z if np.isfinite(Z) and Z > 0 else np.full_like(t, np.nan)
        mom_err = float(np.max(np.abs(np.trapz(tp * f, t, axis=1) - mu)))
        ok = bool(np.all(np.isfinite(f)) and np.isfinite(mom_err) and mom_err < self.mom_tol)
        return MaxEntResult(f=f, lam=res.x, mom_err=mom_err, converged=ok, opt_success=bool(res.success))
