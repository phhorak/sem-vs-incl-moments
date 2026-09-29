"""chi2 fit of one density to the residual moments at several thresholds, shared by step 8 (data)
and step 7 (the same fit on Asimov pseudo-data).

  joint_fit  El, q2: one density f(t) ~ t^a (1-t)^b exp(sum_k c_k t^k) fit to the conditional
             residual moments (orders 1-3) at all thresholds, weighted by the toy covariance.
  mx_fits    Mx: separate fits per El threshold (Mx moments are El-selected sub-populations).
Both refit the given residual toys for a 68% band.
"""
import time

import numpy as np
from scipy.optimize import minimize

from lib.maxent import raw_to_mu01, raw_to_mu01_jacobian

VAR = {"Mx": "mx2", "El": "el", "Q2": "q2"}
M_JOINT, M_MX = 6, 2
L2 = 1e-4          # guards the flat chi2 directions of the 1-threshold Mx fits
BAD_CLOSURE = 0.3  # a toy refit this far from its target sits in a degenerate basin


class JointFit:
    """chi2 fit of one density to the conditional moments (orders 1-3) at several thresholds."""

    def __init__(self, cfg, obs, thresholds, cov_raw, masked, lo, hi, t_n=400):
        mc = cfg["maxent"]
        self.lo, self.hi, self.span = lo, hi, hi - lo
        self.thr = list(thresholds)
        self.t = np.linspace(0, 1, t_n)
        a, b = mc["alpha"][VAR[obs]], mc["beta"][VAR[obs]]
        self.prior = np.maximum(self.t, 1e-300) ** a * np.maximum(1 - self.t, 1e-300) ** b
        self.tp = np.stack([self.t ** k for k in range(1, M_JOINT + 1)])
        self.masks = np.stack([(self.t >= (thr - lo) / self.span) if masked else np.ones_like(self.t)
                               for thr in self.thr]).astype(float)
        self.Cinv = self._cinv(cov_raw)

    def _cinv(self, cov_raw):
        """Raw covariance -> [0,1]-moment space, eigenvalues floored at 1e-3 of the largest."""
        n = len(self.thr)
        if cov_raw is None:
            return np.eye(3 * n)
        J = raw_to_mu01_jacobian(self.lo, self.hi, 4)[1:, 1:]
        B = np.kron(np.eye(n), J)
        w, v = np.linalg.eigh(B @ cov_raw @ B.T)
        return (v / np.maximum(w, 1e-3 * w.max())) @ v.T

    def target(self, raw_gap):
        return np.concatenate([raw_to_mu01(np.r_[1.0, raw_gap[3 * i:3 * i + 3]], self.lo, self.hi)[1:]
                               for i in range(len(self.thr))])

    def density(self, c, M):
        e = c @ self.tp[:M]
        return self.prior * np.exp(e - e.max())

    def model(self, c, M):
        f = self.density(c, M) * self.masks
        Z = np.trapz(f, self.t, axis=1)
        return np.stack([np.trapz(self.t ** k * f, self.t, axis=1) / Z for k in (1, 2, 3)], 1).reshape(-1)

    def chi2(self, c, target, M):
        r = self.model(c, M) - target
        return float(r @ self.Cinv @ r)

    def fit(self, raw_gap, M, c0=None):
        target = self.target(raw_gap)
        obj = lambda c: self.chi2(c, target, M) + L2 * float(c @ c)
        if c0 is not None:
            c = minimize(obj, c0, method="BFGS", options={"maxiter": 2000, "gtol": 1e-8}).x
            if np.max(np.abs(self.model(c, M) - target)) <= BAD_CLOSURE:
                return c, True
        nm = minimize(obj, np.zeros(M), method="Nelder-Mead",
                      options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True})
        bf = minimize(obj, nm.x, method="BFGS", options={"maxiter": 5000, "gtol": 1e-10})
        c = bf.x if bf.fun < nm.fun else nm.x
        return c, bool(np.max(np.abs(self.model(c, M) - target)) <= BAD_CLOSURE)

    def f_phys(self, c, M):
        f = self.density(c, M)
        return f / np.trapz(f, self.t) / self.span

    @property
    def x(self):
        return self.lo + self.t * self.span


def band(curves):
    curves = np.asarray(curves)
    return np.percentile(curves, [16, 84], axis=0) if len(curves) > 10 else None


def joint_fit(cfg, obs, R, n_toys, rng):
    """Joint fit of El or Q2 to R = dict(nominal, cov, toys, thr) on the data support."""
    lo, hi = cfg["data"]["support"][VAR[obs]]
    if obs == "Q2":
        lo = cfg["data"]["joint_fit_q2_lo"]
    jf = JointFit(cfg, obs, R["thr"], R["cov"], True, lo, hi)
    c_best, _ = jf.fit(R["nominal"], M_JOINT)
    chi2 = jf.chi2(c_best, jf.target(R["nominal"]), M_JOINT)
    toys = R["toys"]
    idx = rng.choice(len(toys), min(n_toys, len(toys)), replace=False)
    t0, curves = time.time(), []
    for i in idx:
        c, ok = jf.fit(toys[i], M_JOINT, c0=c_best)
        if ok:
            curves.append(jf.f_phys(c, M_JOINT))
    return dict(x=jf.x, f=jf.f_phys(c_best, M_JOINT), band=band(curves), chi2=chi2, dof=3 * len(R["thr"]) - M_JOINT,
                n_toys=len(idx), n_ok=len(curves), seconds=time.time() - t0, lo=lo, hi=hi)


def mx_fits(cfg, R, n_toys, rng):
    """Per-El-threshold fits of Mx to R on the data support: {thr: dict(x, f, band, chi2)}."""
    lo, hi = cfg["data"]["support"]["mx2"]
    toys = R["toys"]
    idx = rng.choice(len(toys), min(n_toys, len(toys)), replace=False)
    out = {}
    for i, thr in enumerate(R["thr"]):
        s = slice(3 * i, 3 * i + 3)
        jf = JointFit(cfg, "Mx", [thr], R["cov"][s, s], False, lo, hi)
        c_best, _ = jf.fit(R["nominal"][s], M_MX)
        curves = [jf.f_phys(c, M_MX) for c, ok in (jf.fit(toys[j, s], M_MX, c0=c_best) for j in idx) if ok]
        out[thr] = dict(x=jf.x, f=jf.f_phys(c_best, M_MX), band=band(curves),
                        chi2=jf.chi2(c_best, jf.target(R["nominal"][s]), M_MX))
    return out
