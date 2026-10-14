"""Admissibility test of moments m1..m3 on [0, 1]: distance to the moment space of probability measures.

  d^2   Mahalanobis distance (toy covariance) to the moment space; 0 when admissible. For three moments its
        boundary is the set of two-point measures w delta_e + (1-w) delta_t, e in {0, 1}: exact projection by a
        scan in t with w in closed form. Several blocks (observables): block coordinate descent, full covariance.
  p     parametric bootstrap: toy fluctuations re-centred on the closest admissible point. The toys are not
        Gaussian (c_true = B_incl/B_gap has a heavy tail) and P(chi2_k >= d^2) is smaller than the bootstrap p.
"""
import numpy as np
from scipy.stats import chi2, norm

from lib.maxent import hausdorff_check

TG = np.linspace(0, 1, 4001)
G = np.stack([TG, TG ** 2, TG ** 3], 1)


def project(target, P):
    """Closest point (metric P) to target in the moment space of probability measures on [0, 1]."""
    if hausdorff_check(np.r_[1.0, target])[0]:
        return target
    L = np.linalg.cholesky(P).T
    best = (np.inf, None)
    for e in (0.0, 1.0):
        D = np.array([e, e ** 2, e ** 3]) - G
        R, LD = (L @ (target - G).T).T, (L @ D.T).T
        w = np.clip(np.sum(R * LD, 1) / np.maximum(np.sum(LD * LD, 1), 1e-300), 0, 1)
        d2 = np.sum((R - w[:, None] * LD) ** 2, 1)
        i = int(np.argmin(d2))
        if d2[i] < best[0]:
            best = (d2[i], G[i] + w[i] * D[i])
    return best[1]


def distance(mu, P, nb, iters=50):
    """d^2 from mu (scaled moments 1..3 per block) to the product of moment spaces, metric P."""
    nu = mu.copy()
    for _ in range(iters if nb > 1 else 1):
        prev = nu.copy()
        for i in range(nb):
            s, o = slice(3 * i, 3 * i + 3), np.r_[0:3 * i, 3 * i + 3:3 * nb]
            target = mu[s] + np.linalg.solve(P[s, s], P[s][:, o] @ (mu[o] - nu[o])) if nb > 1 else mu[s]
            nu[s] = project(target, P[s, s])
        if np.max(np.abs(nu - prev)) < 1e-10:
            break
    r = mu - nu
    return float(r @ P @ r), nu


def test(mu_nom, mu_toys, power=False):
    nb = len(mu_nom) // 3
    P = np.linalg.inv(np.cov(mu_toys, rowvar=False))
    d2, nu = distance(mu_nom, P, nb)
    out = dict(d2=d2, p_chi2=float(chi2.sf(d2, len(mu_nom))))
    if d2 < 1e-9:
        return out | dict(d2=0.0, p=1.0, z=0.0)
    boot = np.sort([distance(nu + dm, P, nb)[0] for dm in mu_toys - mu_toys.mean(0)])
    p = float(np.mean(boot >= d2))
    out |= dict(p=p, z=float(norm.isf(max(p, 0.5 / len(boot)))))
    if power:
        pt = 1 - np.searchsorted(boot, [distance(m, P, nb)[0] for m in mu_toys[:500]], side="left") / len(boot)
        out |= dict(p_toys_median=float(np.median(pt)), frac_toys_p_below_0p05=float(np.mean(pt < 0.05)))
    return out
