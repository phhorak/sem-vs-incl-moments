#!/usr/bin/env python3
"""
6_joint_fit.py

Step 6: the actual MaxEnt analysis, reading step 5's residual moments/covariance (no toy
generation here -- pure local computation).

  - El, Q2: joint multi-threshold MaxEnt fit. One shared 6-parameter exponentiated-polynomial
    density per observable, fit against all of that observable's thresholds at once, weighted
    by the full (exp+stat+FF+BR) residual covariance from step 5. Includes an Asimov
    (zero-covariance) closure check, the real fit with a delta-method uncertainty band, and a
    per-source (stat_ff / stat_bfmode / stat_ff_bfmode) chi2/dof breakdown for backtracking.

  - Mx: separate, independent per-threshold inversions (NOT a joint fit) -- Mx's thresholds are
    cuts on El, not on Mx2 itself, and this branch's own joint fit showed a systematic
    threshold-dependent trend (chi2/dof~4.3, insensitive to more polynomial flexibility) that
    looks like the conditional Mx2 shape genuinely shifts with the El cut. Each threshold gets
    its own small (M=2) density fit from its own 3 moments + 3x3 covariance block, with its own
    delta-method band -- the old output/10/maxent_data_inversions.json did the equivalent exact
    Lagrangian inversion but with NO covariance at all (point-estimate only, no band); this adds
    the band, made possible because step 5 now gives Mx a real per-threshold covariance.

Usage:
  python3 6_joint_fit.py --config config.yaml
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

OBS_LIST = ["Mx", "El", "Q2"]
SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode"]

# Physics-derived endpoint powers g(t) ~ t^alpha (1-t)^beta (sec:idea+approach derivation --
# see config_stat.yaml's boundary_alpha/beta comment block). NOT config.yaml's
# hausdorff_data.boundary_alpha/beta, which are placeholders without that derivation.
PHYS_BOUNDARY = {"Mx": dict(alpha=0.0, beta=0.0), "El": dict(alpha=2.0, beta=3.5),
                 "Q2": dict(alpha=0.0, beta=1.5)}


def load_common(cfg):
    hd = cfg["hausdorff_data"]
    out_root = Path(cfg["paths"]["output"])
    exp_avg = json.load(open(out_root / "4" / "experimental_average.json"))
    sem_raw = json.load(open(out_root / "5" / "sem_moments_raw.json"))
    avg_raw = exp_avg["average_raw"]
    sem_nom_raw = sem_raw["sem_nominal_raw"]
    sem_el_cuts = np.array(sem_raw["cuts"]["el"])
    sem_q2_cuts = np.array(sem_raw["cuts"]["q2"])

    mx2_lo, mx2_hi = float(hd["mx_support"][0]), float(hd["mx_support"][1])
    el_lo, el_hi = float(hd["el_support"][0]), float(hd["el_support"][1])
    # q2_support[0] from config.yaml is 0.0, but the lowest actual measured q2 threshold is
    # 1.5 GeV^2 -- below that the fit has no data constraining it at all, which is what let the
    # uncertainty band run wild near q2=0. Restrict the density's own support to where data
    # exists instead.
    q2_lo, q2_hi = 1.5, float(hd["q2_support"][1])

    OBS_CFG = {
        "Mx": dict(keys=["mx_1", "mx_2", "mx_3"], cut_ref=sem_el_cuts, lo=mx2_lo, hi=mx2_hi,
                   thr_lo=el_lo, thr_hi=el_hi, masked=False,
                   xlabel=r"$M_X^2\ [\mathrm{GeV}^2]$", ylabel=r"$f(M_X^2)$"),
        "El": dict(keys=["el_1", "el_2", "el_3"], cut_ref=sem_el_cuts, lo=el_lo, hi=el_hi,
                   thr_lo=el_lo, thr_hi=el_hi, masked=True,
                   xlabel=r"$E_\ell\ [\mathrm{GeV}]$", ylabel=r"$f(E_\ell)$"),
        "Q2": dict(keys=["q2_1", "q2_2", "q2_3"], cut_ref=sem_q2_cuts, lo=q2_lo, hi=q2_hi,
                   thr_lo=q2_lo, thr_hi=q2_hi, masked=True,
                   xlabel=r"$q^2\ [\mathrm{GeV}^2]$", ylabel=r"$f(q^2)$"),
    }
    for obs in OBS_LIST:
        OBS_CFG[obs].update(PHYS_BOUNDARY[obs])
    return dict(avg_raw=avg_raw, sem_nom_raw=sem_nom_raw, OBS_CFG=OBS_CFG)


def raw_to_mu01(raw, lo, hi):
    span_ = hi - lo
    nmax = len(raw)
    mu = np.zeros(nmax)
    for k in range(nmax):
        mu[k] = sum(comb(k, j) * (-lo) ** (k - j) * raw[j] for j in range(k + 1)) / span_ ** k
    return mu


def raw_to_mu01_jacobian(lo, hi, nmax):
    span_ = hi - lo
    J = np.zeros((nmax, nmax))
    for k in range(nmax):
        for j in range(k + 1):
            J[k, j] = comb(k, j) * (-lo) ** (k - j) / span_ ** k
    return J


def fit_maxent(obs, C, thresholds, nominal, Cov_raw_gap, M, t_grid_n=400):
    """Core masked/unmasked-basis chi2 fit. `nominal` is the flat (n_thr*3,) residual raw-gap
    vector (order 1..3 per threshold), matching step 5's schema. Cov_raw_gap=None -> identity
    weighting (Asimov / no-uncertainty closure check)."""
    cfg_o = C["OBS_CFG"][obs]
    lo, hi, alpha, beta, masked = cfg_o["lo"], cfg_o["hi"], cfg_o["alpha"], cfg_o["beta"], cfg_o["masked"]
    span = hi - lo
    n_thr = len(thresholds)
    N = 4  # orders 0..3 (order 0 implicit)

    J_full = raw_to_mu01_jacobian(lo, hi, N)
    J_sub = J_full[1:, 1:]
    raw_gap = {thr: nominal[3*i:3*i+3] for i, thr in enumerate(thresholds)}
    mu_cond_t = {thr: raw_to_mu01(np.concatenate([[1.0], raw_gap[thr]]), lo, hi)[1:] for thr in thresholds}
    target = np.concatenate([mu_cond_t[thr] for thr in thresholds])
    labels = [(thr, k) for thr in thresholds for k in range(1, N)]

    if Cov_raw_gap is None:
        Cinv = np.eye(len(target))
        Ct = np.eye(len(target))
        n_floored = 0
    else:
        Ct = np.zeros((3 * n_thr, 3 * n_thr))
        for i in range(n_thr):
            for j in range(n_thr):
                Ct[3*i:3*i+3, 3*j:3*j+3] = J_sub @ Cov_raw_gap[3*i:3*i+3, 3*j:3*j+3] @ J_sub.T
        w_eig, v_eig = np.linalg.eigh(Ct)
        floor_val = 1e-3 * w_eig.max()
        n_floored = int(np.sum(w_eig < floor_val))
        w_reg = np.maximum(w_eig, floor_val)
        Cinv = (v_eig / w_reg) @ v_eig.T

    t_grid = np.linspace(0.0, 1.0, t_grid_n)
    eps = 1e-300
    prior = (np.maximum(t_grid, eps) ** alpha) * (np.maximum(1.0 - t_grid, eps) ** beta)
    mask_by_thr = {thr: (t_grid >= (thr - lo) / span).astype(float) for thr in thresholds} if masked \
        else {thr: np.ones_like(t_grid) for thr in thresholds}

    def model_moments(c):
        tp = np.stack([t_grid ** k for k in range(1, M + 1)])
        expo = c @ tp; expo -= expo.max()
        f = prior * np.exp(expo)
        out = np.empty(len(labels))
        for i, (thr, k) in enumerate(labels):
            mask = mask_by_thr[thr]
            denom = np.trapz(f * mask, t_grid)
            numer = np.trapz((t_grid ** k) * f * mask, t_grid)
            out[i] = numer / denom if denom > 0 else np.nan
        return out

    # Small L2 penalty on the coefficients: with few thresholds (esp. Mx's single-threshold,
    # dof=1 inversions) the chi2 surface can have a genuinely flat direction the data doesn't
    # constrain at all, which an unregularized optimizer wanders arbitrarily far along for
    # free (found empirically: several Mx thresholds converged to identical |c|~1e6-1e7
    # "solutions" with no better chi2 than a much smaller, sane c). This is negligible for
    # well-constrained fits (El/Q2's coefficients stay O(1-10)) and only bites the runaway case.
    l2 = 1e-4

    def chi2(c):
        r = model_moments(c) - target
        return float(r @ Cinv @ r) + l2 * float(np.sum(c ** 2))

    res = minimize(chi2, np.zeros(M), method="Nelder-Mead",
                    options={"maxiter": 30000, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True})
    res2 = minimize(chi2, res.x, method="BFGS", options={"maxiter": 5000, "gtol": 1e-10})
    c_best = res2.x if res2.fun < res.fun else res.x
    r_best = model_moments(c_best) - target
    chi2_best = float(r_best @ Cinv @ r_best)  # raw chi2 for reporting, without the L2 penalty
    dof = len(target) - M

    def f_phys_of_c(c):
        tp = np.stack([t_grid ** k for k in range(1, M + 1)])
        expo = c @ tp; expo -= expo.max()
        f = prior * np.exp(expo)
        Z = np.trapz(f, t_grid)
        return (f / Z) / span

    f_best = f_phys_of_c(c_best)
    x_grid = lo + t_grid * span

    sig_f = None
    if Cov_raw_gap is not None:
        h = 1e-4
        Jr = np.zeros((len(target), M))
        Jf = np.zeros((len(t_grid), M))
        for k in range(M):
            cp = c_best.copy(); cp[k] += h
            cm = c_best.copy(); cm[k] -= h
            Jr[:, k] = (model_moments(cp) - model_moments(cm)) / (2 * h)
            Jf[:, k] = (f_phys_of_c(cp) - f_phys_of_c(cm)) / (2 * h)
        try:
            Cov_c = np.linalg.inv(Jr.T @ Cinv @ Jr)
            var_f = np.einsum("ij,jk,ik->i", Jf, Cov_c, Jf)
            sig_f = np.sqrt(np.maximum(var_f, 0))
        except np.linalg.LinAlgError:
            sig_f = None

    return dict(chi2=chi2_best, dof=dof, n_floored=n_floored, c_best=c_best,
                x_grid=x_grid, f_best=f_best, sig_f=sig_f, thresholds=thresholds)


def load_source_cov(out5, obs, source):
    path = out5 / f"residual_covariance_{obs}_{source}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    return dict(points=d["points"], mean=np.array(d["mean"]), cov=np.array(d["cov"]),
                nominal=np.array(d["nominal"]), thresholds=sorted(set(p["thr"] for p in d["points"])))


def run_joint(cfg, out5, out6, fig6, plt, obs, C):
    """El / Q2: joint multi-threshold fit."""
    full = load_source_cov(out5, obs, "stat_ff_bfmode")
    if full is None:
        print(f"[{obs}] no step-5 output found, skipping"); return None
    thresholds = full["thresholds"]

    asimov = fit_maxent(obs, C, thresholds, full["nominal"], None, M=6)
    print(f"[asimov/{obs}] chi2={asimov['chi2']:.3e}  dof={asimov['dof']}")

    fit = fit_maxent(obs, C, thresholds, full["nominal"], full["cov"], M=6)
    print(f"[{obs}] joint fit: chi2={fit['chi2']:.2f}  dof={fit['dof']}  "
          f"chi2/dof={fit['chi2']/fit['dof']:.3f}  floored {fit['n_floored']}")

    per_source = {}
    for source in SOURCES:
        sc = load_source_cov(out5, obs, source)
        if sc is None:
            continue
        r = fit_maxent(obs, C, thresholds, sc["nominal"], sc["cov"], M=6)
        per_source[source] = dict(chi2=r["chi2"], dof=r["dof"], chi2_per_dof=r["chi2"] / r["dof"])
        print(f"  [{obs}/{source}] chi2/dof={r['chi2']/r['dof']:.3f}")

    cfg_o = C["OBS_CFG"][obs]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(fit["x_grid"], fit["f_best"], color="black", lw=2, label="best fit")
    if fit["sig_f"] is not None:
        ax.fill_between(fit["x_grid"], fit["f_best"] - fit["sig_f"], fit["f_best"] + fit["sig_f"],
                         color="C0", alpha=0.3, label=r"$\pm1\sigma$ (exp. + stat. + FF + BR)")
    for thr in thresholds:
        ax.axvline(thr, color="0.6", lw=0.6, ls="--", zorder=0)
    top = float(np.max(fit["f_best"] + (fit["sig_f"] if fit["sig_f"] is not None else 0)))
    ax.set_ylim(top=1.55 * top)
    formula = (r"$f(t) \propto t^{\alpha}(1-t)^{\beta}\,\exp\!\left(\sum_{k=1}^{6} c_k\, t^k\right)$" "\n"
               rf"$\alpha={cfg_o['alpha']:g},\ \beta={cfg_o['beta']:g}$")
    ax.text(0.03, 0.97, formula, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    ax.set_xlabel(cfg_o["xlabel"]); ax.set_ylabel(cfg_o["ylabel"])
    ax.set_title(rf"{obs} fit, $\chi^2/\mathrm{{ndf}} = {fit['chi2']:.2f}/{fit['dof']} = "
                 rf"{fit['chi2']/fit['dof']:.2f}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig_path = fig6 / f"joint_fit_{obs}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"-> {fig_path}")

    np.savez(out6 / f"joint_fit_{obs}.npz", x_grid=fit["x_grid"], f_best=fit["f_best"],
             sig_f=fit["sig_f"], c_best=fit["c_best"])
    return dict(n_thr=len(thresholds), asimov_chi2=asimov["chi2"], asimov_dof=asimov["dof"],
                chi2=fit["chi2"], dof=fit["dof"], chi2_per_dof=fit["chi2"] / fit["dof"],
                c_best=fit["c_best"].tolist(), per_source=per_source)


def run_mx_inversions(cfg, out5, out6, fig6, plt, C):
    """Mx: separate, independent per-threshold inversions (not a joint fit)."""
    full = load_source_cov(out5, "Mx", "stat_ff_bfmode")
    if full is None:
        print("[Mx] no step-5 output found, skipping"); return None
    thresholds = full["thresholds"]
    M = 2

    entries = {}
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    cmap = plt.get_cmap("viridis")
    for i, thr in enumerate(thresholds):
        nom_i = full["nominal"][3*i:3*i+3]
        cov_i = full["cov"][3*i:3*i+3, 3*i:3*i+3]

        asimov = fit_maxent("Mx", C, [thr], nom_i, None, M=M)
        fit = fit_maxent("Mx", C, [thr], nom_i, cov_i, M=M)
        chi2_per_dof = fit["chi2"] / fit["dof"] if fit["dof"] > 0 else float("nan")
        print(f"[Mx/thr={thr:.1f}] asimov_chi2={asimov['chi2']:.3e}  "
              f"fit chi2/dof={chi2_per_dof:.3f}")

        entries[f"{thr:.2f}"] = dict(
            threshold=thr, asimov_chi2=asimov["chi2"], chi2=fit["chi2"], dof=fit["dof"],
            chi2_per_dof=chi2_per_dof, c_best=fit["c_best"].tolist(),
            x_grid=fit["x_grid"].tolist(), f_best=fit["f_best"].tolist(),
            sig_f=fit["sig_f"].tolist() if fit["sig_f"] is not None else None,
        )
        color = cmap(i / max(len(thresholds) - 1, 1))
        ax.plot(fit["x_grid"], fit["f_best"], color=color, lw=1.5, label=f"El>{thr:.1f}")
        if fit["sig_f"] is not None:
            ax.fill_between(fit["x_grid"], fit["f_best"] - fit["sig_f"], fit["f_best"] + fit["sig_f"],
                             color=color, alpha=0.15)

    ax.set_xlabel(C["OBS_CFG"]["Mx"]["xlabel"]); ax.set_ylabel(C["OBS_CFG"]["Mx"]["ylabel"])
    ax.set_title("Mx: separate per-threshold inversions (El-selected sub-populations)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig_path = fig6 / "inversions_Mx_overlay.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"-> {fig_path}")

    with open(out6 / "inversions_Mx.json", "w") as f:
        json.dump(entries, f, indent=2)
    print(f"-> {out6 / 'inversions_Mx.json'}")
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    C = load_common(cfg)

    out5 = Path(cfg["paths"]["output"]) / "5"
    out6 = Path(cfg["paths"]["output"]) / "6"
    fig6 = Path(cfg["paths"]["figures"]) / "6"
    out6.mkdir(parents=True, exist_ok=True)
    fig6.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt

    summary = {}
    for obs in ["El", "Q2"]:
        r = run_joint(cfg, out5, out6, fig6, plt, obs, C)
        if r is not None:
            summary[obs] = r
    mx = run_mx_inversions(cfg, out5, out6, fig6, plt, C)

    with open(out6 / "joint_fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> {out6 / 'joint_fit_summary.json'}")


if __name__ == "__main__":
    main()
