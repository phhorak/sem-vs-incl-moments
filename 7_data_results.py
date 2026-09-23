#!/usr/bin/env python3
"""Step 7: gap density from the measured moments.

  Joint fit     El, q2: one density f(t) ~ t^a (1-t)^b exp(sum_k c_k t^k) fit by chi2 to the
                conditional residual moments at all thresholds, weighted by the step-5 covariance.
  Mx            separate fits per El threshold (Mx moments are El-selected sub-populations).
  HQE           exact Hausdorff + MaxEnt inversion of the HQE-fit moments at threshold 0
                (Markus Prim's likelihood toys), with and without SEM subtraction.
  Feasibility   Hausdorff check + MaxEnt inversion per threshold (TeX table).

All bands are 68% toy intervals: the step-5 residual toys ("all" source: exp + SEM stat + FF +
BF-mode + B_gap) refit per toy; for HQE, HQE toys paired with step-5 SEM toys.

Usage:
  python3 7_data_results.py --submit
  python3 7_data_results.py [--n-toys 1000]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import yaml
from scipy.optimize import minimize

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from lib.asimov import MAX_ORDER
from lib.maxent import MaxEnt, hausdorff_check, raw_to_mu01, raw_to_mu01_jacobian
from lib.systematics import c_true, load_hqe_raw, read_parquet_downcast, write_budget_tex

HQE_TOYS = HERE / "inputs" / "hqe_likelihood_toys" / "likelihood_toys.h5"
SOURCES = ["stat_ff", "stat_bfmode", "stat_ff_bfmode", "all"]
OBS = ["Mx", "El", "Q2"]
VAR = {"Mx": "mx2", "El": "el", "Q2": "q2"}
M_JOINT, M_MX = 6, 2
L2 = 1e-4          # guards the flat chi2 directions of the 1-threshold Mx fits
BAD_CLOSURE = 0.3  # a toy refit this far from its target sits in a degenerate basin


# ── Joint chi2 fit ───────────────────────────────────────────────────────────

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


def load_res(out5, obs, src):
    d = json.load(open(out5 / f"residual_covariance_{obs}_{src}.json"))
    return dict(nominal=np.array(d["nominal"]), cov=np.array(d["cov"]),
                thr=sorted({p["thr"] for p in d["points"]}))


def band(curves):
    curves = np.asarray(curves)
    return np.percentile(curves, [16, 84], axis=0) if len(curves) > 10 else None


def joint_results(cfg, out5, n_toys, rng):
    sup, res = cfg["data"]["support"], {}
    for obs in ("El", "Q2"):
        lo, hi = sup[VAR[obs]]
        if obs == "Q2":
            lo = cfg["data"]["joint_fit_q2_lo"]
        R = load_res(out5, obs, "all")
        jf = JointFit(cfg, obs, R["thr"], R["cov"], True, lo, hi)
        c_best, _ = jf.fit(R["nominal"], M_JOINT)
        chi2 = jf.chi2(c_best, jf.target(R["nominal"]), M_JOINT)
        dof = 3 * len(R["thr"]) - M_JOINT
        asimov = JointFit(cfg, obs, R["thr"], None, True, lo, hi)
        c_a, _ = asimov.fit(R["nominal"], M_JOINT)
        per_source = {}
        for src in SOURCES:
            S = load_res(out5, obs, src)
            j = JointFit(cfg, obs, S["thr"], S["cov"], True, lo, hi)
            c, _ = j.fit(S["nominal"], M_JOINT)
            per_source[src] = j.chi2(c, j.target(S["nominal"]), M_JOINT) / dof
        toys = np.load(out5 / f"toy_ensemble_{obs}_all.npz")["raw_gap"]
        idx = rng.choice(len(toys), min(n_toys, len(toys)), replace=False)
        t0, curves = time.time(), []
        for i in idx:
            c, ok = jf.fit(toys[i], M_JOINT, c0=c_best)
            if ok:
                curves.append(jf.f_phys(c, M_JOINT))
        print(f"[{obs}] joint fit chi2/ndf = {chi2:.1f}/{dof}; {len(curves)}/{len(idx)} toys refit "
              f"in {time.time() - t0:.0f}s; per source {per_source}", flush=True)
        res[obs] = dict(x=jf.x, f=jf.f_phys(c_best, M_JOINT), band=band(curves), chi2=chi2, dof=dof,
                        asimov_chi2=asimov.chi2(c_a, asimov.target(R["nominal"]), M_JOINT),
                        chi2_per_dof_by_source=per_source, n_toys=len(idx), n_ok=len(curves))
    return res


def mx_results(cfg, out5, n_toys, rng):
    lo, hi = cfg["data"]["support"]["mx2"]
    R = load_res(out5, "Mx", "all")
    toys = np.load(out5 / "toy_ensemble_Mx_all.npz")["raw_gap"]
    idx = rng.choice(len(toys), min(n_toys, len(toys)), replace=False)
    out = {}
    for i, thr in enumerate(R["thr"]):
        s = slice(3 * i, 3 * i + 3)
        jf = JointFit(cfg, "Mx", [thr], R["cov"][s, s], False, lo, hi)
        c_best, _ = jf.fit(R["nominal"][s], M_MX)
        curves = [jf.f_phys(c, M_MX) for c, ok in (jf.fit(toys[j, s], M_MX, c0=c_best) for j in idx) if ok]
        out[thr] = dict(x=jf.x, f=jf.f_phys(c_best, M_MX), band=band(curves),
                        chi2=jf.chi2(c_best, jf.target(R["nominal"][s]), M_MX))
    print(f"[Mx] {len(out)} per-threshold fits", flush=True)
    return out


# ── HQE inversion ────────────────────────────────────────────────────────────

def hqe_results(cfg, T, n_toys, rng, subtract):
    """Exact inversion at threshold 0. subtract=True: gap density (HQE toy x SEM toy x B_gap);
    False: full inclusive density, HQE toys only."""
    raw_c9, raw_t9 = load_hqe_raw(HQE_TOYS)
    mc, sysc, sup = cfg["maxent"], cfg["systematics"], cfg["data"]["support"]
    solver = MaxEnt(np.linspace(0, 1, 400), mom_tol=mc["mom_tol"])
    bf_gap = float(T["bf_gap"])
    out = {}
    for k, obs in enumerate(OBS):
        v = VAR[obs]
        lo, hi = sup[v]
        a, b = mc["alpha"][v], mc["beta"][v]

        def invert(raw):
            m = raw_to_mu01(np.r_[1.0, raw], lo, hi)
            if not hausdorff_check(m)[0]:
                return None, False
            r = solver.solve(m, a, b)
            return r.f / (hi - lo), r.converged

        sem_nom, sem_toys = T["nom_sem0"][MAX_ORDER * k:MAX_ORDER * k + 3], \
            T["sem0_stat_ff_bfmode"][:, MAX_ORDER * k:MAX_ORDER * k + 3]
        c0 = float(c_true(sysc, bf_gap))
        raw_c, hqe = raw_c9[3 * k:3 * k + 3], raw_t9[:, 3 * k:3 * k + 3]
        f_c, ok_c = invert(c0 * raw_c - (c0 - 1) * sem_nom if subtract else raw_c)
        n_hqe = len(hqe)
        n = min(n_toys, n_hqe, len(sem_toys))
        ih, js = rng.choice(n_hqe, n, replace=False), rng.choice(len(sem_toys), n, replace=False)
        curves, n_haus = [], 0
        for i, j in zip(ih, js):
            raw = hqe[i]
            if subtract:
                c = float(c_true(sysc, bf_gap, T["z_gap"][j]))
                raw = c * raw - (c - 1) * sem_toys[j]
            f, ok = invert(raw)
            n_haus += f is not None
            if ok:
                curves.append(f)
        out[obs] = dict(x=lo + solver.t * (hi - lo), f=f_c if ok_c else None, band=band(curves),
                        hausdorff_frac=n_haus / n, converged_frac=len(curves) / n)
        print(f"[HQE {'gap' if subtract else 'full'} {obs}] central {'ok' if ok_c else 'FAILED'}, "
              f"Hausdorff {n_haus}/{n}, converged {len(curves)}/{n}", flush=True)
    return out


def hqe_budget(cfg, T, orders=(1, 2, 3)):
    """Relative uncertainty [%] of the threshold-0 residual gap moments per source (HQE toys,
    SEM template, B_gap), from all toys; FF and BF-mode are taken on top of SEM stat."""
    sysc, bf_gap = cfg["systematics"], float(T["bf_gap"])
    raw_c9, raw_t9 = load_hqe_raw(HQE_TOYS)
    n = len(T["z_gap"])
    raw_h = raw_t9[np.random.default_rng(7).choice(len(raw_t9), n, replace=n > len(raw_t9))]
    c0, ct = float(c_true(sysc, bf_gap)), c_true(sysc, bf_gap, T["z_gap"])[:, None]
    sem_cols = [MAX_ORDER * k + o - 1 for k in range(3) for o in orders]
    hqe_cols = [3 * k + o - 1 for k in range(3) for o in orders]
    sem_nom, raw_c = T["nom_sem0"][sem_cols], raw_c9[hqe_cols]
    sem = lambda v: T[f"sem0_{v}"][:, sem_cols]
    nom = c0 * raw_c - (c0 - 1) * sem_nom
    var = lambda r: r.var(0)
    v_stat = var(c0 * raw_c - (c0 - 1) * sem("stat"))
    rel = lambda v: 100 * np.sqrt(np.clip(v, 0, None)) / np.abs(nom)
    rows = {"hqe": rel(var(c0 * raw_h[:, hqe_cols] - (c0 - 1) * sem_nom)),
            "stat": rel(v_stat),
            "ff": rel(var(c0 * raw_c - (c0 - 1) * sem("stat_ff")) - v_stat),
            "bf_mode": rel(var(c0 * raw_c - (c0 - 1) * sem("stat_bfmode")) - v_stat),
            "bf_gap": rel(var(ct * raw_c - (ct - 1) * sem_nom))}
    rows["quadrature"] = np.sqrt(sum(r ** 2 for r in rows.values()))
    rows["total"] = rel(var(ct * raw_h[:, hqe_cols] - (ct - 1) * sem("stat_ff_bfmode")))
    return {k: v.tolist() for k, v in rows.items()}


BUDGET_LABELS = {"hqe": "HQE fit", "stat": "MC stat.", "ff": "Form factors", "bf_mode": r"$\mathcal{B}$ per mode",
                 "bf_gap": r"$\mathcal{B}_\mathrm{gap}$", "quadrature": "Quadrature sum", "total": "Total"}


# ── Feasibility table ────────────────────────────────────────────────────────

def feasibility(cfg, out5):
    """Per threshold: residual [0,1] moments on the data support, Hausdorff and MaxEnt (N=4)."""
    mc, sup = cfg["maxent"], cfg["data"]["support"]
    solver = MaxEnt(np.linspace(0, 1, mc["grid_size"]), mom_tol=mc["mom_tol"])
    rows = []
    for obs in OBS:
        v = VAR[obs]
        lo, hi = sup[v]
        R = load_res(out5, obs, "all")
        for i, thr in enumerate(R["thr"]):
            m = raw_to_mu01(np.r_[1.0, R["nominal"][3 * i:3 * i + 3]], lo, hi)
            haus = hausdorff_check(m)[0]
            conv = haus and solver.solve(m, mc["alpha"][v], mc["beta"][v]).converged
            rows.append(dict(obs=obs, thr=thr, m=m[1:].tolist(), var=float(m[2] - m[1] ** 2),
                             hausdorff=bool(haus), maxent=bool(conv)))
    return rows


def write_feasibility_tex(rows, path):
    name = {"Mx": r"$M_X^2$", "El": r"$E_\ell$", "Q2": r"$q^2$"}
    ck = lambda b: r"$\checkmark$" if b else r"$\times$"
    lines = [r"\begin{tabular}{cccccccc}", r"\toprule",
             r"Observable & Threshold & $m_1^\text{gap}$ & $m_2^\text{gap}$ & $m_3^\text{gap}$ & "
             r"$\mathrm{Var}[t]$ & Hausdorff & MaxEnt \\", r"\midrule"]
    for r in rows:
        lines.append(f"{name[r['obs']]} & {r['thr']:.2f} & " + " & ".join(f"{x:.4f}" for x in r["m"])
                     + f" & {r['var']:+.2e} & {ck(r['hausdorff'])} & {ck(r['maxent'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


# ── Figures ──────────────────────────────────────────────────────────────────

FIT_COLOR, HQE_COLOR = "#1f77b4", "#d62728"
CATEGORIES = ["D", "D*", "D**", "D(*) pi", "D(*) pi pi", "Ds(*) K"]
CAT_LABELS = [r"$D$", r"$D^*$", r"$D^{**}$", r"$D^{(*)}\pi$", r"$D^{(*)}\pi\pi$", r"$D_s^{(*)}K$"]
CAT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
XLABEL = {"Mx": r"$M_X\ [\mathrm{GeV}]$", "El": r"$E_\ell^B\ [\mathrm{GeV}]$", "Q2": r"$q^2\ [\mathrm{GeV}^2]$"}
YLABEL = {"Mx": r"$f(M_X)\ [\mathrm{GeV}^{-1}]$", "El": r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$",
          "Q2": r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"}
FS_LABEL, FS_TICK, FS_LEG = 26, 19, 20


def style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(True)


def to_mx(x, *ys):
    """Mx2 grid and densities -> Mx grid and densities (Jacobian 2 Mx)."""
    mx = np.sqrt(x)
    return (mx, *[None if y is None else y * 2 * mx for y in ys])


def figure(plt):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.2), dpi=200)
    for ax, obs in zip(axes, OBS):
        ax.set_xlabel(XLABEL[obs], fontsize=FS_LABEL)
        ax.set_ylabel(YLABEL[obs], fontsize=FS_LABEL)
        style(ax)
    return fig, axes


def finish(fig, axes, handles, path, ymax=None):
    for i, ax in enumerate(axes):
        ax.set_ylim(0, None if ymax is None else ymax[i])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FS_LEG,
               bbox_to_anchor=(0.5, -0.11), handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def draw_curve(ax, x, f, bnd, color, obs, ls="-"):
    if obs == "Mx":
        x, f, lo_, hi_ = to_mx(x, f, *(bnd if bnd is not None else (None, None)))
        bnd = None if lo_ is None else (lo_, hi_)
    if bnd is not None:
        ax.fill_between(x, bnd[0], bnd[1], color=color, alpha=0.3, lw=0)
    if f is not None:
        ax.plot(x, f, color=color, lw=2.6, ls=ls)
    return x


def plot_gap(plt, cfg, joint, mx, hqe_gap, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib as mpl
    fig, axes = figure(plt)
    cmap, thrs = plt.get_cmap("viridis"), sorted(mx)
    norm = mpl.colors.Normalize(min(thrs), max(thrs))
    for thr in thrs:
        m = mx[thr]
        xg, fg = to_mx(m["x"], m["f"])
        axes[0].plot(xg, fg, color=cmap(norm(thr)), lw=1.6, alpha=0.9)
    cax = axes[0].inset_axes([0.55, 0.9, 0.4, 0.04])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    cb.set_label(r"$E_\ell$ cut [GeV]", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    axes[0].set_xlim(1.75, 3.0)
    for ax, obs in zip(axes, OBS):
        h = hqe_gap[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], HQE_COLOR, obs)
        if obs in joint:
            j = joint[obs]
            draw_curve(ax, j["x"], j["f"], j["band"], FIT_COLOR, obs)
            ax.text(0.06, 0.94, rf"$\chi^2/\mathrm{{ndf}} = {j['chi2']:.1f}/{j['dof']}$", transform=ax.transAxes,
                    fontsize=20, color=FIT_COLOR, ha="left", va="top")
            ax.set_xlim(j["x"][0] if obs == "El" else 0.0, j["x"][-1])
    if hqe_gap["Mx"]["f"] is None:
        axes[0].text(0.95, 0.74, "HQE moments not\nHausdorff-feasible", transform=axes[0].transAxes,
                     fontsize=17, color=HQE_COLOR, ha="right", va="top")
    sel = lambda x: (x >= 1.75) & (x <= 3.0)
    ymx = max(float(np.max((m["f"] * 2 * np.sqrt(m["x"]))[sel(np.sqrt(m["x"]))])) for m in mx.values())
    yel = max(np.max(v) for v in [joint["El"]["f"], hqe_gap["El"]["f"]] +
              [b[1] for b in (joint["El"]["band"], hqe_gap["El"]["band"]) if b is not None] if v is not None)
    handles = [Line2D([], [], color=FIT_COLOR, lw=2.6, label=r"Joint $\chi^2$ fit"),
               Line2D([], [], color="0.35", lw=1.6, label=r"Per-$E_\ell$-cut fits ($M_X$)"),
               Line2D([], [], color=HQE_COLOR, lw=2.6, label="HQE inversion"),
               Patch(color="0.5", alpha=0.3, lw=0, label="68% (toys)")]
    finish(fig, axes, handles, path, ymax=[1.35 * ymx, 1.3 * yel, 0.7])
    plt.close(fig)


def plot_hqe_full(plt, hqe_full, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = figure(plt)
    for ax, obs in zip(axes, OBS):
        h = hqe_full[obs]
        draw_curve(ax, h["x"], h["f"], h["band"], FIT_COLOR, obs)
        ax.set_xlim(h["x"][0], h["x"][-1])
    axes[0].set_xlim(1.6, 4.0)
    handles = [Line2D([], [], color=FIT_COLOR, lw=2.6, label="HQE inversion (no SEM subtraction)"),
               Patch(color=FIT_COLOR, alpha=0.3, lw=0, label="68% (HQE toys)")]
    finish(fig, axes, handles, path)
    plt.close(fig)


def plot_hqe_vs_stack(plt, cfg, T, hqe_full, path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    df = read_parquet_downcast(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet",
                               ["Mx", "El_B", "q2", "total_weight", "category"], category_cols=("category",))
    df = df[np.isfinite(df[["Mx", "El_B", "q2", "total_weight"]].to_numpy(float)).all(axis=1)
            & (df["total_weight"] > 0)]
    col = {"Mx": "Mx", "El": "El_B", "Q2": "q2"}
    fig, axes = figure(plt)
    for ax, obs in zip(axes, OBS):
        e = T[f"stackedges_{obs}"]
        wsum = df["total_weight"].sum()
        base = np.zeros(len(e) - 1)
        for cat, color in zip(CATEGORIES, CAT_COLORS):
            m = (df["category"] == cat).to_numpy()
            h = np.histogram(df[col[obs]].to_numpy(float)[m], e, weights=df["total_weight"].to_numpy(float)[m])[0]
            h = h / (wsum * np.diff(e))
            ax.stairs(base + h, e, baseline=base, fill=True, color=color, alpha=0.85, lw=0)
            base = base + h
        lo_, hi_ = np.percentile(T[f"stack_{obs}"], [16, 84], axis=0)
        ax.stairs(hi_, e, baseline=lo_, fill=True, color="0.15", alpha=0.45, lw=0, zorder=4)
        h = hqe_full[obs]
        x, f, b0, b1 = h["x"], h["f"], *(h["band"] if h["band"] is not None else (None, None))
        if obs == "Mx":
            x, f, b0, b1 = to_mx(x, f, b0, b1)
        if b0 is not None:
            ax.fill_between(x, b0, b1, facecolor="none", edgecolor="black", hatch="///", lw=0, zorder=5)
        if f is not None:
            ax.plot(x, f, color="black", lw=2.6, zorder=6)
        ax.set_xlim(e[0], e[-1])
    axes[0].set_xlim(1.6, 4.0)
    hm = hqe_full["Mx"]
    top = [y for y in (hm["f"], None if hm["band"] is None else hm["band"][1]) if y is not None]
    ymax = [1.4 * max(float(np.max(y * 2 * np.sqrt(hm["x"]))) for y in top) if top else None, None, None]
    handles = [Patch(color=c, alpha=0.85, lw=0, label=l) for c, l in zip(CAT_COLORS, CAT_LABELS)]
    handles += [Patch(color="0.15", alpha=0.45, lw=0, label="SEM 68%"),
                Line2D([], [], color="black", lw=2.6, label="HQE inversion"),
                Patch(facecolor="none", edgecolor="black", hatch="///", label="HQE 68%")]
    finish(fig, axes, handles, path, ymax=ymax)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(cfg, n_toys, seed):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401  (house style)
    import matplotlib.pyplot as plt

    out5 = Path(cfg["paths"]["output"]) / "5"
    od, fd = Path(cfg["paths"]["output"]) / "7", HERE / "figures" / "7"
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    T = dict(np.load(out5 / "toys.npz"))
    rng = np.random.default_rng(seed)

    budget = hqe_budget(cfg, T)
    write_budget_tex(budget, od / "moment_budget_data_thr0.tex", BUDGET_LABELS)
    rows = feasibility(cfg, out5)
    write_feasibility_tex(rows, od / "feasibility_table.tex")
    joint = joint_results(cfg, out5, n_toys, rng)
    mx = mx_results(cfg, out5, n_toys, rng)
    hqe_gap = hqe_results(cfg, T, n_toys, rng, subtract=True)
    hqe_full = hqe_results(cfg, T, n_toys, rng, subtract=False)

    plot_gap(plt, cfg, joint, mx, hqe_gap, fd / "gap_density.pdf")
    plot_hqe_full(plt, hqe_full, fd / "hqe_inclusive_density.pdf")
    plot_hqe_vs_stack(plt, cfg, T, hqe_full, fd / "hqe_vs_sem_stack.pdf")

    summary = dict(
        joint={o: {k: v for k, v in r.items() if k not in ("x", "f", "band")} for o, r in joint.items()},
        mx={f"{t:.2f}": dict(chi2=r["chi2"]) for t, r in mx.items()},
        hqe_gap={o: {k: r[k] for k in ("hausdorff_frac", "converged_frac")} | {"central_ok": r["f"] is not None}
                 for o, r in hqe_gap.items()},
        hqe_full={o: {k: r[k] for k in ("hausdorff_frac", "converged_frac")} | {"central_ok": r["f"] is not None}
                  for o, r in hqe_full.items()},
        moment_budget_data_thr0_pct=budget, feasibility=rows)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
    print(f"-> {fd}, {od}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--n-toys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after", help="LSF job name the submitted job waits for")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.submit:
        logs = HERE / "logs" / "7"
        logs.mkdir(parents=True, exist_ok=True)
        dep = f' -w "done({args.after})"' if args.after else ""
        cmd = (f'bsub -q {cfg["generation"]["queue"]} -env all -J s7data{dep} -n 4 -oo {logs}/run.log '
               f'"cd {HERE} && python3 7_data_results.py --config {args.config} --n-toys {args.n_toys}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        run(cfg, args.n_toys, args.seed)


if __name__ == "__main__":
    main()
