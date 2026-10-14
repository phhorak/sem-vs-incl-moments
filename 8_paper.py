#!/usr/bin/env python3
"""Step 8: paper figures and numbers.

  figures/8/hqe_moments.pdf               measured moments vs threshold and the HQE fit at threshold 0
  figures/8/closure_n3.pdf                closure test with three moments, both gap truths (step-5 bands)
  figures/8/moment_plane_sem.pdf          gap mean vs variance with the SEM modes
  figures/8/moment_plane_candidates.pdf   the same with candidate gap modes outside the SEM
  paper/numbers.tex                       every number the paper quotes, as TeX macros

Reads the step-4 bundle, output/2 (candidate gap modes), the step 5-7 summaries and the measured moments
(inputs/sem.tar.gz). Toys: the shared pairing of lib/gap.py. Run after steps 5-7.

Usage:
  python3 8_paper.py [--bundle FILE]
  python3 8_paper.py --submit [--after JOB]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.bundle import Bundle, default_path, output_root
from lib.moments import build_exp_moments_df, df_to_plot_dict
from lib import variants as V
from lib.gap import SL, Gap
from lib.systematics import bf_gap_sigma

OBS = ["mx2", "el", "q2"]
SEM_MODES = {"D": (r"$D$", "#1f77b4", "o"), "D*": (r"$D^*$", "#d62728", "o"), "D**": (r"$D^{**}$", "#2ca02c", "^"),
             "D(*) pi pi": (r"$D^{(*)}\pi\pi$", "#9467bd", "^"), "Ds(*) K": (r"$D_s^{(*)}K$", "#8c564b", "s")}
CANDIDATES = {"Dp1DstEta": (r"$D_1'\to D^*\eta$", "#1f77b4", "o"), "D0pipipipenu": (r"$D\pi\pi\pi$", "#9467bd", "^"),
              "Dst0pipipipenu": (r"$D^*\pi\pi\pi$", "#c49cde", "^"), "DsKenu": (r"$D_sK$", "#2ca02c", "s"),
              "LcPenu": (r"$\Lambda_c\bar p$", "#d62728", "P")}


def run(B):
    import matplotlib
    matplotlib.use("Agg")
    import plothist  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg, T = B.cfg, B.toys
    sup, sysc = cfg["data"]["support"], cfg["systematics"]
    out, fd = output_root(cfg, HERE), HERE / "figures" / "8"
    fd.mkdir(parents=True, exist_ok=True)
    hc, ht = B.hqe
    tick = dict(axis="both", which="both", direction="in", top=True, right=True)

    # ── measured moments vs threshold, HQE fit at threshold 0 ──
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(HERE / cfg["data"]["exp_data_tar"]) as tf:
            tf.extractall(tmp, filter="data")
        data = df_to_plot_dict(build_exp_moments_df(Path(tmp)), "calculated")
    EXP = {"babar": ("BaBar", "#8c564b", "D"), "belle": ("Belle", "#1f77b4", "o"),
           "belleII": ("Belle II", "#ff7f0e", "s"), "cleo": ("CLEO", "#2ca02c", "^")}
    ROWS = [("mx", r"$\langle M_X^{2n}\rangle$", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
            ("el", r"$\langle E_\ell^n\rangle$", r"$E_{\ell,\mathrm{cut}}\,[\mathrm{GeV}]$"),
            ("q2", r"$\langle q^{2n}\rangle$", r"$q^2_{\mathrm{cut}}\,[\mathrm{GeV}^2]$")]
    herr = ht.std(0)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), dpi=250)
    for r, (v, ylabel, xlabel) in enumerate(ROWS):
        for c in range(3):
            ax = axes[r, c]
            for exp, e in data[f"{v}_{c + 1}"].items():
                if exp in EXP:
                    _, col, mk = EXP[exp]
                    ax.errorbar(e["cuts"], e["values"], yerr=e["errors"], ls="", marker=mk, color=col, mec="black",
                                ecolor=col, lw=1.6, ms=7, capsize=2, zorder=20)
            ax.errorbar([0.0], [hc[3 * r + c]], yerr=[herr[3 * r + c]], ls="", marker="*", color="#d62728", mec="black",
                        ecolor="#d62728", lw=2, ms=20, capsize=3, zorder=25)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=30, labelpad=8)
            ax.set_xlabel(xlabel, fontsize=28, labelpad=8)
            ax.tick_params(**tick, labelsize=22, pad=6)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.margins(x=0.08, y=0.15)
            ax.text(0.06 if r else 0.94, 0.94, rf"$n={c + 1}$", transform=ax.transAxes, fontsize=32, color="0.15",
                    ha="left" if r else "right", va="top")
    h = [Line2D([], [], marker=mk, color=col, mec="black", ls="", ms=10, label=l) for l, col, mk in EXP.values()]
    h.append(Line2D([], [], marker="*", color="#d62728", mec="black", ls="", ms=22, label=r"HQE fit, $\mathrm{cut}=0$"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=24)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.135, wspace=0.30, hspace=0.35)
    fig.savefig(fd / "hqe_moments.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── closure test with three moments ──
    spec = importlib.util.spec_from_file_location("s5", HERE / "5_closure.py")
    s5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s5)
    cfg5, T5, dev = s5.load(B)
    scens = list(cfg5["asimov"]["scenarios"])
    fig, axes = plt.subplots(len(scens), 3, figsize=(21, 6.2 * len(scens)), dpi=200)
    for r, scen in enumerate(scens):
        P = s5.Panels(cfg5, T5, dev, scen, B.truth(scen))
        for c, obs in enumerate(s5.OBS):
            P.draw(np.array([[axes[r, c]]]), obs, True, [3], headroom=1.3)
            for t in axes[r, c].texts[:]:
                if t.get_text().startswith("$N="):
                    t.remove()
        axes[r, 2].legend(handles=P.legend_handles(True, "el"), fontsize=19, loc="center left", frameon=False,
                          bbox_to_anchor=(1.03, 0.5))
    fig.tight_layout()
    fig.savefig(fd / "closure_n3.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── gap mean vs variance (threshold 0) against SEM, inclusive, and SEM or candidate modes ──
    G = Gap(B)
    c0, c = G.c0, G.c()
    mv = lambda m: np.stack([m[..., 0], m[..., 1] - m[..., 0] ** 2], -1)
    pts = {}
    for o in OBS:
        k = SL[o].start
        hm, htt, sm, st = G.inc_c[k:k + 2], G.inc_t[:, k:k + 2], G.sem_c[k:k + 2], G.sem_t[:, k:k + 2]
        pts[o] = dict(gap=mv(c0 * hm - (c0 - 1) * sm), gap_toys=mv(c * htt - (c - 1) * st), sem=mv(sm), hqe=mv(hm))
    C = V.load()
    s_dss, comp = sum(C["D**:" + k]["share"] for k in V.DSS), V.modes(C)
    refs = {"sem": {k: {o: mv(comp[k][3 * i:3 * i + 2]) for i, o in enumerate(OBS)} for k in SEM_MODES}, "candidates": {}}
    for s in CANDIDATES:
        d = pd.read_parquet(out / "2" / f"{s}.parquet", columns=["Mx", "El_B", "q2", "total_weight"])
        d = d[np.isfinite(d.to_numpy()).all(axis=1) & (d.total_weight > 0)]
        w = d.total_weight.to_numpy() / d.total_weight.sum()
        x = {"mx2": d.Mx.to_numpy() ** 2, "el": d.El_B.to_numpy(), "q2": d.q2.to_numpy()}
        refs["candidates"][s] = {o: mv(np.array([np.sum(w * x[o]), np.sum(w * x[o] ** 2)])) for o in OBS}
    LABEL = {"mx2": (r"$\langle M_X^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(M_X^2)\ [\mathrm{GeV}^4]$"),
             "el": (r"$\langle E_\ell^B\rangle\ [\mathrm{GeV}]$", r"$\mathrm{Var}(E_\ell^B)\ [\mathrm{GeV}^2]$"),
             "q2": (r"$\langle q^2\rangle\ [\mathrm{GeV}^2]$", r"$\mathrm{Var}(q^2)\ [\mathrm{GeV}^4]$")}
    GAP = "#ff7f0e"
    for tag, style in (("sem", SEM_MODES), ("candidates", CANDIDATES)):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7.4), dpi=200)
        for ax, o in zip(axes, OBS):
            P, (lo, hi) = pts[o], sup[o]
            ref = np.array([P["gap"], P["sem"], P["hqe"]] + [r[o] for r in refs[tag].values()])
            q = np.percentile(P["gap_toys"], [1, 99], axis=0)
            (x0, y0), (x1, y1) = np.minimum(ref.min(0), q[0]), np.maximum(ref.max(0), q[1])
            y0 = min(y0, 0)
            px, py = 0.08 * (x1 - x0), 0.1 * (y1 - y0)
            x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + 1.5 * py
            xg = np.linspace(x0, x1, 400)
            ax.fill_between(xg, y0, 0, color="0.88", lw=0, zorder=0)
            ax.fill_between(xg, np.maximum((xg - lo) * (hi - xg), y0), y1, color="0.88", lw=0, zorder=0)
            mu, cov = P["gap_toys"].mean(0), np.cov(P["gap_toys"], rowvar=False)
            w, v = np.linalg.eigh(cov)
            ang = np.linspace(0, 2 * np.pi, 200)
            for chi2, a in ((5.991, 0.2), (2.279, 0.45)):
                xy = mu[:, None] + v @ (np.sqrt(np.clip(w, 0, None) * chi2)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
                ax.fill(*xy, color=GAP, alpha=a, lw=0, zorder=2)
            ax.plot(*P["gap"], "o", color=GAP, ms=13, mec="k", mew=1.4, zorder=6)
            ax.plot(*P["sem"], "s", color="k", ms=12, zorder=6)
            ax.plot(*P["hqe"], "D", color="0.45", ms=12, zorder=6)
            for k, (_, col, mk) in style.items():
                ax.plot(*refs[tag][k][o], mk, color=col, ms=13, mec="k", mew=0.8, zorder=5)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xlabel(LABEL[o][0], fontsize=26)
            ax.set_ylabel(LABEL[o][1], fontsize=26)
            ax.tick_params(**tick, labelsize=19)
        h = [Patch(color=GAP, alpha=0.5, lw=0, label="Gap (68%, 95%)"),
             Line2D([], [], marker="s", color="k", ls="", ms=12, label="SEM"),
             Line2D([], [], marker="D", color="0.45", ls="", ms=12, label="Inclusive (HQE)"),
             Patch(color="0.88", lw=0, label="No positive density")]
        h += [Line2D([], [], marker=mk, color=col, mec="k", mew=0.8, ls="", ms=13, label=l) for l, col, mk in style.values()]
        fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=20, bbox_to_anchor=(0.5, -0.2),
                   handlelength=1.2, columnspacing=1.4)
        fig.tight_layout()
        fig.savefig(fd / f"moment_plane_{tag}.pdf", bbox_inches="tight")
        plt.close(fig)

    # ── numbers ──
    SC, SI, ST = (json.load(open(out / str(k) / "summary.json")) for k in (5, 6, 7))
    f0 = G.f0
    gq = {o: np.percentile(pts[o]["gap_toys"][:, 0], [16, 84]) for o in OBS}
    dmu, dvar = pts["mx2"]["hqe"] - pts["mx2"]["sem"]
    fmin2 = next(f for f in np.arange(0.01, 0.5, 1e-4)
                 if pts["mx2"]["hqe"][1] - (1 - f) * pts["mx2"]["sem"][1] - (1 - f) * dmu ** 2 / f >= 0)
    pct = lambda x: f"{100 * x:.0f}"
    d2f = lambda x: f"{x:.0f}" if x >= 100 else f"{x:.1f}"
    pf = lambda p: r"<0.001" if p == 0 else f"{p:.3f}"
    cl = ST["cdf"]
    sc = SC
    N = {"gapFrac": f"{100 * f0:.1f}", "gapFracUnc": f"{100 * bf_gap_sigma(sysc) / sysc['bf_incl']:.1f}",
         "gapFracDec": f"{f0:.3f}", "cTrue": f"{c0:.2f}", "nToys": f"{len(T['z_gap'])}", "nHqeToys": f"{len(ht)}",
         "inclMeanMx": f"{pts['mx2']['hqe'][0]:.3f}", "inclVarMx": f"{pts['mx2']['hqe'][1]:.2f}",
         "inclVarMxUnc": f"{np.std(ht[:, 1] - ht[:, 0] ** 2):.2f}",
         "inclMeanEl": f"{pts['el']['hqe'][0]:.3f}", "inclVarEl": f"{pts['el']['hqe'][1]:.3f}",
         "inclMeanQ": f"{pts['q2']['hqe'][0]:.3f}", "inclVarQ": f"{pts['q2']['hqe'][1]:.2f}",
         "semMeanMx": f"{pts['mx2']['sem'][0]:.2f}", "semVarMx": f"{pts['mx2']['sem'][1]:.2f}",
         "semMeanEl": f"{pts['el']['sem'][0]:.2f}", "semMeanQ": f"{pts['q2']['sem'][0]:.2f}",
         "gapMeanMx": f"{pts['mx2']['gap'][0]:.2f}", "gapMeanMxLo": f"{gq['mx2'][0]:.2f}", "gapMeanMxHi": f"{gq['mx2'][1]:.2f}",
         "gapMeanEl": f"{pts['el']['gap'][0]:.2f}", "gapMeanQ": f"{pts['q2']['gap'][0]:.2f}",
         "dMeanMx": f"{dmu:.2f}", "fMinTwoMoment": f"{fmin2:.2f}",
         "shareD": f"{100 * C['D']['share']:.1f}", "shareDst": f"{100 * C['D*']['share']:.1f}",
         "shareDss": f"{100 * s_dss:.1f}", "shareDpipi": f"{100 * C['D(*) pi pi']['share']:.1f}",
         "shareDsK": f"{100 * C['Ds(*) K']['share']:.1f}"}
    l1 = [sc[s][f"{o}_N3"]["l1_to_truth_constrained"] for s in sc for o in ("el", "q2")]
    out_mx = [1 - sc[s]["mx2_N3"]["total"]["hausdorff_pass"] for s in sc]
    bfg = [sc[s]["moment_budget_pct"]["bf_gap"] for s in sc]
    N |= {"closureLOneLo": f"{min(l1):.2f}", "closureLOneHi": f"{max(l1):.2f}",
          "closureOutMxLo": pct(min(out_mx)), "closureOutMxHi": pct(max(out_mx)),
          "closureOutQDstEta": pct(1 - sc["Dp1DstEta"]["q2_N3"]["total"]["hausdorff_pass"]),
          "closureBfGapMxLo": f"{min(b[0] for b in bfg):.0f}", "closureBfGapMxHi": f"{max(b[0] for b in bfg):.0f}",
          "closureBfGapQLo": f"{min(b[6] for b in bfg):.0f}", "closureBfGapQHi": f"{max(b[6] for b in bfg):.0f}",
          "gapAllowedMx": pct(SI["hqe_gap"]["Mx"]["hausdorff_frac"]),
          "inclAllowed": pct(min(v["hausdorff_frac"] for v in SI["hqe_full"].values()))}
    q = lambda o, x: cl[o][f"P(x<={x})"]
    N |= {"cdfMxEightLo": f"{q('mx2', 8.0)['nominal'][0]:.2f}", "cdfMxTenLo": f"{q('mx2', 10.0)['nominal'][0]:.2f}",
          "cdfMxTenLoToys": f"{q('mx2', 10.0)['toys68_min_max'][0]:.2f}",
          "cdfMxTenLoPct": pct(q("mx2", 10.0)["nominal"][0]),
          "cdfElHalfHi": f"{q('el', 0.5)['nominal'][1]:.2f}", "cdfElHalfHiToys": f"{q('el', 0.5)['toys68_min_max'][1]:.2f}",
          "aboveElEight": pct(1 - q("el", 0.8)["nominal"][1]),
          "cdfQFourLo": f"{q('q2', 4.0)['nominal'][0]:.2f}", "cdfQFourHi": f"{q('q2', 4.0)['nominal'][1]:.2f}",
          "cdfQSixLo": f"{q('q2', 6.0)['nominal'][0]:.2f}", "cdfQSixLoToys": f"{q('q2', 6.0)['toys68_min_max'][0]:.2f}"}
    su = ST["support"]
    N |= {"lcpOutMx": pct(su["LcP"]["mx2"]["outside_min_nominal"]), "lcpOutQ": pct(su["LcP"]["q2"]["outside_min_nominal"]),
          "lcpEdgeMx": f"{su['LcP']['mx2']['edge']:.1f}", "lcpEdgeQ": f"{su['LcP']['q2']['edge']:.1f}",
          "lcpAllowedMx": pct(su["LcP"]["mx2"]["frac_toys_class_alone_allowed"]),
          "dskAllowedMx": pct(su["DsK"]["mx2"]["frac_toys_class_alone_allowed"])}
    gs, gt = ST["gap_fraction_scan"], ST["gap_fraction_test"]
    fs = np.array(gs["f"])
    N |= {"gapLimitMx": f"{gt['limit_95']['mx2']:.3f}", "gapLimitMxPct": f"{100 * gt['limit_95']['mx2']:.1f}",
          "gapLimitEl": f"{gt['limit_95']['el']:.3f}",
          "toyFracNominal": pct(np.interp(f0, fs, gs["mx2"])), "toyFracHalfAt": f"{gs['f_min_50pct']['mx2']:.2f}",
          "toyFracAtSeven": f"{100 * np.interp(0.07, fs, gs['mx2']):.1f}"}
    MODE = {"D": "D", "D*": "Dst", "D**": "Dss", "D(*) pi pi": "Dpipi", "Ds(*) K": "DsK"}
    COL = {"mx2": "Mx", "el": "El", "q2": "Q", "joint": "All"}
    st = ST["shape_test"]["nominal"]
    for o, k in COL.items():
        N[f"shapeSem{k}"] = d2f(st[o]["d2"])
        for m, mk in MODE.items():
            N[f"shape{mk}{k}"] = d2f(ST["mode_shape_test"][m]["fixed"][o]["d2"])
    N["shapeSemP"] = pf(st["joint"]["p"])
    for m, mk in MODE.items():
        N[f"shape{mk}P"] = pf(ST["mode_shape_test"][m]["fixed"]["joint"]["p"])
    free = [ST["mode_shape_test"][m]["free"]["joint"]["d2"] for m in MODE]
    N |= {"shapeFreeLo": d2f(min(free)), "shapeFreeHi": d2f(max(free))}
    Vs = ST["variants"]
    N |= {"phspDsq": d2f(Vs["phsp_Dpipi"]["mx2"]["d2"]), "phspP": pf(Vs["phsp_Dpipi"]["mx2"]["p"]),
          "phspRejFrac": pct(Vs["phsp_Dpipi"]["mx2"]["frac_toys_p_below_0p05"]),
          "bareDsq": d2f(Vs["bare_evtgen_broad"]["mx2"]["d2"]), "bareP": pf(Vs["bare_evtgen_broad"]["mx2"]["p"]),
          "phspInclP": pf(Vs["phsp_Dpipi_incl_unc_x2"]["mx2"]["p"]),
          "bareInclP": pf(Vs["bare_evtgen_broad_incl_unc_x2"]["mx2"]["p"]), "gInclUnc": f"{ST['g_incl_unc_x2']:.2f}",
          "lineshapeAllowed": "allowed" if all(Vs[v]["mx2"]["d2"] == 0 for v in Vs if not v.startswith(("phsp", "bare")))
          and all(r[o]["d2"] == 0 for k, r in ST["supports"].items() if k != "q2_6_wrong" for o in OBS) else "NOT ALL ALLOWED",
          "wrongSupportDsq": d2f(ST["supports"]["q2_6_wrong"]["joint"]["d2"])}
    lines = ["% generated by 11_paper.py; do not edit"] + [rf"\newcommand{{\n{k[0].upper() + k[1:]}}}{{{v}}}" for k, v in N.items()]
    (HERE / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"-> {fd}, paper/numbers.tex ({len(N)} numbers)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--after")
    args = p.parse_args()
    bundle = Path(args.bundle) if args.bundle else default_path(HERE)
    if args.submit:
        w = f' -w "done({args.after})"' if args.after else ""
        (HERE / "logs" / "8").mkdir(parents=True, exist_ok=True)
        cmd = (f'bsub -q {yaml.safe_load(open(HERE / "config.yaml"))["batch"]["queue"]} -env all -J s8paper{w} -n 1 -oo {HERE}/logs/8/run.log '
               f'"cd {HERE} && python3 8_paper.py --bundle {bundle.resolve()}"')
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        return
    run(Bundle(bundle))


if __name__ == "__main__":
    main()
