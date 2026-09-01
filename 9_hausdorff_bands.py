#!/usr/bin/env python3
"""
9_hausdorff_bands.py

Merge toy chunk JSONs produced by 8_hausdorff_data.py --toy-job and plot
MaxEnt uncertainty bands (68% / 95%) on top of the nominal inversions.

Usage:
  cd clean && python3 9_hausdorff_bands.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from lib.gap_modes import GAP_MODES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def _overlay_gap_modes(ax, parquet_dir, obs_col, xlo, xhi, ymax, extra_filter=None, n_bins=60):
    """Draw each gap mode's weighted, unit-normalized histogram on ax; return updated ymax."""
    for gm in GAP_MODES:
        pq = parquet_dir / f"{gm['mode']}.parquet"
        if not pq.exists():
            continue
        cols = [obs_col, "total_weight"] if extra_filter is None else [obs_col, extra_filter[0], "total_weight"]
        df_gm = pd.read_parquet(pq, columns=cols)
        if extra_filter is not None:
            col, predicate = extra_filter
            df_gm = df_gm[predicate(df_gm[col])]
        vals = df_gm[obs_col].to_numpy()
        wts = df_gm["total_weight"].to_numpy()
        mask = (vals >= xlo) & (vals <= xhi)
        vals, wts = vals[mask], wts[mask]
        if wts.sum() == 0:
            continue
        counts, edges = np.histogram(vals, bins=n_bins, range=(xlo, xhi), weights=wts)
        db = edges[1] - edges[0]
        counts = counts / (counts.sum() * db)
        ax.step(edges, np.append(counts, counts[-1]), where="post",
                color=gm["color"], lw=gm["lw"], ls=gm["ls"], alpha=0.85, label=gm["label"])
        ymax = max(ymax, float(counts.max()))
    return ymax


def main():
    args  = parse_args()
    cfg   = yaml.safe_load(open(args.config))
    hd    = cfg.get("hausdorff_data", {})

    out_dir = Path(cfg["paths"]["output"]) / "8"
    fig_dir = Path(cfg["paths"]["figures"]) / "9"
    fig_dir.mkdir(parents=True, exist_ok=True)

    toy_dir = out_dir / "toys"
    chunk_files = sorted(toy_dir.glob("chunk_*.json"))
    if not chunk_files:
        print(f"No chunk files found in {toy_dir}. Run 8_hausdorff_data.py --submit first.")
        return
    print(f"Found {len(chunk_files)} chunk file(s).")

    # ── Load nominal MaxEnt inversions ────────────────────────────────────────
    with open(out_dir / "maxent_data_inversions.json") as f:
        maxent_out = json.load(f)
    with open(out_dir / "hausdorff_data_check.json") as f:
        haus_results = json.load(f)

    # ── Merge toy chunks ──────────────────────────────────────────────────────
    # toy_curves[obs][thr_str] = list of f_phys arrays
    toy_curves: dict[str, dict[str, list]] = {}
    # toy_curves_joint[obs] = list of f_phys arrays, one per toy (opt-in, 8_hausdorff_data.py's
    # joint_maxent.enable path) -- already living on one common global grid per toy, so unlike
    # toy_curves above this needs NO stitching: just percentile it directly.
    toy_curves_joint: dict[str, list] = {}
    n_toys_total = 0
    n_toys_joint = 0
    for cf in chunk_files:
        with open(cf) as f:
            chunk = json.load(f)
        n_toys_total += chunk["i_end"] - chunk["i_start"]
        for obs, thr_dict in chunk["toy_curves"].items():
            toy_curves.setdefault(obs, {})
            for thr_str, curves in thr_dict.items():
                toy_curves[obs].setdefault(thr_str, []).extend(curves)
        if "toy_curves_joint" in chunk:
            n_toys_joint += chunk["i_end"] - chunk["i_start"]
            for obs, curves in chunk["toy_curves_joint"].items():
                toy_curves_joint.setdefault(obs, []).extend(curves)
    print(f"Merged {n_toys_total} toys across {len(chunk_files)} chunks.")
    if toy_curves_joint:
        print(f"  (joint_maxent data present: {n_toys_joint} toys)")

    # ── Support / label config (mirrors 8_) ──────────────────────────────────
    mx2_lo, mx2_hi = float(hd["mx_support"][0]), float(hd["mx_support"][1])
    el_lo,  el_hi  = float(hd["el_support"][0]),  float(hd["el_support"][1])
    q2_lo,  q2_hi  = float(hd["q2_support"][0]),  float(hd["q2_support"][1])

    mx_lo, mx_hi = np.sqrt(mx2_lo), np.sqrt(mx2_hi)

    obs_support = {
        "Mx": (mx_lo,  mx_hi,  r"$M_X\ [\mathrm{GeV}]$",     r"$f(M_X)\ [\mathrm{GeV}^{-1}]$"),
        "El": (el_lo,  el_hi,  r"$E_\ell\ [\mathrm{GeV}]$",  r"$f(E_\ell)\ [\mathrm{GeV}^{-1}]$"),
        "Q2": (q2_lo,  q2_hi,  r"$q^2\ [\mathrm{GeV}^2]$",   r"$f(q^2)\ [\mathrm{GeV}^{-2}]$"),
    }
    obs_title = {"Mx": r"$M_X$", "El": r"$E_\ell$", "Q2": r"$q^2$"}

    grid_size = int(hd.get("grid_size", 2000))
    t_grid = np.linspace(0., 1., grid_size)

    # ── Plot one figure per observable ────────────────────────────────────────
    print("Plotting uncertainty bands …")

    for obs in ["Mx", "El", "Q2"]:
        if obs not in maxent_out:
            print(f"  {obs}: no nominal inversions, skip")
            continue

        res_obs    = haus_results[obs]
        thresholds = np.array(res_obs["thresholds"])
        pass_flags = np.array(res_obs["pass"])
        passing    = thresholds[pass_flags]
        if len(passing) == 0:
            continue

        xlo_base, xhi, xlabel, ylabel = obs_support[obs]
        ncols = min(len(passing), 5)
        nrows = int(np.ceil(len(passing) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.5, nrows * 3.2),
                                 squeeze=False)

        for idx, thr in enumerate(passing):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            thr_str = f"{thr:.2f}"

            xlo    = float(thr) if obs in ("El", "Q2") else xlo_base
            span   = xhi - xlo
            # For Mx: stored curves are f(Mx²) on a Mx² grid; convert to f(Mx)
            if obs == "Mx":
                mx2_grid = mx2_lo + t_grid * (mx2_hi - mx2_lo)
                x_grid   = np.sqrt(mx2_grid)
                jac      = 2 * x_grid          # dMx²/dMx = 2*Mx  →  f(Mx)=f(Mx²)*2Mx
            else:
                x_grid = xlo + t_grid * span
                jac    = np.ones_like(x_grid)

            # nominal — suppressed for Mx (degenerate spike, not meaningful)
            nom_entry = maxent_out[obs]["thresholds"].get(thr_str, {})
            f_nom = nom_entry.get("f_phys") if obs != "Mx" else None

            toy_list = toy_curves.get(obs, {}).get(thr_str, [])
            n_good = len(toy_list)
            if toy_list:
                arr = np.array(toy_list) * jac
                lo2, lo1, hi1, hi2 = np.percentile(arr, [2.5, 16, 84, 97.5], axis=0)
                ax.fill_between(x_grid, lo2, hi2, color="steelblue", alpha=0.15,
                                label="95% band")
                ax.fill_between(x_grid, lo1, hi1, color="steelblue", alpha=0.30,
                                label="68% band")
                ax.set_ylim(0, 1.3 * float(hi1.max()))

            if f_nom is not None:
                f_nom_arr = np.array(f_nom) * jac
                ax.plot(x_grid, f_nom_arr, color="steelblue", lw=2, label="nominal")
                if not toy_list:
                    ax.set_ylim(0, 2.0 * float(f_nom_arr.max()))

            cut_label = (r"$E_\ell > " + f"{thr:.1f}" + r"\ \mathrm{GeV}$"
                         if obs != "Q2" else
                         r"$q^2 > " + f"{thr:.1f}" + r"\ \mathrm{GeV}^2$")
            ax.set_title(cut_label + f"\n{n_good}/{n_toys_total} good toys", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(xlo, xhi)
            if not toy_list and f_nom is None:
                ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)
            if idx == 0:
                ax.legend(fontsize=7)

        for idx in range(len(passing), nrows * ncols):
            axes[divmod(idx, ncols)].set_visible(False)

        fig.suptitle(
            r"MaxEnt inversion with uncertainty bands — " + obs_title[obs]
            + f"\n({n_toys_total} Gaussian toys from exp. covariance, "
            + r"$\alpha=\beta=0$)",
            fontsize=11)
        fig.tight_layout()
        out_fig = fig_dir / f"maxent_bands_{obs}.pdf"
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"  → {out_fig}")

    # ── Joint multi-threshold MaxEnt bands (opt-in, no stitching needed) ─────────
    if toy_curves_joint:
        print("Plotting joint MaxEnt bands …")
        for obs in ["El", "Q2"]:
            if obs not in toy_curves_joint:
                continue
            curves = np.array(toy_curves_joint[obs])
            ok_mask = np.all(np.isfinite(curves), axis=1) if curves.size else np.zeros(0, bool)
            n_conv = int(ok_mask.sum())
            if n_conv < 4:
                print(f"  {obs}: only {n_conv} converged joint toys, skip")
                continue
            arr = curves[ok_mask]
            xlo_o, xhi_o, xlabel, ylabel = obs_support[obs]
            x_grid = xlo_o + t_grid * (xhi_o - xlo_o)
            lo2, lo1, med, hi1, hi2 = np.percentile(arr, [2.5, 16, 50, 84, 97.5], axis=0)

            fig_j, ax_j = plt.subplots(figsize=(7, 4.5))
            ax_j.fill_between(x_grid, lo2, hi2, color="steelblue", alpha=0.15, label="95% band")
            ax_j.fill_between(x_grid, lo1, hi1, color="steelblue", alpha=0.35, label="68% band")
            ax_j.plot(x_grid, med, color="steelblue", lw=1.8, label="toy median")

            obs_col = {"El": "El_B", "Q2": "q2"}[obs]
            parquet_dir = Path(cfg["paths"]["output"]) / "3"
            ymax = float(np.percentile(hi1, 99))
            ymax = _overlay_gap_modes(ax_j, parquet_dir, obs_col, xlo_o, xhi_o, ymax)

            ax_j.set_ylim(0, 1.3 * ymax)
            ax_j.set_xlim(xlo_o, xhi_o)
            ax_j.set_xlabel(xlabel, fontsize=10); ax_j.set_ylabel(ylabel, fontsize=10)
            ax_j.legend(fontsize=6, ncol=2); ax_j.grid(alpha=0.25)
            obs_var = {"El": r"$E_\ell$", "Q2": r"$q^2$"}[obs]
            fig_j.suptitle(
                f"Joint multi-threshold MaxEnt — {obs_var} ({n_conv}/{len(curves)} converged toys)\n"
                "single global fit per toy, no stitching (lib/joint_maxent.py)", fontsize=10)
            fig_j.tight_layout()
            out_j = fig_dir / f"maxent_joint_bands_{obs}.pdf"
            fig_j.savefig(out_j); plt.close(fig_j)
            print(f"  → {out_j}")

    # ── Stitched combined band (legacy per-threshold + splice) ───────────────────
    print("Plotting stitched combined bands …")

    for obs in ["El", "Q2"]:
        if obs not in toy_curves or obs not in haus_results:
            continue

        xlo_base, xhi, xlabel, ylabel = obs_support[obs]
        res_obs    = haus_results[obs]
        thresholds = np.array(res_obs["thresholds"])
        pass_flags = np.array(res_obs["pass"])
        passing    = sorted(thresholds[pass_flags].tolist())
        if len(passing) < 2:
            continue

        # Collect per-threshold percentile bands, only where enough toys exist
        MIN_TOYS = 50
        bands = []  # (thr, x_grid, lo2, lo1, hi1, hi2) all in physical coords
        for thr in passing:
            thr_str = f"{thr:.2f}"
            toy_list = toy_curves[obs].get(thr_str, [])
            if len(toy_list) < MIN_TOYS:
                continue
            xlo = float(thr) if obs in ("El", "Q2") else xlo_base
            span = xhi - xlo
            if obs == "Mx":
                mx2_grid = mx2_lo + t_grid * (mx2_hi - mx2_lo)
                xg = np.sqrt(mx2_grid); jac = 2 * xg
            else:
                xg = xlo + t_grid * span; jac = np.ones_like(xg)
            arr = np.array(toy_list) * jac
            lo2, lo1, hi1, hi2 = np.percentile(arr, [2.5, 16, 84, 97.5], axis=0)
            bands.append((float(thr), xg, lo2, lo1, hi1, hi2))

        if len(bands) < 2:
            print(f"  {obs}: not enough threshold bands, skip stitched plot")
            continue

        # Normalize: highest threshold band normalized so median total = 1 (each raw per-
        # threshold toy curve is already a conditional density with mu_0=1 over its OWN full
        # domain, so this "total" is already ~1 before scaling -- scales[top] just tidies up
        # discretization noise). Each lower band i is then scaled so its median integral above
        # the NEXT threshold matches band (i+1)'s ALREADY-SCALED total (=scales[i+1], since band
        # i+1's raw total over its own full domain is 1) -- i.e. scales[i] must chain through
        # scales[i+1], not reset to a fresh 1/above every time. Without the chain multiplication
        # (a prior bug here), non-adjacent bands silently drift apart in overall level -- e.g.
        # the lowest El threshold came out ~2x too low relative to the top one on real data.
        def _band_median(lo1, hi1):
            return 0.5 * (lo1 + hi1)

        # scales[i]: factor putting band i on the common scale, chained down from the top band
        scales = [None] * len(bands)
        for i in range(len(bands) - 1, -1, -1):
            thr, xg, lo2, lo1, hi1, hi2 = bands[i]
            med = _band_median(lo1, hi1)
            if i == len(bands) - 1:
                total = np.trapz(med, xg)
                scales[i] = 1.0 / total if total > 0 else 1.0
            else:
                thr_next = bands[i + 1][0]
                mask = xg >= thr_next
                above = np.trapz(med[mask], xg[mask])
                # match band i's mass beyond thr_next to band (i+1)'s already-scaled total
                scales[i] = scales[i + 1] / above if above > 0 else scales[i + 1]

        scaled_bands = []
        for i, (thr, xg, lo2, lo1, hi1, hi2) in enumerate(bands):
            s = scales[i]
            scaled_bands.append((thr, xg, lo2*s, lo1*s, hi1*s, hi2*s))

        x_common = np.linspace(scaled_bands[0][0], xhi, grid_size)

        # Piecewise assignment: for each x, use ONLY the band from the tightest covering cut
        # (the largest threshold <= x), not an intersection over every covering band. The
        # previous max(lo)/min(hi) intersection ANDs together every band that happens to cover
        # a given x -- as x grows, more (increasingly toy-starved, independently rescaled)
        # bands enter that AND, and whichever one is noisiest that day clamps the envelope,
        # producing spurious kinks/spikes right at threshold boundaries (worst for q2, where
        # the good-toy fraction falls off fastest). Picking the single nearest-from-below
        # threshold's band avoids compounding noise from unrelated, farther-away cuts.
        thr_arr = np.array([b[0] for b in scaled_bands])
        band_idx = np.clip(np.searchsorted(thr_arr, x_common, side="right") - 1,
                            0, len(scaled_bands) - 1)

        lo2_stitch = np.full_like(x_common, np.nan)
        lo1_stitch = np.full_like(x_common, np.nan)
        hi1_stitch = np.full_like(x_common, np.nan)
        hi2_stitch = np.full_like(x_common, np.nan)

        for i, (thr, xg, lo2, lo1, hi1, hi2) in enumerate(scaled_bands):
            sel = band_idx == i
            if not sel.any():
                continue
            lo2_stitch[sel] = np.interp(x_common[sel], xg, lo2)
            lo1_stitch[sel] = np.interp(x_common[sel], xg, lo1)
            hi1_stitch[sel] = np.interp(x_common[sel], xg, hi1)
            hi2_stitch[sel] = np.interp(x_common[sel], xg, hi2)

        covered = np.isfinite(lo1_stitch)
        lo2_stitch = np.clip(lo2_stitch, 0, None)
        lo1_stitch = np.clip(lo1_stitch, 0, None)

        obs_col = {"El": "El_B", "Q2": "q2"}[obs]
        parquet_dir = Path(cfg["paths"]["output"]) / "3"

        fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
        ax_s.fill_between(x_common[covered], lo2_stitch[covered], hi2_stitch[covered],
                          color="steelblue", alpha=0.15, label="95% band")
        ax_s.fill_between(x_common[covered], lo1_stitch[covered], hi1_stitch[covered],
                          color="steelblue", alpha=0.35, label="68% band")
        ymax = np.percentile(hi1_stitch[covered], 99)

        x_lo_stitch = scaled_bands[0][0]
        ymax = _overlay_gap_modes(ax_s, parquet_dir, obs_col, x_lo_stitch, xhi, ymax)

        ax_s.set_ylim(0, 1.3 * ymax)
        ax_s.set_xlim(x_lo_stitch, xhi)
        ax_s.set_xlabel(xlabel, fontsize=10)
        ax_s.set_ylabel("rescaled " + ylabel, fontsize=10)
        ax_s.legend(fontsize=6, ncol=2)
        ax_s.grid(alpha=0.25)
        obs_var = {"El": r"$E_\ell$", "Q2": r"$q^2$"}[obs]
        fig_s.suptitle(
            f"Inverted {obs_var} distribution\n"
            r"Overlap of uncertainty bands from every " + obs_var + r" $>$ X threshold",
            fontsize=10)
        fig_s.tight_layout()
        out_s = fig_dir / f"maxent_stitched_{obs}.pdf"
        fig_s.savefig(out_s); plt.close(fig_s)
        print(f"  → {out_s}")

    # ── Mx band at best threshold + gap mode overlay ──────────────────────────
    print("Plotting Mx band with gap mode overlay …")
    if "Mx" in toy_curves and "Mx" in haus_results:
        xlo_base, _, xlabel_mx, ylabel_mx = obs_support["Mx"]
        mx_x_hi_plot = 3.0   # truncate at 3 GeV

        # pick threshold with most good toys
        mx_toy = toy_curves["Mx"]
        best_thr_str = max(mx_toy, key=lambda k: len(mx_toy[k]))
        best_thr     = float(best_thr_str)
        toy_list     = mx_toy[best_thr_str]

        mx2_grid = mx2_lo + t_grid * (mx2_hi - mx2_lo)
        x_grid   = np.sqrt(mx2_grid)
        jac      = 2 * x_grid

        arr = np.array(toy_list) * jac
        lo2, lo1, hi1, hi2 = np.percentile(arr, [2.5, 16, 84, 97.5], axis=0)
        mask_plot = x_grid <= mx_x_hi_plot

        fig_mx, ax_mx = plt.subplots(figsize=(7, 4.5))
        ax_mx.fill_between(x_grid[mask_plot], lo2[mask_plot], hi2[mask_plot],
                           color="steelblue", alpha=0.15, label="95% band")
        ax_mx.fill_between(x_grid[mask_plot], lo1[mask_plot], hi1[mask_plot],
                           color="steelblue", alpha=0.35, label="68% band")
        ymax_mx = float(np.percentile(hi1[mask_plot], 99))

        parquet_dir = Path(cfg["paths"]["output"]) / "3"
        ymax_mx = _overlay_gap_modes(ax_mx, parquet_dir, "Mx", xlo_base, mx_x_hi_plot, ymax_mx,
                                     extra_filter=("El_B", lambda s: s >= best_thr))

        ax_mx.set_ylim(0, 1.3 * ymax_mx)
        ax_mx.set_xlim(xlo_base, mx_x_hi_plot)
        ax_mx.set_xlabel(xlabel_mx, fontsize=10)
        ax_mx.set_ylabel(ylabel_mx, fontsize=10)
        ax_mx.legend(fontsize=6, ncol=2)
        ax_mx.grid(alpha=0.25)
        fig_mx.suptitle(
            r"$M_X$ band ($E_\ell > " + f"{best_thr:.2f}" + r"\ \mathrm{GeV}$, "
            + f"{len(toy_list)} toys) + gap modes",
            fontsize=10)
        fig_mx.tight_layout()
        out_mx = fig_dir / "maxent_stitched_Mx.pdf"
        fig_mx.savefig(out_mx); plt.close(fig_mx)
        print(f"  → {out_mx}")

    print("Done.")


if __name__ == "__main__":
    main()
