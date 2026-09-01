#!/usr/bin/env python3
"""Step 2: Digitize invariant-mass spectra from reference plot images.

Targets:
  inputs/digitize_plots/dsk.png      (black markers with errorbars)
  inputs/digitize_plots/lambdac_p.png (blue trace)

Writes to output/2/ (JSON + CSV) and figures/2/ (overlay + curve PNGs).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config) as _f:
    _cfg = yaml.safe_load(_f)

_d    = _cfg["digitize"]
_out  = Path(_cfg["paths"]["output"]) / "2"
_figs = Path(_cfg["paths"]["figures"]) / "2"
_out.mkdir(parents=True, exist_ok=True)
_figs.mkdir(parents=True, exist_ok=True)


@dataclass
class SpectrumConfig:
    key: str
    image_path: Path
    color: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    n_bins: int = 90
    min_pixels_per_col: int = 1
    inner_box_fracs: tuple | None = None


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _detect_plot_box(rgb):
    gray = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114]))
    dark = gray < 80.0
    row_thr = max(40, int(0.12 * dark.shape[1]))
    col_thr = max(40, int(0.12 * dark.shape[0]))
    row_idx = np.where(dark.sum(axis=1) >= row_thr)[0]
    col_idx = np.where(dark.sum(axis=0) >= col_thr)[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        raise RuntimeError("Could not infer plot box from axis lines.")
    y0, y1 = int(row_idx.min()), int(row_idx.max())
    x0, x1 = int(col_idx.min()), int(col_idx.max())
    pad = 2
    return x0 + pad, x1 - pad, y0 + pad, y1 - pad


def _apply_inner_box_fracs(box, fracs):
    if fracs is None:
        return box
    x0, x1, y0, y1 = box
    fx0, fx1, fy0, fy1 = fracs
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    nx0 = max(x0, min(x0 + int(round(fx0 * w)), x1 - 1))
    nx1 = max(nx0 + 1, min(x0 + int(round(fx1 * w)), x1))
    ny0 = max(y0, min(y0 + int(round(fy0 * h)), y1 - 1))
    ny1 = max(ny0 + 1, min(y0 + int(round(fy1 * h)), y1))
    return nx0, nx1, ny0, ny1


def _color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(np.int16) for i in range(3)]
    if color == "red":
        return (r > 150) & (r - g > 45) & (r - b > 45)
    if color == "blue":
        return (b > 150) & (b - r > 45) & (b - g > 45)
    if color == "black":
        return (r < 70) & (g < 70) & (b < 70) & (np.abs(r - g) < 20) & (np.abs(r - b) < 20)
    raise ValueError(f"Unsupported color: {color}")


def _digitize_trace(rgb, box, color, n_bins, min_pixels_per_col):
    x0, x1, y0, y1 = box
    roi  = rgb[y0:y1 + 1, x0:x1 + 1]
    mask = _color_mask(roi, color)
    if color == "black":
        # Suppress the red template curve/histogram contribution in the dsk plot.
        red_mask = _color_mask(roi, "red")
        if np.any(red_mask):
            red_veto = ndi.binary_dilation(red_mask, iterations=2)
            mask &= ~red_veto

    if color == "red":
        lab, n = ndi.label(mask)
        if n <= 0:
            raise RuntimeError("No red components found.")
        areas = np.bincount(lab.ravel())[1:]
        k     = int(np.argmax(areas)) + 1
        use   = lab == k
        xs, ys = [], []
        for x in range(use.shape[1]):
            y_idx = np.where(use[:, x])[0]
            if y_idx.size < min_pixels_per_col:
                continue
            xs.append(float(x)); ys.append(float(np.median(y_idx)))
        if not xs:
            raise RuntimeError("Largest red component yielded no points.")
        xs, ys = np.asarray(xs), np.asarray(ys)
    elif color == "blue":
        lab, n = ndi.label(mask)
        if n <= 0:
            raise RuntimeError("No blue components found.")
        objs   = ndi.find_objects(lab)
        xs, ys = [], []
        for i, s in enumerate(objs, start=1):
            if s is None:
                continue
            blob = lab[s] == i
            if blob.sum() < 20:
                continue
            yy, xx = np.where(blob)
            xs.append(float(xx.mean() + s[1].start))
            ys.append(float(yy.mean() + s[0].start))
        if not xs:
            raise RuntimeError("No suitable blue marker components found.")
        xs, ys = np.asarray(xs), np.asarray(ys)
    else:  # black
        lab, n = ndi.label(mask)
        if n <= 0:
            raise RuntimeError("No black components found.")
        objs   = ndi.find_objects(lab)
        xs, ys = [], []
        for i, s in enumerate(objs, start=1):
            if s is None:
                continue
            blob = lab[s] == i
            area = int(blob.sum())
            h = int(s[0].stop - s[0].start); w = int(s[1].stop - s[1].start)
            if area < 15 or area > 1200:
                continue
            if max(w / max(h, 1), h / max(w, 1)) > 4.0:
                continue
            if h > 35 or w > 35:
                continue
            yy, xx = np.where(blob)
            xs.append(float(xx.mean() + s[1].start))
            ys.append(float(yy.mean() + s[0].start))
        if not xs:
            raise RuntimeError("No suitable black marker components found.")
        xs, ys = np.asarray(xs), np.asarray(ys)

    bins  = np.linspace(xs.min(), xs.max(), n_bins + 1)
    x_out, y_out = [], []
    for i in range(n_bins):
        m = (xs >= bins[i]) & (xs < bins[i + 1] if i < n_bins - 1 else xs <= bins[i + 1])
        if np.any(m):
            x_out.append(float(np.mean(xs[m]))); y_out.append(float(np.mean(ys[m])))

    x_px   = np.asarray(x_out) + x0
    y_px   = np.asarray(y_out) + y0
    x_norm = (x_px - x0) / max(1.0, x1 - x0)
    y_norm = 1.0 - (y_px - y0) / max(1.0, y1 - y0)
    return pd.DataFrame({"x_px": x_px, "y_px": y_px, "x_norm": x_norm, "y_norm": y_norm}
                        ).sort_values("x_px", kind="mergesort").reset_index(drop=True)


def _extract_dsk_points(rgb, box):
    x0, x1, y0, y1 = box
    roi = rgb[y0:y1 + 1, x0:x1 + 1]
    black = _color_mask(roi, "black")
    red = _color_mask(roi, "red")
    if np.any(red):
        black &= ~ndi.binary_dilation(red, iterations=2)

    # Remove thin lines (error bars, axes) and keep compact marker cores.
    core = ndi.binary_opening(black, structure=np.ones((3, 3), dtype=bool))
    core = ndi.binary_closing(core, structure=np.ones((3, 3), dtype=bool))

    lab, n = ndi.label(core)
    if n <= 0:
        raise RuntimeError("No dsk marker candidates found.")
    objs = ndi.find_objects(lab)
    pts = []
    for i, s in enumerate(objs, start=1):
        if s is None:
            continue
        blob = lab[s] == i
        area = int(blob.sum())
        h = int(s[0].stop - s[0].start)
        w = int(s[1].stop - s[1].start)
        if area < 6 or area > 240:
            continue
        asp = max(w / max(h, 1), h / max(w, 1))
        if asp > 2.6:
            continue
        yy, xx = np.where(blob)
        cx = float(xx.mean() + s[1].start)
        cy = float(yy.mean() + s[0].start)
        pts.append((cx, cy))
    if len(pts) < 10:
        raise RuntimeError("Too few dsk marker candidates after filtering.")

    pts.sort(key=lambda t: t[0])
    merged = []
    for cx, cy in pts:
        if not merged or abs(cx - merged[-1][0]) > 8.0:
            merged.append([cx, cy, 1.0])
        else:
            m = merged[-1]
            wgt = m[2] + 1.0
            m[0] = (m[0] * m[2] + cx) / wgt
            m[1] = (m[1] * m[2] + cy) / wgt
            m[2] = wgt

    arr = np.asarray([[m[0], m[1]] for m in merged], dtype=float)
    x_px = arr[:, 0] + x0
    y_px = arr[:, 1] + y0

    # Estimate y-error from the vertical span of nearby black pixels in the full mask.
    yerr_px = np.empty_like(x_px)
    for i, (xp, yp) in enumerate(zip(x_px, y_px)):
        xr = int(np.clip(round(xp - x0), 0, black.shape[1] - 1))
        xl = max(0, xr - 1)
        xh = min(black.shape[1], xr + 2)
        ys = np.where(np.any(black[:, xl:xh], axis=1))[0]
        if ys.size == 0:
            yerr_px[i] = 1.0
            continue
        # Keep pixels local to the marker center to avoid distant text/axes.
        ys = ys[np.abs(ys - (yp - y0)) < 60]
        if ys.size == 0:
            yerr_px[i] = 1.0
            continue
        yerr_px[i] = max(1.0, 0.5 * float(ys.max() - ys.min()))

    x_norm = (x_px - x0) / max(1.0, x1 - x0)
    y_norm = 1.0 - (y_px - y0) / max(1.0, y1 - y0)
    yerr_norm = yerr_px / max(1.0, y1 - y0)
    df = (
        pd.DataFrame(
            {
                "x_px": x_px,
                "y_px": y_px,
                "yerr_px": yerr_px,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "yerr_norm": yerr_norm,
            }
        )
        .sort_values("x_px", kind="mergesort")
        .reset_index(drop=True)
    )
    # Remove spurious points from the top-right panel label/graphics region.
    keep = ~((df["x_norm"] > 0.80) & (df["y_norm"] > 0.45))
    return df.loc[keep].reset_index(drop=True)


def _extract_red_histogram(rgb, box, scfg):
    """Digitize a red step-function histogram inside the plot box.

    Returns a DataFrame with columns x_lo, x_hi, x_mid, y (physical units).
    """
    x0, x1, y0, y1 = box
    roi = rgb[y0:y1 + 1, x0:x1 + 1]
    mask = _color_mask(roi, "red")

    n_cols = mask.shape[1]

    # Scan each pixel column: find the topmost red pixel (histogram top edge).
    col_top = np.full(n_cols, np.nan)
    for col in range(n_cols):
        ys = np.where(mask[:, col])[0]
        if ys.size > 0:
            col_top[col] = float(ys.min())

    # Smooth out single-pixel gaps with forward-fill, then cluster into bins.
    valid = np.where(~np.isnan(col_top))[0]
    if len(valid) == 0:
        raise RuntimeError("No red histogram pixels found.")

    # Forward-fill gaps ≤ 3 pixels wide.
    filled = col_top.copy()
    last_val, gap = np.nan, 0
    for c in range(n_cols):
        if not np.isnan(filled[c]):
            last_val, gap = filled[c], 0
        else:
            gap += 1
            if gap <= 3 and not np.isnan(last_val):
                filled[c] = last_val

    # Detect step edges: columns where the top-pixel row changes significantly.
    valid2 = np.where(~np.isnan(filled))[0]
    tops = filled[valid2]
    diffs = np.abs(np.diff(tops))
    step_threshold = 3.0  # pixels
    edges = valid2[np.where(diffs > step_threshold)[0] + 1].tolist()
    edges = [int(valid2[0])] + edges + [int(valid2[-1]) + 1]

    bins = []
    for i in range(len(edges) - 1):
        c_lo, c_hi = edges[i], edges[i + 1]
        seg = filled[c_lo:c_hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        y_px_top = float(np.median(seg))
        x_lo_phys = scfg.x_min + (c_lo / max(1.0, x1 - x0)) * (scfg.x_max - scfg.x_min)
        x_hi_phys = scfg.x_min + (c_hi / max(1.0, x1 - x0)) * (scfg.x_max - scfg.x_min)
        y_norm = 1.0 - y_px_top / max(1.0, y1 - y0)
        y_phys = scfg.y_min + y_norm * (scfg.y_max - scfg.y_min)
        bins.append({
            "x_lo": float(x_lo_phys),
            "x_hi": float(x_hi_phys),
            "x_mid": float(0.5 * (x_lo_phys + x_hi_phys)),
            "y": float(max(0.0, y_phys)),
        })

    return pd.DataFrame(bins)


def _plot_overlay(rgb, box, pts, out, title, bkg_df=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(rgb)
    x0, x1, y0, y1 = box
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="yellow", lw=1.2, ls="--")
    if "yerr_px" in pts.columns:
        ax.errorbar(pts["x_px"], pts["y_px"], yerr=pts["yerr_px"],
                    ls="", color="lime", ecolor="lime", lw=0.8, capsize=2)
        ax.scatter(pts["x_px"], pts["y_px"], s=12, c="lime", edgecolors="black", linewidths=0.25)
    else:
        ax.scatter(pts["x_px"], pts["y_px"], s=10, c="lime", edgecolors="black", linewidths=0.25)
    if bkg_df is not None and len(bkg_df) > 0:
        # Draw background bin tops in pixel space as cyan horizontal lines.
        for _, row in bkg_df.iterrows():
            # Convert physical x back to pixel for the overlay.
            xlo_norm = (row["x_lo"] - bkg_df.attrs["scfg_x_min"]) / max(1e-9, bkg_df.attrs["scfg_x_max"] - bkg_df.attrs["scfg_x_min"])
            xhi_norm = (row["x_hi"] - bkg_df.attrs["scfg_x_min"]) / max(1e-9, bkg_df.attrs["scfg_x_max"] - bkg_df.attrs["scfg_x_min"])
            y_norm   = (row["y"]    - bkg_df.attrs["scfg_y_min"]) / max(1e-9, bkg_df.attrs["scfg_y_max"] - bkg_df.attrs["scfg_y_min"])
            xlo_px = x0 + xlo_norm * (x1 - x0)
            xhi_px = x0 + xhi_norm * (x1 - x0)
            y_px   = y1 - y_norm   * (y1 - y0)
            ax.plot([xlo_px, xhi_px], [y_px, y_px], color="cyan", lw=1.5)
    ax.set_title(title); ax.set_axis_off()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def _plot_curve(pts, out, title, key):
    """Plot digitized points + fit curve. Returns (xf, yf) dense fit arrays, or (None, None)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    x = np.asarray(pts["x"], dtype=float) if "x" in pts.columns else np.asarray(pts["x_norm"], dtype=float)
    y = np.asarray(pts["y"], dtype=float) if "y" in pts.columns else np.asarray(pts["y_norm"], dtype=float)
    yerr_col = "yerr" if "yerr" in pts.columns else ("yerr_norm" if "yerr_norm" in pts.columns else None)
    if yerr_col is not None:
        ax.errorbar(x, y, yerr=pts[yerr_col], ls="", marker="o", ms=2.6, lw=0.8, capsize=2, label="Digitized")
    else:
        ax.plot(x, y, marker="o", ms=2.3, lw=1.0, label="Digitized")
    xf, yf = None, None
    if len(x) >= 4:
        xf = np.linspace(float(x.min()), float(x.max()), 300)
        if key == "lambdac_p":
            def _lc_model(xx, a, k):
                return a * np.exp(-k * (xx - x.min()))
            try:
                y_pos = np.clip(y, 1e-8, None)
                p0 = [float(y_pos.max()), 5.0]
                lo = [0.0, 0.1]
                hi = [1e4, 50.0]
                popt, _ = curve_fit(_lc_model, x, y_pos, p0=p0, bounds=(lo, hi), maxfev=50000)
                yf = np.clip(_lc_model(xf, *popt), 0.0, None)
                ax.plot(xf, yf, lw=1.2, label="Exponential fit")
            except Exception:
                coefs = np.polyfit(x, y, deg=min(5, len(x) - 1))
                yf = np.clip(np.polyval(coefs, xf), 0.0, None)
                ax.plot(xf, yf, lw=1.2, label=f"Poly fallback (deg {min(5, len(x)-1)})")
        elif key == "dsk":
            def _rbw(xx, A, M, G):
                return A * (xx * G * M) / ((xx**2 - M**2)**2 + M**2 * G**2)
            try:
                y_pos = np.clip(y, 0.0, None)
                p0 = [float(y_pos.max()) * 0.1, 2.72, 0.10]
                lo = [0.0, 2.55, 0.01]
                hi = [1e6,  3.05, 0.60]
                popt, _ = curve_fit(_rbw, x, y_pos, p0=p0, bounds=(lo, hi), maxfev=50000)
                yf = np.clip(_rbw(xf, *popt), 0.0, None)
                ax.plot(xf, yf, lw=1.2, label=f"Rel. Breit-Wigner (M={popt[1]:.3f}, Γ={popt[2]:.3f})")
            except Exception:
                coefs = np.polyfit(x, y, deg=min(5, len(x) - 1))
                yf = np.clip(np.polyval(coefs, xf), 0.0, None)
                ax.plot(xf, yf, lw=1.2, label=f"Poly fallback (deg {min(5, len(x)-1)})")
        else:
            coefs = np.polyfit(x, y, deg=min(5, len(x) - 1))
            yf = np.clip(np.polyval(coefs, xf), 0.0, None)
            ax.plot(xf, yf, lw=1.2, label=f"Smooth fit (deg {min(5, len(x)-1)})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return xf, yf


def _estimate_trace_yerr_px(pts: pd.DataFrame, min_px: float = 1.0) -> np.ndarray:
    x = pts["x_px"].to_numpy(dtype=float)
    y = pts["y_px"].to_numpy(dtype=float)
    n = len(pts)
    if n < 5:
        return np.full(n, min_px, dtype=float)
    win = max(5, min(11, (n // 5) * 2 + 1))
    half = win // 2
    y_smooth = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        y_smooth[i] = np.median(y[lo:hi])
    resid = y - y_smooth
    sigma = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        r = resid[lo:hi]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        s = 1.4826 * mad if mad > 0 else float(np.std(r))
        if not np.isfinite(s):
            s = min_px
        sigma[i] = max(min_px, s)
    return sigma


def _add_physical_axes(pts: pd.DataFrame, scfg: SpectrumConfig) -> pd.DataFrame:
    out = pts.copy()
    x_norm = out["x_norm"].to_numpy(dtype=float)
    y_norm = out["y_norm"].to_numpy(dtype=float)
    out["x"] = scfg.x_min + x_norm * (scfg.x_max - scfg.x_min)
    out["y"] = scfg.y_min + y_norm * (scfg.y_max - scfg.y_min)
    if "yerr_norm" in out.columns:
        out["yerr"] = out["yerr_norm"].to_numpy(dtype=float) * abs(scfg.y_max - scfg.y_min)
    return out


def _run_spectrum(scfg, out_data, out_figs):
    rgb = _load_rgb(scfg.image_path)
    box = _apply_inner_box_fracs(_detect_plot_box(rgb), scfg.inner_box_fracs)
    if scfg.key == "dsk":
        pts = _extract_dsk_points(rgb, box)
    else:
        pts = _digitize_trace(rgb, box, scfg.color, scfg.n_bins, scfg.min_pixels_per_col)
    if scfg.key in {"lambdac_p", "dsk"} and "yerr_norm" not in pts.columns:
        yerr_px = _estimate_trace_yerr_px(pts, min_px=1.0)
        x0, x1, y0, y1 = box
        pts["yerr_px"] = yerr_px
        pts["yerr_norm"] = yerr_px / max(1.0, y1 - y0)
    pts = _add_physical_axes(pts, scfg)

    stem = scfg.key
    csv_path     = out_data / f"{stem}_digitized.csv"
    json_path    = out_data / f"{stem}_digitized.json"
    overlay_path = out_figs / f"{stem}_overlay.png"
    curve_path   = out_figs / f"{stem}_curve.png"

    bkg_df = None
    if scfg.key == "dsk":
        bkg_df = _extract_red_histogram(rgb, box, scfg)
        bkg_df.attrs["scfg_x_min"] = scfg.x_min
        bkg_df.attrs["scfg_x_max"] = scfg.x_max
        bkg_df.attrs["scfg_y_min"] = scfg.y_min
        bkg_df.attrs["scfg_y_max"] = scfg.y_max
        print(f"[dsk] background: {len(bkg_df)} bins digitized")

        # Interpolate background at each data point x and subtract.
        bkg_x_mid = bkg_df["x_mid"].to_numpy()
        bkg_y     = bkg_df["y"].to_numpy()
        data_x    = pts["x"].to_numpy()
        bkg_at_data = np.interp(data_x, bkg_x_mid, bkg_y,
                                left=bkg_y[0], right=bkg_y[-1])
        pts["y_raw"]    = pts["y"]
        pts["y_bkg"]    = bkg_at_data
        pts["y"]        = np.clip(pts["y"] - bkg_at_data, 0.0, None)

        bkg_df.to_csv(out_data / f"{stem}_background.csv", index=False)
        (out_data / f"{stem}_background.json").write_text(json.dumps({
            "key": f"{stem}_background",
            "n_bins": int(len(bkg_df)),
            "bins": bkg_df.to_dict(orient="records"),
        }, indent=2))
        print(f"[dsk] background → {out_data / f'{stem}_background.csv'}")

    pts.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "key":          scfg.key,
        "image_path":   str(scfg.image_path),
        "axis_ranges":  {"x_min": scfg.x_min, "x_max": scfg.x_max, "y_min": scfg.y_min, "y_max": scfg.y_max},
        "plot_box_px":  {"x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3]},
        "n_points":     int(len(pts)),
        "background_subtracted": scfg.key == "dsk",
        "points":       pts.to_dict(orient="records"),
    }, indent=2))

    _plot_overlay(rgb, box, pts, overlay_path, f"{stem}: extracted points overlay", bkg_df=bkg_df)
    xf, yf = _plot_curve(pts, curve_path, f"{stem}: digitized curve (bkg subtracted)" if scfg.key == "dsk" else f"{stem}: digitized curve", stem)
    if xf is not None and yf is not None:
        x_pts = pts["x"].to_numpy(float) if "x" in pts.columns else pts["x_norm"].to_numpy(float)
        pts["y_fit"] = np.interp(x_pts, xf, yf, left=0.0, right=0.0)
        pts.to_csv(csv_path, index=False)
    print(f"[{stem}] {len(pts)} points → {csv_path}, {overlay_path}, {curve_path}")
    return {"key": stem, "n_points": int(len(pts)),
            "csv": str(csv_path), "json": str(json_path)}


spectrum_configs = [
    SpectrumConfig(key="dsk",       image_path=Path(_d["dsk_plot"]),     color="black",
                   x_min=float(_d["dsk_x_min"]), x_max=float(_d["dsk_x_max"]),
                   y_min=float(_d["dsk_y_min"]), y_max=float(_d["dsk_y_max"]),
                   n_bins=90, inner_box_fracs=(0.115, 1, 0.0, 0.85)),
    SpectrumConfig(key="lambdac_p", image_path=Path(_d["lambdac_plot"]), color="blue",
                   x_min=float(_d["lambdac_x_min"]), x_max=float(_d["lambdac_x_max"]),
                   y_min=float(_d["lambdac_y_min"]), y_max=float(_d["lambdac_y_max"]),
                   n_bins=90, inner_box_fracs=(0.11, 1, 0.0, 0.96)),
]

results = [_run_spectrum(sc, _out, _figs)
           for sc in spectrum_configs]

(_out / "digitize_summary.json").write_text(json.dumps({"results": results}, indent=2))
print(f"Summary → {_out / 'digitize_summary.json'}")
