"""Step-8 gap density, HQE inversion only (no joint threshold fit), E_l and q2."""
import importlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
s8 = importlib.import_module("8_data_results")
from lib.bundle import Bundle, default_path

import matplotlib
matplotlib.use("Agg")
import plothist  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

B = Bundle(default_path(HERE))
hqe = s8.hqe_results(B.cfg, B.toys, 1000, np.random.default_rng(42), subtract=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=200)
for ax, obs in zip(axes, ("El", "Q2")):
    h = hqe[obs]
    s8.draw_curve(ax, h["x"], h["f"], h["band"], s8.HQE_COLOR, obs)
    ax.set_xlim(h["x"][0], h["x"][-1])
    ax.set_xlabel(s8.XLABEL[obs], fontsize=s8.FS_LABEL)
    ax.set_ylabel(s8.YLABEL[obs], fontsize=s8.FS_LABEL)
    s8.style(ax)
handles = [Line2D([], [], color=s8.HQE_COLOR, lw=2.6, label="HQE inversion"),
           Patch(color=s8.HQE_COLOR, alpha=0.3, lw=0, label="68% (toys)")]
s8.finish(fig, axes, handles, Path(__file__).parent / "gap_density_hqe.pdf", ymax=[1.3 * np.max(hqe["El"]["band"][1]), 0.3])
