"""Asimov gap-truth scenarios (config `asimov.scenarios`), shared by steps 6 and 7."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["Mx", "El_B", "q2", "total_weight", "decay_name"]
MAX_ORDER = 6   # raw moment orders stored per observable (step 6) and available to steps 7-8


def power_matrix(mx2, el, q2) -> np.ndarray:
    """(3*MAX_ORDER, N): x^k for x in (mx2, el, q2), k = 1..MAX_ORDER."""
    return np.stack([x ** k for x in (mx2, el, q2) for k in range(1, MAX_ORDER + 1)])


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    return d[np.isfinite(d[COLS[:4]].to_numpy(float)).all(axis=1) & (d["total_weight"] > 0)]


def _split(names, w, fraction, equal_species):
    """[(rows, weights normalized to their fraction)], one entry per B species if requested."""
    if not equal_species:
        return [(np.arange(len(w)), fraction * w / w.sum())]
    species = np.array([n[:2] for n in names])
    groups = [np.flatnonzero(species == s) for s in np.unique(species)]
    return [(g, fraction / len(groups) * w[g] / w[g].sum()) for g in groups]


def truth_groups(cfg: dict, name: str, cocktail: dict | None = None) -> list[dict]:
    """Event groups of one scenario's gap truth. Each group: comp (component index), src
    ("own" or "cocktail"), rows (into the cocktail for "cocktail"), mx2, el, q2, and w (summing
    to the group's share of the truth). `cocktail` = dict(mx2, el, q2, w, codes, cats) is needed
    for components given as `decays`."""
    sc = cfg["asimov"]["scenarios"][name]
    equal = sc.get("equal_species", False)
    out3 = Path(cfg["paths"]["output"]) / "3"
    groups = []
    for i, comp in enumerate(sc["components"]):
        if "gap_mode" in comp:
            d = _clean(pd.read_parquet(out3 / f"{comp['gap_mode']}.parquet", columns=COLS))
            x = dict(mx2=d["Mx"].to_numpy(float) ** 2, el=d["El_B"].to_numpy(float),
                     q2=d["q2"].to_numpy(float))
            names, w, src, base = d["decay_name"].to_numpy(str), d["total_weight"].to_numpy(float), "own", None
        else:
            ck = cocktail
            base = np.flatnonzero(np.isin(ck["codes"], np.flatnonzero(np.isin(ck["cats"], comp["decays"]))))
            x = {k: ck[k][base] for k in ("mx2", "el", "q2")}
            names, w, src = ck["cats"][ck["codes"][base]], ck["w"][base], "cocktail"
        for rows, wg in _split(names, w, comp["fraction"], equal):
            groups.append(dict(comp=i, src=src, rows=rows if base is None else base[rows],
                               w=wg, **{k: v[rows] for k, v in x.items()}))
    return groups


def support(groups: list[dict]) -> dict[str, tuple[float, float]]:
    """Reconstruction support from the truth's own endpoints (El and q2 anchored at 0)."""
    cat = {k: np.concatenate([g[k] for g in groups]) for k in ("mx2", "el", "q2")}
    return {"mx2": (cat["mx2"].min() * 0.999, cat["mx2"].max() * 1.001),
            "el": (0.0, cat["el"].max() * 1.001), "q2": (0.0, cat["q2"].max() * 1.001)}
