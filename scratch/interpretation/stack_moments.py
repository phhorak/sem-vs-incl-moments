"""Per-category raw-moment contributions of the SEM cocktail vs threshold (step-5 stack), cached as JSON."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
CATEGORIES = ["D", "D*", "D**", "D(*) pi", "D(*) pi pi", "Ds(*) K"]
CUTS = {"el": np.linspace(0.0, 2.5, 51), "q2": np.linspace(0.0, 10.0, 41)}

d = pd.read_parquet(HERE / "output" / "3" / "cocktail.parquet", columns=["Mx", "El_B", "q2", "total_weight", "category"])
d = d[np.isfinite(d[["Mx", "El_B", "q2"]].to_numpy()).all(axis=1) & (d.total_weight > 0)]
w, cat = d.total_weight.to_numpy(), d.category.to_numpy()
x = {"mx": d.Mx.to_numpy() ** 2, "el": d.El_B.to_numpy(), "q2": d.q2.to_numpy()}
out = {"cuts": {k: v.tolist() for k, v in CUTS.items()}}
for v, cv in (("mx", "el"), ("el", "el"), ("q2", "q2")):
    for n in (1, 2, 3):
        out[f"{v}_{n}"] = {c: [] for c in CATEGORIES}
        for cut in CUTS[cv]:
            m = x[cv] > cut
            W = w[m].sum()
            for c in CATEGORIES:
                mc = m & (cat == c)
                out[f"{v}_{n}"][c].append(float(np.sum(w[mc] * x[v][mc] ** n) / W))
json.dump(out, open(Path(__file__).parent / "stack_moments.json", "w"))
