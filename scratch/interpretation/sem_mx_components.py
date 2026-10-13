"""Which SEM category drives Var(Mx^2)? Per-category weight, mean, variance, and the scale of
each category's BF at which (1-f) Var_SEM = Var_incl (the positivity limit) at the Table-1 f."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from lib.systematics import read_parquet_downcast, load_hqe_raw
cfg = yaml.safe_load(open("config.yaml"))
df = read_parquet_downcast(Path(cfg["paths"]["output"]) / "3" / "cocktail.parquet", ["Mx", "total_weight", "category"],
                           category_cols=("category",))
df = df[np.isfinite(df.Mx) & np.isfinite(df.total_weight) & (df.total_weight > 0)]
x2 = df.Mx.to_numpy(float) ** 2; w = df.total_weight.to_numpy(float); cat = df.category.astype(str).to_numpy()
g = {c: (w[cat == c].sum(), (w * x2)[cat == c].sum(), (w * x2 ** 2)[cat == c].sum()) for c in np.unique(cat)}
W = sum(v[0] for v in g.values())
def var(scale):
    s0 = sum(v[0] * scale.get(c, 1) for c, v in g.items()); s1 = sum(v[1] * scale.get(c, 1) for c, v in g.items())
    s2 = sum(v[2] * scale.get(c, 1) for c, v in g.items()); m = s1 / s0; return m, s2 / s0 - m ** 2
m, v = var({})
hc, _ = load_hqe_raw("inputs/hqe_likelihood_toys/likelihood_toys.h5"); vi = hc[1] - hc[0] ** 2
T = dict(np.load(Path(cfg["paths"]["output"]) / "5" / "toys.npz")); f = float(T["bf_gap"]) / cfg["systematics"]["bf_incl"]
print(f"SEM <Mx2>={m:.3f} Var={v:.3f} | incl Var={vi:.3f} | f={f:.3f}: need Var_SEM <= {vi/(1-f):.3f}")
print(f"{'category':12s} {'weight':>7s} {'<Mx2>':>7s} {'sd':>6s} {'Var_SEM without it':>19s}")
for c, (w0, w1, w2) in sorted(g.items(), key=lambda kv: -kv[1][0]):
    mc, sd = w1 / w0, np.sqrt(max(w2 / w0 - (w1 / w0) ** 2, 0))
    print(f"{c:12s} {w0 / W:7.3f} {mc:7.2f} {sd:6.2f} {var({c: 0.0})[1]:19.3f}")
for c in [k for k in g if "pi" in k]:
    lam = next((l for l in np.linspace(1, 0, 201) if var({c: l})[1] <= vi / (1 - f)), None)
    print(f"scale of '{c}' BF needed for admissibility (others fixed): {lam}")
lam = next((l for l in np.linspace(1, 0, 201) if var({k: l for k in g if 'pi' in k})[1] <= vi / (1 - f)), None)
print("scale of all D(*)pi / D(*)pipi categories together:", lam)
