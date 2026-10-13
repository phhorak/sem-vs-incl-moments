import sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, ".")
from lib.asimov import MAX_ORDER
from lib.maxent import hausdorff_check, raw_to_mu01
from lib.systematics import load_hqe_raw
cfg = yaml.safe_load(open("config.yaml")); T = dict(np.load(Path(cfg["paths"]["output"]) / "5" / "toys.npz"))
hc, ht = load_hqe_raw("inputs/hqe_likelihood_toys/likelihood_toys.h5")
n = len(T["z_gap"]); ih = np.random.default_rng(11).choice(len(ht), n, replace=False)
f_nom = float(T["bf_gap"]) / cfg["systematics"]["bf_incl"]
for k, o in enumerate(["mx2", "el", "q2"]):
    ic, it = hc[3*k:3*k+3], ht[ih, 3*k:3*k+3]
    sc, st = T["nom_sem0"][MAX_ORDER*k:MAX_ORDER*k+3], T["sem0_stat_ff_bfmode"][:, MAX_ORDER*k:MAX_ORDER*k+3]
    v = lambda m: m[..., 1] - m[..., 0] ** 2
    vi, vs, vit, vst = v(ic), v(sc), v(it), v(st)
    fmin = 1 - vit / vst
    print(f"{o}: mean incl {ic[0]:.3f} SEM {sc[0]:.3f} | Var incl {vi:.3f}+-{vit.std():.3f}  Var SEM {vs:.3f}+-{vst.std():.3f}  "
          f"=> f >= 1-Var_incl/Var_SEM = {1-vi/vs:.3f} (toys 16/50/84%: {np.round(np.percentile(fmin,[16,50,84]),3).tolist()}); Table-1 f = {f_nom:.3f}")
    lo, hi = cfg["data"]["support"][o]
    for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        c = 1 / f; r = c * it - (c - 1) * st
        adm = np.mean([hausdorff_check(raw_to_mu01(np.r_[1.0, x], lo, hi))[0] for x in r[:500]])
        print(f"    f={f:.1f}: admissible {adm:.2f}", end="")
    print()
