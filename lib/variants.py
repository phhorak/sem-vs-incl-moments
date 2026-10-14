"""SEM modes and SEM variants from data/sem_component_moments.json (step 3).

A variant is (changes, hqe_mx2_scale): changes = [(share, raw moments m1..m3 of mx2, el, q2), ...] added to the
nominal SEM, a negative share removing a component; the SEM moments are linear in the component weights.
  phsp_Dpipi        D(*)pipi with the 5-body PHSP lineshape
  noFF_broad        no Hammer weight for the broad states (D0*, D1', D(*)pipi), MCAmbulance kept
  bare_evtgen_broad neither Hammer nor MCAmbulance for the broad states (EvtGen lineshape as generated)
  hqe_mx2_x2        HQE M_X^2 moment fluctuations doubled
  split_Dp1_f       D1':D0* weight fraction of D(*)pipi set to f
  lhcb_*            LHCb-style cocktail: D** fractions (11874430/12873450), ISGW2 D** shapes, higher D** (D(2S),
                    D*(2S), D(2750), D(3000) -> D* pi pi) in place of D(*)pipi, no D_s K; and all together
"""
import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "data" / "sem_component_moments.json"
DSS = ("D1", "D0st", "Dp1", "D2st")
LHCB_DSS = {"D0st": (0.2941 + 0.2970) / 2, "D1": (0.2941 + 0.2772) / 2, "Dp1": (0.2647 + 0.3069) / 2,
            "D2st": (0.1471 + 0.1188) / 2}
LHCB_HIGHER = {"D2S": 0.45, "Dst2S": 0.05, "D2750": 0.45, "D3000": 0.05}


def load():
    return json.load(open(PATH))


def modes(C):
    """Raw moments of the SEM modes; D** combines its four states with their SEM shares."""
    s = sum(C["D**:" + k]["share"] for k in DSS)
    return {k: np.array(C[k]["moments"]) for k in ("D", "D*", "D(*) pi pi", "Ds(*) K")} \
        | {"D**": sum(C["D**:" + k]["share"] * np.array(C["D**:" + k]["moments"]) for k in DSS) / s}


def variants(C):
    m = lambda k, tag="": np.array(C[k]["moments" + ("_" + tag if tag else "")])
    sh = lambda k: C[k]["share"]
    swap = lambda comps, tag: [x for k in comps for x in ((-sh(k), m(k)), (sh(k), m(k, tag)))]
    s_nr, s_dss = sh("D(*) pi pi"), sum(sh("D**:" + k) for k in DSS)
    lhcb = C.get("lhcb", {})
    no_dss = [(-sh("D**:" + k), m("D**:" + k)) for k in DSS]
    higher = [(-s_nr, m("D(*) pi pi"))] + [(s_nr * f, np.array(lhcb[k])) for k, f in LHCB_HIGHER.items() if k in lhcb]
    no_dsk = [(-sh("Ds(*) K"), m("Ds(*) K"))]
    out = {"nominal": ([], 1.0),
           "phsp_Dpipi": ([(-s_nr, m("D(*) pi pi")), (s_nr, m("D(*) pi pi", "phsp"))], 1.0),
           "noFF_broad": (swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noFF"), 1.0),
           "bare_evtgen_broad": (swap(["D**:D0st", "D**:Dp1", "D(*) pi pi"], "noMCA_noFF"), 1.0),
           "hqe_mx2_x2": ([], 2.0)}
    for f in (0.0, 0.25, 0.75, 1.0):
        out[f"split_Dp1_{f:.2f}"] = ([(-s_nr, m("D(*) pi pi")),
                                      (s_nr, f * m("D(*) pi pi", "parent_Dp1") + (1 - f) * m("D(*) pi pi", "parent_D0st"))], 1.0)
    if len(lhcb) == 8:
        out |= {"lhcb_dss_fractions": (no_dss + [(s_dss * f, m("D**:" + k)) for k, f in LHCB_DSS.items()], 1.0),
                "lhcb_isgw2_dss": (no_dss + [(sh("D**:" + k), np.array(lhcb[k])) for k in DSS], 1.0),
                "lhcb_higher_dss": (higher, 1.0), "lhcb_no_dsk": (no_dsk, 1.0),
                "lhcb_all": (no_dss + [(s_dss * f, np.array(lhcb[k])) for k, f in LHCB_DSS.items()] + higher + no_dsk, 1.0)}
    return out
