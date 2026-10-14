"""Threshold-0 inclusive (HQE fit) and SEM moments with one pairing of HQE-fit toys to the step-4 toys,
shared by steps 6-8.

Moments are raw m1..m3 of (mx2, el, q2), shape (9,); toys (n, 9). Toy i pairs SEM toy i (stat + FF + BF per mode)
and gap-normalisation draw z[i] with HQE-fit toy ih[i].
"""
import numpy as np

from lib.asimov import MAX_ORDER
from lib.maxent import raw_to_mu01
from lib.systematics import c_true

OBS = ["mx2", "el", "q2"]
IDX = np.r_[[MAX_ORDER * k + j for k in range(3) for j in range(3)]]
SL = {o: slice(3 * i, 3 * i + 3) for i, o in enumerate(OBS)} | {"joint": slice(0, 9)}


class Gap:
    def __init__(self, B):
        cfg, T = B.cfg, B.toys
        hc, ht = B.hqe
        self.cfg, self.sysc, self.sup = cfg, cfg["systematics"], cfg["data"]["support"]
        self.n = len(T["z_gap"])
        self.ih = np.random.default_rng([cfg["toys"]["seed"], 99]).choice(len(ht), self.n, replace=False)
        self.inc_c, self.inc_t = hc, ht[self.ih]
        self.sem_c = T["nom_sem0"][IDX]
        self.sem_v = {v: T[f"sem0_{v}"][:, IDX] for v in ("stat", "stat_ff", "stat_bfmode", "stat_ff_bfmode")}
        self.sem_t = self.sem_v["stat_ff_bfmode"]
        self.z, self.bf_gap = T["z_gap"], float(T["bf_gap"])
        self.f0 = self.bf_gap / self.sysc["bf_incl"]
        self.c0 = float(c_true(self.sysc, self.bf_gap))

    def c(self, zscale=1.0):
        """c_true per toy, (n, 1); zscale scales the B_gap fluctuation."""
        return c_true(self.sysc, self.bf_gap, zscale * self.z)[:, None]

    def inc(self, mx2_scale=1.0):
        """Inclusive toys with the M_X^2 fluctuations scaled."""
        t = self.inc_t.copy()
        t[:, :3] = self.inc_c[:3] + mx2_scale * (t[:, :3] - self.inc_c[:3])
        return t

    def scale(self, raw, sup=None):
        """Raw moments (n, 9) -> moments m1..m3 of the variables mapped to [0, 1] on the support."""
        s = sup or self.sup
        return np.concatenate([raw_to_mu01(np.c_[np.ones(len(raw)), raw[:, SL[o]]].T, *s[o])[1:].T for o in OBS], axis=1)

    def residual(self, changes=(), mx2_scale=1.0, zscale=1.0, sup=None, c=None):
        """Scaled gap moments (nominal (9,), toys (n, 9)) and c_true for changes = [(share, raw moments), ...] added
        to the nominal SEM (negative share removes; B_SEM and c_true follow). c fixes c_true for every toy."""
        k = 1 + sum(sv for sv, _ in changes)
        add = sum((sv * mv for sv, mv in changes), np.zeros(9))
        b = self.sysc["bf_incl"]
        cs = b / (b - (b - self.bf_gap) * k) if c is None else c
        ct = cs / self.c0 * self.c(zscale) if c is None else c
        return (self.scale((cs * self.inc_c - (cs - 1) * (self.sem_c + add) / k)[None], sup)[0],
                self.scale(ct * self.inc(mx2_scale) - (ct - 1) * (self.sem_t + add) / k, sup), cs)
