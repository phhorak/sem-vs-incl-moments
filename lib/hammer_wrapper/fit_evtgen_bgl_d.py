"""Hammer-BGL (with Blaschke poles) coefficients that reproduce EvtGen's B -> D BGL (Blaschke factors = 1)
for the decfile coefficients, so Hammer can use the generator FF as its input."""
import numpy as np

mB, mD = 5.27934, 1.86484
r = mD / mB
ap = [0.0126, -0.094, 0.34, -0.1]   # decfile BGL a+ and a0
a0 = [0.0115, -0.057, 0.12, 0.4]
nc, chiT, chiL = 2.6, 5.131e-4, 6.332e-3
wmax = (mB**2 + mD**2) / (2 * mB * mD)
w = np.linspace(1, wmax, 400)
z = (np.sqrt(w + 1) - np.sqrt(2)) / (np.sqrt(w + 1) + np.sqrt(2))
sub = (1 + r) * (1 - z) + 2 * np.sqrt(r) * (1 + z)
tp, tm = (mB + mD)**2 / mB**2, (mB - mD)**2 / mB**2
def P(poles):
    out = np.ones_like(z)
    for m in poles:
        zp = (np.sqrt(tp - (m / mB)**2) - np.sqrt(tp - tm)) / (np.sqrt(tp - (m / mB)**2) + np.sqrt(tp - tm))
        out *= (z - zp) / (1 - z * zp)
    return out
f_p = np.polyval(ap[::-1], z) / (1.1213 * (1 + z)**2 * np.sqrt(1 - z) * sub**-5)        # EvtGen
f_0 = np.polyval(a0[::-1], z) / (0.5299 * (1 + z) * (1 - z)**1.5 * sub**-4)
phip = 32 * np.sqrt(nc / (6 * np.pi * chiT * mB**2)) * r**2 * (1 + z)**2 * np.sqrt(1 - z) / sub**5   # Hammer
phi0 = (1 - r**2) * 8 * np.sqrt(nc / (8 * np.pi * chiL)) * r * (1 + z) * (1 - z)**1.5 / sub**4
for name, f, Pz, phi in (("ap", f_p, P([6.329, 6.920, 7.020]), phip), ("a0", f_0, P([6.716, 7.121]), phi0)):
    b = np.polyfit(z, Pz * phi * f, 3)[::-1]
    fh = np.polyval(b[::-1], z) / (Pz * phi)
    print(f"{name}: [{', '.join(f'{x:.6g}' for x in b)}]  max |f_Hammer/f_EvtGen - 1| = {np.max(np.abs(fh / f - 1)):.2e}")
