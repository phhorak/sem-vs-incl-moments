"""One-off: zero the B0 non-resonant D(*)pi l nu modes in output/3/cocktail.parquet, equivalent to
rerunning step 3 with the fixed _BF_TABLE (weight is proportional to bf). Keeps the original as
cocktail_pre_b0dpi_fix.parquet."""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT3 = Path(__file__).resolve().parents[1] / "output" / "3"
SRC, TMP = OUT3 / "cocktail.parquet", OUT3 / "cocktail_patched.parquet"
BAK = OUT3 / "cocktail_pre_b0dpi_fix.parquet"
MODES = [f"B0_{m}{l}" for m in ("D0pi", "Dpi0", "Dst0pi", "Dstpi0") for l in ("enu", "munu")]
ZERO = ["bf", "bf_unc", "weight", "total_weight"]



def record(store, dn, t):
    g = pa.table({"n": dn, "bf": t["bf"]}).group_by("n").aggregate([("bf", "max")])
    for n, b in zip(g["n"].to_pylist(), g["bf_max"].to_pylist()):
        store.setdefault(n, b)


pf = pq.ParquetFile(SRC)
names = pf.schema_arrow.names
assert all(c in names for c in ZERO), [c for c in ZERO if c not in names]
bf_before, bf_after, n_hit = {}, {}, {m: 0 for m in MODES}
with pq.ParquetWriter(TMP, pf.schema_arrow, compression="snappy") as w:
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        dn = pc.cast(t["decay_name"], pa.string())
        hit = pc.is_in(dn, value_set=pa.array(MODES))
        record(bf_before, dn, t)
        if pc.any(hit).as_py():
            for m in MODES:
                n_hit[m] += pc.sum(pc.equal(dn, m)).as_py() or 0
            for c in ZERO:
                col = t[c]
                t = t.set_column(names.index(c), c, pc.if_else(hit, pa.scalar(0, col.type), col))
        record(bf_after, dn, t)
        w.write_table(t)

print("rows zeroed per mode:", n_hit)
sem_b, sem_a = sum(bf_before.values()), sum(bf_after.values())
print(f"B_SEM before {sem_b:.5f}  after {sem_a:.5f}  (Table 1: 0.35620)")
assert all(n_hit[m] > 0 for m in MODES), "a mode was not found"
assert abs(sem_a - 0.3562) < 2e-4, "patched B_SEM does not match Table 1"
assert pq.ParquetFile(TMP).metadata.num_rows == pf.metadata.num_rows
SRC.rename(BAK)
TMP.rename(SRC)
print(f"-> {SRC} (original kept as {BAK.name})")
