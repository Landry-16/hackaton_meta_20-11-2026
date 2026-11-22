"""Build a probe file from a base submission: python make_probe.py BASE.csv OUT.csv add=ep-32,ep-33 remove=ep-14
'add' puts endpoints into their family's base group, 'remove' makes them singletons."""
import csv, sys
base, out, *ops = sys.argv[1:]
add = {e for o in ops if o.startswith("add=") for e in o[4:].split(",") if e}
rem = {e for o in ops if o.startswith("remove=") for e in o[7:].split(",") if e}
rows = list(csv.DictReader(open(base)))
for r in rows:
    if r["endpoint"] in add: r["model"] = f"{r['family']}-base"
    if r["endpoint"] in rem: r["model"] = f"{r['family']}-{r['endpoint']}"
assert sorted(r["endpoint"] for r in rows) == [f"ep-{i:02d}" for i in range(1, 34)]
with open(out, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["endpoint", "family", "model"])
    for r in rows: w.writerow([r["endpoint"], r["family"], r["model"]])
for F in ("F1", "F2", "F3"):
    print(F, sorted(r["endpoint"] for r in rows if r["model"] == f"{F}-base"))
