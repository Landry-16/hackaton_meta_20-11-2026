"""Regression audit helper: list every setting that differs between two checkpoints, from their file names.

  python tag_diff.py <reference checkpoint> <candidate checkpoint>
A clean comparison has exactly ONE difference (the change under test). Anything else (clean-frac, epochs, anchor,
alpha, init, seed...) is a confound: do not conclude from the score before removing it. Names are the tags written
by unlearn_distill.py / soup.py (underscore-separated tokens; the file stem prefix and epoch suffix are ignored).
"""
import re
import sys
from pathlib import Path


def tokens(name: str) -> dict:
    stem = Path(name).stem
    stem = re.sub(r"_e\d+$", "", stem)  # checkpoint epoch suffix
    out = {}
    for tok in stem.split("_")[1:]:     # drop the --out prefix (distill, caltech, ...)
        m = re.match(r"^([a-z]+(?:-[a-z]+)?)(.*)$", tok)
        key, val = (m.group(1), m.group(2)) if m else (tok, "")
        if key in out:
            key = f"{key}#{sum(k.startswith(key) for k in out)}"
        out[key] = val
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, cand = tokens(sys.argv[1]), tokens(sys.argv[2])
    diffs = [(k, ref.get(k, "(absent)"), cand.get(k, "(absent)")) for k in sorted(set(ref) | set(cand))
             if ref.get(k) != cand.get(k)]
    for k, r, c in diffs:
        print(f"  {k:>12}: {r!s:>14} -> {c}")
    print(f"{len(diffs)} difference(s): " + ("clean one-change comparison" if len(diffs) == 1 else
                                              "CONFOUNDED, do not conclude" if diffs else "identical settings"))


if __name__ == "__main__":
    main()
