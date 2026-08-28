#!/usr/bin/env python3
"""Verify the README's numeric claims are internally consistent.

Checks (loud exit 1 on any mismatch):
  1. Every SUPPORT_MATRIX row: coverage == passed / (passed + failed),
     rounded to one decimal (skips excluded by design — the metric is
     "correctness of what runs", per scripts/gen_support_matrix.py).
  2. The TOTAL row is the column-wise sum of the suite rows, and its
     coverage follows the same formula.
  3. The TALLY block's headline percentage matches its own
     passed/failed numbers.

This cannot prove the numbers against reality (only a referee run can —
CONTRIBUTING.md: update numbers only with a tally); it proves the README
never contradicts itself. Run via `make readme-check`; CI runs it on
every push.
"""

import re
import sys

FAIL = 0


def err(msg):
    global FAIL
    FAIL = 1
    print(f"READM-E-CHECK FAIL: {msg}")


def block(text, name):
    m = re.search(rf"<!-- {name} -->(.*?)<!-- /{name} -->", text, re.S)
    if not m:
        err(f"marker block {name} not found")
        return ""
    return m.group(1)


def main():
    text = open("README.md", encoding="utf-8").read()

    # --- SUPPORT_MATRIX ---
    matrix = block(text, "SUPPORT_MATRIX")
    rows = re.findall(
        r"^\| (.+?) \| \**(\d+)\** \| \**(\d+)\** \| \**(\d+)\** \|"
        r" \**([\d.]+)%\** \|",
        matrix,
        re.M,
    )
    suites = [(n, int(p), int(f), int(s), float(c)) for n, p, f, s, c in rows]
    if not suites:
        err("no matrix rows parsed")
    total_row = None
    sums = [0, 0, 0]
    for name, p, f, s, c in suites:
        if name.strip("* ") == "TOTAL":
            total_row = (p, f, s, c)
            continue
        want = round(100.0 * p / (p + f), 1) if (p + f) else 100.0
        if abs(want - c) > 0.05:
            err(f"row {name!r}: coverage {c}% but {p}/{p}+{f} = {want}%")
        sums = [sums[0] + p, sums[1] + f, sums[2] + s]
    if total_row is None:
        err("no TOTAL row found")
    else:
        p, f, s, c = total_row
        if [p, f, s] != sums:
            err(f"TOTAL row {p}/{f}/{s} != column sums {sums}")
        want = round(100.0 * p / (p + f), 1) if (p + f) else 100.0
        if abs(want - c) > 0.05:
            err(f"TOTAL coverage {c}% but computed {want}%")

    # --- TALLY ---
    tally = block(text, "TALLY")
    m = re.search(
        r"([\d,]+) passed /\s*(\d+) failed / (\d+) skipped \(([\d.]+)%\)",
        tally,
    )
    if not m:
        err("TALLY headline numbers not parsed")
    else:
        p = int(m.group(1).replace(",", ""))
        f = int(m.group(2))
        pct = float(m.group(4))
        want = round(100.0 * p / (p + f), 1)
        if abs(want - pct) > 0.05:
            err(f"TALLY says {pct}% but {p}/{p}+{f} = {want}%")

    if FAIL:
        sys.exit(1)
    print(f"readme-check OK: {len(suites) - 1} suite rows + TOTAL + TALLY arithmetically consistent")


if __name__ == "__main__":
    main()
