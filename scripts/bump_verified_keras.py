#!/usr/bin/env python3
"""Append a referee-verified keras version to `VERIFIED_KERAS_VERSIONS` in
src/keras_tinygrad/_loader.py.

Used by .github/workflows/keras-watch.yml — the logic lives here, testable
offline (tests/test_bump_verified_keras.py), because the sed it replaces
silently produced `("3.15.1" "3.16.0")`: Python adjacent-string
concatenation, i.e. the nonsense version "3.15.13.16.0" warned to users.

    python scripts/bump_verified_keras.py 3.16.0   # idempotent
"""

import ast
import os
import re
import sys

LOADER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "keras_tinygrad", "_loader.py"
)
_PATTERN = re.compile(r"VERIFIED_KERAS_VERSIONS = \(([^)]*)\)")


def bump(text, version):
    """Return (new_text, new_tuple); appends `version` if absent."""
    if not re.fullmatch(r"\d+(\.\d+)*([a-z]+\d*)?", version):
        raise SystemExit(f"not a version string: {version!r}")
    match = _PATTERN.search(text)
    if not match:
        raise SystemExit("VERIFIED_KERAS_VERSIONS tuple not found")
    current = ast.literal_eval("(" + match.group(1) + ")")
    if version in current:
        return text, current
    new = current + (version,)
    rendered = "VERIFIED_KERAS_VERSIONS = (" + ", ".join(f'"{v}"' for v in new) + ")"
    out = text[: match.start()] + rendered + text[match.end() :]
    # Receipt: the rewritten tuple must parse back to exactly the intent.
    reparsed = ast.literal_eval("(" + _PATTERN.search(out).group(1) + ")")
    assert reparsed == new, f"rewrite corrupted the tuple: {reparsed}"
    return out, new


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(f"usage: {os.path.basename(sys.argv[0])} <keras version>  (e.g. 3.16.0)")
    version = sys.argv[1]
    with open(LOADER, encoding="utf-8") as f:
        text = f.read()
    out, new = bump(text, version)
    if out == text:
        print(f"{version} already in VERIFIED_KERAS_VERSIONS {new}")
        return
    with open(LOADER, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"VERIFIED_KERAS_VERSIONS -> {new}")


if __name__ == "__main__":
    main()
