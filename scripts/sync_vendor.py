#!/usr/bin/env python3
"""Loader patch-anchor check (formerly also a vendor sync).

The backend sources under ``src/keras_tinygrad/_backend/`` ARE the source
of truth (since 2026-08-30; before that they were a snapshot of a sibling
keras clone, and this tool copied them across).  What remains is the one
check that needs no clone:

  --self-check  Every ``_loader.py`` patch anchor occurs exactly once in
                the *installed* stock keras, so the import hook's
                exact-string patches will apply.  (Also the default.)

The referee (Keras' own test suite against this package) lives in
``scripts/referee.sh``; it clones the pinned keras tag itself.
"""

import argparse
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_DIR = os.path.join(REPO_ROOT, "src", "keras_tinygrad", "_backend")
LOADER_PATH = os.path.join(REPO_ROOT, "src", "keras_tinygrad", "_loader.py")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_patch_table():
    """The _PATCHES dict from _loader.py, loaded without importing keras
    or the keras_tinygrad package (whose __init__ installs the hook)."""
    spec = importlib.util.spec_from_file_location("_kt_loader", LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._PATCHES


def module_source_path(keras_pkg_dir, module_name):
    """File for a `keras.*` module name under a keras package directory."""
    assert module_name == "keras" or module_name.startswith("keras.")
    rel = module_name.split(".")[1:]
    base = os.path.join(keras_pkg_dir, *rel)
    if os.path.isdir(base):
        return os.path.join(base, "__init__.py")
    return base + ".py"


def check_anchors(keras_pkg_dir, label, expect):
    """Verify every patch target is in the expected state. Two states exist:

    expect="unpatched" -- stock keras (installed wheel): every ANCHOR must
    occur exactly once, so the loader's exact-string patch will apply.
    expect="patched" -- a keras tree that already carries the edits
    in-tree: every REPLACEMENT must occur exactly once. (Kept for tooling
    that inspects a patched tree; the hook path never needs it.)

    Returns a list of problem strings (empty = all good).
    """
    problems = []
    for module_name, patches in load_patch_table().items():
        path = module_source_path(keras_pkg_dir, module_name)
        if not os.path.isfile(path):
            problems.append(f"{module_name}: file not found in {label}: {path}")
            continue
        source = read(path)
        for i, (anchor, replacement) in enumerate(patches):
            needle = anchor if expect == "unpatched" else replacement
            count = source.count(needle)
            if count != 1:
                first_line = anchor.splitlines()[0]
                detail = ""
                if expect == "unpatched" and count == 0 and replacement in source:
                    detail = " -- the REPLACEMENT text is present, so the file appears patched in-tree"
                elif expect == "patched" and count == 0 and anchor in source:
                    detail = (
                        " -- the ANCHOR is present unpatched, so the"
                        " clone is missing this in-tree edit (or its edit"
                        " diverged from the loader's replacement text)"
                    )
                problems.append(
                    f"{module_name} (patch {i}, anchor starts {first_line!r}):"
                    f" {expect} text matched {count} times in {label}"
                    f" (expected 1){detail}"
                )
    return problems


def report_anchors(keras_pkg_dir, label, expect):
    problems = check_anchors(keras_pkg_dir, label, expect)
    print(f"Anchor check against {label} ({keras_pkg_dir}, expect {expect}):")
    if problems:
        for p in problems:
            print(f"  DRIFT  {p}")
    else:
        print("  OK  all loader patch targets match exactly once")
    return not problems


def cmd_self_check():
    spec = importlib.util.find_spec("keras")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("error: no installed keras package found")
    keras_pkg_dir = list(spec.submodule_search_locations)[0]
    return 0 if report_anchors(keras_pkg_dir, "installed keras", expect="unpatched") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate loader patch anchors against the installed keras (default)",
    )
    parser.parse_args(argv)
    return cmd_self_check()


if __name__ == "__main__":
    sys.exit(main())
