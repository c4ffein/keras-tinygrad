#!/usr/bin/env python3
"""Keep the vendored backend in sync with the keras clone it snapshots.

The sources under ``src/keras_tinygrad/_backend/`` are a snapshot of
``<keras-clone>/keras/src/backend/tinygrad/`` and go stale as that clone
evolves.  This tool (stdlib only) has three modes:

  --check       Diff vendored files against the source-of-truth clone and
                verify the clone's in-tree keras-core edits are byte-equal
                to ``_loader.py``'s replacement texts (the clone is the
                PATCHED state; anchors are for unpatched stock keras).
                Exit 1 on any drift, with a per-file report.
  --sync        Copy the clone's backend files over the vendored snapshot.
                Vendored files that no longer exist in the clone are NOT
                deleted unless --force is given.
  --self-check  Validate only the ``_loader.py`` patch anchors, against the
                *installed* keras (no clone needed -- usable in CI).

The keras clone defaults to a sibling checkout of this repository
(``../keras``); override with --source PATH (path to the clone root, the
directory that contains ``keras/``).
"""

import argparse
import importlib.util
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_DIR = os.path.join(REPO_ROOT, "src", "keras_tinygrad", "_backend")
LOADER_PATH = os.path.join(REPO_ROOT, "src", "keras_tinygrad", "_loader.py")
DEFAULT_SOURCE = os.path.join(os.path.dirname(REPO_ROOT), "keras")
BACKEND_REL = os.path.join("keras", "src", "backend", "tinygrad")


def list_py(directory):
    """Names of the .py files directly in `directory` (no __pycache__)."""
    # This sync (and the wheel's package-data glob) is deliberately flat;
    # a subpackage appearing in the backend would be silently unsynced,
    # unchecked, and unshipped — so refuse loudly instead.
    subdirs = [
        name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name != "__pycache__"
    ]
    if subdirs:
        raise SystemExit(
            f"sync_vendor: backend directory {directory} contains "
            f"subdirectories {subdirs}; the flat sync/check/package-data "
            "story must be extended before adding subpackages."
        )
    return sorted(
        name for name in os.listdir(directory) if name.endswith(".py") and os.path.isfile(os.path.join(directory, name))
    )


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
    expect="patched" -- the source-of-truth clone, which carries the edits
    in-tree: every REPLACEMENT must occur exactly once, i.e. the clone's
    edit and the loader's replacement text are the same bytes.  (Checking
    anchors against the clone is wrong by construction: replacing patches
    like the standardize_dtype shim consume their anchor.)

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


def compare(source_dir):
    """(changed, added, removed) file-name lists, vendored vs source."""
    vendored = set(list_py(VENDORED_DIR))
    upstream = set(list_py(source_dir))
    changed = sorted(
        name
        for name in vendored & upstream
        if read(os.path.join(VENDORED_DIR, name)) != read(os.path.join(source_dir, name))
    )
    added = sorted(upstream - vendored)  # in the clone, not vendored yet
    removed = sorted(vendored - upstream)  # vendored, gone from the clone
    return changed, added, removed


def resolve_source(args):
    source_dir = os.path.join(args.source, BACKEND_REL)
    if not os.path.isdir(source_dir):
        sys.exit(
            f"error: source-of-truth backend not found: {source_dir}\n"
            "(--source must point at a keras clone root, the directory"
            " containing keras/)"
        )
    return source_dir


def cmd_check(args):
    source_dir = resolve_source(args)
    changed, added, removed = compare(source_dir)
    print(f"Vendored:        {VENDORED_DIR}")
    print(f"Source of truth: {source_dir}")
    if not (changed or added or removed):
        print("  OK  vendored snapshot matches the clone")
    for name in changed:
        print(f"  CHANGED  {name}")
    for name in added:
        print(f"  ADDED    {name}  (in clone, missing from vendored)")
    for name in removed:
        print(f"  REMOVED  {name}  (vendored, missing from clone)")
    anchors_ok = report_anchors(os.path.join(args.source, "keras"), "clone", expect="patched")
    ok = not (changed or added or removed) and anchors_ok
    return 0 if ok else 1


def cmd_sync(args):
    source_dir = resolve_source(args)
    changed, added, removed = compare(source_dir)
    if removed and not args.force:
        sys.exit(
            "error: vendored files not present in the clone: "
            + ", ".join(removed)
            + "\nRefusing to delete them; re-run with --force to remove."
        )
    for name in changed + added:
        shutil.copyfile(os.path.join(source_dir, name), os.path.join(VENDORED_DIR, name))
        print(f"  copied   {name}")
    if args.force:
        for name in removed:
            os.remove(os.path.join(VENDORED_DIR, name))
            print(f"  deleted  {name}")
    if not (changed or added or (args.force and removed)):
        print("  nothing to do; vendored snapshot already matches")
    # Anchors are informational on sync: they live in _loader.py, which this
    # tool never edits, so drift there needs a manual fix.
    anchors_ok = report_anchors(os.path.join(args.source, "keras"), "clone", expect="patched")
    return 0 if anchors_ok else 1


def cmd_self_check():
    spec = importlib.util.find_spec("keras")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("error: no installed keras package found")
    keras_pkg_dir = list(spec.submodule_search_locations)[0]
    return 0 if report_anchors(keras_pkg_dir, "installed keras", expect="unpatched") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="report drift between vendored snapshot and the clone (exit 1)",
    )
    mode.add_argument(
        "--sync",
        action="store_true",
        help="copy the clone's backend files over the vendored snapshot",
    )
    mode.add_argument(
        "--self-check",
        action="store_true",
        help="validate loader patch anchors against the installed keras only",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        metavar="PATH",
        help="keras clone root (default: sibling ../keras of this repo)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --sync: also delete vendored files absent from the clone",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return cmd_self_check()
    if args.sync:
        return cmd_sync(args)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
