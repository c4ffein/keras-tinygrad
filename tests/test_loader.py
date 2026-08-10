"""Tests for the import hook (`keras_tinygrad._loader`).

The hook's guarantees are all-or-nothing: either keras imports fully
patched, or the import fails loudly. A Keras process is locked to one
backend at import time and the hook itself is process-global state, so
every scenario that imports keras runs in a fresh subprocess; only the
pure-function tests (`_apply_patches`, anchor drift against the installed
keras sources) run in-process.

Requires: this package and stock keras installed in the running
interpreter's environment (the same requirement as using the package).
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

from keras_tinygrad._loader import _PATCHES, _apply_patches

import pytest

TIMEOUT = 300  # keras imports compile a lot on first touch


def run_py(code: str, backend: str | None) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter with KERAS_BACKEND controlled."""
    env = dict(os.environ)
    env.pop("KERAS_BACKEND", None)
    if backend is not None:
        env["KERAS_BACKEND"] = backend
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIMEOUT,
    )


def check(proc: subprocess.CompletedProcess) -> str:
    assert proc.returncode == 0, f"child failed:\n{proc.stderr}"
    return proc.stdout


# ---------------------------------------------------------------------------
# Subprocess scenarios
# ---------------------------------------------------------------------------


def test_hook_defaults_backend_to_tinygrad():
    out = check(
        run_py(
            """
        import keras_tinygrad
        import keras
        print(keras.backend.backend())
        """,
            backend=None,
        )
    )
    assert out.strip() == "tinygrad"


def test_explicit_other_backend_is_respected():
    # The hook must not hijack an explicit KERAS_BACKEND — and the patched
    # modules must still import and compute under the other backend.
    out = check(
        run_py(
            """
        import keras_tinygrad
        import keras
        print(keras.backend.backend())
        print(int(keras.ops.convert_to_numpy(keras.ops.add(2, 3))))
        """,
            backend="numpy",
        )
    )
    assert out.splitlines() == ["numpy", "5"]


def test_keras_imported_first_raises():
    proc = run_py(
        """
        import keras
        try:
            import keras_tinygrad
        except RuntimeError as exc:
            print("RuntimeError:", exc)
        else:
            raise SystemExit("no RuntimeError — half-patched install possible")
        """,
        backend="numpy",
    )
    out = check(proc)
    assert "RuntimeError:" in out
    assert "BEFORE keras" in out


def test_double_import_is_idempotent():
    out = check(
        run_py(
            """
        import sys
        import keras_tinygrad
        import keras_tinygrad  # no-op
        from keras_tinygrad._loader import TinygradBackendFinder, install
        install()  # explicit second call is also a no-op
        finders = [f for f in sys.meta_path
                   if isinstance(f, TinygradBackendFinder)]
        print(len(finders))
        """,
            backend=None,
        )
    )
    assert out.strip() == "1"


def test_backend_package_served_from_this_package():
    out = check(
        run_py(
            """
        import keras_tinygrad
        import keras.src.backend.tinygrad as b
        print(b.__file__)
        """,
            backend=None,
        )
    )
    assert os.path.join("keras_tinygrad", "_backend") in out


# ---------------------------------------------------------------------------
# In-process: the anchor rule itself
# ---------------------------------------------------------------------------

MODULE = "keras.src.backend"
ANCHOR = _PATCHES[MODULE][0][0]


def test_apply_patches_zero_matches_fails_loudly():
    with pytest.raises(ImportError, match="matched 0 times"):
        _apply_patches(MODULE, "def backend():\n    return 'x'\n")


def test_apply_patches_two_matches_fails_loudly():
    with pytest.raises(ImportError, match="matched 2 times"):
        _apply_patches(MODULE, ANCHOR + "\n\n" + ANCHOR)


def test_apply_patches_exact_match_rewrites():
    patched = _apply_patches(MODULE, "prefix\n" + ANCHOR + "\nsuffix")
    assert 'backend() == "tinygrad"' in patched
    # The unknown-backend raise survives the patch, once, after our branch.
    assert patched.count(ANCHOR) == 1
    assert patched.index('backend() == "tinygrad"') < patched.index(ANCHOR)


def test_anchors_match_installed_keras_exactly_once():
    # File-level check against the installed keras (no keras import: reading
    # source only). This is the same guarantee sync_vendor.py --self-check
    # gives, kept here so plain `pytest` catches an unsupported keras.
    spec = importlib.util.find_spec("keras")
    assert spec and spec.submodule_search_locations
    root = spec.submodule_search_locations[0]
    for module, patches in _PATCHES.items():
        rel = module.split(".", 1)[1].replace(".", os.sep)
        for candidate in (rel + ".py", os.path.join(rel, "__init__.py")):
            path = os.path.join(root, candidate)
            if os.path.exists(path):
                break
        else:
            pytest.fail(f"no source file found for {module}")
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        for anchor, _ in patches:
            count = source.count(anchor)
            assert count == 1, f"{module}: anchor matched {count} times (expected 1) in {path}"
