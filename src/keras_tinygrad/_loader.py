"""Import-time machinery that grafts the tinygrad backend onto stock Keras.

Stock Keras 3 hardcodes its backend dispatch as ``elif`` chains that
``raise`` at import time for unknown backends, in four modules, plus two
behavioral special-cases: ``standardize_dtype`` and ``DynamicBackend``'s
per-backend branch.  There is no plugin hook, so we install a
:class:`importlib.abc.MetaPathFinder` at the front of ``sys.meta_path``
(before Keras is imported) that does two things:

1. Serves ``keras.src.backend.tinygrad`` from this package's ``_backend/``
   directory (verbatim backend sources).  Only the package name itself is
   intercepted; once its ``__path__`` points at ``_backend/``, the stock
   ``PathFinder`` resolves every submodule import normally.

2. Intercepts exactly six Keras modules and execs a patched copy of their
   source, produced by exact-string replacement (see ``_PATCHES``).  Every
   anchor must match exactly once or the import fails loudly with a
   version-mismatch error -- we never guess and never let Keras fall
   through to its own "Unable to import backend" error.

Everything else imports untouched.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_BACKEND_PKG = "keras.src.backend.tinygrad"
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "_backend")

# ---------------------------------------------------------------------------
# The patch table: module name -> list of (anchor, replacement).
# Anchors are exact source strings from stock keras (verified against the
# 3.15.0 and 3.15.1 wheels); each must occur exactly once.
# ---------------------------------------------------------------------------

_TINYGRAD_BACKEND_BRANCH = """elif backend() == "tinygrad":
    from keras.src.backend.tinygrad import *  # noqa: F403
    from keras.src.backend.tinygrad.core import Variable as BackendVariable

    distribution_lib = None
else:
    raise ValueError(f"Unable to import backend : {backend()}")"""

_TINYGRAD_DTYPE_SHIM = """    dtype = dtypes.PYTHON_DTYPES_MAP.get(dtype, dtype)
    if type(dtype).__module__.split(".")[0] == "tinygrad":
        # tinygrad DType.name spellings ("float", "half", ...) don't match
        # the Keras names; map through the backend's table.
        from keras.src.backend.tinygrad.core import to_keras_dtype

        dtype = to_keras_dtype(dtype)
    if hasattr(dtype, "name"):"""

_PATCHES = {
    # The backend loader itself: the elif chain that star-imports the
    # selected backend module and raises ValueError for unknown names.
    "keras.src.backend": [
        (
            'else:\n    raise ValueError(f"Unable to import backend : {backend()}")',
            _TINYGRAD_BACKEND_BRANCH,
        ),
    ],
    # standardize_dtype(): tinygrad DType objects have a .name attribute
    # whose spellings ("float", "half", "char", ...) are not Keras dtype
    # names, so the generic `hasattr(dtype, "name")` path would produce
    # invalid dtypes.  Map them through the backend's table first.
    "keras.src.backend.common.variables": [
        (
            '    dtype = dtypes.PYTHON_DTYPES_MAP.get(dtype, dtype)\n    if hasattr(dtype, "name"):',
            _TINYGRAD_DTYPE_SHIM,
        ),
    ],
    # Layer's per-backend mixin class: elif chain raising RuntimeError.
    "keras.src.layers.layer": [
        (
            'else:\n    raise RuntimeError(\n        f"Backend'
            " '{backend.backend()}' must implement a layer mixin class.\"",
            'elif backend.backend() == "tinygrad":\n'
            "    from keras.src.backend.tinygrad.layer import"
            " TinygradLayer as BackendLayer\n"
            'else:\n    raise RuntimeError(\n        f"Backend'
            " '{backend.backend()}' must implement a layer mixin class.\"",
        ),
    ],
    # Model's per-backend Trainer class: elif chain raising RuntimeError.
    "keras.src.models.model": [
        (
            'else:\n    raise RuntimeError(\n        f"Backend'
            " '{backend.backend()}' must implement the Trainer class.\"",
            'elif backend.backend() == "tinygrad":\n'
            "    from keras.src.backend.tinygrad.trainer import"
            " TinygradTrainer as Trainer\n"
            'else:\n    raise RuntimeError(\n        f"Backend'
            " '{backend.backend()}' must implement the Trainer class.\"",
        ),
    ],
    # DynamicBackend.__getattr__: if/return chain with NO else-raise — an
    # unknown backend silently returns None, so preprocessing layers that
    # route through DynamicBackend (image augmentation, MelSpectrogram)
    # would crash with "'NoneType' object has no attribute ...".  Append a
    # tinygrad branch after the openvino one.
    "keras.src.utils.backend_utils": [
        (
            '        if self._backend == "openvino":\n'
            "            module = importlib.import_module("
            '"keras.src.backend.openvino")\n'
            "            return getattr(module, name)",
            '        if self._backend == "openvino":\n'
            "            module = importlib.import_module("
            '"keras.src.backend.openvino")\n'
            "            return getattr(module, name)\n"
            '        if self._backend == "tinygrad":\n'
            "            module = importlib.import_module("
            '"keras.src.backend.tinygrad")\n'
            "            return getattr(module, name)",
        ),
    ],
    # ExportArchive: elif chain raising RuntimeError at import time (the
    # module is imported unconditionally via keras.src.models.model ->
    # ... -> keras.src.export).
    "keras.src.export.saved_model": [
        (
            "else:\n    raise RuntimeError(\n        f\"Backend '{backend.backend()}' must implement ExportArchive.\"",
            'elif backend.backend() == "tinygrad":\n'
            "    from keras.src.backend.tinygrad.export import (\n"
            "        TinygradExportArchive as BackendSavedModelExportArchive,\n"
            "    )\n"
            'else:\n    raise RuntimeError(\n        f"Backend'
            " '{backend.backend()}' must implement ExportArchive.\"",
        ),
    ],
}


class _PatchedLoader(importlib.abc.Loader):
    """Execs a patched copy of a stock keras module's source."""

    def __init__(self, fullname, real_spec):
        self._fullname = fullname
        self._real_spec = real_spec

    def create_module(self, spec):
        return None  # default module creation

    def get_source(self, fullname):
        source = self._real_spec.loader.get_source(fullname)
        if source is None:
            raise ImportError(f"keras_tinygrad: cannot read source of {fullname!r} (bytecode-only install?)")
        return _apply_patches(fullname, source)

    def exec_module(self, module):
        source = self.get_source(self._fullname)
        code = compile(source, self._real_spec.origin or f"<{self._fullname}>", "exec")
        exec(code, module.__dict__)


def _apply_patches(fullname, source):
    for anchor, replacement in _PATCHES[fullname]:
        count = source.count(anchor)
        if count != 1:
            raise ImportError(
                f"keras_tinygrad: patch anchor for {fullname!r} matched"
                f" {count} times (expected 1). Your installed keras version"
                " is not supported by this keras_tinygrad build --"
                " the upstream source changed. Refusing to continue."
            )
        source = source.replace(anchor, replacement)
    return source


class TinygradBackendFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _BACKEND_PKG:
            return importlib.util.spec_from_file_location(
                fullname,
                os.path.join(_BACKEND_DIR, "__init__.py"),
                submodule_search_locations=[_BACKEND_DIR],
            )
        if fullname not in _PATCHES:
            return None
        # Locate the real module without re-entering sys.meta_path.
        real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if real_spec is None or real_spec.loader is None:
            return None  # let the normal machinery produce its error
        loader = _PatchedLoader(fullname, real_spec)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=real_spec.origin,
            is_package=bool(real_spec.submodule_search_locations),
        )
        if real_spec.submodule_search_locations:
            # Keep the real package __path__ (e.g. keras/src/backend/) so
            # sibling submodules resolve from the stock install.
            spec.submodule_search_locations = list(real_spec.submodule_search_locations)
        return spec


_FINDER = None

# Keras versions the full referee battery has been run against (the
# keras-watch canary opens a PR appending here when a new release passes
# its checks; the FULL battery is still a manual pre-merge step). This
# list gates only a WARNING: the anchor exactly-once checks below are the
# real, never-bypassable compatibility proof — an unlisted version whose
# anchors match works and merely warns; one whose anchors drifted fails
# loudly regardless of any list.
VERIFIED_KERAS_VERSIONS = ("3.15.0", "3.15.1")


def _warn_if_unverified_keras():
    if os.environ.get("KERAS_TINYGRAD_NO_VERSION_WARNING") == "1":
        return
    try:
        from importlib.metadata import version

        installed = version("keras")
    except Exception:
        return  # no metadata -> the anchor checks will speak for themselves
    if installed not in VERIFIED_KERAS_VERSIONS:
        import warnings

        warnings.warn(
            f"keras {installed} has not been referee-verified with this "
            f"keras-tinygrad release (verified: "
            f"{', '.join(VERIFIED_KERAS_VERSIONS)}). The loader's "
            "exact-once anchor checks still verify patchability — a "
            "mismatch fails loudly at import. Silence this warning with "
            "KERAS_TINYGRAD_NO_VERSION_WARNING=1.",
            stacklevel=3,
        )


def install():
    """Install the finder (idempotent). Must run before keras is imported."""
    global _FINDER
    if _FINDER is not None:
        return
    already = [name for name in list(_PATCHES) + [_BACKEND_PKG] if name in sys.modules]
    if already:
        raise RuntimeError(
            f"keras_tinygrad must be imported BEFORE keras: these keras modules are already loaded unpatched: {already}"
        )
    _warn_if_unverified_keras()
    _FINDER = TinygradBackendFinder()
    sys.meta_path.insert(0, _FINDER)
