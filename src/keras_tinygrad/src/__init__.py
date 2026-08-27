"""Protocol-layout shim: `keras_tinygrad.src` in the keras-mlx package shape.

The pluggable-backend keras resolves out-of-tree backends as
`keras_<name>.src` (star surface) with `trainer`/`layer`/`export`
submodules. Our backend sources live in `_backend/` with in-tree-style
internal imports (`keras.src.backend.tinygrad.*`); until the planned full
restructure (see docs/upstream/keras-plugin-poc.md), this shim registers
the `_backend` package under that in-tree alias BEFORE executing it —
plugin-side, no keras patching — and re-exports the protocol surface.
"""

import importlib.util
import sys

_ALIAS = "keras.src.backend.tinygrad"

if _ALIAS not in sys.modules:
    _spec = importlib.util.find_spec("keras_tinygrad._backend")
    _module = importlib.util.module_from_spec(_spec)
    # Both names registered before exec: the sources import themselves
    # through the in-tree alias path.
    sys.modules["keras_tinygrad._backend"] = _module
    sys.modules[_ALIAS] = _module
    try:
        _spec.loader.exec_module(_module)
    except BaseException:
        sys.modules.pop("keras_tinygrad._backend", None)
        sys.modules.pop(_ALIAS, None)
        raise

from keras_tinygrad.src import ops  # noqa: F401, E402

from keras.src.backend.common.name_scope import (  # noqa: E402
    name_scope,  # noqa: F401
)
from keras.src.backend.tinygrad import *  # noqa: F401, F403, E402
from keras.src.backend.tinygrad.core import (  # noqa: F401, E402
    Variable,  # noqa: F401, E402
    standardize_dtype_hook,
)

distribution_lib = None
