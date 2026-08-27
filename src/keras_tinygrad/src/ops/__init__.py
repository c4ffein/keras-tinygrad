"""Protocol `ops` namespace (keras calls `backend.ops.is_tensor`,
`backend.ops.numpy.where`, ...). Aggregates the existing flat `_backend`
modules into the pluggable-era layout without moving them; the real
restructure replaces this shim (docs/upstream/keras-plugin-poc.md)."""

import keras_tinygrad.src  # noqa: F401 -- registers the backend alias

from keras.src.backend.tinygrad import (
    core,  # noqa: F401
    image,  # noqa: F401
    linalg,  # noqa: F401
    math,  # noqa: F401
    nn,  # noqa: F401
    numpy,  # noqa: F401
)
from keras.src.backend.tinygrad.core import *  # noqa: F401, F403
