# Star import FIRST, submodules after, as plain `import` statements: `core`
# has a bare `import math` that the star import (no `__all__`) re-exports
# as `ops.math`; a later `from keras_tinygrad.src.ops import math` would
# return that stdlib attribute without loading the submodule, whereas
# `import keras_tinygrad.src.ops.math` loads it and rebinds the attribute.
from keras_tinygrad.src.ops.core import *  # noqa: F403

import keras_tinygrad.src.ops.core  # noqa: E402, F401
import keras_tinygrad.src.ops.image  # noqa: E402, F401
import keras_tinygrad.src.ops.linalg  # noqa: E402, F401
import keras_tinygrad.src.ops.math  # noqa: E402, F401
import keras_tinygrad.src.ops.nn  # noqa: E402, F401
import keras_tinygrad.src.ops.numpy  # noqa: E402, F401
