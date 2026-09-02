"""tinygrad backend for stock Keras 3.

Usage (order matters -- this package must be imported before keras):

    import keras_tinygrad  # installs the import hook
    import keras           # KERAS_BACKEND=tinygrad now works

Importing this package also defaults KERAS_BACKEND to "tinygrad" if the
variable is unset (an explicit KERAS_BACKEND always wins).

If the installed keras supports backend plugins natively (a
`keras/src/backend/plugins.py` resolving the `keras.backends` entry-point
group), the import hook is NOT installed: keras discovers the backend
through the entry point declared in pyproject.toml, and `import keras`
alone suffices. The probe below is filesystem-only -- importing keras here
would defeat the hook's install-before-keras requirement on stock keras.
"""

import importlib.util
import os

from keras_tinygrad._loader import install

__version__ = "0.1.1"


def reset_device_rng():
    """Re-seed the on-device RNG stream at the next train-step draw.

    The trainer does this itself whenever it builds a train function, so
    `keras.utils.set_random_seed(s)` + build + `fit` is reproducible. Call
    this after `set_random_seed` only when no train function is rebuilt
    (e.g. a `train_on_batch` loop on an already-compiled model). Details:
    docs/device-rng.md.
    """
    from keras.src.backend.tinygrad import random as backend_random

    backend_random.reset_device_stream()


def _keras_supports_backend_plugins():
    spec = importlib.util.find_spec("keras")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        os.path.isfile(os.path.join(root, "src", "backend", "plugins.py")) for root in spec.submodule_search_locations
    )


os.environ.setdefault("KERAS_BACKEND", "tinygrad")
# KERAS_TINYGRAD_NO_HOOK=1: skip the import hook entirely. Needed when a
# pluggable-backend keras imports this package ITSELF (as `keras_
# tinygrad.src`, mid-keras-import — the hook's keras-first guard would
# fire); the filesystem probe below only detects OUR fork's marker, not
# the keras team's mechanism (their branch has no plugins.py). A proper
# marker/version probe replaces this when their design ships.
if os.environ.get("KERAS_TINYGRAD_NO_HOOK") != "1":
    if not _keras_supports_backend_plugins():
        install()
