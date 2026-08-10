"""tinygrad backend for stock Keras 3.

Usage (order matters -- this package must be imported before keras):

    import keras_tinygrad  # installs the import hook
    import keras           # KERAS_BACKEND=tinygrad now works

Importing this package also defaults KERAS_BACKEND to "tinygrad" if the
variable is unset (an explicit KERAS_BACKEND always wins).
"""

import os

from keras_tinygrad._loader import install

__version__ = "0.1.0"

os.environ.setdefault("KERAS_BACKEND", "tinygrad")
install()
