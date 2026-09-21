"""The tinygrad backend for Keras, in the pluggable-backend package shape.

`keras_tinygrad.src` is what keras' pluggable_backend branch imports for
`KERAS_BACKEND=tinygrad` (`keras_<name>.src`), and what the import hook's
six patches import on stock keras 3.15.x. Same layout as keras' in-tree
backends and keras-team/keras-openvino: `ops/` (core, image, linalg,
math, nn, numpy), `random`, `rnn`, plus the optional `layer`, `trainer`,
`export` modules.

Keras >= 3.16 reads `backend.ops.numpy.x`; keras 3.15.x reads
`backend.numpy.x`. Both spellings are exported here, so one package
serves both.
"""

from keras.src.backend.common.name_scope import name_scope
from keras_tinygrad.src import ops
from keras_tinygrad.src import random
from keras_tinygrad.src.ops import core
from keras_tinygrad.src.ops import image
from keras_tinygrad.src.ops import linalg
from keras_tinygrad.src.ops import math
from keras_tinygrad.src.ops import nn
from keras_tinygrad.src.ops import numpy
from keras_tinygrad.src.ops.core import IS_THREAD_SAFE
from keras_tinygrad.src.ops.core import SUPPORTS_COMPLEX_DTYPES
from keras_tinygrad.src.ops.core import SUPPORTS_RAGGED_TENSORS
from keras_tinygrad.src.ops.core import SUPPORTS_SPARSE_TENSORS
from keras_tinygrad.src.ops.core import Variable
from keras_tinygrad.src.ops.core import cast
from keras_tinygrad.src.ops.core import compute_output_spec
from keras_tinygrad.src.ops.core import cond
from keras_tinygrad.src.ops.core import convert_to_numpy
from keras_tinygrad.src.ops.core import convert_to_tensor
from keras_tinygrad.src.ops.core import device_scope
from keras_tinygrad.src.ops.core import is_tensor
from keras_tinygrad.src.ops.core import random_seed_dtype
from keras_tinygrad.src.ops.core import shape
from keras_tinygrad.src.ops.core import standardize_dtype_hook
from keras_tinygrad.src.ops.core import vectorized_map
from keras_tinygrad.src.rnn import bidirectional_gru
from keras_tinygrad.src.rnn import bidirectional_lstm
from keras_tinygrad.src.rnn import cudnn_ok
from keras_tinygrad.src.rnn import gru
from keras_tinygrad.src.rnn import lstm
from keras_tinygrad.src.rnn import rnn

distribution_lib = None
