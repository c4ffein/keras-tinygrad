# keras-tinygrad examples

Small, self-contained Keras 3 scripts running on the tinygrad backend. Every
example needs this two-line environment preamble on this box:

```bash
export KERAS_BACKEND=tinygrad   # route Keras 3 onto the tinygrad backend
export CC=zigcc                 # clang-less machines only: tinygrad's CPU device shells out to $CC; the zigcc shim stands in for clang
```

Then run any script with the venv Python, e.g.:

```bash
python examples/convnet_mnist.py --epochs 2
```

Every script starts with the same header — `import keras_tinygrad` **must**
come before `import keras` so the backend is registered when Keras loads:

```python
import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
```

## Index

| File | What it shows | Runtime expectation |
|---|---|---|
| [`mlp_smoke.py`](mlp_smoke.py) | Minimal compile/fit/predict smoke test (Dense regression) | seconds |
| [`convnet_mnist.py`](convnet_mnist.py) | Small CNN (Conv2D/MaxPooling2D) on MNIST-shaped data; `--real` fetches actual MNIST via `keras.datasets` | seconds synthetic; ~minutes with `--real` on CPU |
| [`char_rnn.py`](char_rnn.py) | Character-level SimpleRNN language model on an embedded corpus, plus temperature sampling | tens of seconds; sampling is one predict per character |
| [`autoencoder.py`](autoencoder.py) | Dense autoencoder on synthetic manifold data; predict + per-sample reconstruction error vs noise | seconds |
| [`quantized_inference.py`](quantized_inference.py) | Train float32, `model.quantize("int8")`, measure prediction drift — int8 works on this backend today | seconds |

All scripts accept `--epochs` and `--batch-size` where sensible; defaults are
tuned to finish quickly on a CPU device while still showing the loss move.

QR-based initializers (e.g. `Orthogonal`, SimpleRNN's default recurrent
initializer) are supported since the linalg wave — `char_rnn.py` uses the
default on purpose, as a standing regression test.
