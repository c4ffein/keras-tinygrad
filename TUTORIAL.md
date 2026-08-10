# keras-tinygrad tutorial

Every python block on this page is executed, in order, in a single fresh
process by `tests/test_tutorial.py` (run it with `make tutorial`). If the
tutorial and the code ever disagree, CI goes red — this page cannot rot.

## Install

```sh
pip install keras-tinygrad     # or: uv add keras-tinygrad
```

That pulls stock PyPI `keras` (3.15.x) and `tinygrad` next to it. Nothing
is forked or vendored; see the README for how the import hook works.

## The one rule

`import keras_tinygrad` **before** `import keras`. Importing it installs
the import hook and — if `KERAS_BACKEND` is unset — defaults it to
`tinygrad` (an explicit setting always wins; the hook never hijacks a
backend you chose). If keras was already imported you get an immediate
`RuntimeError`, never a half-patched install.

```python
import keras_tinygrad  # must come first: installs the import hook

import keras
import numpy as np

assert keras.backend.backend() == "tinygrad"
print("keras", keras.__version__, "on tinygrad")
```

## Train something

Everything after the import is literally just Keras:

```python
rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
y = (x @ rng.normal(size=(8, 1)).astype("float32")).astype("float32")

model = keras.Sequential(
    [
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
history = model.fit(x, y, epochs=3, batch_size=32, verbose=0)
assert history.history["loss"][-1] < history.history["loss"][0]
print("loss went down:", [round(v, 3) for v in history.history["loss"]])
```

Evaluation and prediction work the same way:

```python
loss = model.evaluate(x, y, verbose=0)
preds = model.predict(x[:4], verbose=0)
assert preds.shape == (4, 1)
print("eval loss:", round(float(loss), 4))
```

## Quantization

int8 / int4 / float8 post-training quantization is supported:

```python
qmodel = keras.Sequential(
    [
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)
qmodel.set_weights(model.get_weights())
qmodel.quantize("int8")
qpreds = qmodel.predict(x[:4], verbose=0)
assert qpreds.shape == (4, 1)
print("int8 predictions:", np.round(np.asarray(qpreds).ravel(), 2).tolist())
```

## When something is missing, you hear about it

The backend's contract: **an error, never a wrong answer**. Anything not
implemented raises `NotImplementedError` loudly — there are no silent
numpy fallbacks, because a fallback would silently detach gradients.

```python
try:
    keras.ops.unique(np.array([1, 2, 2], dtype="int32"))
    raise AssertionError("unique unexpectedly worked — update the tutorial!")
except NotImplementedError as exc:
    print("unimplemented op raised loudly:", type(exc).__name__)
```

(If this block ever breaks the build, that's the tutorial working as
intended: the op got implemented and this page must catch up. Pick the
next loud stub from the README's known-gaps list.)

The same philosophy bounds complex-number support: complex values can
enter and leave (`view_as_complex` / `view_as_real` work), but complex
*arithmetic* is out of scope and refuses loudly rather than computing
float math on complex values (see `docs/complex-support.md`):

```python
z = keras.ops.view_as_complex(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
assert str(z.dtype) == "complex64"
back = keras.ops.view_as_real(z)
print("complex round-trip ok:", keras.ops.convert_to_numpy(back).tolist())

try:
    keras.ops.matmul(z, z)
    raise AssertionError("complex matmul unexpectedly worked!")
except NotImplementedError as exc:
    print("complex arithmetic refused loudly")
```

## Picking a different backend

`keras_tinygrad` only *defaults* the backend. `KERAS_BACKEND=numpy` (or
jax, torch, tensorflow) in the environment wins, and the import hook stays
inert for that process. This makes A/B-ing backends a matter of an env var.

## No clang on your box?

tinygrad's CPU jit shells out to `clang`. The `ziglang` PyPI wheel is a
drop-in substitute: a shim that translates the target triple and execs
`zig cc`. Point `CC` at the shim (the repo Makefile does this
automatically when clang is absent). Details in the README.

## What's supported

The README's support matrix is the authoritative, referee-verified list —
Keras' own test suite, per-op (see the README's tally for the current
numbers; this page deliberately repeats none of them). The
honest gaps: fused RNN kernels (correct-but-slow generic scan), sparse and
ragged tensors, TF-string preprocessing layers, complex arithmetic.

## Going further

- `examples/` — convnet, char-RNN, autoencoder, quantized inference.
- `tools/parity_fuzz.py` — randomized cross-backend parity fuzzing
  (`make fuzz`; finite-difference gradient checks via `make fuzz-grad`).
- `CONTRIBUTING.md` — the method: numpy backend is the reference, Keras'
  tests are the referee, stubs stay loud.
