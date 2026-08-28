# keras-tinygrad

---

*WARNING — this is a vibe-engineering experiment - not affiliated with the Keras team or the tiny corp.*  
It may still be useful to you, and will be updated as
[the RFC for Keras backends as plugins](https://github.com/keras-team/keras/issues/23523) advances.

---

A [tinygrad](https://github.com/tinygrad/tinygrad) backend for stock Keras 3.
No fork, no vendored Keras — installs next to the stock PyPI wheel:
`pip install keras-tinygrad` / `uv pip install keras-tinygrad`.
Every release is certified by a public workflow that installs the package
**from PyPI** — no repo checkout — and trains with it:
[pypi-verify](https://github.com/c4ffein/keras-tinygrad/actions/workflows/pypi-verify.yml).

Also runs plugin-style on Keras' in-development
[pluggable-backend branch](https://github.com/keras-team/keras/tree/pluggable_backend)
with zero patches (see the [pilot report](https://github.com/c4ffein/keras-tinygrad/blob/main/docs/upstream/pluggable-branch-pilot.md)).

```python
import keras_tinygrad  # must come first: installs the import hook

import keras  # KERAS_BACKEND=tinygrad
import numpy as np

x = np.random.normal(size=(256, 8)).astype("float32")
y = x @ np.random.normal(size=(8, 1)).astype("float32")

model = keras.Sequential(
    [
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
model.fit(x, y, epochs=5, batch_size=32)
```

That is the whole API. Everything after the first line is literally just Keras.

## Install

```sh
pip install keras-tinygrad
```

New here? Start with **[TUTORIAL.md](https://github.com/c4ffein/keras-tinygrad/blob/main/TUTORIAL.md)** — every code block on
that page is executed by CI, so it cannot rot.

Works against the stock PyPI `keras` wheel (3.15.x — 3.15.0 verified by the
full test tally below; 3.15.1 verified by the loader test suite and the
executable tutorial, training included) and `tinygrad` 0.13; the dependency
pin says the same thing (`keras>=3.15,<3.16`). On any other Keras version
the import fails loudly at the anchor check — see below.
The only rule: `import keras_tinygrad` **before** `import keras` (importing it
also defaults `KERAS_BACKEND=tinygrad`; an explicit setting always wins). If
Keras was already imported, you get a `RuntimeError` — never a half-patched
install.

## How it can be a backend without a fork

Keras 3 has no backend plugin hook. This package installs a
`sys.meta_path` finder that serves `keras.src.backend.tinygrad` from its own
sources and surgically patches the six Keras modules that hardcode backend
dispatch. Each patch is an exact-string anchor that must match **exactly
once** — on an unsupported Keras version the import fails loudly with a
version-mismatch error instead of guessing. Details in
[docs/how-it-works.md](https://github.com/c4ffein/keras-tinygrad/blob/main/docs/how-it-works.md).

## Status

<!-- TALLY -->
**Keras' own full layers test tree (preprocessing included): 1,989 passed /
5 failed / 215 skipped (99.7%).** (python 3.12 + tensorflow installed for
collection; first run 2026-08-03, re-verified identical 2026-08-27 —
same 5 failures, no regressions.) All 5 failures are individually
documented: 2× upstream `test_quantize_float8` (test-side `train_one_step`
only defined for tf/jax/torch — fix drafted in `docs/upstream/keras-pr/`),
2× RandomCrop (tinygrad `__getitem__` lacks Tensor slice bounds — upstream
tinygrad item), 1× AutoContrast (FMA-contraction residual 1.9e-06 vs atol
1e-06). Cross-backend parity fuzz vs the numpy reference (`make fuzz`;
finite-difference gradient checks via `make fuzz-grad`): green; the one
former flag was the documented keras 64→32 promotion policy
(`docs/float64-promotion.md`).
<!-- /TALLY -->

Verified against Keras' own test suite, per-op:

- Core layer families green: conv/pooling 100%, activations 100%,
  losses 166/166, RNN layers (incl. default Orthogonal init) via the
  generic scan, attention (flash accepted as a hint), CTC with beam search,
  image ops (all five resize interpolations, antialias included).
- Training uses tinygrad 0.13's explicit `loss.gradient()` — gradients are
  pure outputs, no tape bookkeeping; `custom_gradient` honored via the
  trainer's tape (quantized training works).
- int8 / int4 / float8 quantization working.
- **No silent fallbacks.** An unimplemented op raises `NotImplementedError`.
  You get an error, never a wrong answer or a silently detached gradient.

<!-- SUPPORT_MATRIX -->
| Suite | ✅ passed | ❌ failed | skipped | coverage |
|---|---:|---:|---:|---:|
| activations | 51 | 0 | 0 | 100.0% |
| Dense | 70 | 1 | 1 | 98.6% |
| EinsumDense | 98 | 1 | 0 | 99.0% |
| Embedding | 48 | 0 | 3 | 100.0% |
| BatchNormalization | 33 | 0 | 0 | 100.0% |
| Dropout | 14 | 0 | 0 | 100.0% |
| Conv | 41 | 0 | 0 | 100.0% |
| pooling | 135 | 0 | 0 | 100.0% |
| SimpleRNN | 5 | 0 | 0 | 100.0% |
| MultiHeadAttention | 49 | 0 | 1 | 100.0% |
| losses | 166 | 0 | 1 | 100.0% |
| Adam | 8 | 0 | 1 | 100.0% |
| SGD | 7 | 0 | 0 | 100.0% |
| accuracy metrics | 35 | 0 | 0 | 100.0% |
| ops/core | 165 | 0 | 9 | 100.0% |
| ops/image | 331 | 0 | 5 | 100.0% |
| ops/math | 208 | 0 | 4 | 100.0% |
| ops/numpy | 5502 | 3 | 708 | 99.9% |
| preprocessing layers (tree) | 689 | 4 | 29 | 99.4% |
| **TOTAL** | **7655** | **9** | **762** | **99.9%** |

The Dense/EinsumDense failures are the upstream float8 test-side gap.
`view_as_complex` / `view_as_real` work via complex-lite interop (a
real/imag wrapper value); complex ARITHMETIC remains out of scope — any
complex op beyond that interop set raises `NotImplementedError` loudly,
never silently. The 4 remaining preprocessing failures: RandomCrop ×2
(needs tinygrad Tensor slice bounds — upstream conversation), one
FMA-precision residual (1.9e-06 vs atol 1e-06), one grain-thread ×
tinygrad-sqlite-cache clash. Preprocessing/ops-image runs need tensorflow
installed for test *collection* only.
<!-- /SUPPORT_MATRIX -->

### Known gaps

Honest list:

- `keras.ops.unique` / `keras.ops.vectorize` (data-dependent output
  shapes — loud stubs pending a design decision; the rest of the numpy
  tail landed, see [docs/ops-numpy-triage.md](https://github.com/c4ffein/keras-tinygrad/blob/main/docs/ops-numpy-triage.md)).
- Fused RNN kernels (recurrent layers take the generic scan path — correct,
  not fast; the TinyJit train step recovers most of the gap).
- Sparse and ragged tensors.
- TF-string preprocessing layers.
- Complex arithmetic (interop works; see
  [docs/complex-support.md](https://github.com/c4ffein/keras-tinygrad/blob/main/docs/complex-support.md)).

## No clang? Use zig

tinygrad's CPU jit shells out to `clang`. On boxes without it, the
`ziglang` PyPI wheel works as a drop-in: a small shim script translates the
target triple to zig's spelling, adds `-g0`, and execs `zig cc`. Point
tinygrad's `CC` at the shim and CPU jit works with zero system packages.

## Contributing

The method is fixed: the numpy backend is the semantic reference, Keras' own
tests are the referee, stubs stay loud. Dev loop: `uv sync`, then
`make verify` (lint + format + loader tests) before review; `make tutorial`,
`make smoke`, and `make fuzz` for the heavier checks. See
[CONTRIBUTING.md](https://github.com/c4ffein/keras-tinygrad/blob/main/CONTRIBUTING.md).

## License

Backend sources subclass and patch Keras (Apache-2.0) and drive tinygrad
(MIT). This package's own code: see [LICENSE](https://github.com/c4ffein/keras-tinygrad/blob/main/LICENSE).
