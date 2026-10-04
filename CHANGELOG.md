# Changelog

Releases of the Python package (`keras-tinygrad` on PyPI, tags `v*`). The
JS package (`js/`, npm, tags `js-v*`) has its own line at the bottom. Dates
are tag dates; "Unreleased" is the working tree. The referee tally of
record for every release is in README's status section, taken on the
pinned keras tag with `make referee`.

## Unreleased — 0.2.0

The version has been 0.2.0 in `pyproject.toml` since 2026-09-21; nothing
between 0.1.0 and it was tagged (a 0.1.1 was prepared on 2026-09-01 and
folded in).

### Breaking

- **`keras.src.backend.tinygrad` no longer exists.** The backend is the
  package `keras_tinygrad.src` (pluggable-backend shape: `ops/{core, image,
  linalg, math, nn, numpy}` + `random`, `rnn`, `trainer`, `layer`,
  `export`), mirroring keras master's `backend.ops.*` layout and the
  out-of-tree plugin repos. Anything that imported the old alias breaks
  loudly.
- **tinygrad pin moved to `>=0.14,<0.15`** (was `>=0.13,<0.14`). On 0.14
  `.contiguous().realize()` no longer turns a CONST into a buffer, so
  Variables assigned from python scalars were baked into the jitted train
  step (the loss tracker's count froze and `fit` reported 0.0 from epoch
  2; Adam's iteration froze). Fixed via `core._concrete` (architecture
  invariant 5); receipt: the JIT-vs-eager every-variable test.
- Losses handed to the WebGPU exporter must be built with `reduction=None`
  (checked before tracing).

### Fixed

- **`train_step` zeroed gradients upstream of any forward barrier**: the
  loss tracker was updated (a realize) before the gradient graph was
  built. Any model with a `log1p`, a linalg loop or a recurrent layer in
  its forward trained only the weights downstream of the barrier. The
  gradient graph is built first now.
- **Recurrent layers**: the train-step JIT was silently off for every
  LSTM/GRU/SimpleRNN model, and the scan was one lazy graph over all
  timesteps (35 s per eager step for LSTM(32) over 10 steps). Cells join
  the device-RNG cover, the scan cuts per timestep: 1.0 s per jitted step.
- **`IS_THREAD_SAFE` is now False**: keras ran callback hooks on a thread
  pool and a host read from that thread raced tinygrad's in-place kernel
  arguments on an asynchronous device (wrong `evaluate` about one run in
  three on OpenCL; the CPU device's synchronous launches hid it).
- `train_on_batch` / `test_on_batch` returned a running mean over calls
  instead of the batch's own metrics (they reset metrics first now, like
  the tensorflow, jax and torch trainers).
- **linalg gradients were silently zero** for inv, det, solve, lu_factor,
  cholesky, eigh, svd, lstsq, pinv: per-step `.realize()` and host validity
  checks swapped realized graphs for buffers. Now `_cut` in the loops, host
  checks on a side graph of the detached input, none under the trainer's
  tape (architecture invariant 12). Jacobi above 110 rounds still realizes
  per round (no gradient; raises inside a train step).
- `gammainc` gradients were NaN at the domain edges and through a
  broadcast `a`.
- `elastic_transform` NameError (a dropped import); `make lint-check` now
  runs an undefined-names pass over the backend sources.
- On tinygrad 0.14: python-scalar operands came back typed
  `weakint`/`weakfloat` (`_pair`, `clip` now cast to keras' `result_type`
  first); `int ** int` rendered invalid C (exact `_int_power`,
  square-and-multiply with numpy's wrap-around).
- Vendored exporter ported to tinygrad 0.14's `extra/export_model.py`, and
  kernel-name collisions in one trace (0.14 reuses shape-derived names)
  no longer drop kernel bodies (WebGPU `GPUPipelineError` at setup).
- Export initial values come from each weight's declared initializer
  (BatchNormalization's gamma and moving_variance exported as 0.0 before).
- Device RNG re-seeds at every train-function build, so `set_random_seed;
  build; fit` twice in one process agree (numpy-backend shape).

### Added

- `tinygrad_tensor <op> keras.Variable` works (the Variable on the right of
  a python operator, a keras-core idiom): tinygrad's dunders defer to the
  Variable's reflected op instead of raising "Could not infer dtype";
  `abs(variable)` works (`Tensor.__abs__`).
- `bench/`: CPU benchmark against the tensorflow, jax and torch backends
  (same cores, init and data; per-step loss agreement checked first).
- `make examples`; `make referee-cl`: the referee's quick slice on tinygrad's OpenCL
  renderer with no GPU (PoCL from PyPI); `scripts/run_keras_guide.py`
  runs one keras.io guide under the backend, capped.
- keras-master-only ops: `copysign`, `float_power`, `cov`, `lgamma`,
  `gammainc`; `core.MissingOpError` for master's `hasattr` probes.
- `keras.backends` entry point: on a keras with native plugin support the
  import hook stands down (`docs/history/upstream/keras-plugin-poc-2026-08-10.md`).
- `keras_tinygrad.webgpu`: export a Keras train step as a self-contained
  WebGPU bundle; the wheel ships `webgpu.py` and the vendored exporter.
- Browser end-to-end suite (`make e2e`, `tests/test_webgpu_export.py`):
  headless Chromium drives the demo hub through five proof flows, bundle
  vs tf.js, dropout fresh masks, browser vs local Python, Pyodide in-tab
  trace vs bundle.
- Every test subprocess runs under resource caps (`tests/_limits.py`);
  the referee caps its run too.
- Loader: a version warning and two canaries for keras drift.

## 0.1.0 — 2026-08-28

First release on PyPI. Stock keras 3.15.x + tinygrad 0.13 via the
meta-path import hook (six match-exactly-once patches). Keras' own layers
tree: 1,988 passed / 5 failed / 215 skipped on the v3.15.1 tag; ops/numpy,
math, linalg, image suites green but for the three known (unique,
vectorize, test_cross). TinyJit train step with bit-for-bit eager parity.

## JS package (`js/`, npm `keras-tinygrad`)

- 0.0.1 — unreleased: runner (`loadBundle`, `fetchBundle`, `importRunner`,
  `lcg32` bit-identical to the Python twin) and the in-tab tracer
  (`keras-tinygrad/trace`: real Keras under Pyodide, needs the Python
  wheel's URL). First publish pending (`docs/npm-publishing.md`).
