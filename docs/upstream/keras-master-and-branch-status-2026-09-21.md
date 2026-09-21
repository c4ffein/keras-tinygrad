# Keras master (3.16-dev) and the `pluggable_backend` branch vs this backend — 2026-09-21

Static comparison (no pilot re-run) of three things against the packaged
backend: keras **master** at `c6a3b948` (2026-09-19, version 3.16.0-dev,
251 commits past the v3.15.1 pin), the **`pluggable_backend`** branch at
`60be5d35` (2026-09-17, 80 commits past master), and the split-out
reference backends **`keras-team/keras-openvino`** (`c00c9f92`) and
**`keras-team/keras-mlx`** (`1f29133`). The diff series that came out of
it is described at the end.

## 1. Op surface

Against the numpy backend at the v3.15.1 pin the tinygrad backend was
name-complete (aliases included: `abs`, `amax`, `amin`, `conj`,
`true_divide`; `fmax` / `fmin` via `_nan_preferring`; the only absences
`unique` / `vectorize`, the documented decision items).

Ops master added since the pin and this backend lacked (now landed, see
§5): `copysign`, `float_power`, `cov` (numpy), `gammainc`, `lgamma`
(math). `column_stack` and `matrix_power` were already here.

Master also grew OPTIONAL fused hooks probed with `hasattr`:
`backend.ops.nn.rms_normalization` / `layer_normalization` (#23510, torch
only in-tree). A perf lever, not a gap.

## 2. Keras master (the next stock release) — what breaks the hook path

1. **`ops/` subpackage layout (#23642).** Every in-tree backend moved its
   `core/image/linalg/math/nn/numpy` into `backend/<name>/ops/`, exported
   as `ops` from the backend `__init__`; keras now calls
   `backend.ops.numpy.x`, `backend.ops.convert_to_tensor`, ... (471
   `backend.ops.` references in `ops/numpy.py` alone; `backend.numpy.x`
   is gone). DONE in this series: the package now has that shape
   (`keras_tinygrad/src/ops/`), exports both spellings, and the loader's
   patches import `keras_tinygrad.src` directly (no alias — stock keras
   never builds a backend module name dynamically).
2. **One loader anchor drifted:** the `DynamicBackend.__getattr__` block
   in `utils/backend_utils.py` was rewritten (the per-backend `return`s
   collapsed into one; numpy no longer special-cased). The other five
   anchors still match master exactly once.
3. **`hasattr` probes on the backend (#23643 and friends).** Master
   decides between a backend op and its own backend-agnostic fallback
   with `hasattr(backend.ops.numpy, "copysign")`, `... "allclose"`,
   `"cbrt"`, `"hsplit"`, `"vsplit"`, `"dsplit"`, `"float_power"`,
   `hasattr(backend.ops.math, "lgamma")`, `hasattr(backend.ops.nn,
   "rms_normalization")`. `hasattr` only swallows `AttributeError`; the
   PEP 562 loud stubs raised a plain `NotImplementedError`, which would
   have escaped the probe and failed even the ops keras could compute
   without us. Fixed in this series: `core.MissingOpError` is both.
4. `Trainer` base gained a `state_sync()` no-op hook (#23620; the jax
   trainer's `jax_state_sync` generalized), and the numpy `Trainer.predict`
   concatenation was reworked. The tinygrad trainer inherits the base;
   nothing to do unless it wants the hook.
5. Data adapters now handle native backend tensors (#23628, #23691) —
   worth a referee run on 3.16 when it ships.

## 3. The `pluggable_backend` branch — current protocol

The branch is where the plugin protocol actually lives (RFC #23523 is
still open with no decision recorded). Since the 2026-08-27 pilot the
protocol changed in ways the shim did not track:

1. **Discovery is a hard-coded frozenset** — `config._PLUGGABLE_BACKENDS
   = {"mlx", "openvino", "paddle"}` — plus the `keras_<name>.src` naming
   convention; **no entry points**. `tinygrad` is not in the set, so plain
   `KERAS_BACKEND=tinygrad` raises "Unsupported backend" on the branch
   head; the pilot's generic-`else` patch (`exports/05`) is still the only
   way in. `DynamicBackend.set_backend` whitelists the same union.
2. **OpenVINO moved OUT of the keras tree** on the branch (`keras/src/
   backend/openvino` is gone; master still carries it). It lives in
   `keras-team/keras-openvino`, next to `keras-mlx` and `keras-paddle`.
   Their layout is the reference for a plugin package:
   `keras_<name>/src/{ops/{core,image,linalg,math,nn,numpy}.py, random.py,
   rnn.py, trainer.py, variable.py, version.py}`; `Variable` split out of
   `ops/core.py`; `src/__init__.py` star-exports `ops`, `random`, `rnn`,
   `compute_output_spec`, `device_scope`, `Variable`, `name_scope`, the
   `SUPPORTS_*` / `IS_THREAD_SAFE` flags and `distribution_lib = None`;
   `keras_<name>/__init__.py` only re-exports `__version__`.
3. **`layer.py` and `export.py` are optional**: keras does
   `importlib.util.find_spec(f"keras_{name}.src.layer")` and falls back
   to an empty `BackendLayer`; likewise `.src.export` falls back to
   `BaseSavedModelExportArchive`. Neither reference package ships them.
   When present, the attribute names are **`BackendLayer`** and
   **`SavedModelExportArchive`** — the shim exported `Layer` /
   `TinygradLayer` and `ExportArchive` / `TinygradExportArchive`, so both
   `getattr`s would have failed at `import keras`. Fixed in this series:
   `BackendLayer` added, `keras_tinygrad/src/export.py` removed (the base
   fallback raises the same loud `NotImplementedError` for both hooks).
   `Trainer` already matched.
4. **Test exclusions** are a per-package `excluded_tests.txt` at the repo
   root (exact pytest node ids; keras' conftest reads it only for a
   checked-out backend, an installed one has none — #23671). Same idea as
   `scripts/referee-baseline.txt`, opposite polarity (skip vs. expect-fail).
   `integration_tests/import_test.py` takes `KERAS_BACKEND_PACKAGES` for
   the plugin's pip requirements.
5. **Reference CI** (`keras-openvino/.github/workflows/actions.yml`):
   check out `keras@pluggable_backend` next to the backend, `pip install
   -e .`, `KERAS_HOME` pointing at a `keras.json` that sets the backend,
   then `pytest keras -n auto --dist loadfile --ignore keras/src/
   applications --ignore keras/src/wrappers` from the keras root. Ruff at
   line length 80, `pre-commit`, Apache-2.0.
6. Still open on the branch from the pilot: `standardize_dtype` reads
   `.name` first and its string heuristic lists `"mlx"` by name (the
   `standardize_dtype_hook` remains a local patch); the float8
   `train_one_step` test-side assumption.

## 4. `unique` / `vectorize` on master

Unchanged decision items. On master both are called directly
(`backend.ops.numpy.unique(...)`, no agnostic fallback), so they stay
loud — `MissingOpError` changes nothing for them.

## 5. What the 2026-09-21 series does

1. `copysign`, `float_power`, `cov`, `lgamma`, `gammainc` in the backend,
   numpy/scipy semantics, with `tests/test_ops_beyond_pin.py` carrying
   keras master's own test cases (the pinned suite cannot referee them)
   plus gradient receipts. Notes: `copysign` and the magnitude of `-0.0`
   are bit-level (tinygrad's `abs` keeps `-0.0`); `lgamma` is the Lanczos
   form keras' agnostic `_lgamma` uses, with the unselected reflection
   branch fed a benign argument so `lgamma'(n)` is `digamma(n)` and not
   nan; `gammainc` is Numerical Recipes' series + Lentz continued
   fraction at fixed term counts (200 / 60: float32-exact to a ~ 1000),
   both branches clamped into their convergence regions for a finite
   discarded branch, the unrolled recurrences cut into identical
   20-iteration segments (stacked `contiguous` barriers; `realize` would
   silently stop the gradient at the last segment) — forward compiles 5
   kernels per call shape, the gradient ~25 and ~30 s of clang on first
   use. Float32 cancellation costs ~1e-5 relative from a ~ 50 up.
2. `core.MissingOpError(NotImplementedError, AttributeError)` raised by
   the five PEP 562 stubs (§2.3).
3. Shim names for the current branch protocol (§3.3).
4. This note; HANDOFF remainder 9.

5. The restructure: backend sources moved to `keras_tinygrad/src/` in the
   reference layout (`ops/` subpackage; `Variable` stays in `ops/core.py`
   as in keras' in-tree backends — keras-openvino's separate
   `variable.py` is that package's choice, not protocol), the aliasing
   shim deleted, the loader reduced to its six patches.

Not done, deliberately: the drifted `DynamicBackend` anchor (§2.2) — 3.16
is not on PyPI and the pin says `<3.16`; it is a one-line replacement
once the released tree is known.
