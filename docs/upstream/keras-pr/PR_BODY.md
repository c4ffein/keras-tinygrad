# PR title

```
Fix test-side backend and numpy-version assumptions (np.cross on numpy >= 2.5, float8 train_one_step, trainer_test Trainer dispatch)
```

# Summary

Three small test-side fixes that remove hidden environment and backend
assumptions from the test suite: `test_cross` builds its reference with a
`np.cross` call that no longer exists on numpy >= 2.5; the two
`test_quantize_float8` tests define their `train_one_step` helper only for
the tensorflow/jax/torch backends and crash with `UnboundLocalError` on any
other trainable backend; and `trainer_test.py` duplicates the
backend-to-Trainer dispatch with a hard `raise ImportError` that kills
collection of all 173 tests in the module for any backend not named in its
`elif` chain. None of the fixes change what is tested on the in-tree
backends; each one only makes the tests portable across numpy versions and
backends.

# Fix 1: `test_cross` reference crashes on numpy >= 2.5

**Failing test:**
`keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_cross`

**Problem:** the test verifies `keras.ops.cross` against `np.cross` with
2-dimensional vectors (`y3 = np.ones([1, 5, 4, 2])`, used at
numpy_test.py:3721-3722 and 3729-3730). numpy deprecated 2-dimensional
vectors in `np.cross` in numpy 2.0 and removed them in numpy 2.5:

```
$ python -c "import numpy as np; np.cross(np.ones([4,2]), np.ones([4,2]))"
# numpy 2.4.0: DeprecationWarning: Arrays of 2-dimensional vectors are
#   deprecated. Use arrays of 3-dimensional vectors instead.
#   (deprecated in NumPy 2.0)
# numpy 2.5.0: ValueError: Both input arrays must be (arrays of)
#   3-dimensional vectors, but they are 2 and 2 dimensional instead.
```

Since the crash is in the test's own *reference*, the test fails on every
backend that reaches those lines (all except torch, which is excluded by
the existing `backend.backend() != "torch"` guard) as soon as the
environment resolves numpy >= 2.5 — which keras' unpinned numpy requirement
now does. `keras.ops.cross` itself still documents and supports
2-dimensional vectors.

**Fix:** a local `np_cross` helper that emulates the numpy < 2.5 reference
behavior (zero-pad the third component; when both inputs are 2-dimensional,
return the z-component), used only on the four 2-d assertions. The pure 3-d
assertions keep calling `np.cross` directly.

**Prior art:** the closed MLX-backend PR keras-team/keras#23193 bundled the
same guard ("`numpy_test.py` (guard `np.cross` for NumPy >= 2.5, which
dropped the dim-2 fallback)") among ~8,000 lines of backend code; no
standalone fix has been proposed.

**Note for reviewers:** after this fix, the test still fails on the *numpy
backend* under numpy >= 2.5 — but now for a real reason:
`keras/src/backend/numpy/numpy.py::cross` (line 554) also forwards
2-dimensional vectors to `np.cross`. That is an implementation bug of the
same root cause, and a one-function companion fix (emulating the removed
behavior with zero-padding, verified: all 58 `-k test_cross` tests pass
with it) can be included in this PR if maintainers prefer; it is kept out
of the test-side diff here.

# Fix 2: `test_quantize_float8` crashes on backends it does not know

**Failing tests:**
- `keras/src/layers/core/dense_test.py::DenseTest::test_quantize_float8`
- `keras/src/layers/core/einsum_dense_test.py::EinsumDenseTest::test_quantize_float8`

**Problem:** both tests define a local `train_one_step` only inside
`if backend.backend() == "tensorflow": / elif "jax": / elif "torch":`
branches (dense_test.py:720/730/762, einsum_dense_test.py:972/982/1014)
and then call it unconditionally (dense_test.py:809,
einsum_dense_test.py:1061). `@pytest.mark.requires_trainable_backend`
hides this on numpy/openvino, but any *other* trainable backend crashes
with:

```
UnboundLocalError: cannot access local variable 'train_one_step' where it
is not associated with a value
```

— an obscure failure that points at the test rather than at anything the
backend does wrong (the float8 quantization path under test can be fully
functional).

**Fix:** a `skipif` decorator on both tests, skipping backends outside
`("tensorflow", "jax", "torch")` with an explicit reason. The test's
gradient plumbing is inherently backend-specific (`tf.GradientTape`,
`jax.grad`, `loss.backward()`), so a skip-with-reason states the actual
situation; the same pattern is already used for backend-specific paths
elsewhere in the suite (e.g. the openvino skipif on
`ViewAsComplexRealTest`, math_test.py:1861). Behavior on
tensorflow/jax/torch is unchanged — the tests still run there.

# Fix 3: `trainer_test.py` fails collection for backends not in its `elif` chain

**Failing:** collection of the whole module (173 tests) —
`keras/src/trainers/trainer_test.py`.

**Problem:** trainer_test.py:28-43 duplicates the backend-to-Trainer
dispatch that `keras/src/models/model.py` already performs, and ends with
`raise ImportError(f"Invalid backend: {backend.backend()}")` (line 43).
For any backend not named there, pytest aborts the entire module at
collection time — even though every test in it drives models through the
public `Model`/`Trainer` machinery that keras itself already resolved for
that backend.

**Fix:** replace the `raise` with a fallback import of the `Trainer` name
that `keras.src.models.model` has already bound for the current backend.
All in-tree backends keep their explicit branches (byte-identical
behavior, including jax's extra distribution imports); a genuinely unknown
backend still fails loudly — in `keras.src.models.model`, the canonical
place, which must have resolved a Trainer for `import keras` to have
succeeded at all.

# How verified

Environment: keras at master, numpy 2.5.1; backends: tensorflow-cpu 2.21.0,
jax (CPU), numpy, plus an out-of-tree trainable backend selected via
`KERAS_BACKEND` to exercise the unknown-backend paths. torch is unaffected
by fix 1 (guarded lines) and keeps its explicit branches in fixes 2-3.

Before (stock test files):

```
KERAS_BACKEND=numpy pytest keras/src/ops/numpy_test.py -k "test_cross and Correctness"
# FAILED ...::NumpyTwoInputOpsCorrectnessTest::test_cross
# ValueError: Both input arrays must be (arrays of) 3-dimensional vectors,
#   but they are 3 and 2 dimensional instead.   (numpy 2.5.1)

KERAS_BACKEND=<out-of-tree> pytest keras/src/layers/core/dense_test.py::DenseTest::test_quantize_float8 \
  keras/src/layers/core/einsum_dense_test.py::EinsumDenseTest::test_quantize_float8
# 2 failed — UnboundLocalError at dense_test.py:809 / einsum_dense_test.py:1061

KERAS_BACKEND=<out-of-tree> pytest keras/src/trainers/trainer_test.py --collect-only
# ImportError: Invalid backend: <out-of-tree>
# no tests collected, 1 error
```

After (this patch):

```
KERAS_BACKEND=tensorflow pytest keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_cross
# 1 passed
KERAS_BACKEND=jax pytest keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_cross
# 1 passed
KERAS_BACKEND=numpy pytest ...::test_cross
# still fails — see the numpy-backend `cross` companion note in Fix 1
# (with the one-function companion fix applied: 58 passed for -k test_cross)

KERAS_BACKEND=tensorflow pytest <both test_quantize_float8 tests>
# 2 passed (unchanged)
KERAS_BACKEND=jax pytest keras/src/layers/core/dense_test.py::DenseTest::test_quantize_float8
# 1 passed (unchanged)
KERAS_BACKEND=<out-of-tree> pytest <both test_quantize_float8 tests>
# 2 skipped: "`train_one_step` is only implemented for the tensorflow, jax
#   and torch backends"

KERAS_BACKEND=numpy pytest keras/src/trainers/trainer_test.py --collect-only
# 173 tests collected (unchanged)
KERAS_BACKEND=<out-of-tree> pytest keras/src/trainers/trainer_test.py --collect-only
# 173 tests collected (previously: collection error)
```

Diffstat:

```
 keras/src/layers/core/dense_test.py        |  5 +++++
 keras/src/layers/core/einsum_dense_test.py |  5 +++++
 keras/src/ops/numpy_test.py                | 24 ++++++++++++++++++++----
 keras/src/trainers/trainer_test.py         |  6 +++++-
 4 files changed, 35 insertions(+), 5 deletions(-)
```
