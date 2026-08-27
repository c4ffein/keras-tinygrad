# Upstream draft — keras-team/keras test-side fixes (NOT submitted)

Status: local draft for the owner's review. Nothing has been submitted,
no issue opened, nothing pushed.

**UPDATE 2026-08-21:** item 1 (`test_cross`) is DONE upstream — keras PR
#23408 (merged 2026-08-11) fixed the test AND the numpy backend's
`cross`, i.e. both our patch and the held-in-reserve companion.
Do not submit it. Items 2 and 4 (float8 skipif, trainer_test dispatch)
remain unfixed on both master and the team's `pluggable_backend` branch
(whose trainer_test still ends in `raise ImportError`) — retargeted per
`/home/dev/workspace/KERAS_COMMITS_AND_ORDER_GUIDE.md` §1/§3: offer them
during engagement with the pluggable-backend effort, targeting whichever
tree the maintainers prefer. A ready-to-apply PR package (patch +
neutral PR body) lives in `docs/upstream/keras-pr/` (`tests-fix.patch`,
`PR_BODY.md`); this document is the keras-tinygrad-side rationale with
full citations. All line numbers are against the reference clone at
`/home/dev/workspace/keras` (master, `abd068b3`), whose test files are
pristine stock keras; verified 2026-08-03.

Four mechanical items were investigated. **Three made the PR package**
(items 1, 2, 4). Item 3 (`ViewAsComplexRealTest`) was **dropped** after
re-verification — the backend now passes it, and the generic argument
does not survive scrutiny; the analysis is kept below.

---

## 1. `test_cross`: the test's own numpy reference crashes on numpy >= 2.5

**Failing test:**
`keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_cross`
(class at numpy_test.py:3405, test at :3710).

**Root cause:** the test builds `y3 = np.ones([1, 5, 4, 2])` and calls
`np.cross(x1, y3)` / `np.cross(x2, y3)` as the *reference* at
numpy_test.py:3721-3722 and :3729-3730. `np.cross` support for
2-dimensional vectors was deprecated in numpy 2.0 and **removed in numpy
2.5** — verified empirically:

- numpy 2.4.0: `DeprecationWarning: Arrays of 2-dimensional vectors are
  deprecated. Use arrays of 3-dimensional vectors instead. (deprecated in
  NumPy 2.0)`
- numpy 2.5.0: `ValueError: Both input arrays must be (arrays of)
  3-dimensional vectors, but they are 2 and 2 dimensional instead.`

Keras does not pin numpy below 2.5, so any freshly resolved environment
hits this. It is why the failure is only surfacing now: CI environments
resolved numpy < 2.5 until recently.

**Why backend-agnostic:** the crash is in the reference expression, before
any backend output is compared. It reproduces on the stock numpy backend
with no tinygrad anywhere:

```
KERAS_BACKEND=numpy pytest keras/src/ops/numpy_test.py -k "test_cross and Correctness"
# FAILED — ValueError from np.cross (numpy 2.5.1)
```

Every backend that reaches those lines fails; the one exception is torch,
which the test already excludes from the 2-d assertions via the
`backend.backend() != "torch"` guard at :3718 and :3726 (so "every
backend" in the strict sense is: all except torch).

**Proposed patch** (in `tests-fix.patch`; emulates the numpy < 2.5
reference semantics — zero-pad the third component, z-component result
when both inputs are 2-d):

```diff
--- a/keras/src/ops/numpy_test.py
+++ b/keras/src/ops/numpy_test.py
@@ -3713,21 +3713,37 @@ class NumpyTwoInputOpsCorrectnessTest(testing.TestCase):
         y1 = np.ones([2, 1, 4, 3])
         y2 = np.ones([1, 5, 4, 3])
         y3 = np.ones([1, 5, 4, 2])
+
+        def np_cross(a, b):
+            # `np.cross` removed support for 2-dimensional vectors in
+            # numpy 2.5 (deprecated in 2.0), but `keras.ops.cross` still
+            # supports them. Emulate the numpy < 2.5 reference behavior.
+            if a.shape[-1] == 2 and b.shape[-1] == 2:
+                return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
+
+            def pad3(v):
+                if v.shape[-1] == 2:
+                    zeros = np.zeros(v.shape[:-1] + (1,), dtype=v.dtype)
+                    v = np.concatenate([v, zeros], axis=-1)
+                return v
+
+            return np.cross(pad3(a), pad3(b))
+
         self.assertAllClose(knp.cross(x1, y1), np.cross(x1, y1))
         self.assertAllClose(knp.cross(x1, y2), np.cross(x1, y2))
         if backend.backend() != "torch":
             # API divergence between `torch.cross` and `np.cross`
             # `torch.cross` only allows dim 3, `np.cross` allows dim 2 or 3
-            self.assertAllClose(knp.cross(x1, y3), np.cross(x1, y3))
-            self.assertAllClose(knp.cross(x2, y3), np.cross(x2, y3))
+            self.assertAllClose(knp.cross(x1, y3), np_cross(x1, y3))
+            self.assertAllClose(knp.cross(x2, y3), np_cross(x2, y3))
 
         self.assertAllClose(knp.Cross()(x1, y1), np.cross(x1, y1))
         self.assertAllClose(knp.Cross()(x1, y2), np.cross(x1, y2))
         if backend.backend() != "torch":
             # API divergence between `torch.cross` and `np.cross`
             # `torch.cross` only allows dim 3, `np.cross` allows dim 2 or 3
-            self.assertAllClose(knp.Cross()(x1, y3), np.cross(x1, y3))
-            self.assertAllClose(knp.Cross()(x2, y3), np.cross(x2, y3))
+            self.assertAllClose(knp.Cross()(x1, y3), np_cross(x1, y3))
+            self.assertAllClose(knp.Cross()(x2, y3), np_cross(x2, y3))
 
         # Test axis is not None
         self.assertAllClose(
```

Verified after the fix: tensorflow 1 passed, jax 1 passed, tinygrad 1
passed (our backend implements 2-d cross natively).

**Discovered companion bug (not test-side, NOT in `tests-fix.patch`):**
the *numpy backend's own* `cross` forwards 2-d vectors to `np.cross`
(`keras/src/backend/numpy/numpy.py:547-561`, the `np.cross` call at
:554), so on numpy >= 2.5 `keras.ops.cross` itself crashes on the numpy
backend and the test stays red there even with the reference fixed. A
one-function fix was written and verified (all 58 `-k test_cross` tests
pass on the numpy backend with it, including the dtype matrix), then
reverted from the clone. The hunk, for the owner to bundle or file
separately:

```diff
--- a/keras/src/backend/numpy/numpy.py
+++ b/keras/src/backend/numpy/numpy.py
@@ -551,14 +551,26 @@ def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
     dtype = dtypes.result_type(x1.dtype, x2.dtype)
     x1 = x1.astype(dtype)
     x2 = x2.astype(dtype)
-    return np.cross(
-        x1,
-        x2,
-        axisa=axisa,
-        axisb=axisb,
-        axisc=axisc,
-        axis=axis,
-    )
+    if axis is not None:
+        axisa = axisb = axisc = axis
+
+    def move_and_pad3(v, src_axis):
+        # `np.cross` removed support for 2-dimensional vectors in
+        # numpy 2.5 (deprecated in 2.0); emulate them by zero-padding
+        # the third component.
+        v = np.moveaxis(v, src_axis, -1)
+        padded = v.shape[-1] == 2
+        if padded:
+            zeros = np.zeros(v.shape[:-1] + (1,), dtype=v.dtype)
+            v = np.concatenate([v, zeros], axis=-1)
+        return v, padded
+
+    x1, padded1 = move_and_pad3(x1, axisa)
+    x2, padded2 = move_and_pad3(x2, axisb)
+    out = np.cross(x1, x2)
+    if padded1 and padded2:
+        return out[..., 2]
+    return np.moveaxis(out, -1, axisc)
 
 
 def cumprod(x, axis=None, dtype=None):
```

**Prior art:** keras-team/keras PR **#23193** ("[Feature] Add MLX backend
(training-capable, Apple Silicon)", closed, not merged, +8132/-38)
included the same guard in its shared-file fixes — its body lists
"`numpy_test.py` (guard `np.cross` for NumPy >= 2.5, which dropped the
dim-2 fallback)". No standalone issue or PR for the breakage exists; a
small standalone fix is exactly the piece of that PR that should not have
died with it. (Our helper was derived independently; the approach —
guarding the reference, not the op — matches theirs as described.)

## 2. `test_quantize_float8`: test-side `train_one_step` only defined for tf/jax/torch

**Failing tests:**
- `keras/src/layers/core/dense_test.py::DenseTest::test_quantize_float8`
  (test at dense_test.py:704)
- `keras/src/layers/core/einsum_dense_test.py::EinsumDenseTest::test_quantize_float8`
  (test at einsum_dense_test.py:952)

These are the 2 of the 4 red tests in the clone's 2,127-test layers
referee run (HANDOFF "Verified state").

**Root cause:** each test defines a local `train_one_step` under
`if backend.backend() == "tensorflow":` (dense_test.py:720,
einsum_dense_test.py:972), `elif ... == "jax":` (:730 / :982), `elif ...
== "torch":` (:762 / :1014) — with no `else` — then calls it
unconditionally at dense_test.py:809 / einsum_dense_test.py:1061. On the
tinygrad backend:

```
UnboundLocalError: cannot access local variable 'train_one_step' where it
is not associated with a value
```

The backend's float8 path is proven correct (HANDOFF); only the test's
private gradient plumbing is missing a branch.

**Why backend-agnostic:** any trainable backend that is not one of the
three named crashes identically —
`@pytest.mark.requires_trainable_backend` (dense_test.py:700,
einsum_dense_test.py:951) only shields numpy/openvino. The helper is
written directly against `tf.GradientTape` / `jax.grad` /
`loss.backward()`, i.e. it is *inherently* backend-specific test code, so
the generic fix is an explicit skip-with-reason for unknown backends
rather than a per-backend branch (upstream does not know tinygrad
exists, and a skip states the true situation: this test's harness, not
the feature, is tf/jax/torch-only).

**Proposed patch** (in `tests-fix.patch`; the einsum hunk is identical
modulo context):

```diff
--- a/keras/src/layers/core/dense_test.py
+++ b/keras/src/layers/core/dense_test.py
@@ -701,6 +701,11 @@ class DenseTest(testing.TestCase):
     @pytest.mark.skipif(
         testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
     )
+    @pytest.mark.skipif(
+        backend.backend() not in ("tensorflow", "jax", "torch"),
+        reason="`train_one_step` is only implemented for the tensorflow, "
+        "jax and torch backends",
+    )
     def test_quantize_float8(self):
```

Verified: tinygrad 2 skipped (with the reason shown), tensorflow 2
passed, jax 1 passed (dense; unchanged behavior).

Note on our own house rule ("a test that can't go green stays loudly
failing — never skipped"): that rule governs *backend* gaps. Here the gap
is in the test's own harness code; the skip reason names the harness, and
the actual float8 backend path stays covered by
`test_quantize_float8_fitting` / `test_quantize_float8_inference`
(dense_test.py:819/:872, einsum_dense_test.py:1071/:1128), which run and
pass on tinygrad.

## 3. `ViewAsComplexRealTest` skipif — investigated, DROPPED from the PR

**The idea:** `keras/src/ops/math_test.py` skips the whole
`ViewAsComplexRealTest` class for openvino by name:

```python
@pytest.mark.skipif(                      # math_test.py:1861
    backend.backend() == "openvino",      # :1862
    reason="Complex dtype is not supported on OpenVINO backend.",
)
class ViewAsComplexRealTest(testing.TestCase):   # :1865
```

A capability-based condition (`not backend.SUPPORTS_COMPLEX_DTYPES`,
which every backend module exports — numpy/core.py:19 `True`,
openvino/core.py:25 `False` — and which the suite already uses at
`keras/src/backend/common/dtypes_test.py:38`) would generalize the skip
to any backend without complex support.

**Why it was dropped (2026-08-03 re-verification):** the premise is
stale. The tinygrad backend now **passes all 7 tests in the class**
(verified: `ViewAsComplexRealTest` → 7 passed under KERAS_BACKEND=tinygrad)
via the complex-lite `ComplexTensor` interop (docs/complex-support.md),
while correctly declaring `SUPPORTS_COMPLEX_DTYPES = False` — the flag
means "full complex dtype support", but `view_as_complex`/`view_as_real`
only need interop-level support. So the capability-flag skipif would
*regress 7 green tests into skips* for us, and for a generic minimal
backend it would hide a loud, honest failure. The remaining
openvino-by-name skip is accurate as-is. No test-side change is
proposed; if a future backend genuinely needs it, the right fix is a
finer-grained capability flag, which is not worth proposing without a
second consumer.

## 4. `trainer_test.py:43`: backend `elif` chain kills collection

**Failing:** module collection —
`pytest keras/src/trainers/trainer_test.py` errors out before collecting
any of its 173 tests:

```
ImportError: Invalid backend: tinygrad
```

(HANDOFF item 7's "known residue".)

**Root cause:** trainer_test.py:28-43 re-implements the backend→Trainer
dispatch (jax :28, torch :33, tensorflow :35, numpy :38, openvino :40)
and ends with `raise ImportError(f"Invalid backend: {backend.backend()}")`
at :43. This duplicates `keras/src/models/model.py`, which performs the
same dispatch at import time and binds the chosen class to the
module-level name `Trainer` — for any backend that can `import keras` and
build a `Model` at all, the canonical answer already exists.

**Why backend-agnostic:** every test in the module drives training
through `Model`/`Trainer` mixins; nothing else in the file needs the
backend name. Any out-of-tree backend — which is what keras#20793
contemplates — is locked out of the entire trainer suite by this one
line, while genuinely invalid backends already fail earlier and louder
inside keras itself.

**Proposed patch — the least-invasive fix** (in `tests-fix.patch`): keep
all existing branches byte-identical (jax's branch also re-imports
`DataParallel`/`DeviceMesh`, so it cannot be collapsed blindly) and turn
only the `else` from a raise into a fallback to keras' own resolution:

```diff
--- a/keras/src/trainers/trainer_test.py
+++ b/keras/src/trainers/trainer_test.py
@@ -40,7 +40,11 @@ elif backend.backend() == "numpy":
 elif backend.backend() == "openvino":
     from keras.src.backend.openvino.trainer import OpenVINOTrainer as Trainer
 else:
-    raise ImportError(f"Invalid backend: {backend.backend()}")
+    # Reuse the Trainer class that `keras.src.models.model` already resolved
+    # for the current backend instead of duplicating the dispatch here. Any
+    # backend that can build a `Model` can run this file; truly unknown
+    # backends still fail loudly, in `keras.src.models.model`.
+    from keras.src.models.model import Trainer
```

Verified: tinygrad collection goes from `1 error, 0 collected` to `173
collected`, and a sampled test
(`TestTrainer::test_callback_methods_keys`) passes; numpy-backend
collection unchanged at 173.

---

# The strategic ask (separate from the mechanical fixes above)

The four items above are deliberately tiny and self-contained. The larger
conversation — kept out of any mechanical PR — is a **backend-plugin
entry point**: today a third-party backend physically cannot exist
against stock keras, because the backend surface is hardcoded `elif`
chains at exactly six keras-core touchpoints (`keras.src.backend`
loader, `models/model.py` Trainer dispatch, `layers/layer.py` mixin
dispatch, `export/saved_model.py`, `common/variables.py`
`standardize_dtype`, `utils/backend_utils.py` `DynamicBackend` — the
contract surface documented in our `docs/architecture.md`). keras-tinygrad
ships as a meta-path import hook that patches those six sites in stock
keras 3.15 — working, but a hack that no backend author should need.

In **keras#20793** Chollet laid out criteria for accepting new backends;
the realistic path he points to for everything below that bar is
out-of-tree. **PR #23193** (MLX backend, closed unmerged) is the direct
evidence for why out-of-tree needs an entry point: a complete,
training-capable backend with a ~12,700-tests-green tally died in process
friction — CLA state, rolling merge conflicts with six-plus hardcoded
dispatch sites, and a codecov patch-coverage fail (0.3%) because CI never
runs a new backend's tests — with no maintainer commitment to review
8,000 lines. A `keras.backends` entry-point group (setuptools entry
points resolving the same six touchpoints) would let such backends ship,
version, and CI themselves, and would shrink "add a backend" PRs to one
registration line. The proposal should be raised as a design
issue/discussion referencing #20793 first — not as a cold PR — and the
mechanical fixes above intentionally do not depend on it (though fix 4 is
a small concrete instance of the same principle: don't duplicate backend
dispatch, resolve it once).

---

# Verification environments (for reproducing the numbers)

- Clone: `/home/dev/workspace/keras` @ `abd068b3` (master), test files
  restored pristine after patch generation (`git diff` on all touched
  test files: empty; only the six known touchpoint files remain locally
  modified).
- tinygrad runs: `KERAS_BACKEND=tinygrad CC=/home/dev/.local/bin/zigcc`
  with `/home/dev/workspace/ktg-venv/bin/python` (numpy 2.5.1);
  preprocessing tests via `/home/dev/workspace/tf-venv/bin/python`.
- tensorflow/jax/numpy runs: `/home/dev/workspace/tf-venv/bin/python`
  (tensorflow-cpu 2.21.0, numpy 2.5.1).
- numpy 2.4.0 vs 2.5.0 `np.cross` check: `uv run --with numpy==2.4.0`
  / `--with numpy==2.5.0`.
