# tools/ — cross-backend parity fuzzing

## parity_fuzz.py

Keras' own tests are example-based; this tool hunts numerical drift the
examples miss. It generates seeded, randomized configurations and compares
the backend under test (default: `tinygrad`) against a reference backend
(default: `numpy` — any `KERAS_BACKEND` value is accepted) on identical
inputs and identical weights.

### Architecture

A Keras process is locked to one backend at import time, so there is no way
to compare two backends inside a single process. The design is therefore:

```
parity_fuzz.py (parent — stdlib + numpy ONLY, never imports keras)
  ├─ generates cases from --seed  (specs → JSON, tensors/weights → .npz)
  ├─ per (backend, case-batch): spawns a fresh subprocess running
  │    _parity_child.py            (the only module that imports keras)
  │    with KERAS_BACKEND set; the child executes the batch and writes
  │    outputs to .npz + a per-case status manifest to JSON
  └─ loads both sides' .npz and compares in the parent
```

For `--backend-under-test tinygrad` the child imports `keras_tinygrad`
before `keras` (the package's import hook), so the tool works from any cwd
where both are importable.

### Case kinds

- **op** — single `keras.ops.*` calls over a curated list (`matmul`, `conv`,
  `softmax`, elementwise, reductions, shape ops, `argmax`, …) with randomized
  shapes, dtypes (float32/float16/float64 mix), axes and params drawn from
  sane distributions.
- **layer** — single-layer forward passes (`Dense`, `Conv2D`,
  `BatchNormalization`, `LayerNormalization`, `Embedding`, `SimpleRNN`) with
  weights **generated in the parent** and installed via `layer.set_weights`,
  so both backends compute with byte-identical weights. Weight shapes are
  derived in the parent from the stable Keras weight-order contracts.
- **grad** — scalar loss = sum of all outputs; gradients w.r.t. the float
  inputs are compared. Two checking paths:
  - if the **reference** backend has autograd (`torch`, `jax`, `tensorflow`,
    `tinygrad`), analytic gradients are compared across backends;
  - `--slow` enables the workhorse: **central finite differences computed on
    the backend-under-test side itself** (forward evals only — no second
    backend needed), compared against that backend's analytic gradients.
    Grad-case inputs are deliberately tiny (FD is O(N) forwards).

  If the reference is `numpy` (no autograd) and `--slow` is not given,
  gradient cases are auto-skipped with a printed note.

### Comparison and tolerances

Per-dtype `(rtol, atol)` defaults (scalable with `--tol-scale`):

| dtype    | rtol  | atol  |
|----------|-------|-------|
| float64  | 1e-6  | 1e-9  |
| float32  | 1e-4  | 1e-6  |
| float16  | 1e-2  | 1e-3  |

Cross-backend gradients use the forward tolerance × 10; analytic-vs-FD uses
`(2e-2, 1e-3)` (central differences in float32 are inherently noisy; FD step
is `1e-3 · (1 + |x|)`). Integer/bool outputs (e.g. `argmax`) must match
exactly (width differences like int32 vs int64 are tolerated, values are
not). A case fails when `max |diff| / (atol + rtol·|ref|) > 1`, on a shape
mismatch, a float-dtype mismatch, or NaN/inf placement differences.

One float-dtype mismatch is deliberately benign: **ref float64 vs test
float32, that exact direction only**. Keras demotes implicit float64 to
float32 on every backend except tensorflow (the 64→32 entries of the
promotion table in `keras/src/backend/common/dtypes.py`); the numpy
reference keeps float64 only because its raw ops skip the conversion step,
so the mismatch is an accident of the reference, not a backend bug. Values
are then compared under **float32** tolerances (the weaker dtype judges) and
the case passes with a visible `keras 64->32 demotion` note in the summary
message and JSON report. The reverse direction (ref float32 vs test float64)
and every other dtype mismatch still fail. See `docs/float64-promotion.md`.

**`NotImplementedError` from a backend is counted as `UNSUPPORTED`, never as
a failure** — that is the honest state of a partial backend. Any other
exception in the backend under test is an `ERROR` (nonzero exit); reference-
side errors mark the case `SKIPPED`.

### Usage

```
python tools/parity_fuzz.py [options]

  --seed N                  RNG seed; every case is a pure function of (seed, index)
  --cases N                 number of cases (default 100)
  --ops a,b,...             restrict the op list (also drops layer cases unless --layers/--kinds given)
  --layers A,B,...          restrict the layer list
  --kinds op,layer,grad     explicit kind selection
  --backend-under-test B    default: tinygrad
  --reference B             default: numpy (any KERAS_BACKEND value)
  --slow                    finite-difference gradient checking (the workhorse)
  --top-k K                 worst cases shown in the summary (default 10)
  --json PATH               machine-readable full report
  --repro CASE_ID           re-run ONE case verbosely (same seed/filters required)
  --batch-size N            cases per child subprocess (default 32)
  --python PATH             interpreter for children (default: this one)
  --tol-scale F             multiply all tolerances
  --child-timeout SEC       per-subprocess timeout (default 600)
  --keep-artifacts DIR      keep the JSON/.npz exchange files for inspection
```

Exit codes: `0` all compared cases within tolerance; `1` any `FAIL`/`ERROR`;
`2` usage or infrastructure problems — including a run where **no case was
compared at all** (everything skipped/unsupported, e.g. a dead reference
child). An all-skipped run is never a pass.

Examples:

```sh
# the standard hunt (tinygrad vs numpy, forwards only)
python tools/parity_fuzz.py --seed 7 --cases 500 --json report.json

# gradients via finite differences (the workhorse when reference is numpy)
python tools/parity_fuzz.py --seed 7 --cases 200 --kinds grad --slow

# cross-backend analytic gradients when torch is installed
python tools/parity_fuzz.py --reference torch --cases 300

# reproduce one offender verbosely (copy the printed repro command, or:)
python tools/parity_fuzz.py --seed 7 --cases 500 --repro 0042-op-conv

# plumbing self-test, no tinygrad involved
python tools/parity_fuzz.py --ops matmul --cases 2 \
    --backend-under-test numpy --reference numpy
```

Reproduction requires the same `--seed`, `--cases` and op/layer/kind filters
as the original run — case IDs (`0042-op-conv`) are positional. The summary
prints a ready-made repro command per offender carrying every
generation-affecting flag (`--kinds` resolved explicitly, `--tol-scale` when
non-default), so the printed command reproduces both the case and its
verdict verbatim.

### Files

- `parity_fuzz.py` — parent orchestrator (stdlib + numpy only).
- `_parity_child.py` — subprocess worker; the only module that imports keras.

### Extending

Add an op: write a sampler in `parity_fuzz.py` returning
`(inputs, kwargs, list_input)` and register it in `OP_SAMPLERS` (add it to
`GRAD_OPS` if differentiable — keep grad shapes tiny). Add a layer: a
sampler returning `(ctor_kwargs, [input], weights)` registered in
`LAYER_SAMPLERS`; the weights list must match the layer's
`get_weights()` order exactly. The child needs no changes for either.
