# Decision memo — the float64 promotion flag (owner decides)

Status: **decided & implemented — Option A, fuzzer side (2026-08-03)**.
`tools/parity_fuzz.py` now treats the exact ref-float64 / test-float32
mismatch as ok-with-note ("keras 64->32 demotion"), compared under float32
tolerances; the reverse direction and every other mismatch still fail. The
backend is unchanged. This memo documents the one formerly persistent
parity-fuzz flag ("float64-vs-float32 promotion, values equal to ~1e-7")
with the mechanism pinned down.

## What actually happens (all verified 2026-08-03, keras 3.15.0)

Feed a **float64 numpy array** into any op:

| Path | Result dtype |
|---|---|
| tinygrad backend, implicit (`ops.reshape(np_f64)`) | **float32** |
| numpy backend, implicit (`ops.reshape(np_f64)`) | **float64** |
| numpy backend, `convert_to_tensor(np_f64)` | **float32** |
| tinygrad backend, explicit (`convert_to_tensor(x, "float64")` → op chain) | **float64**, correct to 1e-12 |

## The mechanism

Keras' own promotion deliberately demotes 64-bit dtypes on every backend
except tensorflow — `keras/src/backend/common/dtypes.py:240`:

```python
"float64": "float64" if config.backend() == "tensorflow" else "float32",
```

Our `convert_to_tensor` routes implicit dtypes through that `result_type`
(`_backend/core.py`), so float64 inputs compute in float32 — exactly what
the jax backend does with x64 disabled (the policy Keras' lattice is
documented to match: "attempts to match `jnp.result_type`").

The numpy backend keeps float64 **only because its ops operate on raw
arrays without a conversion step** — its own `convert_to_tensor` demotes
identically to ours. The "reference behavior" the fuzzer flags is an
accident of the reference's implementation, not a policy we're violating.

## The options

**A. Keep the backend as-is; teach the fuzzer the policy.** (Recommended.)
The backend is conformant with Keras' documented promotion and with
jax/torch precedent, and the 2,127-test green tally was earned under this
behavior. The fuzzer change is small: when the reference output is float64,
the backend under test follows the Keras 64→32 demotion, and values agree
under float32 tolerances, report PASS with a note instead of FAIL
(alternatively: generate float64 cases through explicit
`convert_to_tensor(x, "float64")`, which both backends honor end-to-end).
Blast radius: `tools/parity_fuzz.py` comparison only; no backend change; no
test-tally risk.

**B. Preserve concrete float64 array dtypes at conversion.** Matches the
numpy backend's raw-op behavior, diverges from Keras' promotion table and
from jax/torch. Blast radius: `convert_to_tensor` semantics change under
every implicit conversion (trainer batches included); re-run of the full
Keras tally required with real risk to currently-green dtype-expectation
tests; slower compute wherever float64 sneaks in; float64 kernel support
varies across tinygrad devices.

## Addendum (2026-08-03): the both-demoted case

The original implementation keyed the tolerance demotion on a ref-float64 /
test-float32 OUTPUT mismatch. But some reference-backend ops apply the same
64→32 conversion (e.g. `tanh` routes through `convert_to_tensor`), so both
sides emit float32, no mismatch exists, and the case was judged under the
float64 tolerances of its INPUTS — two float32 computations one ulp apart
became a FAIL (`0017-op-tanh` at `--cases 100`). The comparison now judges
by the common output float dtype whenever both sides agree on one, which is
the same "the weaker dtype judges" rule this memo decided.

## The invariant tension (named, per the decision)

Invariant 1 says the numpy backend is the semantic reference — option A
knowingly deviates from the reference's *observed* behavior on the grounds
that Keras' own promotion table (and the reference's own conversion path)
contradict it. That deviation-with-a-named-reason is the owner's call.
