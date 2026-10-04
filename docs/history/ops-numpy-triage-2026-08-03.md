# Triage — the keras/src/ops/numpy_test.py failure tail

Status: **triage only, 2026-08-03** — no backend code was changed. This suite
was never part of the support matrix; the ops/math wave logged it as an
untriaged candidate work item. This memo classifies every failure so the
follow-up waves can be planned and sized. Numbers verified with one full run
on write day; re-verify with the command below rather than trusting them.

```sh
scripts/referee.sh keras/src/ops/numpy_test.py   # clones the pinned keras tag itself
```

## The tally (2026-08-03, keras clone, tinygrad backend)

**4309 passed / 1196 failed / 708 skipped** (20m30s wall). Matches the number
logged by the ops/math wave exactly. The 708 skips are the backend-gated
requirements (SparseTest and friends — `SUPPORTS_SPARSE_TENSORS` is False by
design); **zero failures come from sparse**, so the "test-side requirements"
bucket is empty on the failure side.

Every one of the 1196 failures was classified by parsing the full `-vv`
FAILURES section (all 1196 blocks accounted for; parser in the session
scratchpad).

## Bucket table

| Bucket | Count | Share | Effort shape |
|---|---|---|---|
| (a) missing op — loud `NotImplementedError` via PEP 562 `__getattr__` | 1047 | 87.5% | mechanical port from the numpy backend for ~48 of 51 ops; see exceptions below |
| (a2) implemented op, loud `NotImplementedError` on a mode — `pad` reflect/symmetric | 53 | 4.4% | moderate: flip+concat/gather indexing, one op |
| (b) dtype-promotion mismatches (expected jax-lattice dtype, got another) | 60 | 5.0% | policy plumbing through `result_type`; low risk, needs the lattice read carefully |
| (c) shape/value disagreements on implemented ops — candidate real bugs | 4 | 0.3% | small targeted fixes; **investigated below, 3 of 4 are real wrong-answer bugs** |
| (e) loud crashes on argument forms the backend never handled | 32 | 2.7% | mechanical input-handling hardening |
| (d) test-side requirements (sparse, tf-only) | 0 | — | already absorbed by the 708 skips |

Nothing silent anywhere: every failure is either a loud raise or a test
assertion. Invariant 2 held.

## (a) Missing ops — 1047 tests over 51 ops

Per-op failing-test counts (dtype-suite tests dominate, so counts track how
parametrized each op is, not difficulty):

| Count | Ops |
|---|---|
| 122 | `nextafter` |
| 56 each | `append`, `cross`, `fmax`, `fmin`, `fmod`, `column_stack`, `dstack`, `logaddexp2` |
| 37 | `view` (bitwise reinterpret; tinygrad `bitcast`) |
| 17 | `rot90` |
| 16 each | `gcd`, `kron`, `lcm` |
| 14 each | `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser` (window fns — host-built constant tables, sanctioned numpy role) |
| 12 each | `percentile`, `quantile`, `nanpercentile`, `nanquantile`, `angle`, `cbrt`, `corrcoef`, `diagflat`, `dsplit`, `hsplit`, `vsplit`, `fabs`, `isneginf`, `isposinf`, `nanargmax`, `nanargmin`, `nancumsum`, `nanprod`, `nanstd`, `nanvar`, `sinc`, `trapezoid` |
| 11 each | `argpartition`, `vander`, `isreal` |
| 1 each | `allclose`, `array_split`, `fliplr`, `flipud`, `i0`, `slogdet`, `unique`, `vectorize` |

Effort notes:

- **Mechanical** (port the numpy-backend body onto Tensor ops): everything in
  the 56/16/14/12/11 rows plus `rot90`, `fliplr`, `flipud`, `array_split`,
  `allclose` (compose on `isclose` — fix its bug first, see below), `i0`,
  `slogdet` (compose on existing `logdet`/linalg machinery).
- **Needs a design decision or real work**: `nextafter` (bit-level float
  increment — wants `bitcast` tricks, largest single op at 122 tests);
  `unique` (data-dependent output shape vs lazy graphs — may be honest to
  leave loud, like other backends' dynamic-shape limits); `vectorize`
  (host-side python looping — decide whether it belongs in a lazy backend
  at all).
- When porting these, conform to the jax dtype lattice at the same time —
  most of the 1047 are `NumpyDtypeTest` cases, and a port with wrong
  promotion just moves the test from bucket (a) to bucket (b).

## (a2) pad modes — 53 tests

`pad` exists but raises loudly for `mode="reflect"` (21) and
`mode="symmetric"` (32). One implementation (flip + concat per axis, or a
gather with reflected indices) clears the whole bucket. Gradient flows
through gather/concat, so no invariant tension.

## (b) dtype-promotion mismatches — 60 tests

Implemented ops whose result dtype disagrees with the jax-lattice
expectation: `arctan2` (14), `average` (14), `einsum` (11), `power` with
python scalars (7 — weak-typing: `power(int_tensor, 1.0)` must promote to
floatx, backend keeps the int/bool dtype), `prod` (5), `cumsum` (4), `full`,
`matmul`, `max`, `min`, `square` (1 each). Typical shape: `average(float16,
weights=int16)` must stay float16, backend returns float32 (the `_float`
helper jumps to floatx instead of consulting `result_type`). All fixes are
promotion plumbing, not numerics; same family as the settled
`docs/float64-promotion.md` policy, so no new policy decision is expected —
just apply the lattice.

## (c) shape/value disagreements — 4 tests, investigated

The dangerous bucket: implemented ops that **return an answer** and the
answer is wrong. Three investigated in depth (the fourth, `signbit`, fell
out trivially during the sweep):

- **`diag` / negative k — REAL BUG.** `numpy.py:1452`: for 1-D construction
  the pad offset is `max(k, 0)`, which is 0 for negative k; the
  column-broadcast path needs `abs(k)` leading zeros, so every `diag(x, k<0)`
  places the diagonal shifted by |k| and drops elements
  (`diag([1,2,3], k=-1)` loses the 1 and misplaces the rest). Silent wrong
  values from a differentiable op — invariant-1 violation, **urgent**;
  the fix is a one-liner (`pad_before = abs(k)`).
- **`dot` / N-D semantics — REAL BUG.** `numpy.py:1545` falls through to
  `matmul` for ndim >= 2. numpy `dot` with a >2-D right operand is the
  outer-stacked contraction (result shape `a.shape[:-1] + b.shape[:-2] +
  b.shape[-1:]`); matmul broadcasts batch dims instead — (2,3,4)·(2,4,5)
  returns (2,3,5) where numpy returns (2,3,2,5). Wrong shape, and where
  shapes collide, silently wrong values. Real bug; fix by routing ndim>2
  cases through `tensordot`-style reshape/matmul. Rare call pattern in
  models, but it answers instead of raising — **urgent by invariant 1**.
- **`isclose` / non-finite inputs — REAL BUG.** `numpy.py:779` uses the
  naive `|a-b| <= atol + rtol*|b|` with no non-finite special-casing:
  `inf <= inf` is True, so `isclose(0, inf)`, `isclose(inf, -inf)` and the
  nan row all report True where numpy says False (4 of 7 special-value
  elements wrong). Finite inputs are correct. Fix is mechanical: require
  both finite for the tolerance test, OR exact equality for infinities,
  AND out nans unless `equal_nan`. Also gates `allclose` (bucket a).
- **`signbit` / negative zero — benign edge, named cause.** `numpy.py:1802`
  implements it as `x < 0`, which cannot see the sign bit of `-0.0`
  (1 of 6 test elements). Needs actual bit inspection (`bitcast` to int and
  mask). Not gradient-relevant; low priority but keep it loud in the tally
  until fixed.

## (e) loud crashes — 32 tests

Argument forms that blow up before producing a tensor (loud, so no invariant
violation — but they gate real API surface):

- `linspace` (25): array-valued `start`/`stop` hit `float(stop)` →
  `TypeError` (`numpy.py:1194`, 24 tests), and `num=0` divides by zero
  (1 test).
- `where` / `select` (2): when the condition is already a Tensor, branch
  args skip `convert_to_tensor` and raw numpy arrays reach
  `Tensor.where` → tinygrad-internal assert. Same one-line-shape fix as the
  isclose family: convert both branches.
- `full` / `full_like` (2): array-valued `fill_value` → `TypeError`
  (scalar-only path).
- `average` (1): tuple `axis` indexes a python list with a tuple →
  `TypeError`.
- `split` (1): list-of-indices `indices_or_sections` form unsupported.
- `eye` (1): float `N`/`M` raises `AttributeError` where the test expects
  the numpy-style error type.

All mechanical input-handling ports from the numpy backend.

## Recommended attack order

1. **Bucket (c) first — it's tiny and it's the only silent-wrong-answer
   surface.** `diag` (one-liner), `isclose` (small mask), `dot` N-D routing,
   `signbit` last. ~4 tests green, but this is the invariant-1 debt.
2. **Bucket (e)** — 32 tests of mechanical argument hardening; `where`/
   `select` first since raw-array branches could bite real users.
3. **Bucket (a) mechanical tier, largest ops first** — the eight 56-test ops
   (`append`, `cross`, `fmax/fmin/fmod`, `column_stack`, `dstack`,
   `logaddexp2`), then `view`, then the 16/14/12/11 rows. Port each with its
   jax-lattice dtype behavior so tests don't migrate to bucket (b).
   ~900 tests.
4. **Bucket (a2)** — pad reflect/symmetric, one focused implementation,
   53 tests.
5. **Bucket (b)** — the 60 promotion mismatches on existing ops; do after
   the ports so the lattice work is done once, with fresh context.
6. **Deferred/decide**: `nextafter` (122 tests, bit tricks), `unique` +
   `vectorize` (may stay loud by design — if so, record the decision the
   way complex support was recorded).

Ceiling check: everything above except item 6 is ~1070 of the 1196; with
item 6 landed too the suite should reach the same "green or loudly,
deliberately red" state as ops/math.
