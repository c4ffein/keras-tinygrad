# Decision memo — `unique` and `vectorize` (the last two loud ops.numpy stubs)

Status: **open, owner reading**. These are the only remaining
`keras.ops.numpy` failures that are decisions rather than work
(`test_unique`, `test_vectorize`; everything else in the suite is green or
upstream-broken). Both raise loud `NotImplementedError` today.

## `unique` — the data-dependent-shape problem

`np.unique([1, 2, 2])` → `[1, 2]` (length 2); `np.unique([1, 1, 1])` →
`[1]` (length 1). The output length depends on the input's *values*.
tinygrad, like JAX, builds shape-static lazy graphs: every shape must be
known at graph-build time, before any value is computed. "A tensor whose
length I learn by running" is unrepresentable in that model — this is a
framework-class limitation, not a backend gap.

### The padded-form trick (the "add a len" idea — it's real)

The standard shape-static workaround: return a **fixed-size padded array
plus a count**. Pre-declare a maximum size (worst case: the input length —
all elements distinct); the op returns `(values_padded_to_size, n_unique)`
with the tail filled by a sentinel. Shape is static; the data-dependence
now lives in a *value* (the count), which lazy graphs handle fine.

This is exactly `jax.numpy.unique(x, size=…, fill_value=…)` — jax refuses
to run `unique` under `jit` *unless* `size` is given. Real-world precedent
for the host-side option (a): keras PR #23193's MLX backend materializes
`unique`/`nonzero`/`signbit` to numpy as documented inline policy.
Reading list:
- jax.numpy.unique docs (the `size`/`fill_value` parameters).
- JAX "Thinking in JAX" / sharp-bits docs on shape polymorphism and
  data-dependent shapes.
- The same pattern appears in XLA's `SetDimensionSize` and torch's
  `torch.unique` being incompatible with `torch.compile` fullgraph mode.

### The catch, and the three options

Keras' public `keras.ops.unique` API has **no `size` parameter** — the
contract is the numpy one: exact-length output. So:

- **(a) Host-side implementation.** Realize the input, `np.unique` on
  host, wrap the result. API-conforming, exact. Cost: numpy between input
  and output — normally invariant 3's banned zone — so the sanctioned-host
  -roles list must be formally extended with "data-dependent-shape
  structural ops". Mitigating: `unique` is not usefully differentiable on
  ANY framework (torch's breaks graphs; jax's is constant-folded), so no
  gradient capability is being silently lost, and the op is
  overwhelmingly used on int/index data. Also unavailable inside a future
  jitted context (host access raises `JitError` at capture — loudly, so
  no silent staleness).
- **(b) Stay loud** (today's behavior). jax-without-size precedent. Users
  who need it call `np.unique(convert_to_numpy(x))` themselves — same
  computation, honesty preserved, the detach is in *their* code.
- **(c) Padded backend extra.** Implement the `(padded, count)` form as a
  documented backend-specific extension while `keras.ops.unique` stays
  loud. Pure-tinygrad (sort + neighbor-diff + cumsum builds it lazily),
  differentiability irrelevant. Most engineering for least API value —
  only worth it if something (NNVP?) needs unique *inside* a lazy graph.

Recommendation to compare against after reading: **(a)** with the
invariant amendment written in the same diff, falling back to (b) if the
amendment feels like scope creep. (c) only on demonstrated need.

## `vectorize` — not a tensor op at all

`np.vectorize(f)` maps an **arbitrary Python function** elementwise — a
Python loop in a trench coat. The numpy backend passes trivially because
a host loop is its native execution mode. For a lazy-graph backend the
options are:

- Per-element host loop: realize every scalar, call Python `f` on each —
  catastrophically slow, gradient-destroying, and would freeze under any
  jit. A trap dressed as a feature.
- Symbolic trace of `f` (what `jax.vmap` actually does): real compiler
  engineering, and only valid when `f` is composed of keras ops — at
  which point the user could have written the composed expression
  directly.

Recommendation: **stay loud, permanently**. A slow-and-detached
`vectorize` is precisely the "plausible but wrong" artifact the
invariants exist to refuse. Revisit only if keras itself grows a
vmap-style contract.

## Referee note

`test_unique` and `test_vectorize` each cost exactly one test in the
suite tally (5502/3/708 as of 2026-08-03; the third red is the
upstream-broken `test_cross`). Neither decision moves the number
meaningfully — this memo is about API honesty, not coverage.
