# Complex dtype support — the two tiers

Status: **tier 1 landed (2026-08-03)** — `ViewAsComplexRealTest` 7/7,
ops/math 208/0/4 (100%), no keras-core touchpoints, no new monkeypatches,
`SUPPORTS_COMPLEX_DTYPES` stays False. Tier 2 deliberately not attempted.
This memo defines the boundary so nobody re-derives it — or accidentally
crosses it — later.

## Why there is a boundary at all

tinygrad has no complex dtype: nothing in its dtype system, no kernels
that could compute with one. Keras, meanwhile, exposes complex in two very
different ways, and they demand very different amounts of machinery.

## Tier 1 — complex interop (bounded; the current attempt)

A complex **value** can exist, enter, and leave the backend. Concretely: a
`ComplexTensor` wrapper holding a `(real, imag)` pair of float tinygrad
Tensors, plus exactly the closed op set that `keras.ops.view_as_complex` /
`view_as_real` exercise (keras-core implements both ops itself and only
calls the backend for: `convert_to_tensor` on complex numpy arrays,
`convert_to_numpy` back out, `cast` to `"complex64"`, `real`/`imag`,
wrapper+wrapper addition, and scalar-complex multiply for the `1j * x`
step). Everything else a wrapper touches raises a clean, specific
`NotImplementedError` — complex values must never silently flow into
float math.

Precedents for this containment pattern already in the backend: the fft
family does complex *semantics* on `(real, imag)` tensor tuples (Keras
designed those APIs tuple-shaped for exactly this reason), and float8 is
a recast dance over float32 buffers. Tier 1 is the same move: represent,
convert, guard.

What tier 1 buys: the 4 `ViewAsComplexRealTest` referee tests, correct
interop for users passing complex arrays at the boundary, and honest loud
errors instead of a `ValueError` at the dtype table.

## Tier 2 — complex arithmetic (not attempted; upstream-gated)

Complex **math**: `(a+bi)(c+di)` in every multiply/matmul, conjugation
rules in reductions and norms, complex `abs`/`angle`/`exp`, complex
`einsum`, and — the genuinely hard part — gradients, where the
Wirtinger-calculus conventions differ between torch and jax, so there is
no single "reference" to port. That is a parallel arithmetic layer across
the whole ops surface, with a correctness surface Keras' own test suite
barely referees (invariant 1's "the referee exists" method has nothing to
referee it against).

Tier 2 is gated on tinygrad growing a native complex dtype upstream. If
that happens, the wrapper becomes a thin shim and this memo gets rewritten;
until then, any wrapper-based tier 2 would be a large, permanently
hand-verified surface — rejected.

## The rule this leaves behind

The `ComplexTensor` wrapper (if tier 1 lands) is an interop container,
not a number type. Extending its op set beyond what `view_as_complex` /
`view_as_real` require is a tier boundary crossing and needs this memo
updated first — with the referee that will judge the new ops named.
