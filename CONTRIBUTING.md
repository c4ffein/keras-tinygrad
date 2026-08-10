# Contributing

The project runs on one method. Follow it and reviews are fast.

## The method

1. **The numpy backend is the reference.** Keras' in-tree numpy backend
   defines the expected semantics for every op — shapes, dtype promotion,
   edge cases. When in doubt, read `keras/src/backend/numpy/` and port that
   behavior to tinygrad tensors end-to-end. Never port by intuition.
2. **Keras' own tests are the referee.** An op is done when the Keras test
   suite says so, not when a hand-rolled example looks right. We don't
   write parallel tests for op semantics; upstream already wrote them.
3. **One op at a time.** Implement an op → run its Keras tests → green →
   next op. Don't batch ten ops into one PR; a failing tally is impossible
   to review.
4. **Loud stubs over silent fallbacks.** An unimplemented op must raise
   `NotImplementedError` (the module-level `__getattr__` in `numpy.py`
   handles the common case). Never fall back to numpy inside an op — it
   silently detaches gradients and quietly lies in the coverage tally.
   Wrong answers are the one bug class this project refuses to ship.
5. **Monkeypatches: additive and guarded only.** Patching tinygrad or Keras
   at runtime is a last resort. A patch must add behavior, never change
   existing behavior for non-Keras users (`DType.__repr__` stays untouched
   while `__str__` gains Keras spellings — that's the template), must live
   next to a comment explaining exactly which Keras expectation forces it,
   and must be installed once at backend import time.
6. **Anchors match exactly once.** Any change to the `_loader.py` patch
   table keeps the invariant: every anchor occurs exactly once in the
   targeted Keras release, and a mismatch raises `ImportError` with the
   version-mismatch message. Widening an anchor to "probably matches" is a
   rejected PR.

## Workflow

Setup (uv-based; `uv run` auto-creates `.venv` with the project + dev tools):

```sh
make verify     # lint + format check + loader tests — the pre-review gate
make tutorial   # executes every code block in TUTORIAL.md
make smoke      # must print SMOKE OK before and after your change
make fuzz       # randomized parity hunt vs the numpy reference
```

No clang on your box? The Makefile automatically points `CC` at the zig
shim (see README). All checks together: `make verify tutorial smoke`.

Running Keras' tests against this backend: clone `keras` at the supported
release tag, make sure `keras_tinygrad` is imported before `keras` (a
root-level `conftest.py` containing `import keras_tinygrad` does it —
conftest loads before test modules), then run the test file that owns your
op, e.g. the ops tests under `keras/src/ops/` or the layer family's tests
under `keras/src/layers/`.

Implementing an op:

1. Find its numpy-backend implementation and its Keras test(s).
2. Write the tinygrad version in the matching `_backend/` module. Stay on
   `Tensor` the whole way — `Tensor.__array__` exists for the test harness
   and tuple-returning ops, not for op internals.
3. Run the op's Keras tests. Fix until green. If a test documents a
   semantic the backend can't support yet (see `SUPPORTS_*` flags in
   `core.py`), say so in the PR instead of skipping quietly.
4. The README status tables (the `<!-- SUPPORT_MATRIX -->` / `<!-- TALLY -->`
   token blocks) state referee results. Update them only together with the
   test run that produced the numbers, and say which run that was in the PR
   — never adjust a number without a tally to back it.

## PR checklist

- [ ] Op(s) green in Keras' own test file; test file named in the PR.
- [ ] No numpy round-trips inside op implementations.
- [ ] Stubs for anything intentionally unimplemented raise
      `NotImplementedError`.
- [ ] New monkeypatches (if any): additive, guarded, commented with the
      forcing Keras expectation.
- [ ] `_loader.py` anchors untouched, or re-verified to match exactly once
      on all supported Keras releases.
- [ ] `python examples/mlp_smoke.py` still prints `SMOKE OK`.

## Where things live

- `src/keras_tinygrad/_loader.py` — meta-path finder + patch table.
- `src/keras_tinygrad/_backend/` — the backend, one module per Keras
  backend surface (`core`, `numpy`, `nn`, `math`, `rnn`, `random`,
  `image`, `linalg`, `layer`, `trainer`, `export`).
- `examples/` — runnable smoke and demo scripts.
- `scripts/` — support-matrix / tally generation and dev tooling.

No clang on your box? The `ziglang` wheel + a `zig cc` shim (translate the
target triple, add `-g0`) gives tinygrad a working CPU jit — see the README.
