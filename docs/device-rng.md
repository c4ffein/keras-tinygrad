# Device RNG — the tinygrad-native path for random ops

Status: **DEFAULT ON since 2026-08-30** (`KERAS_TINYGRAD_DEVICE_RNG=0`
restores the host/numpy path everywhere; read the flag only through
`core.device_rng_enabled()`).
Both flip criteria were met the same day, data below. **Wave 3
(same day):** device paths extended to `random.normal` and
`random.uniform`, but ONLY inside the trainer's `device_rng_scope` (the
train step) AND for `SeedGenerator` seeds — seed-TYPE alone cannot make
the split (the global generator and initializers ride SeedGenerators and
None too; the first attempt broke `test_global_seed_generator` and
`test_random_normal`, both fixed by the scope). Covered layers now:
Dropout, GaussianNoise, GaussianDropout, AlphaDropout — EXACT types only
(2026-08-30 review hardening: a subclass may add host randomness in
call() that a capture would freeze silently, so a class joins the list
only with its own A/B receipt; SpatialDropout et al. therefore run
eager, correctly). All four JIT (A/B-verified, plus an lr=0 same-batch
isolation test proving 6/6 distinct losses = fresh masks under replay).
The preprocessing Random* family forces eager (mixed op usage incl.
host-only samplers; inside the scope its dropout/normal/uniform draws
ARE on-device — fresh and correct every eager step, the host-only ops
sample numpy). Lifting the gate wholesale would let host-drawn noise be
baked into a capture and silently replayed frozen — the exact bug class
the gate prevents.

**Seeding, revision 2 (2026-09-01 review):** the stream is re-seeded at
every train-function build (`random.reset_device_stream`, called by the
trainer's `make_train_function`; `keras_tinygrad.reset_device_rng()` for
loops that never rebuild one), from the first device draw's SeedGenerator,
and that seeding ADVANCES the generator (`next()`), so a continued fit on
the same model moves to a new stream instead of replaying — the same
shape as the numpy backend, whose generators advance across fits. Receipt:
`test_dropout_training_is_reproducible_in_one_process` (the once-per-
process seeding of revision 1 made `set_random_seed(0); build; fit` twice
in one process disagree — a notebook re-running a cell got new masks).
Also new: a None seed inside the scope (a custom layer's seedless
`keras.random.*` call) takes the device path — it resolves to the global
generator, and on the host path a capture would have baked it (tinygrad's
JitError made that loud: eager forever). Receipt:
`test_seedless_random_op_in_step_jits_and_stays_fresh`. Re-seeding
replaces tinygrad's counter buffers, so it must never land inside a
capture: the trainer's first step after a build is the eager probe, and
exporters seed before capturing state (`keras_tinygrad.webgpu`).

**Seeding, revision 1 (corrected 2026-08-30, found in review):** Keras wraps EVERY
layer seed in a `SeedGenerator` — `Dropout(0.5, seed=42)` included — so
explicit layer seeds take the device path too; an earlier revision of
this doc claimed they kept the host path, and worse, nothing seeded
tinygrad's stream (default = wall clock): device-RNG training was
irreproducible across processes. Fixed: the device stream is seeded once
per process from the first device draw's SeedGenerator state
(`random._ensure_device_stream_seeded`) — same seeds, same script ⇒ same
run, cross-process (receipts: `tests/test_backend_regressions.py`,
which runs the same dropout fit in two subprocesses). The contract is
DETERMINISM under Keras seeding, not numpy-bit-parity. Draws outside the
scope (initializers, build-time draws, direct keras.random calls outside
a train step) keep the host/numpy reference path.

## The mechanism

Invariant 9 makes random ops sample on the host with numpy, bit-identical
to the reference backend. Host RNG is invisible to a TinyJit capture, so
the trainer permanently gates Dropout models to eager — forfeiting the
~5–15× steady-state train-step speedup — and makes dropout unexportable
(a captured mask would replay forever).

tinygrad's own answer is `Tensor.rand`: a threefry counter-based RNG
computed ON DEVICE, whose counter advance is part of the graph. Verified
on tinygrad 0.13 (probe, 2026-08-30): a jitted function containing
`Tensor.rand` produces distinct values on all replays — the counter is a
realized buffer that advances like a weight. Under the flag,
`random.dropout` builds its mask this way; the trainer's seed-generator
gate lifts (`trainer.py::_check_gates`), verified A/B: Dropout model
`_disabled=True` without the flag, `False` with it, training green.

## The trade (named, per invariant 1's rule)

- Masks are NO LONGER bit-identical to the numpy reference backend.
- The Keras per-call seed (`SeedGenerator`) is consumed ONCE per training
  run (the seeding draw), not per call; determinism follows tinygrad's
  stream (`Tensor.manual_seed` + counter).
- Referee: `dropout_test.py` + `random_test.py` = 89 passed / 4 skipped
  in BOTH modes (2026-08-30) — the suite does not depend on mask
  bit-parity.

## Scope and sequencing

Inside the train step only, deliberately (dropout first, then normal and
uniform — the ops the regularization layers draw through). Initializers
and data-side sampling stay on the host path (they run once / outside
capture, and their bit-parity is what cross-backend weight-init
comparisons actually use).

Sequencing (owner-agreed): clean tinygrad-native solution in Python
FIRST — this document — then the export/JS story: an exported bundle
containing the threefry kernels + counter buffer as state is
self-contained browser dropout (no per-step mask uploads), the
alternative to mask-as-input. Unverified through export yet.

## Criteria to flip the default

1. ~~Full layers-tree referee~~ **MET 2026-08-30, twice**: wave 2
   (dropout-only) and wave 3 (scoped normal/uniform + all four
   regularization layers covered) each ran the full tree: 5 failed /
   1,989 passed / 215 skipped / 1 xpassed — the exact baseline
   fingerprint, failures named and identical to the documented five
   (float8 x2, RandomCrop x2, AutoContrast). Zero regressions.
2. ~~Export path validated~~ **MET 2026-08-30**: a Dense+Dropout+Dense
   Keras step exported on NULL:WGSL (27 kernels; `rng/seed` and
   `rng/counter` as named state buffers, safetensors gained a U32 case;
   `validate_runner_js` passed), and in headless Chromium the SAME batch
   fed three times produced 3/3 distinct finite losses — the counter
   buffer advances across `step()` calls in WebGPU. Probe:
   `experiments/m0-keras-trainstep/export_dropout_probe.py`.
3. A note in README's status section naming the deviation, same-diff.
