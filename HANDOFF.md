# HANDOFF — state of the keras-tinygrad bridge (2026-08-03)

Written by the session that built all of this in one day, for whoever picks it
up next (human or agent). Everything below was true at write time; verify with
the commands given rather than trusting numbers blindly.

## What this is

The first Keras 3 backend for tinygrad. Two forms:

1. **In-tree**: `/home/dev/workspace/keras` — a keras-team/keras clone with
   `keras/src/backend/tinygrad/` (~10 modules) plus 6 small keras-core edits
   (4 dispatch elif chains, `standardize_dtype`, `DynamicBackend`). This is
   the SOURCE OF TRUTH for backend code. ALL UNCOMMITTED — the owner commits
   himself, never commit for him.
2. **Packaged**: this repo — pip package running the same backend against
   STOCK pypi keras via a meta-path import hook (6 match-exactly-once source
   patches; see `docs/how-it-works.md`). Vendored copy under
   `src/keras_tinygrad/_backend/` is synced from the clone by
   `scripts/sync_vendor.py` (last synced after wave 2; run `--check`).

## Verified state (2026-07-27)

- Keras' own FULL layers tree on the clone, preprocessing included
  (tf-venv, py3.12+tf, re-certified 2026-08-03 after the jit + op waves):
  **1,989 passed / 5 failed / 215 skipped / 1 xpassed (99.7%)**. The 5 =
  2× upstream float8 (PR package drafted), 2× RandomCrop (tinygrad slice
  bounds, upstream item), 1× AutoContrast (FMA 1.9e-06 vs atol 1e-06).
  The grain/sqlite flake did not reproduce (warm kernel cache). The old
  2,127/4/201 headline was the py3.14 ktg-venv profile WITHOUT
  preprocessing collected — different skip/collection profile, superseded.
- Support matrix (in README between the SUPPORT_MATRIX tokens — the README
  table is the authoritative row list and totals; don't trust counts
  repeated here): the 2026-08-03 dip from 99.8% is the DENOMINATOR growing:
  ops/image and the
  preprocessing tree entered the matrix 2026-08-03 (TF venv unlocked their
  collection) and bring their honest tails with them. ops/math is 208/0/4
  (100%) as of 2026-08-03 — cdist, logdet and
  the segment_max/min/prod family landed (masked broadcast-reduce with
  identity fill, differentiable w.r.t. data), and view_as_complex/
  view_as_real went green via complex-lite interop: core's `ComplexTensor`
  wrapper (real/imag Tensor pair, dtype the string "complex64", closed op
  set, loud NotImplementedError on all complex arithmetic). Tier boundary
  + extension rule: docs/complex-support.md.
- Parity fuzz vs numpy reference: `make fuzz` 100/100 (op+layer),
  `make fuzz-grad` 60/60 (finite-difference gradients), legacy
  `--cases 80 --kinds op` 80/80 — after the float64-promotion decision and
  its both-demoted addendum (both docs/float64-promotion.md; the superseded
  39/40 run's one flag was that policy).
- Package smoke vs stock keras 3.15: green (`examples/mlp_smoke.py`).
- Loader test suite (`tests/test_loader.py`, 9 tests, in CI): green. Covers
  keras-first RuntimeError, idempotent double-import, explicit-backend
  respect, anchor-mismatch loudness, anchor drift vs installed keras.
- Loader anchors verified against BOTH keras 3.15.0 (installed) and the
  3.15.1 wheel (downloaded, file-level check); README/pyproject/docs now
  all state 3.15.x — the old "3.15 / 3.16" claim was wrong, 3.16 does not
  exist on PyPI as of 2026-08-03.
- `sync_vendor.py --check` was crying wolf: it checked ANCHORS against the
  clone, but the clone is the patched state (the standardize_dtype patch
  consumes its anchor). It now checks the clone for the REPLACEMENT texts
  (byte-equal to the loader's) and found real drift doing it: the clone's
  layer/model elif edits used one-line imports vs the loader's wrapped
  form — loader aligned to the clone. `--check` and `--self-check` both
  exit 0 now.

## How to run anything

```sh
# in-tree tests (the referee):
cd /home/dev/workspace/keras && KERAS_BACKEND=tinygrad \
  CC=/home/dev/.local/bin/zigcc /home/dev/workspace/ktg-venv/bin/python \
  -m pytest keras/src/layers/... -q --no-header -p no:cacheprovider
# packaged smoke (stock keras venv):
cd /home/dev/workspace/keras-tinygrad-pkg && CC=/home/dev/.local/bin/zigcc \
  KERAS_BACKEND=tinygrad /home/dev/workspace/pkg-test-venv/bin/python \
  examples/mlp_smoke.py
# loader test suite (same venv; pip+pytest were ensurepip'd into it 2026-08-03):
CC=/home/dev/.local/bin/zigcc /home/dev/workspace/pkg-test-venv/bin/python \
  -m pytest tests -q --no-header
# dev loop (uv-based, since 2026-08-03; .venv resolves keras 3.15.1):
make verify     # ruff lint + format check + loader tests
make tutorial   # executes every python block in TUTORIAL.md
make smoke fuzz vendor-check
```

`CC=zigcc` matters: this box has no clang; the shim at
`/home/dev/.local/bin/zigcc` makes tinygrad's CPU jit compile via the
ziglang wheel (translated target triple, `-g0`). Real CI uses real clang.

## The rules that shaped the code (do not regress them)

Full list: `docs/architecture.md` (11 invariants). The load-bearing ones:
numpy backend is the semantic reference; Keras' own tests are the referee;
NO silent numpy fallbacks in differentiable paths (loud NotImplementedError);
copy-on-convert (tinygrad wraps numpy zero-copy + lazy reads);
monkeypatches additive+guarded only; every keras-core touchpoint gets a
loader patch-table anchor in the same change.

## Known remainders (smallest first)

1. ~~Fuzzer papercuts~~ RESOLVED 2026-08-03: the "exits 0 despite FAIL"
   claim was stale — the real bug was the printed repro command omitting
   `--kinds`/`--tol-scale` (a FAIL reproduced as a silent PASS, or "case
   not found"); fixed, plus a new guard: a run where NO case compares
   (dead reference → all SKIPPED) now exits 2, never 0. float64 promotion:
   DECIDED + IMPLEMENTED (Option A, fuzzer side) — ref-float64/test-float32
   is ok-with-note under float32 tolerances, reverse direction still fails;
   backend unchanged; fuzz `--seed 0 --cases 80 --kinds op` now 80/80.
   Mechanism + decision record: `docs/float64-promotion.md`.
2. ~~ops/math stub tail~~ RESOLVED 2026-08-03, fully green: 49 of 53 red
   fixed mechanically (cdist, logdet, segment_max/min/prod; segment_sum
   refactored onto a shared prepare helper), the last 4 via the
   complex-lite ComplexTensor interop (docs/complex-support.md).
3. ~~ops/numpy tail~~ RESOLVED 2026-08-03 in three waves (fix wave, port
   wave A, port wave B): **5502 passed / 3 failed / 708 skipped** from
   4309/1196/708 at triage. All silent-wrong bugs fixed (diag/dot/
   isclose+allclose/signbit-bitcast), pad reflect/symmetric, 32 argument
   crashes, ~45 ops ported (nextafter ulp-bitcast, tensor Euclid gcd/lcm,
   sort-based percentile/quantile family, window fns, nan-family, etc.),
   full bucket-(b) promotion alignment (arctan2/average/einsum/power/
   prod/cumsum/matmul/max/min/square). The 3 red: unique + vectorize
   (DECISION ITEMS: data-dependent output shapes — implement host-side
   with realize, or stay loud; owner call) and test_cross (test-side:
   numpy 2.x removed 2-element np.cross, the test's own reference crashes;
   upstream-PR bucket). Zero regressions at every wave; math_test 208/0/4
   throughout. NOTE: tutorial's loud-stub demo now uses ops.unique (its
   rot90 demo broke when rot90 landed — the executable tutorial caught it).
4. Dev tooling landed 2026-08-03: uv + ruff (line-length 120, _backend/
   excluded — it must stay byte-identical to the clone), Makefile
   (verify/tutorial/smoke/fuzz/vendor-check), executable TUTORIAL.md
   enforced by tests/test_tutorial.py, CI rewritten onto uv+make.
   Repo-owned code ruff-formatted; anchors re-verified after.
   requires-python fixed to >=3.11 (keras 3.15's own floor). keras 3.15.1
   now runtime-verified (uv .venv resolves it; loader suite + tutorial
   train against it).
5. ~~TF venv for preprocessing/ops-image collection~~ RESOLVED 2026-08-03:
   /home/dev/workspace/tf-venv (uv-managed CPython 3.12.13, tensorflow-cpu
   2.21 + jax + grain, clone keras editable). Referee results AT VENV
   CREATION (superseded — current numbers live in the README matrix):
   ops/image 306/25/5 (92.4%), preprocessing tree 679/14/29 (98.0%) after
   3 real bug fixes in the clone's numpy.py (TrackedList shapes in
   reshape/broadcast_to — tinygrad argfix does exact-class checks; 0-d
   Tensor pad widths read out as host ints). Remaining red: 35 tests =
   four missing image ops (perspective_transform, sobel_edges,
   gaussian_blur, elastic_transform — next mechanical work item);
   RandomCrop x2 (tinygrad __getitem__ rejects Tensor slice bounds —
   add to the tinygrad-upstream dunders conversation); AutoContrast x1
   (FMA contraction residual 1.9e-06 vs atol 1e-06); grain x1 (thread
   pool vs tinygrad's process-global sqlite kernel cache).
6. Fused RNN kernels (lstm/gru fast path) — generic scan is correct, slower.
7. ~~TinyJit on the train step~~ LANDED 2026-08-03: `_TrainStepJit` in
   trainer.py — per-batch-signature captures, pinned-buffer weight
   propagation, loud-JitError-at-capture fallback to eager (tape/schedules/
   RNG hazards), 12-scenario bit-for-bit eager parity, ~5–15x steady-state
   steps/sec (~2.3x even on the warmup epoch). Escape hatch
   KERAS_TINYGRAD_TRAINER_JIT=0. Known residue: keras/src/trainers
   collection needs a test-side backend branch (same upstream bucket as
   float8); validation scripts preserved in the session scratchpad.
   test/predict paths still eager — the remaining smaller lever.
8. Upstream float8 test branch (keras test files) — part of any upstream PR.

## Decision queue (owner's, not yours)

- Commits/checkpoints in the clone and this repo.
- Publishing this repo + PyPI name claim.
- tinygrad upstream PR (draft ready in `/home/dev/workspace/tinygrad-upstream/DRAFT_PR.md`;
  their rules require AI-assistance disclosure; `__bool__` half needs an
  issue first — it reverts a deliberate upstream ban). Two more candidates
  found 2026-08-03: Tensor slice bounds in `__getitem__` (blocks
  RandomCrop) and argfix's exact-class tuple/list check (worked around
  backend-side for TrackedList).
- keras upstream: realistic path is a backend-plugin entry-point PR, not the
  backend itself (see Chollet's criteria in keras#20793; confirmed by
  keras#23193 — a complete MLX backend PR, 12k tests green, closed unmerged
  on process friction). READY TO DROP: docs/upstream/keras-pr/ has
  tests-fix.patch (git-apply-clean, 4 files) + PR_BODY.md for the
  test-side bundle (test_cross numpy>=2.5 guard, float8 skipif,
  trainer_test fallback; ViewAsComplex skip deliberately dropped — we pass
  those now). Full analysis: docs/upstream-keras-draft.md — including a
  verified companion fix for the numpy BACKEND's own broken cross
  (numpy.py:554, red under numpy>=2.5), excluded from the test-side patch,
  owner decides whether to bundle it. tinygrad additions:
  docs/upstream-tinygrad-draft.md (slice bounds path traced to
  tensor.py:878/movement.py:87, argfix isinstance one-liner).
- NNVP integration M0: trace a Keras TrainStep on tinygrad NULL device →
  WebGPU runner (design + unknowns in the nnvp session scratchpad doc
  `nnvp-keras-tinygrad-design.md`; the Pyodide probe results in
  `pyodide-keras/`). This is the one genuinely unproven link.

## Related artifacts elsewhere

- `/home/dev/workspace/tinygrad-upstream` — dunders patch + DRAFT_PR.md.
- nnvp scratchpad (session-tied, may be gone): design doc, build story,
  Pyodide harness. The build story was delivered to the owner directly.
- The owner's nnvp project memory has a condensed version of this file.
