# HANDOFF — state of the keras-tinygrad bridge (2026-09-01)

Written by the session that built all of this in one day, for whoever picks it
up next (human or agent). Everything below was true at write time; verify with
the commands given rather than trusting numbers blindly.

## What this is

The first Keras 3 backend for tinygrad. Two forms:

1. **In-tree** (historical): `/home/dev/workspace/keras` — a keras-team/keras
   clone (master 2026-07-25) with `keras/src/backend/tinygrad/` plus 6 small
   keras-core edits. It was the source of truth until 2026-08-30; it is now
   a leftover. Nothing in this repo reads it.
2. **Packaged** (the only form): this repo — pip package running the
   backend against STOCK pypi keras via a meta-path import hook (6
   match-exactly-once source patches; see `docs/how-it-works.md`). The
   backend sources under `src/keras_tinygrad/_backend/` ARE the source of
   truth. The referee (`make referee` → `scripts/referee.sh`) clones the
   pinned keras tag into `.referee/` and runs Keras' own tests from inside
   that tree with the hook active — no hand-edited checkout anywhere.

## Verified state (2026-08-30, RNG/export items re-verified 2026-09-01)

- Keras' own FULL layers tree, preprocessing included — since 2026-08-30
  on the keras **v3.15.1 tag tree** via `make referee` (`scripts/referee.sh`,
  py3.12 + tensorflow for collection, failed set compared to
  `scripts/referee-baseline.txt`): **1,988 passed / 5 failed / 215 skipped /
  1 xpassed (99.7%)**. (Earlier tallies on a July keras master snapshot:
  1,989 passed, same 5 failures — the tree has one test fewer.) The 5 =
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
- `sync_vendor.py` (history): its `--check` mode against the sibling clone
  is gone with the clone. What remains is the one real guard — every
  loader anchor occurs exactly once in the INSTALLED stock keras
  (`make vendor-check`; `--self-check` is the same thing, kept for the
  workflows' command lines).

## 2026-09-01 review pass (Fable 5.1 over the whole uncommitted diff)

Owner approved every item; all landed in this diff, all uncommitted.
Receipts: `make verify` (22 tests, ~45 s — five new subprocess
receipts in `tests/test_backend_regressions.py`), `make browser-assets`
(green, wheel name derived as 0.1.1), and the full referee:
**5 failed / 1,988 passed / 215 skipped / 1 xpassed** (0:26:54, v3.15.1
tree, 2026-09-01) — the FAILED set is exactly the five baseline entries,
zero regressions from the RNG changes. Run twice: the first run (sharing
the box with `make verify` + `make browser-assets`) ended with pytest's
own exit status 127 after a complete summary, so the script stopped
before its comparison; the clean rerun (0:26:23, same tally) exited 0
through the script's own check: "OK — failed set == baseline (5 known)".

- **Device RNG revision 2** (`docs/device-rng.md`): the stream re-seeds at
  every train-function build (`random.reset_device_stream`, called from
  `make_train_function`; `keras_tinygrad.reset_device_rng()` for loops
  that never rebuild one) and the seeding draw ADVANCES the generator, so
  `set_random_seed; build; fit` twice in one process agree while a
  continued fit moves on (numpy-backend shape). Before: once-per-process
  seeding — a notebook re-running a cell got new masks. None seeds inside
  the scope take the device path (a seedless custom layer now JITs with
  fresh masks instead of the loud eager fallback). Device path coerces
  numpy-int dims (tinygrad's exact-class argfix).
- **Export hardening** (`keras_tinygrad.webgpu`): initial values come from
  each weight's declared initializer (`_initializer_map` resolves the
  owning layer's `<name>_initializer`; `_host_initial_values` ports Zeros/
  Ones/Constant/GlorotUniform/RandomUniform/RandomNormal, anything else is
  a NotImplementedError) — the "Glorot for kernels, zeros elsewhere" rule
  had exported BatchNormalization's gamma and moving_variance as 0.0, a
  dead network, silently. The loss must be built with `reduction=None`
  (checked before tracing: Keras' default divides by a host-tuple tensor
  that exported as an unwritten buffer). Labels derive from the loss
  (`Sparse*` → int32 ids, else float32 like the output; `label_shape`/
  `label_dtype` override). Placeholder batches are `Tensor.empty`, not
  `randn` (which advanced the captured RNG counter).
- **One exporter**: m0's `export_model.py` copy deleted, `m0.py` imports
  `keras_tinygrad._vendor`; the byte-equality test is gone with it.
  `experiments/pyodide-tinygrad/` keeps a frozen copy on purpose (runs
  without this package inside Pyodide).
- **Version 0.1.1** (pyproject + `__version__`): the wheel now ships
  `webgpu.py` + `_vendor/`. The hub (`gen_hub.py`) and the Pyodide page
  (`main.js` reads `wheels/latest.txt`) derive the wheel name; `make
  browser-assets` builds it. Tag `v0.1.1` publishes.
- **CI**: keras-watch prunes the orphan `ci-state` checkout (its first
  commit would have carried main's whole tree), rebase-retries the push,
  reads the tinygrad pin from pyproject; its PR step needs the repo
  setting "Allow GitHub Actions to create and approve pull requests"
  (`gh api` one-liner in the workflow header — owner's to flip).
  referee.sh judges `PYTEST_ARGS` runs as subsets (a `-k` run reported
  unexecuted baseline entries as NOW PASSING) and strips trailing slashes;
  referee.yml passes the dispatch input through env. bump script: usage
  error instead of IndexError; its last test compared a call to itself.
- **Docs**: README (3.15.1 is the version of record; browser limits),
  HANDOFF dates + the dead `sync_vendor --check` paragraph, architecture
  invariant 9 + RNG-state row, device-rng.md revision 2, experiments/
  README's "2-4× faster" line (steady-state single-box numbers, no run of
  record — the bench harness already splits setup / first step / steady
  state, what is missing is a real-GPU run), `run-bench.sh` tf.min.js
  source (was an nnvp path).

## How to run anything

```sh
# the referee (Keras' own layers tree, ~25 min; clones + builds .referee/
# on first use, compares FAILED set to scripts/referee-baseline.txt):
make referee
# ~1 min slice of the same suite (known failures stay green, any other is red):
make referee-quick
# any keras path, e.g. the ops suites:
scripts/referee.sh keras/src/ops/numpy_test.py
# dev loop (uv-based; .venv resolves keras 3.15.1):
make verify     # ruff lint + format + loader tests + backend regression receipts (~45 s)
make smoke tutorial fuzz vendor-check readme-check
make browser-assets   # regenerate every browser artifact (bundles, pages, tf.js) — none are in git
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

## Plugin-backends PoC (2026-08-10) — WORKING

`docs/upstream/keras-plugin-poc.md`. A keras fork (worktree
`/home/dev/workspace/keras-plugin-fork`, branch `plugin-backends`, all
uncommitted) adds `keras/src/backend/plugins.py` + generalizes the six
dispatch `else:` tails to resolve the `keras.backends` entry-point group.
Result: on the fork, plain `KERAS_BACKEND=tinygrad python -c "import
keras"` loads this backend with ZERO patches and no hook (referee
dense_test 70/1/1, identical to stock-path). The pip package now declares
the entry point (inert on stock keras) and the hook stands down when it
detects native plugin support; stock path re-verified green. This branch
is the living demo for the eventual upstream design issue — sequencing in
the PoC doc. NOTE: the zigcc shim now execs via
`/home/dev/workspace/zig-venv` (old ktg-venv is gone).

## Decision queue (owner's, not yours)

- Commits/checkpoints in the clone and this repo.
- Publishing this repo + PyPI name claim.
- tinygrad upstream PR (draft ready in `/home/dev/workspace/tinygrad-upstream/DRAFT_PR.md`;
  their rules require AI-assistance disclosure; `__bool__` half needs an
  issue first — it reverts a deliberate upstream ban). Two more candidates
  found 2026-08-03: Tensor slice bounds in `__getitem__` (blocks
  RandomCrop) and argfix's exact-class tuple/list check (worked around
  backend-side for TrackedList).
- keras upstream — LANDSCAPE CHANGED 2026-08-21 (see
  /home/dev/workspace/KERAS_COMMITS_AND_ORDER_GUIDE.md §0): the keras team
  is building pluggable backends itself (PRs #23397/#23410 + branches;
  official keras-team/keras-mlx and keras-team/keras-openvino plugin
  repos; `keras_<name>` naming convention — ours already matches; master
  landing rewound 2026-08-20, effort continues). The old plan (design
  issue proposing entry points) is obsolete; new plan: engage as an
  independent pilot plugin. Also: test_cross was fixed upstream (#23408
  merged 2026-08-11, incl. the numpy-backend companion) — our PR-1a is
  dead. Original reasoning kept for the record: (Chollet's criteria in
  keras#20793; keras#23193, the closed MLX PR, has been reincarnated as
  the pilot plugin of the official program). READY TO DROP: docs/upstream/keras-pr/ has
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
  WebGPU runner. **PROVEN 2026-08-30** — `experiments/m0-keras-trainstep/`
  (Keras Sequential + SGD + sparse CE exported to a self-contained WebGPU
  runner that trains in headless Chromium; two real bugs found and fixed,
  see its README). The browser-training story and what to reuse from the
  migrated nnvp experiments: `docs/browser-training.md`.

## Related artifacts elsewhere

- `/home/dev/workspace/tinygrad-upstream` — dunders patch + DRAFT_PR.md.
- nnvp scratchpad (session-tied, may be gone): design doc, build story,
  Pyodide harness. The build story was delivered to the owner directly.
- The owner's nnvp project memory has a condensed version of this file.
