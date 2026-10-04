# Handoff log — every pass, in order (2026-08 → 2026-09-28)

> The dated record behind `HANDOFF.md`. Frozen on 2026-09-28 when the
> handoff was rewritten as a short current-state file; nothing here is
> maintained, paths are as of each section's date.

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
   backend sources under `src/keras_tinygrad/src/` ARE the source of
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

## 2026-09-21 — cloud patch series reviewed + integrated, linalg gradients fixed

All uncommitted in the working tree (the owner commits). Source: the
5-patch series `c4ffein-work/playground`, branch
`claude/keras-tinygrad-openvino-check-ay9203`,
`patches/c4ffein__keras-tinygrad/2026-09-21-keras-master-branch-compat/`
(its REPORT.md is the why; remainder 9 below is the summary). Applied whole,
then reviewed locally. What the review changed:

- **`elastic_transform` was broken by the series**: patch 1 dropped
  `image.py`'s `draw_seed` import while reordering the block → NameError on
  first call. Invisible to `make verify` (backend sources are excluded from
  ruff, no local test called the op). Restored; `make lint-check` now also
  runs `ruff --isolated --select F821` over `src/keras_tinygrad/src`
  (undefined names only — formatting still never touches those files);
  receipt `test_elastic_transform_runs`.
- **`gammainc` gradients were NaN at the domain edges** (x = 0, x = inf),
  and through a broadcast `a` the NaN reached every element: the edges were
  patched in with `where` AFTER the expansions had seen them, and the
  broadcast was `a + x * 0.0` (inf * 0). Now: static-shape `expand`, edge
  elements run the expansions on an interior point (gradient exactly 0
  there), and `gammainc(0, 0)` is nan like scipy. Receipt:
  `gammainc_grad_edges_finite` in `tests/test_ops_beyond_pin.py`.
- Small: `MissingOpError` moved above the `SUPPORTS_*` flags it had split;
  a dead `rnn` module import in `src/__init__.py`; NOTICE's path.
- **Pre-existing, found through the series' REPORT aside — linalg gradients
  were silently ZERO** for inv / det / solve / lu_factor / cholesky / eigh /
  svd / lstsq / pinv (qr, norm, solve_triangular were fine). Mechanism, with
  minimal repros in linalg's module docstring: tinygrad 0.13 swaps a
  realized tensor's graph for its buffer and `Tensor.gradient` returns zeros
  upstream, no error; `.item()` does the same to every `.contiguous()` node
  upstream of what it reads. The loops called `.realize()` per step and the
  validity checks read the returned factors. Fix = architecture invariant
  12 (new): `_cut` (`contiguous().contiguous_backward()`) in the loops, host
  checks on a side graph of the detached input, none under the trainer's
  tape (`core.in_custom_gradient_tape`; `eig` raises there). The ops
  now match finite differences of the numpy reference
  (`test_linalg_gradients_are_not_silently_zero`).
- **The first version of that fix stalled the box** (load average 174, the
  owner killed the process from the host): fully lazy Jacobi is superlinear
  in rotation rounds and keras' `test_svd` (4, 30, 20) is 190 of them. Now
  `linalg._JACOBI_LAZY_ROUNDS = 110`: below, lazy + differentiable; above,
  realize per round as before (no gradient; RAISES inside a train step).
  Measurements and the not-done real fix (closed-form gradient
  re-attachment): architecture.md "known leaks". Lesson kept in code, at
  the owner's suggestion: every test subprocess runs under
  `tests/_limits.py` (4 GB address space then, 8 GB since 2026-09-22 — see its docstring; 1800 CPU-s, nice; receipts in
  `tests/test_limits.py`), and `scripts/referee.sh` caps its run
  (`REFEREE_MEM_MB`=8192, per-test `REFEREE_TEST_TIMEOUT`=900 via
  pytest-timeout, nice). Re-run of the stalling test under a 6 GB cap
  before the fix: MemoryError in 75 s, load 2. `make referee` now calls
  `bash scripts/referee.sh` (the file is mode 644 in git).
- NOT fixed, same mechanism, needs an audit: every other mid-forward host
  read — `core.cond` / `while_loop` predicates (`.numpy().item()`), the
  index/argument reads in `ops/numpy.py` (13 `.item()` sites). They only
  bite when the value read sits downstream of a `.contiguous()` node that
  is also upstream of the loss (nn.py has 3 such nodes, numpy.py 2, core
  5), and the jitted train step already refuses host reads loudly — the
  exposure is the EAGER fallback path. `make fuzz-grad` could not have
  caught the linalg zeros: `tools/parity_fuzz.py` has no linalg case at all.
- **Version 0.2.0** (pyproject, `__version__`, uv.lock — owner's call,
  2026-09-21): `keras.src.backend.tinygrad`, importable on the PyPI
  0.1.0 / 0.1.1 wheels, no longer exists; the backend is
  `keras_tinygrad.src`. Not tagged, not published.
- **The referee had been running on the wrong tinygrad**: referee.sh took
  the first `"tinygrad…"` string in pyproject — the KEYWORD — as the pin,
  so `.referee/venv-*` got the latest release (0.14.0 since 2026-08-30;
  the 2026-09-01 tally of record was taken on it). On 0.14 the ops/numpy
  suite has 49 extra failures (`Cannot map tinygrad dtype dtypes.weakint /
  weakfloat to Keras` — python-scalar dtypes, a real item for the day the
  pin moves) plus a zig cc compile error in test_power. Fixed: the grep
  wants a version operator, an existing venv is re-pinned on every run,
  and the preflight prints the tinygrad version.
- Owner call left open: whether the gammainc gradient receipt (~1 min of
  first-use compile) belongs in `make verify`.

Receipts (2026-09-21, final working tree, this box with `CC=zigcc`; every
referee number below is on the PINNED tinygrad 0.13.0 — runs earlier the
same day were on the accidental 0.14.0 and are void):

- `make verify`: ruff + the F821 pass over the backend sources + format
  clean, **53 passed** (2:41). `make smoke`, `make tutorial`,
  `make vendor-check`, `make readme-check`: green.
- `bash scripts/referee.sh keras/src/ops/numpy_test.py`: **3 failed /
  5,444 passed / 708 skipped** — the three known (unique, vectorize,
  test_cross). The script prints them as NEW because
  `scripts/referee-baseline.txt` only covers the layers tree.
- `… keras/src/ops/{linalg,math,image}_test.py`: **695 passed / 17
  skipped**, the same tally as the unmodified `cf75be8` tree (that
  comparison run was on 0.14.0); includes the `test_svd` that stalled the
  box.
- Known wart, also on the unmodified tree: a process that imported
  tensorflow (ops/image_test, the preprocessing tree) aborts AT EXIT with
  tinygrad 0.13 — `free(): invalid pointer`, exit 134, after pytest's
  complete summary. referee.sh tolerates exactly that shape with a loud
  WARNING (any abort without a final summary is still a crash) and sets
  `ulimit -c 0` (each abort left a ~1 GB core file in `.referee/`).
- Layers tree, the tally of record (`bash scripts/referee.sh`, 0:25:36):
  **5 failed / 1,988 passed / 215 skipped / 1 xpassed** — "OK — failed
  set == baseline (5 known)", identical to the 2026-09-01 tally (which was
  taken on tinygrad 0.14.0; this one is the first on the pinned 0.13.0).
  The exit-time abort above fired after the summary, as expected.

## 2026-09-21 (later) — tinygrad 0.13 → 0.14, owner's decision

Uncommitted, on top of `be50656` (which holds everything in the section
above). Pin: `tinygrad>=0.14,<0.15`; `.venv`, uv.lock and the referee venv
follow. What the move broke, smallest blast radius last:

- **ops/numpy dtype leaks (49 referee failures)**: a python scalar is typed
  `weakint` / `weakfloat` on 0.14 and `int8_tensor + 1.0` comes back typed
  `weakfloat` — a dtype that exists only in tinygrad's promotion lattice.
  `_pair` and `clip` had left that promotion to tinygrad; they now cast the
  tensor operand to keras' own `result_type` first (right on any version).
- **int tensor ** int tensor**: tinygrad has no integer pow (`# TODO: int
  pow` in both versions); 0.14 renders the float route as C no compiler
  accepts (keras' `test_power`). `numpy._int_power`: exact
  square-and-multiply, numpy's wrap-around, no float — also exact past
  2**24, where the old route was not.
- **The vendored exporter**: upstream's v0.13.0..v0.14.0 diff of
  `extra/export_model.py` (eight hunks: PARAM slots, `is_bound_var`,
  `prg.src[2]`, `NUM_CPU_THREADS`, AddrSpace-based symbolic vars) applied to
  `_vendor/export_model.py`. Python-side tests green; the exported bundle
  RUNNING in a browser is NOT re-verified (`make browser-assets` + the m0
  experiment is the check — owner's box).
- **The one that matters — the jitted train step silently froze state.**
  On 0.14 `.contiguous().realize()` no longer turns a CONST into a buffer,
  so a Variable assigned from a python scalar had no buffer to pin and was
  baked into the capture: `Mean.reset_state()` assigns 0 → the loss
  tracker's `count` froze at the first replay and `fit` reported a loss of
  exactly 0.0 from epoch 2; Adam's `iteration` (initialized from 0) froze →
  stale bias correction, a different trajectory. Weights kept training, so
  NOTHING failed: `make smoke` printed SMOKE OK with losses
  `[13.2, 0.0, 0.0, 0.0, 0.0]` (its only assert is last < first — worth
  tightening, owner's call), and the referee had been running on 0.14 by
  accident since 2026-08-30 without a red test. Fix: `core._concrete`
  (architecture invariant 5). Receipt:
  `test_jitted_train_step_keeps_every_variable_in_step_with_eager` — all
  model / optimizer / metric variables, JIT vs eager. How it was found, for
  the next bump: diff every variable between `KERAS_TINYGRAD_TRAINER_JIT=1`
  and `=0` after a 2-epoch Adam fit. Do that FIRST.
- Unchanged on 0.14 (re-probed): realize / host reads zero the gradients
  upstream (invariant 12), `JitError` on a host read at capture. Gone on
  0.14: the tensorflow exit-time abort (referee.sh still tolerates it).
  New on 0.14: the CPU device runs a worker pool — load average ~20 during
  a referee run where 0.13 showed ~2; pin with `taskset`.

Receipts (all on tinygrad 0.14.0, final tree): `make verify` **54 passed**;
`make smoke` (losses 5.80 → 0.64), `make tutorial`, `make vendor-check`,
`make readme-check` green; ops suites numpy + linalg + math + image
**3 failed / 6,139 passed / 725 skipped** — the three known (unique,
vectorize, test_cross), the same totals as on 0.13; layers tree, the tally
of record, **5 failed / 1,988 passed / 215 skipped / 1 xpassed**, "OK —
failed set == baseline (5 known)" (0:36:49 on 5 pinned cores).

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
make verify     # ruff lint + format + loader tests + backend regression receipts (~3 min since 2026-09-21)
make smoke tutorial fuzz vendor-check readme-check
make browser-assets   # regenerate every generated browser artifact (bundles, pages, tf.js) — outputs stay out of git
make e2e        # browser end-to-end: headless Chromium + SwiftShader drives the hub's proof flows
```

`CC=zigcc` matters: this box has no clang; the shim at
`/home/dev/.local/bin/zigcc` makes tinygrad's CPU jit compile via the
ziglang wheel (translated target triple, `-g0`). Real CI uses real clang.

## 2026-09-22 — browser story promoted out of experiments/, e2e suite born

The two directories the Makefile/README already depended on stopped
pretending to be experiments: `experiments/m0-keras-trainstep/` →
`js/demo/` (the demo hub app: generators, `check.sh`, `demo-server.mjs`)
and `experiments/pyodide-keras/` → `js/src/trace/` (the in-tab
engine). `.gitignore` narrowed from `experiments/**` to generated
outputs only — the SOURCES of the browser story are now tracked (before
this they existed only on the dev box and the owner's mirror);
`experiments/` keeps the four archaeology spikes, also tracked.

Born with the move:

- **`tests/test_webgpu_export.py` + `make e2e` + a CI job**: headless Chromium
  (SwiftShader WebGPU) drives the hub through the four proof flows and
  asserts the page's own checks JSON — train-and-tfjs-equivalence, the
  dense probe (negative control), the dropout fresh-masks probe, and the
  pyodide in-tab-trace-vs-bundle identity (network-gated:
  `KERAS_TINYGRAD_E2E_PYODIDE=1`; CI sets it).
- **The hub consumes the npm package**: `gen_hub.py` embeds
  `js/src/index.mjs` and routes ALL bundle loading (`loadBundle`) and
  batch streams (`lcg32`) through it — every hub run, including e2e, now
  exercises the exact code `npm install keras-tinygrad` ships.
- **`js/test/run.mjs`** (`bun test/run.mjs`, framework-free): export
  surface, LCG bit-parity with the Python twin, `loadBundle`'s
  input contract. The npm-publish verify job runs it; a CI `js-tests`
  job runs it on every push.
- js/demo/ is ruff-linted like the rest of the repo (E501 excepted —
  embedded HTML/JS templates); `demo-server.mjs` honors `PORT=`.

## 2026-09-23 — tracer folded into the npm package; one build/ dir

- **`keras-tinygrad/trace` is real**: the Pyodide engine moved from
  `js/src/trace/` into the package — `js/src/trace/` holds
  `client.mjs` (`createTraceEngine` + `traceModel`, which composes with
  the runner's `loadBundle`), `worker.js`, `driver.py`, and `shims/`
  (pure-Python optree/ml_dtypes; NEITHER is in Pyodide 0.28.3's package
  index — checked 2026-09-23, 343 packages). The hub's pyodide engine
  imports `client.mjs` from the served package sources (`/js/` mapping),
  so the e2e suite asserts the shipped tracer, not a copy. The wheel URL
  stays a parameter (`versions.wheel`); tarball-shipping the wheel is an
  open owner decision.
- **Everything generated lands in `js/demo/build/`** (bundles, pages,
  tf.min.js, wheels/, run_local output) — one gitignore line, and
  `rm -rf js/demo/build && make browser-assets` is the clean rebuild.
  Historical URLs survive: `demo-server.mjs` resolves build/ first.
- **A ruff-costs-correctness lesson**: `ruff --fix`'s isort hoisted
  `from keras_tinygrad.src.ops.core import ...` above `import keras`
  (the dedicated keras_tinygrad section sorts first), and
  `keras_tinygrad.src.__init__` imports keras itself → circular partial
  init → the backend star-import came up empty (NameError `name_scope`).
  Guard: `# isort: split` after `import keras` in m0.py/gen_demo.py with
  a comment naming the hazard.
- `js/src/trace/` now holds only the standalone arbitrary-config page
  (`index.html`/`main.js`/`server.mjs`) + `run_local.py`; the page is
  NOT fully covered by the hub (the hub has two fixed models, the page
  traces arbitrary units/activation/rate/lr/batch configs) — superseded
  same-day, next section.

## 2026-09-23 (later) — browser/ dissolved; final layout

Owner's call: "browser has to disappear... either js or tests or
/dev/null, your call." Final shape:

- **`js/demo/`** — the demo/proof rig for the JS version (excluded from
  the npm tarball, which remains exactly `js/src/**`): `gen_hub.py`,
  `m0.py`, `export_dropout_probe.py` (page mode removed), `server.mjs`
  (PORT-aware; `/` = `build/hub.html`, `/js/*` = package sources,
  `/build/*` = generated), `gen_ref_curve.py` (extracted from the deleted
  gen_demo.py: same exported init + same lcg32 stream through LOCAL
  Python keras-tinygrad on CPU -> `build/ref_curve.json`), and
  **`build/`** — the ONE output dir for everything generated.
- **`tests/test_webgpu_export.py`** — now FIVE tests: train+tfjs verdict, dense probe,
  dropout probe, **browser-vs-local-Python seal** (steps 0-1 < 1e-3
  against ref_curve.json + both tails converge), pyodide identity
  (network-gated).
- **The hub gained a "custom" model** (JSON config textarea, `&config=`
  URL param): any layers/dims/optimizer, pyodide engine only — this
  absorbed the standalone pyodide page's unique ability, so that page
  (old browser/pyodide's `index.html`/`main.js`/`server.mjs`), the
  `check.sh` probe pair, `gen_demo.py`'s demo.html and the dropout probe
  page are all DELETED (never committed — no git history holds them; the
  pyodide page's measurement record survives as
  `docs/history/pyodide-keras-record-2026-08-30.md`). Probe semantics for custom:
  distinct losses expected iff the config has a dropout layer.
- `run_local.py` -> `js/test/run_local.py` (manual harness: driver +
  shims natively, no browser).

## 2026-09-28 — "anything missing?" pass (Fable 5.1, owner-approved list)

Uncommitted, on top of `4742f0e`. The owner asked for a gap review, then
approved every item; three forks ran the independent tracks in parallel.
State of the world found first: PyPI holds **0.1.0 only** (2026-08-27),
the remote has the `v0.1.0` tag only — 0.2.0 was never tagged; **npm has
no `keras-tinygrad`** (404, no `js-v*` tag); `gh` auth on this box is
dead (401), so CI status could not be read.

- **Host reads gave WRONG gradients, not zero** (`tests/test_host_reads.py`,
  `core.host_read`): the 09-21 audit item. 22 sites in ops/numpy.py and
  ops/core.py read a data-dependent value (`cond`/`while_loop` predicates,
  `nonzero`/`bincount` counts, tensor-valued shape arguments of `split`/
  `repeat`/`roll`/`pad`/`full`/`arange`/`eye`/`quantile`, the `__bool__`/
  `__float__`/`__int__`/`__array__` dunders, `_index_int`). Probed on
  0.14 through a shared `log1p` contiguous node: 22/22 wrong before
  (the factor that bypassed the materialized node survived, so the
  gradient was non-zero and wrong), 22/22 correct after. `x.detach()`
  before the read does NOT help (same UOp objects). The fix snapshots
  every live tensor's `uop` (tinygrad's `all_tensors` registry), reads,
  restores whatever got swapped for a buffer — value unchanged,
  intermediates recomputed once later. The test's `mechanism` entry
  asserts a raw `.item()` still zeroes gradients on the pinned tinygrad;
  when that stops, `host_read` retires. Not affected, verified: `clip`,
  `trapezoid`, `fori_loop`, the numpy-scalar branches; `convert_to_numpy`
  stays the terminal read by design. Invariant 12 reworded
  (architecture.md). Referee slice: `core_test.py + numpy_test.py` **3
  failed / 5,609 passed / 717 skipped** — the three known.
- **Parity fuzz has linalg now** (tools/): 12 ops (inv, det, solve,
  solve_triangular, lu_factor, cholesky, eigh, svd, qr, lstsq, pinv, norm)
  in both op and grad kinds; conditioning by construction, sign-invariant
  `abs` post for eigh/svd/qr, grad loss over float outputs only.
  Would-have-caught-it receipt: a scratchpad child wrapping every linalg
  output in `stop_gradient` → 24/24 FAIL. `make fuzz` 100/100, `make
  fuzz-grad` 60/60 — note those counts now spread over 35 / 21 ops;
  raising `--cases` in the Makefile is the owner's call.
- **Examples run in CI** (`make examples`, ~3.5 min: autoencoder,
  char_rnn, convnet_mnist synthetic, quantized_inference at `--epochs 1`),
  every script ends in a real assertion (char_rnn: loss below chance and
  in-vocab samples; convnet: finite losses, softmax rows sum to 1).
- **Receipts for the 2026-09-22 post-push fixes** (`tests/test_export_vendor.py`):
  the exporter's kernel-name collision fix (m0 trace: raw names collide,
  emitted names unique, no body dropped, `_ktgN` fired — the precondition
  is asserted so a reverted fix cannot pass), `driver.py`'s `PARALLEL=0`
  before the tinygrad import (static AST check; Pyodide can't run in CI),
  `gen_hub.py` baking the tinygrad version from the environment.
- **`make smoke` is a real check**: no loss may be exactly 0.0 (the
  09-21 frozen-tracker shape), all finite, last < half of first, and
  `evaluate` must agree with the reported curve.
- **e2e caps**: the bun server runs under `child_limits`; Chromium under
  the CPU cap + nice only — it dies with SIGTRAP under ANY `RLIMIT_AS`
  (measured at 8/16/32 GB; `child_limits(address_space=False)`).
  `make e2e`: 4 passed / 1 skipped (pyodide, network-gated) with the caps.
- **referee.yml** invokes `bash scripts/referee.sh` (the file was mode
  644 in git — every scheduled run since would have failed on the exec
  bit; the working tree also flips referee.sh and fetch_tfjs.sh to 755).
- **CHANGELOG.md** born (0.1.0 → unreleased 0.2.0 → js 0.0.1); README
  points at it. Version stays **0.2.0**: nothing after 0.1.0 was ever
  released, so there is nothing to bump past.
- **npm readiness**: in-repo everything is ready (workflow now also runs
  `npm pack --dry-run` + `npm test` under node, the tool the publish job
  uses; package.json description no longer calls the tracer a roadmap
  item); docs/npm-publishing.md has a "Status" block. What is left is
  account-side (steps 1–8 there).
- **Doc drift fixed**: README's dead `check.sh`, browser-training.md's
  `check.sh` + tinygrad 0.13, the `test_webgpu_export.py ` typo (×6), this
  file's date/cap/decision queue, the FABLE analysis marked as a 2026-08-03
  snapshot.

Receipts (final tree, `CC=zigcc`, this box): `make verify` **60 passed**
(ruff + F821 + format clean); `make smoke`, `make readme-check` green;
`make e2e` 4 passed / 1 skipped; `make fuzz` 100/100, `make fuzz-grad`
60/60; `make examples` 4/4; referee slice above. The full layers referee
was NOT rerun this pass (nothing under keras/src/layers is touched by
`host_read` beyond what core/numpy_test cover — owner's call to rerun
before tagging 0.2.0).

## The rules that shaped the code (do not regress them)

Full list: `docs/architecture.md` (12 invariants). The load-bearing ones:
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
4. Dev tooling landed 2026-08-03: uv + ruff (line-length 120, the backend
   sources
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
9. Keras 3.16 / `pluggable_backend` drift (2026-09-21, static check —
   `docs/history/upstream/keras-master-and-branch-status-2026-09-21.md`): master
   moved every backend's ops into an `ops/` subpackage (`backend.ops.numpy.x`)
   and rewrote the `DynamicBackend` anchor — the package now HAS that shape
   (`keras_tinygrad/src/ops/`, both spellings exported, the loader's
   patches import `keras_tinygrad.src` directly, no alias), so only the
   anchor is left for 3.16; the branch resolves plugins from a hard-coded frozenset
   (`tinygrad` not in it) and OpenVINO now lives out of tree in
   `keras-team/keras-openvino`, the layout to mirror. Landed alongside:
   the master-only ops (`copysign`, `float_power`, `cov`, `lgamma`,
   `gammainc`; `tests/test_ops_beyond_pin.py`), `core.MissingOpError`
   (master's `hasattr` probes on the backend), the shim's `BackendLayer`
   name and the removed `src/export.py` (branch protocol).

## Plugin-backends PoC (2026-08-10) — WORKING

`docs/history/upstream/keras-plugin-poc-2026-08-10.md`. A keras fork (worktree
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

- ~~Commits/checkpoints in the clone~~ (clone gone) — commits in this repo
  stay the owner's; ~~publishing + PyPI name claim~~ done: 0.1.0 on PyPI
  since 2026-08-28.
- **Release 0.2.0** (`v0.2.0` tag → publish.yml): the working tree has been
  0.2.0 since 2026-09-21 but the tag was never pushed (remote tags:
  `v0.1.0` only; the planned `v0.1.1` never happened either). CHANGELOG.md
  holds the notes. Breaking for 0.1.0 users: `keras.src.backend.tinygrad`
  is gone (the backend is `keras_tinygrad.src`).
- **First npm publish** (`js-v0.0.1`): `npm view keras-tinygrad` is a 404
  as of 2026-09-28; the in-repo side is ready (docs/npm-publishing.md
  "Status"), the remaining steps are account-side and need a browser.
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
  the pilot plugin of the official program). READY TO DROP: docs/history/upstream/keras-pr/ has
  tests-fix.patch (git-apply-clean, 4 files) + PR_BODY.md for the
  test-side bundle (test_cross numpy>=2.5 guard, float8 skipif,
  trainer_test fallback; ViewAsComplex skip deliberately dropped — we pass
  those now). Full analysis: docs/history/upstream/keras-draft.md — including a
  verified companion fix for the numpy BACKEND's own broken cross
  (numpy.py:554, red under numpy>=2.5), excluded from the test-side patch,
  owner decides whether to bundle it. tinygrad additions:
  docs/history/upstream/tinygrad-draft.md (slice bounds path traced to
  tensor.py:878/movement.py:87, argfix isinstance one-liner).
- NNVP integration M0: trace a Keras TrainStep on tinygrad NULL device →
  WebGPU runner. **PROVEN 2026-08-30** — `js/demo/`
  (Keras Sequential + SGD + sparse CE exported to a self-contained WebGPU
  runner that trains in headless Chromium; two real bugs found and fixed,
  see its README). The browser-training story and what to reuse from the
  migrated nnvp experiments: `docs/browser-training.md`.

## Related artifacts elsewhere

- `/home/dev/workspace/tinygrad-upstream` — dunders patch + DRAFT_PR.md.
- nnvp scratchpad (session-tied, may be gone): design doc, build story,
  Pyodide harness. The build story was delivered to the owner directly.
- The owner's nnvp project memory has a condensed version of this file.
