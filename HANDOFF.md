# HANDOFF — current state of keras-tinygrad (2026-09-28)

The short version for whoever picks this up next, human or agent. Numbers
are the tallies of record; re-verify with the commands given rather than
trusting them. The full dated log of how every piece came to be — every
review pass, every bug and its receipt — is `docs/history/handoff-log.md`.

## What this is

A tinygrad backend for stock Keras 3, shipped as a pip package. The backend
sources are `src/keras_tinygrad/src/` (pluggable-backend layout:
`ops/{core,image,linalg,math,nn,numpy}` + `random`, `rnn`, `trainer`,
`layer`, `export`). A meta-path import hook grafts them onto the PyPI keras
wheel with six match-exactly-once source patches (`docs/how-it-works.md`);
on keras' pluggable-backend branch the hook stands down and the package
loads as a plain plugin. Keras' own test suite is the referee.

Two more halves: `keras_tinygrad.webgpu` exports a Keras train step as a
self-contained WebGPU bundle, and `js/` is the npm package that runs such
bundles in a browser (`keras-tinygrad`) and traces models in-tab under
Pyodide (`keras-tinygrad/trace`). `js/demo/` is the proof rig for both.

## Pins and versions

| what | value |
|---|---|
| keras | `>=3.15,<3.16` — 3.15.1 is the version of record |
| tinygrad | `>=0.14,<0.15` (moved from 0.13 on 2026-09-21) |
| python | `>=3.11` |
| package version | 0.2.0 in pyproject, **untagged** — PyPI holds 0.1.0 only |
| npm package | 0.0.1, **never published** |

## Tallies of record (tinygrad 0.14.0, keras v3.15.1 tag, 2026-09-21)

- Keras' layers tree, preprocessing included (`make referee`): **5 failed /
  1,988 passed / 215 skipped / 1 xpassed** — the failed set equals
  `scripts/referee-baseline.txt` (2× upstream float8, 2× RandomCrop, 1×
  AutoContrast; reasons in README "Status").
- ops suites numpy + linalg + math + image: **3 failed / 6,139 passed /
  725 skipped** — unique, vectorize (data-dependent shapes, decision
  pending: `docs/unique-vectorize.md`), test_cross (fixed upstream).
- `make verify` 60 passed; `make e2e` 4 passed / 1 skipped (Pyodide flow
  is network-gated); `make fuzz` 100/100, `make fuzz-grad` 60/60; `make
  examples` 4/4 (all 2026-09-28).

## How to run anything

```sh
make verify          # lint + format + undefined-names pass + fast tests (~3 min)
make smoke tutorial examples fuzz fuzz-grad vendor-check readme-check
make referee         # Keras' layers tree (~30 min); clones the pinned tag into .referee/
make referee-quick   # ~1 min slice: backend/optimizer/core ops + Dense
bash scripts/referee.sh keras/src/ops/numpy_test.py   # any keras path
make browser-assets  # every generated browser artifact -> js/demo/build/ (gitignored)
make e2e             # headless Chromium drives the hub through five asserted flows
make referee-cl      # the quick slice on tinygrad's OpenCL device, no GPU needed (PoCL from PyPI)
make bench           # CPU benchmark vs the tensorflow/jax/torch backends (~8 min; bench/results/)
uv run python scripts/run_keras_guide.py <path/to/keras-io/guides/x.py>   # one keras.io guide, capped
cd js && bun test/run.mjs                              # the npm package's own tests
```

On this box `CC=zigcc` (no clang; the shim execs `zig cc`). Real CI uses
clang. Every test subprocess and the referee run under resource caps
(`tests/_limits.py`): an uncapped runaway graph pinned this box once.

## The rules (do not regress them)

`docs/architecture.md` holds the twelve invariants; update it in the same
diff that moves a boundary. The load-bearing ones: the numpy backend is the
semantic reference; Keras' own tests are the referee, a red test stays red
(never skipped, never worked around); no silent numpy fallbacks in a
differentiable path, loud `NotImplementedError` over a wrong answer;
copy-on-convert; monkeypatches additive and guarded only; every keras-core
touchpoint has a loader anchor; a differentiable op never realizes on the
path to what it returns (host reads go through `core.host_read`).

The owner reviews and commits every diff himself. Never commit or push.

## Open items

Technical, smallest first:

1. `ops.unique` / `ops.vectorize` — loud stubs; data-dependent output
   shapes need a design decision (`docs/unique-vectorize.md`).
2. Fused RNN kernels — the generic scan is correct and slow.
3. Jacobi eigh/svd above `linalg._JACOBI_LAZY_ROUNDS` carry no gradient
   (raise inside a train step); the real fix is closed-form re-attachment.
4. test/predict paths are eager per step; only the train step is jitted.
5. **CPU benchmark of record exists** (`make bench`, `bench/README.md`,
   `bench/results/2026-09-28-cpu-3cd22935c208.md`): MLP / small CNN /
   LSTM across tinygrad, tensorflow, jax, torch, same 4 pinned cores, same
   init, same data, one capped process each; per-step losses agree across
   all four backends to ~1e-7 for the first 10 steps (the like-with-like
   proof). Steady train step, this box: MLP 0.048 s vs 0.004-0.007 s;
   CNN 0.48 s vs 0.02 s; LSTM 1.16 s vs 0.006-0.018 s; tinygrad's first
   step is compile-bound (118 s for the LSTM). Read as facts about this
   run on this box. No real-GPU run of record for anything; the
   browser bench is SwiftShader-only. This box has no GPU and no root.
   Two no-hardware paths exist since 2026-09-28: `make referee-cl`
   (`scripts/with_cl.sh`: PoCL from PyPI behind the system ICD loader →
   tinygrad's OpenCL renderer on the CPU; referee-quick slice **244
   passed / 1 failed (the known float8) / 18 skipped** in 3:06, same
   failed set as CPU) and `DEV=PYTHON::sm_80|METAL|gfx1100` (tinygrad's
   UOp interpreter with that GPU's renderer rules; a 4-step MLP fit takes
   10-20 s, so spot checks only). Neither is a GPU: memory model, driver
   and timing stay unverified until the owner's Metal box or a cloud
   runner runs the referee.
   The CL path already paid for itself: `model.evaluate` was
   intermittently WRONG there (224.0 or 0.0 instead of the predict-MSE,
   ~1 run in 3). Root-caused 2026-10-02, OURS: the backend declared
   `IS_THREAD_SAFE = True` (copied from the numpy backend; tensorflow says
   False), so keras' `CallbackList` dispatched batch-end hooks to a
   ThreadPoolExecutor and the per-batch metric read ran on a worker thread
   while the main thread launched the next batch's kernels — tinygrad sets
   kernel arguments in place on one `cl_kernel` per program and its LRU
   allocator has no lock, so the worker's launch ran with the other
   thread's arguments and the copyout returned a stale buffer (7×32 =
   224). The CPU device's launch path is synchronous, which is the only
   reason it never misfired there. Fix: `IS_THREAD_SAFE = False` — every
   host read stays on the thread that drives the device. Receipts:
   `tests/test_device_cl.py` (thread-affinity spy, runs everywhere; a
   40-evaluate agreement loop under `KERAS_TINYGRAD_TEST_CL=1` +
   `scripts/with_cl.sh`, the one skip the suite allows, reason stated),
   probe 10/10 after vs ~1 in 3 wrong before, `make referee-quick` same
   tally as before. This is exactly the class of bug a real GPU would have
   surfaced in production; the emulated device found it first.
6. ~~`tensor + Variable` raises~~ FIXED 2026-09-28: tinygrad's binary
   dunders answer `NotImplemented` for a KerasVariable operand so python
   reaches the Variable's reflected op (ops/core.py, next to the DType
   patch; receipts `tests/test_variable_interop.py`: 17 operators, a
   layer written `inputs * self.w + self.b` trains bit-for-bit like its
   keras.ops twin, junk operands keep tinygrad's own error; `abs(Variable)`
   too — tinygrad had `abs()` but no `__abs__`, keras' own
   `variables_test.test__abs__` was red). Referee: `VariableOpsCorrectnessTest`
   37/37; the slice's other 114 failures are complex64/128 dtypes (declared
   unsupported), the torch-dtype test and the known float8. The keras.io
   guide that found it now gets past the backend and fails on a
   keras-io-vs-3.15.1 issue that the TENSORFLOW backend reproduces
   identically (its `load_own_variables` store key resolves to the
   kernel) — upstream script, not ours.
7. ~~LSTM is unusably slow on CPU~~ FIXED 2026-09-28, three stacked causes
   (`tests/test_rnn_perf.py`, referee `keras/src/layers/rnn` 76 passed / 3
   skipped): (a) the JIT was silently OFF for every recurrent model — the
   cells' `DropoutRNNCell` mix-in owns a SeedGenerator, the gate saw it and
   fell back without a warning; the cells now join `_device_rng_covers`;
   (b) the scan was one lazy graph over all timesteps and tinygrad's
   rewriter is superlinear in it (one steady step: 451 kernels, 90 s of 86
   in `unified_rewrite`, 0.35 s executing) — `rnn._cut` per timestep;
   (c) found by the cuts, PRE-EXISTING and silent-wrong: `train_step`
   updated the loss tracker (a realize) BEFORE `compute_gradients`, which
   zeroed gradients upstream of any forward barrier — rnn cuts, `log1p`,
   linalg loops. Gradient graph is built first now (receipt
   `test_train_step_gradients_survive_forward_barriers`). LSTM(32), batch
   32, `train_on_batch`, first / eager steady / JIT replay: T=10 85 s /
   6-7 s / **1.0 s** (was 35 s), T=25 106 / 9-10 / **1.5 s**, T=50 144 /
   35 / **6.9 s** (was >500 s). The first step is still ~300 kernels
   through zig cc. Gradients match the tensorflow backend to ~1e-7.
7b. **`train_on_batch` reported a running mean** (found chasing an "Adam
   diverges from tensorflow after step 1" that was nothing of the kind:
   Adam's arithmetic on fixed gradients is identical to numpy and
   tensorflow to the last digit; our reported loss at step k was the mean
   of the per-batch losses 1..k). The numpy trainer this one was ported
   from never resets metrics in `*_on_batch`; tensorflow/jax/torch do.
   Fixed, receipt `tests/test_on_batch_semantics.py`; keras' own
   `trainer_test.py -k on_batch` 6/6 with the drafted test-side patch
   applied to a scratch referee tree (the module refuses unknown backends
   at import — the upstream item in `docs/history/upstream/keras-pr/`).
8. keras.io guides as a second referee: `scripts/run_keras_guide.py`
   (capped runner, `--no-plot-model` stands in for graphviz) ran 8
   canonical backend-agnostic guides — 6 pass (5 in ~1 min total;
   `functional_api` in 12 min on 2026-10-02 with the real 170 MB CIFAR
   download and its Embedding+LSTM(128) section, after the LSTM fix —
   it timed out at 15 min before), 2 fail on keras-io-vs-3.15.1 script
   issues the tensorflow/numpy backends reproduce. A `scripts/guides.sh` +
   `guides-baseline.txt` in the referee's shape is the proposal; 23 of
   the 35 guides import tensorflow/jax/torch at the top and would need
   the referee venv.
9. **Browser runner conveniences the July spikes had and `js/` does not**
   (audit 2026-10-02, `docs/history/experiments-2026-07/README.md`): weight
   readback/write-in on a live runner (`COPY_SRC|COPY_DST` + a weightBufs
   map), the optimized-I/O runner (`queue.writeBuffer` uploads, optional
   loss readback — 3-30x on Firefox's 100 ms fence tick), a forward-only
   export, a fake-WebGPU plumbing test of the emitted runner, a JS
   safetensors writer, conv nets through the Keras browser path. None is
   a backend feature; the first two are the ones worth having back.
10. Upstream tinygrad candidates: Tensor slice bounds in `__getitem__`
   (blocks RandomCrop), argfix's exact-class check, the dunders patch
   (`docs/history/upstream/tinygrad-draft.md`).

Owner's decisions pending:

- Tag `v0.2.0` (rerun the full referee first; `CHANGELOG.md` has the
  notes; it removes the `keras.src.backend.tinygrad` path 0.1.0 exposed).
- First npm publish, `js-v0.0.1` — account-side steps in
  `docs/npm-publishing.md` "Status".
- Engage on keras' pluggable-backends RFC (keras#23523) as the external
  pilot; the history of that thread and the drafts: `docs/history/upstream/`.
- Whether the gammainc gradient receipt (~1 min compile) belongs in
  `make verify`; whether the fuzz case counts should rise now that linalg
  cases share them.
- The repo setting "Allow GitHub Actions to create and approve pull
  requests" for keras-watch's PR step.
