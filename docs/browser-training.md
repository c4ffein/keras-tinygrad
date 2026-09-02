# Browser training — what to steal, and the road from M0 to a demo

For the session that picks up the "you can train in the browser" story for
keras-tinygrad. Written 2026-08-30, the day M0 was proven. **This file is the
single entrypoint** — read it top to bottom, then dig only where it points.

## Orientation (cold start)

You are in `~/workspace/keras-tinygrad`: the pip package `keras-tinygrad`
(0.1.0 live on PyPI), a Keras 3 backend for tinygrad. Repo-wide state and
history: `HANDOFF.md`. House rules: `CLAUDE.md` — in particular, NEVER
commit; the owner reviews and commits every diff himself.

What exists already, in reading order:

1. `experiments/m0-keras-trainstep/README.md` — the proof this doc builds
   on: a Keras train step exported to a WebGPU runner that trains in
   headless Chromium, plus the two bugs it found. Its `m0.py` is the code
   to generalize.
2. `experiments/README.md` — the lineage: four spikes migrated from nnvp
   (2026-07) that already solved tracing, Pyodide, benchmarking, and the
   real-graph bridge for RAW tinygrad steps. The M0 spike married them to
   Keras.
3. The canonical productized runtime lives OUTSIDE this repo, in
   `~/workspace/nnvp/nnvp-client-vue/src/lib/TinygradRuntime/` — steal its
   patterns (listed below), never edit it from here.

Verify before building — all three must pass (run from
`experiments/m0-keras-trainstep/`, python = the repo `.venv` — keras 3.15.1
+ tinygrad 0.13 + editable keras_tinygrad):

```sh
python m0.py export out                  # NULL:WGSL trace -> runner + weights
CC=~/.local/bin/zigcc python m0.py cpu   # numeric proof: loss -> 0, acc 1.00
./check.sh                               # trains in headless Chromium (bun + playwright chromium)
```

## The claim we can now make

**A Keras model trains in the browser, on WebGPU, with no server and no
tensorflow.js.** Pipeline: Keras APIs (Sequential / loss / SGD.apply) run on
the tinygrad backend against the GPU-less `NULL:WGSL` device, one train step
is traced and exported as a self-contained WebGPU JS runner (WGSL kernels,
weights as safetensors under real Keras names — `sequential/dense/kernel`,
`sgd/learning_rate`, momentum slots), and the browser loops `step(x, y)`:
loss 4.04 → 0.001 over 300 steps in headless Chromium on *software* WebGPU
(21.9 ms/step on SwiftShader; the same dense net measured 0.8 ms/step on real
Chrome WebGPU hardware in the pyodide experiment — tfjs-webgl was 7.3).

Perf, stated as measured and nothing more (2026-08-30 correction — the
earlier "2–4× faster than tfjs" line here was not supported): the nnvp
bench REPORT measured tfjs-webgl ~1.45× FASTER than us on the software
stack; the m0 hub on the owner's Intel Gen-9 measured ours 11.1 ms/step vs
tfjs-webgl 13.3 ms/step for the same dense step (tfjs first step 826 ms
shader compile vs ours 23 ms); the pyodide-tinygrad README's table has
per-browser numbers for RAW tinygrad steps. No multiplier headline until a
hardware bench of record exists (planned in nnvp).

## What to steal, per source

### `experiments/m0-keras-trainstep/` (native, the proof)

The load-bearing piece is `KerasTrainStep` in `m0.py` — the shape any future
`keras_tinygrad.export_train_step(model, optimizer, loss)` API should
generalize:

1. **Pinning** (from `_backend/trainer.py`'s `_TrainStepJit`): Keras
   Variables rebind `_value` on every assign; inside the traced call, copy
   each new value back into the buffer the capture read (in-place
   `Tensor.assign`) and repoint `_value`. That turns Keras' functional
   update into the stable-buffer in-place form an exported graph needs.
2. **Realize loss + grads BEFORE `optimizer.apply`** builds any update
   graph. Realizing them together with the update assigns let the scheduler
   recompute the loss from moved weights (measured: 2.927 → 0.378 at step
   one). With no update graph in existence yet, the wrong order is
   unbuildable. This is the Keras-side analogue of driver.py's
   "realize the loss WITH the update" comment — the raw-tinygrad protection
   does not transfer.
3. **No realized const chains.** Anything computed outside the captured call
   is fake-executed to zeros on NULL *and* dropped by the WebGPU emitter
   (`bufs_to_save` is honored only for `get_state_dict` names). Keras' stock
   loss reduction (`sum_over_batch_size`) hits this with its 1/batch scalar;
   `reduction=None` + tinygrad `.mean()` folds it as a kernel immediate.
   `validate_runner_js` is the general fail-loud guard: no pass may read a
   never-written empty buffer. A real export API should either run this
   guard or teach the emitter to carry `bufs_to_save` values (which only
   helps on devices that really execute — on NULL the values are already
   gone; the guard is the honest answer there).
4. **Int state rides along as i32.** (2026-08-30 correction: it was
   believed `optimizer.iterations`' unrealized assign dropped out of the
   trace — it never did; every exported runner carried its `+1` kernel
   reading an uninitialized buffer, benign only because fixed-lr SGD never
   reads it.) `iterations` is int32, so it is pinned like the floats and
   exported as I32 state; the counter counts, which is the prerequisite for
   lr schedules through the export.
5. **`get_state_dict` must never walk Keras objects** (parent references
   cycle): hand `export_model` a bare function whose `__dict__` holds only
   the pinned buffers under `variable.path` names.

### nnvp `src/lib/TinygradRuntime/` (the productized runtime — canonical)

Don't copy it wholesale; it's nnvp's engine. Steal patterns:

- `py/driver.py` — `build_safetensors` (NULL traces come out all-zeros, so
  real Glorot/zero/one/lr values are substituted by name pattern; note the
  BatchNorm rules: `running_var` = 1, gamma = 1, and the alias problem —
  the same tensor appears under multiple state names, key generated values
  by tensor identity), `patch_runner_for_weight_readback` (COPY_SRC/COPY_DST
  on weight buffers + a `weightBufs` name→buffer map on the step fn — the
  fail-loud match-exactly-once patch style), and
  `patch_runner_optimize_io` (`writeBuffer` uploads + a `_readLoss` flag;
  worth 4–30× on immature WebGPU stacks; Firefox's ~100 ms fence makes loss
  batching mandatory there).
- `py/README.md` + `worker.ts` — the Pyodide recipe: micropip-install the
  tinygrad wheel, set `DEV=NULL:WGSL` **before** import, trace in a worker,
  blob-import the emitted module on the main thread. Warm boot ≈ 4 s,
  in-tab retrace 3.4–6 s. This is how "in the browser" becomes literal —
  no Python server anywhere.
- `check_runner.ts` — byte-exact fake-WebGPU plumbing test that runs under
  bun; catches emitter drift without a GPU.
- BatchNorm/dropout traps (pinned by nnvp's `tinygradRuntime.spec.ts`):
  running-stat update assigns are NOT loss dependencies — realize them
  explicitly or they freeze at init; dropout's RNG counter lives outside
  restorable weight state.

### `experiments/bench-tfjs-vs-tinygrad/` + `pyodide-tinygrad/`

- The harness shape `m0` already copies: `server.mjs` (serve + `/report`
  collector), headless Chromium flags for software WebGPU
  (`--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader
  --enable-unsafe-swiftshader`), assert on the POSTed curve.
- The honest benchmark discipline: SwiftShader numbers are for CI proof
  only; quote real-hardware numbers (pyodide README's table: Chrome 152 /
  Firefox 152, dense + conv) and always state which stack produced them.

## Suggested build order for the demo (the README showpiece)

> Status 2026-08-30: the single entry point for everything testable in a
> browser is `experiments/m0-keras-trainstep/hub.html` (`gen_hub.py`):
> model (dense / dense+dropout) × task (easy / hard) × engines (prebuilt
> keras-tinygrad bundle / tf.js webgl / keras-tinygrad traced IN the tab
> under Pyodide) × mode (train / lr=0 same-batch probe), with in-page
> verdicts (bundle vs tf.js steps 0–1; in-tab trace vs bundle identical).
> It is the experiment-grade ancestor of step 2 below.

1. ~~`keras_tinygrad.export`~~ **DONE 2026-08-31: `keras_tinygrad.webgpu`**
   — `export_train_step(model, optimizer, loss_fn, batch_size=…,
   input_shape=…) → {js, weights, meta}`, `validate_runner_js` inside,
   exporter vendored in the package (`_vendor/`, byte-pinned to the
   experiment copies by a loader test). The dropout probe and the Pyodide
   driver are thin callers now; the wheel micropip-installed into the tab
   runs the exact shipped code (in-tab dropout bundle re-verified
   byte-identical). m0.py keeps its own copy as the owner's original
   proof. Still open from the old plan: the eval-runner variant
   (forward-only trace) and the weight-readback runner patches.
2. A `demo/` page: MNIST dense (or the conv net from `convnet_mnist.py`),
   real dataset fetched client-side, loss chart, weights download.
   Everything static — host it on GitHub Pages from this repo.
3. The Pyodide variant: same export running IN the tab (the pyodide-tinygrad
   recipe with keras+keras_tinygrad wheels via micropip) — that's the
   "arbitrary Keras model, no build step" version.
   **DONE 2026-08-30 — `experiments/pyodide-keras/`.** Measured cold on
   SwiftShader: Pyodide boot 2.3 s, packages+wheels 2.9 s, `import keras`
   in WASM 3.8 s, trace+export 4.2 s (dense) / 5.1 s (dense+dropout),
   setupNet 1.0 s — ≈14 s tab-to-training, ≈5 s per config change. Two
   pure-Python shims replace the wasm-less C extensions (`optree`,
   `ml_dtypes`); the in-tab traces are byte-identical to the Python-side
   exports (kernel sources and loss curves). Dense + Dropout only so far;
   main-thread Pyodide (worker recipe pending).
4. Then NNVP: it already emits Keras Python as a codegen target. When 1–3
   exist, nnvp's tinygrad engine can swap its bespoke raw-tinygrad emitter
   for the Keras source + this export path — one emitter fewer, and the pip
   package becomes the engine. That work lives in nnvp, but everything it
   calls should be here.

## Known limits (state them in the demo, don't hide them)

- Batch size is baked into the trace (nnvp's `dynamicBatch: false`
  precedent); partial final batches need padding or a second trace.
- Fixed learning rate only. `iterations` DOES ride the trace as i32 state
  since 2026-08-30 (it always emitted a `+1` kernel; now it reads real
  state and counts), so LR schedules are structurally within reach — but
  none has been traced or verified yet.
- BatchNorm needs the driver.py treatment (explicit stat realizes) —
  untested through the KERAS path yet. Dropout IS proven through the Keras
  path since 2026-08-30: with device RNG (docs/device-rng.md) the threefry
  counter advance is part of the trace, the seed/counter ride along as U32
  state buffers, and the exported step draws a fresh mask per call
  (`export_dropout_probe.py`; hub probe mode asserts it with lr=0).
- The traced-step export is training-loop-only: metrics, callbacks, and
  `fit()` ergonomics stay host-side (in the browser page, or under Pyodide).

## Why Dense "just worked" — the capture-hazard taxonomy (added 2026-08-30)

A question that will recur: why did full Keras `Dense` layers export with
zero special handling? Because TinyJit (and hence the export trace)
captures **tensor operations, not Python**. All the Keras machinery that
looks intimidating — `build()`, autocast scopes, name scoping, the call
context — runs at *trace time* in Python and evaporates; only the lazy
tensor graph it emitted remains. A layer is export-transparent iff its
step touches nothing but lazy tensor ops. Dense is exactly that:
matmul + bias add + activation. Nothing to handle, so nothing was.

What makes a layer NOT transparent — the four hazard classes:

1. **Host reads** (`.item()`, `.numpy()` mid-step): `ops.cond`,
   `while_loop`, some metrics. Fails LOUD at capture (JitError) — the
   trainer's whole safety story, inherited by export for free.
2. **Host RNG**: `Dropout` (backend `random.*` draws numpy samples per
   step — invariant 9's bit-parity-with-numpy design). Loud at capture.
   Export needs either mask-as-input (JS supplies randomness per step) or
   an in-graph device RNG under an explicit, documented deviation from
   invariant 9 — a design decision, not a patch.
3. **Non-trainable state mutated in the forward**: `BatchNorm` moving
   stats. Not a hazard per se — just more pins (the `_TrainStepJit`
   scheme extends; `driver.py` in nnvp's runtime has the treatment).
4. **Data-dependent shapes**: `unique`-style ops. Loud stubs already.

Everything hazard-free inherits the export path automatically —
Conv2D/pooling forwards are pure tensor ops and are expected to
export like Dense did (the nnvp bridge already trained conv nets through
the RAW-tinygrad version of this pipeline); unverified through the KERAS
path until someone runs the m0 recipe on a conv net.

And the two SILENT export-specific failure modes found by the m0 spike
(post-update loss scheduling; NULL-baked consts zeroing) are exactly why
`validate_runner_js` + the cpu-mode numeric proof exist: capture hazards
fail loud, but export bookkeeping must be *proven*, per bundle, every time.
