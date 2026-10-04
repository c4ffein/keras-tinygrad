# Browser-training plan (2026-08-30) — what to steal, build order

> Split out of `docs/browser-training.md` on 2026-09-28: the planning
> sections that shaped `js/demo/` and `js/src/trace/`. Every step below
> is done; kept for the reasoning. Paths are as of the date.

## What to steal, per source

### `js/demo/` (native, the proof)

The load-bearing piece is `KerasTrainStep` in `m0.py` — the shape any future
`keras_tinygrad.export_train_step(model, optimizer, loss)` API should
generalize:

1. **Pinning** (from `keras_tinygrad/src/trainer.py`'s `_TrainStepJit`): Keras
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
> browser is the hub (`js/demo/gen_hub.py` -> `js/demo/build/hub.html`):
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
   **DONE 2026-08-30 — `js/src/trace/`.** Measured cold on
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

