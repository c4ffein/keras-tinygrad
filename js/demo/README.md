# M0 — a Keras train step, traced and trained in the browser

**Status: PROVEN, 2026-08-30.** The last unproven link between keras-tinygrad
and the browser-training story (see `../../docs/browser-training.md`): a train
step built from **Keras APIs** — `keras.Sequential` forward, Keras loss,
backend `compute_gradients`, `keras.optimizers.SGD.apply` — traced through the
tinygrad backend on the GPU-less `NULL:WGSL` device, exported as a
self-contained WebGPU runner, and **trained in headless Chromium**:

```
step 0  loss 4.0450  →  step 290  loss 0.0011
first10 0.619 → last10 0.001 over 300 steps
21.9 ms/step on SwiftShader (software WebGPU — a real GPU is far faster;
the July pyodide-tinygrad spike measured 0.8 ms/step for this net on Chrome,
loss read every 10th step — docs/history/experiments-2026-07/)
```

## Files

- `m0.py` — the spike. `export` mode traces on `NULL:WGSL` and emits
  `out.js` (WebGPU runner, 26 WGSL kernels incl. the iteration counter) + `out.safetensors` (real
  Glorot weights under Keras names — `sequential/dense/kernel`,
  `sgd/learning_rate`, momentum slots) + `out.meta.json`. `cpu` mode runs
  the SAME step object with real execution and asserts loss falls to ~0 and
  a fresh batch classifies at >80% (measured: 100%).
- the exporter is NOT copied here any more: `m0.py` imports the shipped
  one, `keras_tinygrad/_vendor/export_model.py` (tinygrad `extra/` at
  v0.13.0; nnvp's own copy has since drifted, so do NOT assume byte-parity
  with it). The July pyodide-tinygrad spike shipped its own frozen 0.13
  copy because it ran without this package inside Pyodide; that spike is
  a document now (docs/history/experiments-2026-07/).
- the browser proof and the equivalence seal are `../../tests/test_webgpu_export.py`
  now (2026-09-23): the suite serves this dir, drives the hub headless
  and asserts training, the tf.js verdict, both probes, the pyodide
  identity, and the browser-vs-local-Python curve (`gen_ref_curve.py` →
  `build/ref_curve.json`, same init bytes + same lcg32 stream through the
  Python backend on CPU). Their standalone ancestors (`check.sh` +
  probe page, `gen_demo.py`'s demo.html, the dropout probe page, the
  standalone pyodide page) are deleted — the hub + e2e strictly supersede
  them; the pyodide page's arbitrary-config ability lives on as the hub's
  "custom" model.
- `export_dropout_probe.py` → `build/dropout.{js,safetensors}` — the
  Dropout model export (on-device threefry RNG as U32 state buffers, see
  `../../docs/device-rng.md`).
- `gen_hub.py` → `hub.html` — the **single entry point for everything
  testable in a browser** (~3.9 MB; tf.min.js — fetched by `scripts/fetch_tfjs.sh` — and both
  Keras-traced bundles embedded). Menus: model (dense / dense+dropout / **custom** — any
  layers/dims/optimizer JSON, traced live by the pyodide engine;
  `&config=` drives it headless) × task × **engines** × mode, dark mode
  toggle.
  Engines — any subset, all fed the same model, task, lr and LCG batches,
  curves overlaid: *bundle* (keras-tinygrad traced in Python on the dev
  box, embedded); *tf.js webgl* (same init parsed from the safetensors
  into `tf.layers`; dropout draws its own masks); *pyodide* (keras 3.15.1 +
  this tree's wheel running IN the tab under Pyodide 0.28.3 inside a module
  Web Worker — the npm package's TRACER half (`js/src/trace/`:
  `client.mjs` + `worker.js` + `driver.py` + `shims/`), imported through
  `server.mjs`'s `/js/` mapping so the hub runs the exact files the
  package ships — tracing the same model on demand; the only engine that
  touches the network. Its JSON carries `mainThread: {maxGapMs,
  ticks, expectedTicks}` from a 20 ms heartbeat during boot + trace — the
  proof the page no longer freezes. Control measurement 2026-08-30: with
  Pyodide on the main thread, 0 of ~185 ticks fire during 3.7 s of Python
  (`runPython` and `runPythonAsync` alike, JSPI enabled) — fully blocked.
  The first meter version sampled only inside the tick callback and
  reported 0 for a solid block; the fixed one also samples at stop and
  counts ticks).
  Tasks: *easy* (prototypes ±1, noise 0.35 — ~20σ margin, solved in ~5
  steps; the frozen equivalence task, comparable with `ref_curve.json`; lr
  0.05) and *hard* (prototypes ±1/12, unit noise — ~1.7σ margin, real
  overlap; lr 0.005). Hardness had to come from margin, not magnitude (12×
  inputs diverged to NaN in both frameworks), and lr had to drop for the
  overlapping task (0.05·0.9 oscillated upward) — the page rewrites the
  live `learning_rate` state buffer per task; the Pyodide engine traces
  with that lr directly.
  Modes: *train* (N steps) and *same-batch probe ×3 with lr=0* (dense must
  give identical losses, dropout must give 3 distinct — fresh threefry
  masks are the only possible source; bundle + pyodide engines).
  In-page verdicts, also in the JSON: bundle vs tf.js steps 0–1 < 1e-3
  (dense; deterministic) and **in-tab trace vs prebuilt bundle: every loss
  identical** (dropout included — same kernels, same zeroed rng state).
  Owner's Intel Gen-9 (2026-08-30): dense/easy bundle 11.1 ms/step vs tf.js
  13.3, curves identical; dropout/hard 11.3 vs 13.4, 3.11→1.51 vs 3.05→1.48.
  All three engines, dropout/hard, 300 steps: bundle 11.5 ms/step, tf.js
  webgl 14.0, in-tab Pyodide trace 11.3 — **Pyodide-traced losses identical
  to the bundle's for all 300 steps (max diff 0)**; Pyodide stages on the
  Mac: boot 5.3 s, packages 1.7 s, wheels 5.6 s, import keras 4.2 s, trace
  5.3 s, setupNet 38 ms (vs 247 ms for the bundle seconds earlier: Chrome's
  pipeline cache already held the byte-identical WGSL).
  `?report=1&model=&task=&mode=&steps=&engines=bundle,tfjs,pyodide` drives
  it headless. The rigorous tfjs bench of record stays in nnvp.
- `server.mjs` — serves this dir on 0.0.0.0:8080 (`PORT=` overrides);
  `/` is the generated hub (`build/hub.html`), `/js/*` the npm package
  sources (tracer half included), `/build/*` everything generated;
  collects `/report` POSTs (tests/test_webgpu_export.py reads them from `/results`).

## Run it

```sh
make browser-assets     # repo root: tf.js fetch + wheel + both bundles + ref curve + hub -> build/
make e2e                # the browser proof: hub driven headless, five assertions
bun server.mjs          # then open http://<box>:8080/ for real-GPU runs
# the pieces, individually:
../../scripts/fetch_tfjs.sh                   # tf.js 4.22.0 (pinned, sha256-verified)
python m0.py export build/out                 # needs keras_tinygrad importable
CC=~/.local/bin/zigcc python m0.py cpu        # numeric proof (CPU device needs a C compiler)
```

## The three bugs the spike found (all matter for the real integration)

1. **Post-update loss.** Scheduling the loss realize together with the
   pinned-buffer update assigns let the scheduler recompute the loss from
   moved weights (measured 2.927 → 0.378 on step one). Keras' functional
   update path doesn't give the ordering protection raw tinygrad's
   `opt.schedule_step()` does. Fix: realize loss + grads BEFORE
   `optimizer.apply` builds any update graph — the wrong order becomes
   unbuildable.
2. **Baked consts silently zeroed.** Keras' stock loss reduction
   (`sum_over_batch_size`) realizes its 1/batch scalar as a const buffer;
   NULL fake-computes it to zeros, and `export_model_webgpu` honors
   `bufs_to_save` only for `get_state_dict` names — everything else becomes
   `createEmptyBuf`. The exported loss read ×0. Fix here: `reduction=None`
   + tinygrad `.mean()` (folds as a kernel immediate); the general guard is
   `validate_runner_js` — no pass may read a never-written empty buffer.

3. **The momentum scalar exported as zero — the bundle trained plain SGD**
   (found 2026-08-30, by the hub's vs-tfjs mode). Bug #2's mechanism, second
   bite: `SGD.update_step` runs `momentum=0.9` through the backend's
   `convert_to_tensor`, whose numpy path wraps scalars as shape-(1,)
   BUFFER-backed tensors (`np.ascontiguousarray` promotes 0-d); on NULL that
   anonymous buffer exports as a zeroed `createEmptyBuf`, so every step
   computed `v = 0.0*v - lr*g`. Steps 0-1 can't catch it (velocity is zero
   then) and "both converge" tolerances can't either (plain SGD also
   converges, just slower: browser tail ~1e-3 vs the true trajectory's
   ~1e-6). What caught it: our CPU reference and tfjs — two unrelated
   stacks — agreed to 4 decimals from step 2 on, and the bundle was the
   odd one out; a float32 numpy twin with the velocity zeroed reproduced
   the bundle's curve digit-for-digit. Fixes, all three needed: the
   backend converts Python scalars to CONST uops (they fold into kernels
   as immediates and survive export); `Variable._initialize` forces
   `.contiguous()` so scalar-INITIALIZED state (the lr) stays a real
   mutable buffer instead of baking in; and `validate_runner_js`'s pass
   parser — written to catch exactly this — had a regex bug
   (`[^[]*\[` stops at `pipelines[N]`'s bracket) that made it pass
   vacuously, now fixed and convicting the old bundles retroactively.
   Post-fix the browser curve matches the CPU reference AND tf.js at 4
   decimals for the early steps, and all three tails reach ~0.

   Related cleanup: `optimizer.iterations` (int32) never actually dropped
   out of the trace — every exported runner contained its `+1` kernel,
   reading garbage (benign under fixed lr: nothing consumes it). It now
   rides in the export state and is pinned like the floats, so the
   counter counts.

## Which pieces are load-bearing

The `KerasTrainStep.__call__` shape: pin every float Variable to the buffer
the capture read (the `_TrainStepJit` scheme from the package trainer —
in-place `Tensor.assign`, `_value` repointed — `iterations` included, as
int32 state; see bug #3), and hand `export_model` a bare function whose `__dict__`
holds only the pinned buffers (get_state_dict must never walk the Keras
object graph — parent references cycle).
