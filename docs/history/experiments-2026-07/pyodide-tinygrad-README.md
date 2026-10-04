# pyodide-tinygrad — retrace & train in the browser, zero servers

**LAYOUT (since 2026-07-16)**: everything lives HERE, standalone. In dev the
app's vite server exposes this directory at `/tinygrad-bench/` via a small
dev-only plugin in `nnvp-client-vue/vite.config.js` (`apply: 'serve'` — it
does NOT ship in dist and needs no symlink). Serve standalone instead with
`python3 -m http.server` from this directory if the app isn't running.

**TWO RUNNER VARIANTS** (`driver.build_both` — one trace, two post-processes):
legacy (2× mapAsync(WRITE) uploads + mapAsync(READ) per step) and optimized
(`patch_runner_optimize_io`: `queue.writeBuffer` uploads + a `_readLoss=false`
flag to skip the readback fence). Motivated by Firefox 2026-07 numbers where
per-step mapAsync fences made tinygrad 6× slower than tfjs-webgl. The FULL
BENCHMARK button races OLD vs NEW vs NEW+loss-every-10 vs tfjs-webgl vs
tfjs-webgpu, per template, plus staged init timings annotated with when each
cost is paid again (browser / page load / graph edit / params / dataset).


The toy that answers the strategic question from the tinygrad investigation
(`../tinygrad-webgpu-export/`): can NNVP get **client-side retracing** —
tinygrad training that adapts to graph edits *without* a Python server?

Architecture (the part that makes it plausible): Python never touches the
GPU. tinygrad runs under **Pyodide** (it is pure, dependency-free Python) and
does tracing + WGSL codegen on the GPU-less `NULL:WGSL` device — the trick
validated in the sibling experiment. It emits a self-contained JS runner,
which the browser imports and loops on **WebGPU**. No JSPI/async bridging
anywhere: the Python↔JS boundary is crossed once per (re)trace, with a string
of JS and a safetensors blob.

## Run it

```bash
cd experiments/pyodide-tinygrad
python3 -m http.server 8090       # any static server; file:// won't work
# open http://localhost:8090 in a WebGPU browser (recent Chrome/Edge),
# click the button, watch the loss fall on a fixed random batch.
```

First load downloads Pyodide (~15 MB) + the tinygrad wheel from CDN/PyPI.

## Files

- `driver.py` — runs inside Pyodide: builds the MLP + SGD TrainStep, traces
  on `NULL:WGSL`, exports via the vendored `export_model.py`, and hand-rolls
  a safetensors blob with REAL values (NULL-device copyout is all zeros, and
  a zero-initialized network cannot train). Gotchas encoded: Glorot init is
  keyed by *tensor identity* because the state dict aliases every weight
  under two names (`model.*.weight` and `opt.params.N` — the emitted runner
  loads from the `opt.params.*` ones!); momentum buffers are 2-D but must
  stay zero; `opt.lr` must carry the real learning rate.
  It also post-processes the emitted runner for **weight access in both
  directions** (`patch_runner_for_weight_readback`): weight buffers get
  `COPY_SRC` + `COPY_DST` (read out for saving/viz, write in for
  loading/editing — training continues from whatever was written), and the
  step function exposes `step.weightBufs` ({stateName: GPUBuffer}). The
  vendored file stays pristine (re-vendoring stays a plain copy); every
  substitution asserts it matched exactly once, so an export_model bump
  fails loudly at trace time, not in the tab.
- `check_runner.js` — bun smoke test of the patched runner against a fake
  WebGPU device with real backing memory that ENFORCES usage flags:
  readback must return the exact safetensors bytes for every weightBufs
  entry, a write must round-trip new bytes through a weight buffer, and one
  full `step()` call must run. Verifies the patch plumbing without a browser
  (kernels are no-ops — it does not check the math).
  `PYTHONPATH=<wheel> python3 run_local.py <prefix>` then
  `bun check_runner.js <prefix>`.
- `export_model.py` — vendored from tinygrad tag `v0.13.0` (`extra/` is NOT
  in the PyPI wheel). **Version-locked pair**: `main.js` pins
  `micropip.install("tinygrad==0.13.0")`; master's export_model does NOT run
  against 0.13.0 (UOp `PARAM.arg` changed shape) — when bumping the wheel,
  re-vendor from the matching tag and re-run `run_local.py`.
- `main.js` / `index.html` — browser harness: Pyodide → trace → blob-import
  the emitted module → `setupNet(device, weights)` → loop
  `step(Float32Array images, Int32Array labels)` 60× on a fixed random batch.
  After training it reads every layer's weights/biases back off the GPU and
  renders init-vs-current heatmaps on a shared ±max|w| scale (diverging:
  cool blue negative / warm orange positive on the dark background):
  `[n, 784]` matrices as per-unit 28×28 input maps, other 2-D as matrix
  heatmaps, biases as strips, each captioned with mean|Δ| vs init. A controls
  row then makes the weights PLAYABLE: load the init or trained snapshot
  (full optimizer state — params, momentum, lr), perturb everything ±5%,
  train 20 more steps from whatever is currently in the buffers, download
  the live state as a .safetensors file, and load a .safetensors file back
  in. Load-init → train shows the loss restart high; load-trained → train
  shows it resume low — that IS the read→write→train round-trip.
- `run_local.py` — validates the whole Python side under plain CPython
  against an unpacked copy of the SAME wheel:
  `PYTHONPATH=/path/to/unpacked-wheel python3 run_local.py [out-prefix]`
  (checks the emitted runner, the weight aliasing, nonzero init, lr, and the
  readback patches; the optional prefix dumps runner + weights for
  `check_runner.js`). Passed against tinygrad 0.13.0 on 2026-07-13.

## Status: VERIFIED END-TO-END IN A REAL BROWSER (2026-07-13)

Run on the maintainer's machine: trace+compile took **7.8 s in-tab**, and 60
WebGPU training steps took the loss **2.54 → 0.029** in a clean monotonic
descent — real gradient descent, retraced and executed entirely client-side.
As far as the July 2026 investigation could find, nobody had shipped
browser-side tinygrad training in any form, let alone with in-browser
retracing. The "edit graph → retrace in browser → train on WebGPU, zero
servers" endgame is FEASIBLE.

One environment fix was needed: Pyodide unvendors `sqlite3` (tinygrad's
compile cache imports it) — `main.js` loads it explicitly.

**Weight read/write + viz + playground added 2026-07-13, verified under bun
only** (fake-GPU `check_runner.js`: byte-exact readback of all 9 state
entries, write round-trip through a weight buffer, full `step()` command
flow; a run WITHOUT the patches fails it). The browser half — real-device
copies, the heatmaps, the controls row, the safetensors download/upload —
still needs one click on a WebGPU machine; the 2.54 → 0.029 training claim
above predates it and remains valid.

**NNVP bridge added 2026-07-16 (Python side + fake-GPU verified; browser
half needs one click)**: the page now has a "Train an NNVP model" panel —
paste the .py from NNVP's Generate > Tinygrad, set the per-sample input shape
(channels-first for conv, e.g. `1,28,28`) and class count, run.
`driver.build(model_source=…)` execs the pasted code, strips the final
`.softmax()` (the loss log-softmaxes internally — training through a double
softmax cripples gradients; this mirrors Keras folding softmax into CCE), and
Glorot init generalizes to any-rank weight tensors (fan_in = prod(shape[1:]),
so Conv 4-D kernels init correctly). Verified against the 0.13.0 wheel with
REAL generator output for both templates: "2D Dense for MNIST" (20 kernels)
and — the big one — "2D Conv for MNIST": Conv2d + max_pool2d + dropout +
flatten traced to 36 WGSL kernels (dropout's threefry RNG traced fine), and
the patched conv runner passes the fake-GPU `check_runner.js` byte-exact.

**Dual-engine MNIST benchmark (2026-07-16, unverified in browser)**: a model
selector offers the built-in MLP plus BOTH NNVP templates, pre-extracted with
the app's own generators into `models.gen.js` (regenerate with
`bun gen-models.mjs` after generator changes) — tinygrad source AND the
matching generated tfjs code, so both engines run the same architecture the
way each ecosystem uses it (tinygrad: logits + fused CE, one pre-recorded
submission/step; tfjs: softmax + categoricalCrossentropy via trainOnBatch,
same eval contract as TrainingZone). The "benchmark tfjs vs tinygrad (MNIST)"
button loads REAL MNIST through the app's own dataset loader (imported from
/src via vite — works because the page is vite-served at /tinygrad-bench/;
falls back to synthetic with a warning when served standalone), slices 10
fixed batches (MNIST is 1-channel: the same flat bytes feed NHWC tfjs and
NCHW tinygrad), runs warmup+60 timed steps per engine, and prints a
copy-pastable result block: median ms/step, final losses, ratio. The page
also falls back to the SOFTWARE WebGPU adapter (forceFallbackAdapter) when no
hardware GPU exists — results then measure SwiftShader, not the GPU.

Remaining next steps: feed real datasets (the app already loads MNIST
client-side); an eval/predict second export (today the only loss probe is a
training step, so probing nudges the weights); per-layer ops coverage; weight
TRANSFER across a retrace (save → edit graph → retrace → load the compatible
tensors back), which the save/load half already enables; and a perf
comparison against tfjs before believing anything. The app itself still has
NO weight/bias viz (its Inspector shows activations only) — its
`summarizeActivation`/`drawInspection` pipeline takes plain Float32Arrays and
would render these readbacks as-is.

Known trade-offs measured/expected: ~seconds per retrace (fine behind a
"recompile" button), ~15 MB one-time runtime download, kernels are default
schedules (unoptimized; no BEAM without a timing device).

## FULL BENCHMARK results (2026-07-16, Windows, Chrome 152 / Firefox 152)

Per-step medians, real MNIST, batch 32 (ms/step):

| engine                    | Chrome dense | Chrome conv | Firefox dense | Firefox conv |
|---------------------------|---:|---:|---:|---:|
| tinygrad OLD (3 fences)   | 4.0 | 6.2 | 301.0 | 301.0 |
| tinygrad NEW (writeBuffer)| 3.6 | 5.7 | 100.0 | 100.0 |
| tinygrad NEW, loss/10     | **0.8** | **3.5** | **10.0** | **10.0** |
| tfjs-webgl                | 7.3 | 15.4 | 12.0 | 25.0 |
| tfjs-webgpu               | 5.7 | 14.5 | 100.0 | 100.0 |

Findings: (1) Firefox's numbers are model-INDEPENDENT and land on exact
multiples of ~100 ms — its mapAsync completion appears to poll on a ~100 ms
tick, so each fence costs one tick: OLD = 3 fences = ~300 ms, NEW = 1 = ~100,
loss-every-10 = 0.1 fence/step = ~10. GPU compute is essentially free there;
the fence tick is everything (tfjs-webgpu pays the same 100 ms). (2) With
both I/O fixes tinygrad wins EVERYWHERE WebGPU exists: Chrome 2–4×
(loss-batched: 7–18×) faster than tfjs; Firefox ties/beats tfjs-webgl.
(3) OLD and NEW losses are bit-identical (0.312/0.242) — the patches change
I/O only. (4) Warm-cache boot ≈ 4 s total; in-tab retrace 3.4 s dense /
6.0 s conv (Chrome). Engine recommendation simplifies to: tinygrad wherever
WebGPU exists (batched loss readback), tfjs-webgl as the no-WebGPU fallback.
