# pyodide standalone page — Keras + keras-tinygrad running IN the tab

**The ENGINE moved (2026-09-23): it is the npm package's tracer half now —
`js/src/trace/` (`client.mjs`, `worker.js`, `driver.py`, `shims/`),
importable as `keras-tinygrad/trace`.** What remains here is the
standalone arbitrary-config page (`index.html` + `main.js` + `server.mjs`,
port 8090) and `run_local.py` (the native fast dev loop for the driver,
which also proves keras runs on the shims alone). Original record below.

---

**Status: WORKS, 2026-08-30 (first run).** This is step 3 of
`docs/browser-training.md`, and the answer to "can we have tfjs-like option
selection?": the Keras API runs in the browser under Pyodide, the user's
config (layers, units, activation, dropout, lr, momentum, batch size) becomes
a `keras.Sequential` + `SGD`, one training step is traced through the
tinygrad backend on the GPU-less `NULL:WGSL` device, and WebGPU loops on the
exported kernels. Every option change is a **retrace (seconds)**, never a
redeploy; nothing runs on a server.

Measured headless (Chromium 149 + SwiftShader, cold caches, this box):

| stage | ms |
|---|---|
| Pyodide 0.28.3 runtime (download + boot) | 2309 |
| Pyodide-built packages (numpy, h5py, rich, packaging, sqlite3, micropip) | 1508 |
| PyPI wheels via micropip (tinygrad 0.13.0, keras 3.15.1, absl-py, namex) + ours | 1351 |
| `import keras_tinygrad` + `import keras` (in WASM) | 3839 |
| **trace + export in Python (NULL:WGSL), default 784-128-10 model** | **4170** |
| WebGPU setupNet | 1002 |
| 100 training steps (SwiftShader) | 2180 (18.8 ms/step) |

Cold tab → training model ≈ 14 s; a config change ≈ 5 s (retrace + setup).
Second config through the same page (dropout 0.3, hard task, lr 0.005):
28 kernels, trace 5073 ms, 150 steps 3.11 → 1.92 — and its first steps
(`3.016, 3.3319, 3.0702, 3.2491`) are digit-identical to the m0 hub's
Python-exported dropout bundle on the same task: the in-tab trace reproduces
the Python-side export exactly, threefry state included.
The trained curve is the equivalence curve digit for digit (`2.5546, 0.8578,
0.0126, 0.0014, 0.0003, 0 …`): same kernels as the Python-side export —
`run_local.py` asserts the default config's 26 kernel sources are byte-identical
to `../out.js`.

## What had to be built

- **Two pure-Python shims** (`shims/`): keras 3.15.1 requires `optree` and
  `ml_dtypes`, both C extensions with no wasm wheels (checked PyPI + the
  Pyodide 0.28.3 lock). `optree` covers exactly the nine calls
  `keras/src/tree/optree_impl.py` makes (sorted-dict-key flattening,
  `flatten_up_to` semantics for `tree_map`, registered node classes in the
  `keras` namespace); `ml_dtypes` delegates `finfo`/`iinfo` to numpy and
  raises loudly for bfloat16/float8 (unavailable without the extension —
  never a silent wrong answer). `absl-py` and `namex` are pure wheels
  micropip fetches from PyPI. `micropip.install("keras", deps=False)`
  skips the two unsatisfiable requirements.
- **`driver.py`** — config → model → `KerasTrainStep` (m0's recipe: realize
  loss+grads BEFORE `optimizer.apply`, pin every Variable incl. `iterations`,
  `device_rng_scope` so dropout masks are on-device threefry, rng seed/counter
  as U32 state) → `export_model` → `validate_runner_js` → safetensors with real
  Glorot init. Identical under CPython and Pyodide.
- **`run_local.py`** — the dev loop AND the shim test: forces the shims to
  the front of `sys.path` and blocks `jax`/`tensorflow`/`torch` (absent under
  Pyodide; present in the dev venv, where keras would import jax eagerly).
- **The wheel of THIS tree**: `uv build --wheel -o js/src/trace/wheels`.
  The PyPI 0.1.0 wheel predates the momentum/const fix of 2026-08-30 and
  would train plain SGD (see `../README.md`, bug #3).

## Where it lives for users

The m0 hub (`../hub.html`, served at `:8080/`) has this
as its third **engine** — same model/task/lr/batches as the prebuilt bundle
and tf.js, curves overlaid, and an in-page verdict that the in-tab trace's
losses are identical to the prebuilt bundle's. `demo-server.mjs` maps
`/pyodide/*` to this directory; the standalone page below is the dev form.

## Run it

```sh
# 1. wheel from the working tree (wheels/ is gitignored)
uv build --wheel -o js/src/trace/wheels
# 2. serve (Pyodide from jsDelivr, keras/tinygrad from PyPI, ours from ./wheels)
cd browser/pyodide && bun server.mjs        # 0.0.0.0:8090
# 3. open http://<box>:8090/ in Chrome/Edge — boot, pick options, trace + train.
# headless / CI:  ?auto=1&report=1[&units=64&dropout=0.3&task=hard&lr=0.005&steps=100]
# python-side dev loop (shims forced, no browser):
CC=~/.local/bin/zigcc ../../.venv/bin/python run_local.py '{"layers":[{"type":"dense","units":64,"activation":"relu"},{"type":"dropout","rate":0.3}]}'
```

## Honest limits (today)

- Layers: Dense + Dropout only (what the export path has proven); adding a
  type = adding it to `driver.build_model` and proving its capture hazards
  (`docs/browser-training.md`, taxonomy section).
- Batch size is still baked per trace (a change = retrace, which is now
  cheap). Fixed lr per trace; it IS a live state buffer, so a page can
  rewrite it without retracing (the m0 hub does).
- Pyodide runs in a module Web Worker (`pyodide-worker.js`, driven through
  `pyodide-client.js` — the one boot/trace recipe both pages use); the page
  stays responsive and the results JSON carries the main thread's longest
  stall and heartbeat ticks fired vs expected during boot and trace
  (`bootMainThread`, `traceMainThread`); on the main thread Pyodide fires 0
  ticks during `import keras` or a trace — fully blocked, JSPI or not.
- bfloat16/float8 anything raises (shim); no saving/loading yet; synthetic
  tasks only (the m0 hub's), no MNIST in this page yet.
- Numbers above are one cold SwiftShader run; real-GPU step times come from
  the owner's machine, the Python-side times are WASM-bound regardless.
