# nnvp-tinygrad-bridge — the app's real graphs through the tinygrad pipeline

Answers "would `../pyodide-tinygrad/` work with the regular NNVP?" by feeding
the EXACT Python that NNVP's tinygrad generator emits for the app's real
templates into the Pyodide → NULL:WGSL trace → WebGPU training pipeline.

## Result (2026-07-15, headless Chromium + SwiftShader WebGPU): YES

All four shipped templates train end-to-end in the browser, 30 steps each,
batch 32, SGD momentum 0.9 lr 0.01, matched (probability) loss:

| template | kernels | trace (wasm) | pipelines | step (software GPU) | loss, 30 steps |
|---|---|---|---|---|---|
| 2D Dense for MNIST | 21 | 15.5 s | 1.6 s | 47 ms | 2.67 → **0.69** |
| 2D Conv for MNIST | 37 | 27.0 s | 7.9 s | 731 ms | 2.51 → **1.22** |
| Fashion MNIST | 50 | 47.2 s | 13.5 s | 1.27 s | 2.46 → 2.34 (falling; dropout-heavy, needs more steps) |
| CIFAR-10 CNN | 64 | 82.7 s | 17.5 s | 5.46 s | 2.69 → 2.23 (falling) |

Step times are SwiftShader (software) — conv workloads suffer most there;
real-GPU numbers are the ones that matter and this harness produces them
unchanged (see Run below).

## What it took — generator gaps found and FIXED in the app

Getting here required real fixes to `KerasGeneratorTinygradHelper` /
`KerasGeneratorDimInference` (all dual-tested in the app's suite):

1. **MaxPooling2D / AveragePooling2D / Dropout** were TODO placeholders —
   now emitted as `x.max_pool2d(...)` / `.avg_pool2d(...)` / `.dropout(p)`.
2. **Dense/Conv2D `activation` params were silently dropped** — now chained
   as Tensor methods (`self.layer_2(x).relu()`), unknown ones get loud TODOs.
3. **Conv/pool spatial arithmetic** was "deliberately not attempted" in dim
   inference — now computed (valid/same, strides), which is what makes
   `Linear(5408, …)` after a conv stack a computed value instead of a TODO.
4. **Conv2D ignored `strides`/`padding='same'`** — now emitted, including the
   end-heavy 4-tuple Keras alignment for even kernels; only stride>1 'same'
   (input-size-dependent) remains a loud TODO.

Bridge-side findings encoded in `driver_nnvp.py`:

- Keras models end in softmax; tinygrad's sparse CCE wants logits. The
  mismatch trains almost not at all (2.31→2.25/30 steps); the matched
  NLL-on-probabilities loss descends properly (2.67→0.69). A product
  integration must match the loss (or strip the final softmax).
- NNVP graphs are channels-LAST; the emitted tinygrad model is channels-FIRST
  — `generate-models.mjs` transposes the Input shape.
- Conv kernels are 4-D: Glorot init generalized (`build_safetensors_nd`).

## Run

```bash
bun generate-models.mjs      # regenerate models/*.py via the app's generator
bun server.mjs &             # serves this dir + ../pyodide-tinygrad, collects /report
chromium --headless=new --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-unsafe-swiftshader http://127.0.0.1:8091/nnvp-tinygrad-bridge/bridge.html
curl -s http://127.0.0.1:8091/results   # or watch the on-page log in a real browser
```

Local (no browser) trace validation of all models:
`DEV=NULL:WGSL uv run --with tinygrad==0.13.0 python3` + `driver_nnvp.build_from_source`
with `../pyodide-tinygrad` on `sys.path`.
