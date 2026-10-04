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

1. `js/demo/README.md` — the proof this doc builds
   on: a Keras train step exported to a WebGPU runner that trains in
   headless Chromium, plus the two bugs it found. Its `m0.py` is the code
   to generalize.
2. `docs/history/experiments-2026-07/` — the lineage: four spikes from nnvp
   (2026-07) that already solved tracing, Pyodide, benchmarking, and the
   real-graph bridge for RAW tinygrad steps, kept as their own documents
   (the code was removed 2026-10; the index says what it could do that
   `js/` cannot). The M0 spike married them to Keras.
3. The canonical productized runtime lives OUTSIDE this repo, in
   `~/workspace/nnvp/nnvp-client-vue/src/lib/TinygradRuntime/` — steal its
   patterns (listed below), never edit it from here.

Verify before building — all three must pass (python = the repo `.venv` —
keras 3.15.1 + tinygrad 0.14 + editable keras_tinygrad):

```sh
cd js/demo && python m0.py export build/out   # NULL:WGSL trace -> runner + weights
cd js/demo && CC=~/.local/bin/zigcc python m0.py cpu   # numeric proof: loss -> 0, acc 1.00
make e2e                                 # trains in headless Chromium (bun + playwright chromium), five asserted flows
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
shader compile vs ours 23 ms); the table in
`docs/history/experiments-2026-07/pyodide-tinygrad-README.md` has
per-browser numbers for RAW tinygrad steps (its 0.8 ms/step row syncs the
loss every 10th step, not every step). No multiplier headline until a
hardware bench of record exists (planned in nnvp).

## Where the plan went

The per-source "what to steal" notes and the build order that produced
`js/demo/` and `js/src/trace/` are in
`docs/history/browser-training-plan-2026-08-30.md`; every step there is done.

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
