# The four browser-training spikes (2026-07-13 → 07-16) — the record

Audit of 2026-10-02, before `experiments/` is removed: every file of the
four spikes was read in full and each thing it held was classified against
the current repo (SUPERSEDED / RECORD-ONLY / MISSING-CAPABILITY /
NNVP-OWNED / GENERATED). This directory keeps everything that was
RECORD-ONLY — the dated findings, numbers and conclusions no living doc
carried — as the spikes' own documents, verbatim, plus this index. The
code is gone; what it could do that the current `js/` cannot is listed at
the end, honestly.

Provenance: the spikes were written in the nnvp repo (`nnvp/experiments/`,
untracked there), copied verbatim into this repo on 2026-08-30, and never
committed anywhere. The 2026-09-22 index that travelled with them:
`experiments-index-as-of-2026-09-22.md`.

## The documents, in order

| file | spike | what it holds |
|---|---|---|
| `tinygrad-webgpu-export-README.md` | 2026-07-13, the origin | NULL:WGSL codegen with no GPU; a raw-tinygrad train step exported as WGSL (24 passes / 20 kernels, weights updated in place); "no prior art found"; and the recommendation on record that tinygrad "cannot retrace in the browser" — overturned two days later |
| `pyodide-tinygrad-README.md` | 2026-07-15/16, the pivot | tinygrad under Pyodide traces and exports IN the tab; the full per-browser benchmark table (Chrome/Firefox × dense/conv × five engines); the Firefox 100 ms fence-tick finding; weight readback, playground and viz |
| `bench-tfjs-vs-tinygrad-REPORT.md` + `-results-2026-07-15.json` | 2026-07-15 | the SwiftShader-only tf.js-vs-tinygrad bench: method, environment, raw numbers (p10/p90/mean), the "±30% is decision-irrelevant" argument, the tf.js maintenance facts, the "tfjs stays default" recommendation — later inverted on real hardware |
| `nnvp-tinygrad-bridge-README.md` | 2026-07-15 | nnvp's four generated templates trained in the browser through the raw pipeline (the only conv-in-browser record this repo holds), the four generator bugs fixed in nnvp, the softmax-plus-logits-loss finding |

## Record-only items that lived in code, not in a document

Verbatim from `pyodide-tinygrad/main.js` (`CACHE_NOTES`, 2026-07-16): when
each cost of the in-tab pipeline is paid again —

- device: per page load (cheap); nothing to cache
- pyodideBoot: download: per BROWSER (HTTP cache) · WASM boot: re-paid every PAGE LOAD — keep the tab or a worker alive across graph edits
- wheel: download (2.5 MB): per BROWSER (HTTP cache) · install: per PAGE LOAD (a Service Worker could pin it)
- pyenv: per PAGE LOAD — surviving graph edits is free if the Python worker stays up
- trace: per EDITED GRAPH; also invalidated by batch size / optimizer / loss (baked into the trace) — NOT by learning rate (lr is a weight buffer, editable live); partial reuse via tinygrad schedule cache (in-memory; persistable to IndexedDB)
- pipelines: per PAGE LOAD per traced graph; the browser shader-caches compiled WGSL per origin, so repeats are cheaper
- mnist: download (~50 MB): per BROWSER (HTTP cache) · decode: per PAGE LOAD · invalidated per EDITED DATASET
- tfjs: download: per BROWSER (HTTP cache) · init: per PAGE LOAD
- train: the workload itself — never cacheable

Other code-level facts worth one line each:

- The origin spike was tested against tinygrad commit
  `223c6d74c3ecc2fd096925b73cd089128d1f68ee` (2026-07-12); "a shallow clone
  + PYTHONPATH suffices". Its inference-only export (`net.js`) was 5 WGSL
  kernels, ~13 KB, zero runtime deps.
- tinygrad 0.13 → master API note of the time: `Tensor.training = True` is
  gone, `Tensor.train()` was the 0.13 context manager, master uses
  `Context(TRAINING=1)`. Nothing in the repo uses either now (Keras passes
  `training=True`).
- The 0.8 ms/step "Chrome WebGPU" number quoted in `docs/browser-training.md`
  and `js/demo/README.md` is the table's "NEW, loss every 10th step" row,
  not a per-step-readback number.
- The frozen 0.13 exporter the Pyodide spike shipped is byte-derivable from
  tinygrad tag v0.13.0 `extra/export_model.py`; the vendored copy in
  `src/keras_tinygrad/_vendor/` is the 0.14 port plus the kernel-name
  collision fix. Nothing in the frozen copy is missing from the vendored one.
- The bench server set COOP/COEP headers (cross-origin isolation); the hub
  server does not, because nothing in the hub needs `SharedArrayBuffer`.
  If Pyodide threading ever needs it, that header pair is what to copy back.

## Everything else was superseded

Every capability the spikes had on the KERAS path exists in the repo with
a test: NULL:WGSL export (`keras_tinygrad.webgpu`), the realize-before-
capture and realize-loss-before-apply rules, real initial values in the
safetensors, in-tab tracing in a module worker with the optree/ml_dtypes
shims (`js/src/trace/`), blob-import of the runner (`js/src/index.mjs`),
the serve-and-POST harness and the six Chromium flags
(`tests/test_webgpu_export.py`), the dropout freshness probe, the tf.js
engine on identical init and batches, the Pyodide/tinygrad/tf.js pins
(`client.mjs`, `scripts/fetch_tfjs.sh`). The nnvp-owned parts (template
generators, the paste bridge, absolute paths into the nnvp checkout, the
four generator fixes) live in nnvp.

## What the current repo cannot do that the spikes could

Listed so the loss is a decision, not an accident. None is a Keras-backend
feature; all are browser-runner conveniences, and the first two are the
ones worth having back.

1. **Weight readback / write-in on a live runner** — `COPY_SRC|COPY_DST` on
   the weight buffers plus a `step.weightBufs` map. Enabled save/load/edit/
   perturb/continue and the weight heatmaps. The vendored runner exposes
   nothing; the hub's learning-rate change is a byte patch before `setupNet`.
2. **The optimized-I/O runner** — `queue.writeBuffer` uploads instead of two
   `mapAsync(WRITE)` fences per step, and a flag to skip the loss readback.
   Measured 3–30× on Firefox in July 2026 (its `mapAsync` completes on a
   ~100 ms tick, so each fence costs one tick). The vendored 0.14 exporter
   still emits the mapped uploads and an unconditional loss readback.
3. Inference-only (forward) export — one `export_model` call in raw
   tinygrad; nothing exports a predict-only bundle today.
4. A fake-WebGPU plumbing test of the EMITTED runner without a browser
   (`check_runner.js`: usage flags enforced, byte-exact readback). The
   current node test round-trips a stub runner, not the emitted code.
5. A JS safetensors WRITER and the playground built on it (snapshot incl.
   momentum and lr, ±5% perturbation, download/upload).
6. A browser-side benchmark that writes a dated results file with p10/p90/
   mean and a fixed step protocol; a tf.js-cpu and a tf.js-webgpu column;
   `forceFallbackAdapter` in the page. The hub reports first-step and steady
   median per engine and nothing more; `bench/` is CPU-only Python.
7. Conv nets through the browser pipeline — the bridge trained four conv
   templates (raw tinygrad). The hub's models are dense and dense+dropout;
   conv through the Keras path is unverified.
8. A raw-tinygrad (non-Keras) model path in the tab — nnvp's, by design.
