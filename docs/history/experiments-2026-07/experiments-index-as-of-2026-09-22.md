# experiments/ — the browser-training lineage (archaeology)

Four spikes migrated verbatim from `nnvp/experiments/` on 2026-08-30 (they
were untracked there; the originals and `nnvp/experiments-bak-1/` can be
deleted once this is committed). Read `../docs/browser-training.md` for
what to reuse and why.

**Promotion note (2026-09-22):** the two directories that stopped being
experiments — `m0-keras-trainstep/` (the hub app + exports) and
`pyodide-keras/` (the in-tab engine) — moved to `js/demo/` and
`js/src/trace/` at the repo root: the Makefile (`browser-assets`), the
README's browser claims, and the e2e suite (`tests/test_webgpu_export.py`) all depend on
them. Their entries below are kept for the timeline; the paths are the
old ones.

Chronological:

- `tinygrad-webgpu-export/` (2026-07-13) — the origin: codegen on the
  GPU-less `NULL:WGSL` device; exported a full raw-tinygrad training step
  (forward → sparse CCE → backward → SGD momentum) as WGSL. Its "can't
  retrace in the browser" conclusion was overturned two days later.
- `pyodide-tinygrad/` (2026-07-15/16) — the pivotal one: tinygrad under
  Pyodide traces + exports IN the tab. Its README's table has per-browser
  STEADY-STATE numbers (0.8 ms/step dense, 3.5 conv on one Chrome WebGPU
  box vs tfjs-webgl 7.3/15.4) — single machine, no run of record, and the
  SwiftShader bench below measured the opposite ordering; no multiplier
  headline until a hardware bench exists (`docs/browser-training.md`,
  "Perf, stated as measured"). Productized as nnvp's canonical
  `src/lib/TinygradRuntime/` (that copy, not this one, is maintained).
- `bench-tfjs-vs-tinygrad/` (2026-07-15) — SwiftShader-only benchmark whose
  "tfjs stays default" numbers were superseded by the real-hardware run
  above; its harness (serve + headless Chromium with software WebGPU +
  result POST) is the reusable part, and is what `m0-keras-trainstep/`
  copies.
- `nnvp-tinygrad-bridge/` (2026-07-15) — proved nnvp's real generated graphs
  survive the pipeline (all four templates trained); its four generator-bug
  fixes landed in the nnvp app.
- `m0-keras-trainstep/` (2026-08-30, native to this repo) — **the missing
  link, proven**: a KERAS train step through keras-tinygrad, exported the
  same way, trained in headless Chromium. See its README.

- `pyodide-keras/` (2026-08-30, native to this repo) — step 3 of
  `docs/browser-training.md`: keras 3.15.1 + this package running IN the tab
  under Pyodide 0.28.3 (module Web Worker), tracing Keras models from a JSON
  config on demand (~4-5 s per retrace) with pure-Python shims for the two
  wasm-less C extensions (`optree`, `ml_dtypes`). Its traces are
  byte-identical to the Python-side exports. Mounted in the m0 hub as the
  third engine (`/pyodide/*` on the hub server).