# tfjs vs tinygrad-in-the-browser — benchmark & decision report

**Date:** 2026-07-15 · **Question:** should NNVP's in-browser training move from
tfjs to the tinygrad/Pyodide/WebGPU stack (`../pyodide-tinygrad/`)?

## TL;DR

On identical work, **tfjs-webgl is ~1.45× faster steady-state and ~8× faster to
first-step** than tinygrad-WGSL on this machine's software GPU stack. The
±30% throughput question turns out to be the wrong axis: **setup latency and
per-edit retrace cost dominate the interactive experience**, and there tfjs
wins by a wide margin today. What tinygrad has is the **future**: tfjs has had
no stable release since **2024-10-21** (a 4.23 RC in Jan 2025 never shipped —
21 months frozen as of today), while the tinygrad path runs on WebGPU, is
retraceable client-side, and its kernels are *unoptimized defaults* with known
headroom. Recommendation: **tfjs stays the default engine; tinygrad becomes the
opt-in "experimental engine" on a real timeline — adopted for strategic reasons
(tfjs freeze, WebGPU), not throughput.** Re-run `./run-bench.sh` on a real GPU
before finalizing.

## Method

Same training problem on both sides, mirrored from `../pyodide-tinygrad/driver.py`:

- MLP 784 → 128 (relu) → 10 logits, Glorot init
- SGD momentum 0.9, lr 0.01, sparse categorical crossentropy
- Batch 32, **identical bytes** (shared LCG-seeded batch in `bench-common.js`)
- 210 steps; first 10 excluded from steady-state; per-step **loss readback on
  both sides** (tinygrad's step returns the loss, so tfjs must `data()` too —
  removing it would flatter tfjs)
- Both must actually train: loss ≥ 2.4 → ≤ 0.007 verified on every run

Environment: headless Chromium 149 (Playwright build), **SwiftShader software
GPU** for both — WebGPU via `--use-webgpu-adapter=swiftshader`, WebGL via
SwiftShader GL. No real GPU exists on this box. That makes the comparison
software-stack vs software-stack: honest between the two, but **not predictive
of real-GPU ratios** (see caveats).

## Results (this box, software GPU, 2 runs each — raw JSON in `results/`)

| | tfjs webgl | tinygrad WGSL (Pyodide) | tfjs cpu |
|---|---|---|---|
| Steady median / step | **38.0 / 38.4 ms** | 55.0 / 56.1 ms | 105.1 ms |
| Steady steps/sec | **25.8 / 25.3** | 17.5 / 17.5 | 9.4 |
| First step | 3 662 / 3 799 ms (shader compile) | **102 / 63 ms** (pipelines prebuilt) | 505 ms |
| Setup before first step | ~50 ms | **29.5 / 29.9 s** | ~20 ms |
| … setup breakdown | — | boot 5.4 s · install 2.3 s · **trace 19.7 s** · pipelines 2.1 s | — |
| One-time download | 1.5 MB (tf.min.js) | ~16.5 MB (Pyodide + wheel) | 1.5 MB |
| Trained (loss) | 2.45 → 0.006 | 2.66 → 0.007 | 2.50 → 0.006 |

Time to *usable training* from cold page: **tfjs ≈ 3.7 s** (setup + warmup step)
vs **tinygrad ≈ 29.6 s**. tinygrad's WGSL comfortably beats tfjs-**cpu** (1.9×).

## The "what if ±30%?" question

Suppose real-GPU numbers came back with tinygrad **30% faster** per step. A
typical interactive session here (60–200 steps on MNIST-scale batches) spends
2–8 s stepping. 30% saves ~1–2 s — while the tinygrad path costs **+26 s of
setup** and **+15 MB download**, plus a **re-trace on every architecture edit**
(that's its defining property; tfjs rebuilds in milliseconds). 30% **slower**
inverts nothing either. **Throughput within ±30% is decision-irrelevant for
NNVP's interactive use.** It would start to matter for long runs (≳10 min of
stepping), where setup amortizes — not this app's primary mode today.

What actually decides:

| Axis | tfjs | tinygrad/Pyodide | Weight |
|---|---|---|---|
| Cold start to training | ~4 s | ~30 s (≈15 s on the maintainer's machine: real-GPU trace was 7.8 s) | High for interactive |
| Cost per architecture edit | ~ms (rebuild) | one full re-trace (≈8–20 s) | High |
| Maintenance trajectory | **frozen since 2024-10** | active upstream, vendored pair pinned + loud-fail | High, long-term |
| Platform | WebGL (legacy path) | WebGPU (the future; tfjs's own WebGPU backend never left experimental) | Medium |
| Kernel headroom | hand-tuned, done | default schedules, BEAM etc. untapped | Medium |
| Ops coverage for NNVP's catalog | full | MLP proven only | Blocker for default |
| Download | 1.5 MB | 16.5 MB (cacheable via service worker) | Medium |

## On tfjs being unmaintained

Registry facts: latest stable **4.22.0 (2024-10-21)**; **4.23.0-rc.0
(2025-01-08)** never went stable; nothing since. Frozen ≠ broken — browsers
keep WebGL compatible for years, and 4.22 works fine today (these very numbers
prove it). But it means: no fixes when platforms shift, no WebGPU future, and
the app's training engine is a dependency with no upstream. That is the real
argument for investing in the tinygrad path **now, at experimental priority** —
so the replacement is proven before it is ever needed, rather than after tfjs
breaks.

## Caveats

- **Software GPU on both sides.** SwiftShader-Vulkan (Dawn/WGSL) and
  SwiftShader-GL (ANGLE) are different software stacks; real-GPU ratios can
  differ in either direction. The harness is one command on real hardware.
- tinygrad kernels are **unoptimized defaults** (no BEAM search — it needs a
  timing device, plausible on real WebGPU); tfjs-webgl kernels are years of
  hand-tuning. The 1.45× gap is tinygrad's *floor*, not its ceiling.
- Trace time (19.7 s here) is CPython-under-wasm on a weak CPU; the maintainer's
  earlier real-browser run traced in 7.8 s.
- Batch 32 / MLP only — conv workloads may reorder the ranking on both axes.

## Recommendation

1. **Today: tfjs remains the interactive engine.** Faster where it counts
   (time-to-first-step, per-edit cost), full ops coverage, 1.5 MB.
2. **Adopt tinygrad as the opt-in experimental engine** ("Train with tinygrad
   (experimental)" in the Training window), gated on WebGPU — justified by the
   tfjs freeze and WebGPU, not by speed. Prerequisites, in order: per-layer
   coverage against NNVP's generated tinygrad code; an eval-only export;
   service-worker caching of Pyodide + wheel (cuts ~8 s of the 30 s; the trace
   is the irreducible per-edit cost).
3. **Re-run `./run-bench.sh` on a real GPU** and append the results to
   `results/` before treating any throughput conclusion as final.
4. Revisit the default once (a) coverage is proven and (b) real-GPU steady
   state lands within ~2× of tfjs — at that point the maintenance argument
   should win.

## Reproduce

```bash
cd experiments/bench-tfjs-vs-tinygrad
./run-bench.sh          # headless, software GPU (this report's setup)
# or serve the dir (bun server.mjs) and open the two bench pages in a real
# browser on GPU hardware; results POST to /results.
```
