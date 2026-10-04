"""Generate build/hub.html — the single entry point for everything testable
in a browser (the SSOT page). server.mjs serves it at "/" on :8080;
tests/test_webgpu_export.py drives it headless.

The page consumes the npm package (js/src/index.mjs, embedded verbatim and
imported as a module) for bundle loading (`loadBundle`) and the LCG batch
stream (`lcg32`) — so every hub run, including the e2e suite's headless
ones (tests/test_webgpu_export.py ), exercises the exact code `npm install keras-tinygrad`
ships, not a parallel copy of it.

MODEL   dense 784→128→10 / dense+Dropout(0.3) / custom (any layers/dims/
        optimizer as JSON — pyodide engine only, since only the in-tab
        tracer can build arbitrary models; absorbed from the deleted
        standalone pyodide page 2026-09-23; &config= URL param drives it
        headless)
TASK    easy (prototypes ±1, noise 0.35: ~20σ margin, solved in ~5 steps; the
        frozen equivalence task) / hard (prototypes ±1/12, unit noise: ~1.7σ
        margin, real class overlap). lr is per task: bundles bake lr 0.05
        but it is a LIVE state buffer, rewritten in the weights blob before
        setupNet (hard uses 0.005 — 0.05·momentum 0.9 oscillated upward on
        the overlapping task in both frameworks).
ENGINES any subset, same model/task/lr/LCG batches for all, curves overlaid:
  bundle   keras-tinygrad, traced in Python on the dev box, embedded here
  tfjs     tf.js webgl (tf.min.js embedded), same init parsed from the
           safetensors into tf.layers; dropout: its own masks
  pyodide  keras-tinygrad traced IN THE TAB: keras 3.15.1 + this repo's
           wheel under Pyodide 0.28.3, in a Web Worker (the page stays
           responsive; the JSON reports the main thread's longest stall
           and heartbeat ticks fired vs expected — on the main thread
           Pyodide fires 0 ticks: fully blocked, JSPI or not).
           The only engine that needs network — Pyodide/PyPI from CDN;
           worker+shims+driver come from the npm package's tracer half
           (js/src/trace/, served at /js/), the wheel from /build/wheels/
MODE    train (N steps) / probe (same batch ×3 with lr=0: dense must give
        identical losses, dropout must give distinct ones — fresh threefry
        masks are the only possible source; bundle + pyodide engines)
VERDICTS (in-page, in the JSON):
  bundle vs tfjs steps 0-1 < 1e-3 (dense only — deterministic)
  pyodide vs bundle: EVERY loss identical (dropout included — same kernels,
  same zeroed rng state)

  python gen_hub.py   ->  hub.html (~3.9 MB; ?report=1&model=&mode=&task=
  &steps=&engines=bundle,tfjs,pyodide drives it headless)
"""

import base64
import importlib.metadata
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, BATCH, DIM, CLASSES = 300, 32, 28 * 28, 10
LR, MOM = 0.05, 0.9

TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>keras-tinygrad — browser demo hub</title>
<style>
 :root { --bg:#fff; --fg:#111; --muted:#666; --border:#888; --c1:#2244cc; --c2:#cc3333; --c3:#228833; --panel:#f4f4f4; }
 @media (prefers-color-scheme: dark) {
  :root:not([data-theme=light]) { --bg:#111; --fg:#ddd; --muted:#999; --border:#555; --c1:#77aaff; --c2:#ff7777; --c3:#66dd66; --panel:#1c1c1c; }
 }
 :root[data-theme=dark] { --bg:#111; --fg:#ddd; --muted:#999; --border:#555; --c1:#77aaff; --c2:#ff7777; --c3:#66dd66; --panel:#1c1c1c; }
 body { font: 14px/1.5 monospace; margin: 2rem auto; max-width: 62rem; padding: 0 1rem; background: var(--bg); color: var(--fg); }
 a { color: var(--c1); }
 fieldset { border: 1px solid var(--border); margin: .5rem 1rem .5rem 0; display: inline-block; vertical-align: top; }
 textarea { width: 100%; height: 9rem; background: var(--panel); color: var(--fg); border: 1px solid var(--border); }
 button { font: inherit; padding: .3rem 1.2rem; background: var(--panel); color: var(--fg); border: 1px solid var(--border); cursor: pointer; }
 #curve { border: 1px solid var(--border); width: 100%; height: 160px; background: var(--panel); }
 #log { background: var(--panel); padding: .5rem; white-space: pre-wrap; }
 .note { color: var(--muted); }
 .c1 { color: var(--c1); } .c2 { color: var(--c2); } .c3 { color: var(--c3); }
 #theme { float: right; }
</style>
<button id="theme" title="theme: auto → dark → light">theme: auto</button>
<h1>keras-tinygrad browser demo hub</h1>
<p><small>build __BUILD_STAMP__ — the single page for everything testable in a browser.
Embedded: two Keras-traced WebGPU bundles + tf.js. The <b>Pyodide engine</b> downloads
Pyodide + wheels (~25 MB, first use) — the only network this page ever does.</small></p>
<p>A Keras training step, traced through the
<a href="https://github.com/c4ffein/keras-tinygrad">tinygrad backend</a> and exported as
WebGPU kernels. Weight buffers update in place: looping <code>step(x, y)</code> <i>is</i>
SGD training. Pick a model, a task, the engines to race, and a mode.</p>

<fieldset><legend>model</legend>
 <label><input type="radio" name="model" value="dense" checked> Dense 784&rarr;128&rarr;10</label><br>
 <label><input type="radio" name="model" value="dropout"> Dense + Dropout(0.3) <span class="note">(on-device threefry masks)</span></label><br>
 <label><input type="radio" name="model" value="custom"> custom <span class="note">(any layers/dims/optimizer JSON — traced live in the tab; pyodide engine only)</span></label><br>
 <textarea id="custom-config" rows="4" style="width:100%;font:inherit" spellcheck="false">{ "input_dim": 784, "classes": 10, "batch": 32,
  "layers": [ { "type": "dense", "units": 64, "activation": "tanh" },
              { "type": "dropout", "rate": 0.2 } ],
  "optimizer": { "lr": 0.02, "momentum": 0.9 } }</textarea>
</fieldset>
<fieldset><legend>task</legend>
 <label><input type="radio" name="task" value="easy" checked> easy <span class="note">(±1 prototypes, noise 0.35: ~20&sigma; margin, solved in ~5 steps; lr 0.05)</span></label><br>
 <label><input type="radio" name="task" value="hard"> hard <span class="note">(±1/12 prototypes, unit noise: ~1.7&sigma; margin, classes overlap; lr 0.005)</span></label>
</fieldset>
<fieldset><legend>engines</legend>
 <label><input type="checkbox" name="engine" value="bundle" checked> <span class="c1">keras-tinygrad — prebuilt bundle</span> <span class="note">(traced in Python, embedded)</span></label><br>
 <label><input type="checkbox" name="engine" value="tfjs"> <span class="c2">tf.js webgl</span> <span class="note">(same init &amp; batches; dropout: its own masks)</span></label><br>
 <label><input type="checkbox" name="engine" value="pyodide"> <span class="c3">keras-tinygrad — traced in-tab (Pyodide)</span> <span class="note">(keras 3.15.1 in WASM; needs network)</span></label>
</fieldset>
<fieldset><legend>mode</legend>
 <label><input type="radio" name="mode" value="train" checked> train <input id="steps" type="number" value="__STEPS__" style="width:5rem;font:inherit"> steps</label><br>
 <label><input type="radio" name="mode" value="probe"> same-batch probe &times;3, lr=0 <span class="note">(dense: identical; dropout: distinct = fresh RNG; bundle + pyodide)</span></label>
</fieldset>
<br><button id="run">run</button>
<pre id="log"></pre>
<canvas id="curve" width="920" height="160"></canvas>
<textarea id="out" readonly placeholder="results JSON appears here"></textarea>
<button id="copy">copy results</button>

<script>__TFJS__</script>
<script type="module">
// The npm package (js/src/index.mjs), embedded verbatim at generation time
// and imported through the same blob-URL trick it uses itself. The hub's
// bundle loading and batch stream go through it — the page IS the e2e of
// the published JS API surface.
const KTG_SRC = "__KTG_JS__";
const ktg = await import(URL.createObjectURL(new Blob([Uint8Array.from(atob(KTG_SRC), c => c.charCodeAt(0))], { type: "text/javascript" })));

const BUNDLES = {
  dense:   { js: "__DENSE_JS__",   w: "__DENSE_W__" },
  dropout: { js: "__DROP_JS__",    w: "__DROP_W__" },
};
const BATCH = __BATCH__, DIM = __DIM__, CLASSES = __CLASSES__, MOM = __MOM__;
// TINYGRAD_VERSION is baked at generation time from the build venv — the same
// tinygrad the embedded wheel's vendored exporter targets. A hardcoded pin
// skewed once (wheel vendored the 0.14 exporter, tab installed 0.13:
// KeyError NUM_CPU_THREADS inside Context, deep in the in-tab trace).
const PYODIDE_VERSION = "0.28.3", TINYGRAD_VERSION = "__TINYGRAD_VERSION__", KERAS_VERSION = "3.15.1";
// The tracer half of the npm package: worker.js + driver.py + shims/ are
// PACKAGE assets (js/src/trace/), served by demo-server's /js/ mapping.
const TRACE_ASSETS = "/js/src/trace/";

// x = pscale*prototype + nscale*gauss. Hardness comes from MARGIN, not input
// magnitude (12x larger inputs diverged to NaN in both frameworks). lr per
// task: rewritten into the live learning_rate state buffer (see setLr).
const TASKS = { easy: { pscale: 1, nscale: 0.35, lr: 0.05 }, hard: { pscale: 1 / 12, nscale: 1, lr: 0.005 } };
const COLORS = { bundle: "--c1", tfjs: "--c2", pyodide: "--c3" };
const LABELS = { bundle: "keras-tinygrad bundle", tfjs: "tf.js webgl", pyodide: "keras-tinygrad in-tab (Pyodide)" };

const $ = (id) => document.getElementById(id);
const log = (m) => { $("log").textContent += m + "\n"; };
const b64 = (s) => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---- theme ---------------------------------------------------------------
const THEMES = ["auto", "dark", "light"];
const applyTheme = (t) => {
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
  $("theme").textContent = "theme: " + t;
  try { localStorage.setItem("hub-theme", t); } catch {}
};
let theme = "auto"; try { theme = localStorage.getItem("hub-theme") || "auto"; } catch {}
applyTheme(theme);
$("theme").onclick = () => { theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]; applyTheme(theme); replot(); };

// ---- model dims -------------------------------------------------------------
// Fixed models use the baked constants; "custom" reads its JSON config (the
// pyodide engine traces whatever it says, so dims/batch are the config's).
const modelCfg = (model) => {
  if (model !== "custom") return { batch: BATCH, dim: DIM, classes: CLASSES, custom: null };
  let cfg;
  try { cfg = JSON.parse($("custom-config").value); } catch (e) { throw new Error("custom config is not valid JSON: " + e.message); }
  for (const k of ["input_dim", "classes", "batch", "layers", "optimizer"])
    if (!(k in cfg)) throw new Error(`custom config: missing "${k}"`);
  return { batch: cfg.batch, dim: cfg.input_dim, classes: cfg.classes, custom: cfg };
};

// ---- deterministic data (bit-identical to the Python twin) -----------------
// The stream itself is the npm package's lcg32 — one implementation, tested
// against the Python twin both in js/test/ (node) and here (browser).
const makeData = (task, { batch, dim, classes }) => {
  const { pscale, nscale } = TASKS[task];
  const g = ktg.lcg32();
  const protos = new Float32Array(classes * dim);
  for (let i = 0; i < protos.length; i++) protos[i] = g.gauss();
  const x = new Float32Array(batch * dim), y = new Int32Array(batch);
  return () => {
    for (let b = 0; b < batch; b++) {
      y[b] = Math.floor(g.rand() * classes);
      for (let i = 0; i < dim; i++) x[b * dim + i] = pscale * protos[y[b] * dim + i] + nscale * g.gauss();
    }
    return [x, y];
  };
};

// ---- safetensors helpers --------------------------------------------------
const parseSafetensors = (bytes) => {
  const n = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  const out = {};
  for (const [k, v] of Object.entries(meta)) {
    if (k === "__metadata__" || v.dtype !== "F32") continue;
    const [a, b] = v.data_offsets;
    out[k] = { shape: v.shape, data: new Float32Array(bytes.slice(8 + n + a, 8 + n + b).buffer) };  // copy: alignment
  }
  return out;
};
// learning_rate is a real state buffer read by the update kernels every
// step: rewriting it in the blob picks the lr; lr=0 freezes the weights.
const setLr = (bytes, lr) => {
  const out = bytes.slice();
  const n = Number(new DataView(out.buffer).getBigUint64(0, true));
  const meta = JSON.parse(new TextDecoder().decode(out.subarray(8, 8 + n)));
  const key = Object.keys(meta).find(k => k.endsWith("learning_rate"));
  if (!key) throw new Error("learning_rate not found in safetensors header");
  new DataView(out.buffer).setFloat32(8 + n + meta[key].data_offsets[0], lr, true);
  return out;
};

let gpu = null;
const getGpu = async () => {
  if (gpu) return gpu;
  if (!navigator.gpu) throw new Error("no WebGPU (Chrome/Edge, secure context)");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  const info = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture || ""}`.trim() : "unknown";
  return (gpu = { device, info });
};
const runnerStep = async (js, weights) => {
  const { device } = await getGpu();
  const { step } = await ktg.loadBundle({ runnerJs: js, weights, device });
  return step;
};

// ---- engines: load(model, lr, mc) -> { step(x,y)->loss, info } --------------
// mc is modelCfg(model); only the pyodide engine can serve "custom" (the
// other two need a pre-baked bundle / a hardcoded tf.layers mirror).
const engines = {
  async bundle(model, lr) {
    if (model === "custom") throw new Error("custom model: only the pyodide engine can trace it (no pre-baked bundle)");
    const js = new TextDecoder().decode(b64(BUNDLES[model].js));
    const t0 = performance.now();
    const step = await runnerStep(js, setLr(b64(BUNDLES[model].w), lr));
    return { step: async (x, y) => (await step(x, y))[0][0], info: { setupMs: +(performance.now() - t0).toFixed(0) } };
  },

  async tfjs(model, lr) {
    if (model === "custom") throw new Error("custom model: only the pyodide engine can trace it");
    const t0 = performance.now();
    await tf.setBackend("webgl"); await tf.ready();
    const st = parseSafetensors(b64(BUNDLES[model].w));
    const find = (suffix) => {
      const hit = Object.entries(st).find(([k]) => k.endsWith(suffix));
      if (!hit) throw new Error("weight not found: " + suffix);
      return hit[1];
    };
    const m = tf.sequential();
    m.add(tf.layers.dense({ inputShape: [DIM], units: 128, activation: "relu" }));
    if (model === "dropout") m.add(tf.layers.dropout({ rate: 0.3 }));
    m.add(tf.layers.dense({ units: CLASSES }));  // logits
    m.layers[0].setWeights([tf.tensor(find("dense/kernel").data, find("dense/kernel").shape), tf.tensor(find("dense/bias").data, find("dense/bias").shape)]);
    const last = m.layers[m.layers.length - 1];
    last.setWeights([tf.tensor(find("dense_1/kernel").data, find("dense_1/kernel").shape), tf.tensor(find("dense_1/bias").data, find("dense_1/bias").shape)]);
    const opt = tf.train.momentum(lr, MOM);
    const step = async (xa, ya) => {
      const xs = tf.tensor2d(xa, [BATCH, DIM]);
      const ys = tf.oneHot(tf.tensor1d(new Int32Array(ya), "int32"), CLASSES);
      const lt = opt.minimize(() => tf.losses.softmaxCrossEntropy(ys, m.apply(xs, { training: true })), true);
      const lv = (await lt.data())[0];
      tf.dispose([xs, ys, lt]);
      return lv;
    };
    return { step, info: { setupMs: +(performance.now() - t0).toFixed(0), backend: tf.getBackend() } };
  },

  async pyodide(model, lr, mc) {
    const cfg = mc?.custom
      ? { ...mc.custom, optimizer: { ...mc.custom.optimizer, lr } }  // lr wins: probe forces 0, train uses the config's
      : {
          input_dim: DIM, classes: CLASSES, batch: BATCH,
          layers: [{ type: "dense", units: 128, activation: "relu" }, ...(model === "dropout" ? [{ type: "dropout", rate: 0.3 }] : [])],
          optimizer: { lr, momentum: MOM },
        };
    // Boot + trace run in a Web Worker; the heartbeat measures how long the
    // MAIN thread was ever blocked meanwhile (was ~10-17 s when Pyodide ran
    // on the main thread; a responsive page stays in the tens of ms).
    const stopGap = gapMeter();
    const eng = await pyodideEngine();
    await eng.boot();
    log(`  pyodide: tracing ${JSON.stringify(cfg.layers)} lr ${lr} in the worker…`);
    const { js, weights, meta, traceMs } = await eng.trace(cfg);
    const stall = stopGap();
    log(`  pyodide: trace+export ${traceMs} ms, ${meta.kernels} kernels; main thread: max stall ${stall.maxGapMs} ms, ${stall.ticks}/${stall.expectedTicks} heartbeat ticks fired`);
    const t1 = performance.now();
    const step = await runnerStep(js, weights);  // lr already in the blob (driver wrote it)
    return { step: async (x, y) => (await step(x, y))[0][0],
      info: { ...eng.timings, traceMs, kernels: meta.kernels, setupMs: +(performance.now() - t1).toFixed(0), mainThread: stall, pythonSide: meta.timings } };
  },
};

// ---- Pyodide engine (the npm package's tracer half, keras-tinygrad/trace) ---
// Imported from its served URL, not embedded: the tracer needs real URLs
// anyway (its worker, driver.py, shims/, the wheel), and this way the e2e
// runs exercise the exact package files a bundler would ship.
let pyodideEngineInstance = null;
async function pyodideEngine() {
  if (pyodideEngineInstance) return pyodideEngineInstance;
  const assetsBase = new URL(TRACE_ASSETS, location.href).href;
  const { createTraceEngine } = await import(new URL("client.mjs", assetsBase).href);
  pyodideEngineInstance = createTraceEngine({
    assetsBase,
    versions: { pyodideVersion: PYODIDE_VERSION, tinygradVersion: TINYGRAD_VERSION, kerasVersion: KERAS_VERSION, wheel: "/build/wheels/__WHEEL__" },
    onStage: (m) => log(`  pyodide: ${m.label} ${m.ms} ms`),
  });
  return pyodideEngineInstance;
}
// Longest main-thread stall while running. A 20 ms heartbeat; the gap is
// taken at every tick AND at stop (an overdue tick is cancelled by
// clearInterval — a meter that only samples inside the callback reports 0
// for a solid block, which is exactly what the first version did). The
// tick count vs the expected count is the second witness: a blocked thread
// fires ~0 ticks.
const gapMeter = () => {
  let last = performance.now(), max = 0, ticks = 0; const t0 = last;
  const id = setInterval(() => { const n = performance.now(); max = Math.max(max, n - last - 20); last = n; ticks++; }, 20);
  return () => {
    clearInterval(id);
    const n = performance.now(); max = Math.max(max, n - last - 20);
    return { maxGapMs: +max.toFixed(0), ticks, expectedTicks: Math.round((n - t0) / 20) };
  };
};

// ---- plotting ----------------------------------------------------------------
let lastCurves = [];
const replot = () => plot(lastCurves);
const plot = (curves) => {
  lastCurves = curves;
  const ctx = $("curve").getContext("2d");
  ctx.clearRect(0, 0, 920, 160);
  const finite = curves.flatMap(c => c.data).filter(Number.isFinite);
  if (!finite.length) return;
  const max = Math.max(...finite);
  ctx.font = "12px monospace";
  curves.forEach((c, i) => {
    ctx.strokeStyle = ctx.fillStyle = cssVar(COLORS[c.engine]);
    ctx.beginPath();
    c.data.forEach((v, j) => ctx.lineTo(j * 920 / c.data.length, 152 - 140 * v / max));
    ctx.stroke();
    ctx.fillText(c.label, 8, 14 + 14 * i);
  });
};

// ---- modes -------------------------------------------------------------------
const selectedEngines = () => [...document.querySelectorAll("input[name=engine]:checked")].map(e => e.value);
const trainOne = async (engine, model, task, steps) => {
  const mc = modelCfg(model);
  const lr = mc.custom ? mc.custom.optimizer.lr : TASKS[task].lr;
  const { step, info } = await engines[engine](model, lr, mc);
  const next = makeData(task, mc);
  const losses = [], times = [];
  for (let i = 0; i < steps; i++) {
    const [x, y] = next();
    const t = performance.now();
    losses.push(await step(x, y));
    times.push(performance.now() - t);
    if (i % 50 === 0) log(`  ${engine} step ${i}  loss ${losses[i].toFixed(4)}`);
  }
  const steady = times.slice(Math.min(10, times.length - 1)).sort((a, b) => a - b);
  return { info, steadyMedianMs: +steady[Math.floor(steady.length / 2)].toFixed(2), firstStepMs: +times[0].toFixed(1),
    first10: +mean(losses.slice(0, 10)).toFixed(4), last10: +mean(losses.slice(-10)).toFixed(4),
    allFinite: losses.every(Number.isFinite), losses: losses.map(v => +v.toFixed(4)), rawLosses: losses };
};

const modes = {
  async train(model, task, engineList, steps) {
    const results = {}, curves = [];
    for (const engine of engineList) {
      log(`— ${LABELS[engine]} —`);
      const r = await trainOne(engine, model, task, steps);
      results[engine] = r;
      curves.push({ engine, label: `${LABELS[engine]}  ${r.steadyMedianMs} ms/step  last10 ${r.last10}`, data: r.losses });
      plot(curves);
    }
    const checks = {};
    if (results.bundle && results.tfjs) {
      if (model === "dense") {
        checks.bundleVsTfjsSteps01 = Math.abs(results.bundle.rawLosses[0] - results.tfjs.rawLosses[0]) < 1e-3
                                  && Math.abs(results.bundle.rawLosses[1] - results.tfjs.rawLosses[1]) < 1e-3;
        log(`verdict bundle vs tf.js, steps 0-1 < 1e-3: ${checks.bundleVsTfjsSteps01 ? "PASS" : "FAIL"}`);
      } else log(`bundle vs tf.js: dropout masks are independent per framework — compare tails only (${results.bundle.last10} vs ${results.tfjs.last10})`);
    }
    if (results.bundle && results.pyodide) {
      const maxDiff = Math.max(...results.bundle.rawLosses.map((v, i) => Math.abs(v - results.pyodide.rawLosses[i])));
      checks.pyodideVsBundleMaxDiff = maxDiff;
      checks.pyodideVsBundleIdentical = maxDiff === 0;
      log(`verdict in-tab trace vs prebuilt bundle, all ${steps} losses: max |diff| ${maxDiff} → ${maxDiff === 0 ? "IDENTICAL" : maxDiff < 1e-5 ? "equal to float32 rounding" : "DIFFERENT"}`);
    }
    for (const r of Object.values(results)) delete r.rawLosses;
    const mc = modelCfg(model);
    return { mode: "train", model, task, lr: mc.custom ? mc.custom.optimizer.lr : TASKS[task].lr,
      ...(mc.custom ? { config: mc.custom } : {}), steps, engines: results, checks };
  },

  async probe(model, task, engineList) {
    const results = {};
    const mc = modelCfg(model);
    for (const engine of engineList.filter(e => e !== "tfjs")) {
      log(`— ${LABELS[engine]} —`);
      const { step } = await engines[engine](model, 0, mc);  // lr=0: weights frozen
      const x = new Float32Array(mc.batch * mc.dim).fill(0.5), y = new Int32Array(mc.batch).map((_, i) => i % mc.classes);
      const losses = [];
      for (let i = 0; i < 3; i++) { losses.push(await step(x, y)); log(`  probe ${i}  loss ${losses[i]}`); }
      const distinct = new Set(losses.map(v => v.toFixed(6))).size;
      // custom: any dropout layer in the config means fresh masks expected
      const hasRng = model === "dropout" || (mc.custom && mc.custom.layers.some(l => l.type === "dropout"));
      const pass = hasRng ? distinct === 3 : distinct === 1;
      log(`  ${hasRng ? "fresh threefry masks: expected distinct" : "no RNG: expected identical"} → ${distinct}/3 ${pass ? "PASS" : "FAIL"}`);
      results[engine] = { losses, distinct, pass, allFinite: losses.every(Number.isFinite) };
    }
    return { mode: "probe", model, task, ...(mc.custom ? { config: mc.custom } : {}), engines: results };
  },
};

// ---- run --------------------------------------------------------------------
const report = (payload) => fetch("/report", { method: "POST", body: JSON.stringify({ id: "hub", ...payload }) }).catch(() => {});
$("run").onclick = async () => {
  $("log").textContent = ""; $("run").disabled = true;
  const model = document.querySelector("input[name=model]:checked").value;
  const task = document.querySelector("input[name=task]:checked").value;
  const mode = document.querySelector("input[name=mode]:checked").value;
  const steps = +$("steps").value || __STEPS__;
  const engineList = selectedEngines();
  let res = { ok: false, ua: navigator.userAgent };
  try {
    if (!engineList.length) throw new Error("select at least one engine");
    const { info } = await getGpu();
    res = { ok: true, ua: navigator.userAgent, adapter: info, ...(await modes[mode](model, task, engineList, steps)) };
  } catch (e) { res.error = String(e && e.stack || e); log("FAILED: " + res.error); }
  $("out").value = JSON.stringify(res);
  if (location.search.includes("report")) report(res);
  $("run").disabled = false;
};
$("copy").onclick = () => navigator.clipboard.writeText($("out").value);

// headless / CI: ?report=1&model=dense&task=easy&mode=train&steps=100&engines=bundle,tfjs,pyodide
const p = new URLSearchParams(location.search);
if (p.get("model")) document.querySelector(`input[name=model][value=${p.get("model")}]`).checked = true;
if (p.get("config")) $("custom-config").value = p.get("config");  // pairs with model=custom (URI-encoded JSON)
if (p.get("task")) document.querySelector(`input[name=task][value=${p.get("task")}]`).checked = true;
if (p.get("steps")) $("steps").value = p.get("steps");
if (p.get("mode") === "tfjs") { p.set("mode", "train"); p.set("engines", "bundle,tfjs"); }  // old URL form
if (p.get("mode")) document.querySelector(`input[name=mode][value=${p.get("mode")}]`).checked = true;
if (p.get("engines")) {
  const want = new Set(p.get("engines").split(","));
  document.querySelectorAll("input[name=engine]").forEach(e => { e.checked = want.has(e.value); });
}
if (location.search.includes("report")) $("run").click();
</script>
"""


BUILD = os.path.join(HERE, "build")


def b64(path):
    full = os.path.join(BUILD, path)
    if not os.path.exists(full):
        raise SystemExit(
            f"{path} missing — generated bundles are not in git; run `make browser-assets` from the repo root"
        )
    with open(full, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    import datetime

    tfjs_path = os.path.join(BUILD, "tf.min.js")
    if not os.path.exists(tfjs_path):
        raise SystemExit("tf.min.js missing — run scripts/fetch_tfjs.sh (pinned tfjs 4.22.0, sha256-verified)")
    with open(tfjs_path) as f:
        tfjs = f.read()
    # The npm package source, embedded so the hub exercises the shipped API.
    with open(os.path.join(HERE, "..", "src", "index.mjs"), "rb") as f:
        ktg_js = base64.b64encode(f.read()).decode()
    html = (
        TEMPLATE.replace("__TFJS__", tfjs)
        .replace("__KTG_JS__", ktg_js)
        .replace("__DENSE_JS__", b64("out.js"))
        .replace("__DENSE_W__", b64("out.safetensors"))
        .replace("__DROP_JS__", b64("dropout.js"))
        .replace("__DROP_W__", b64("dropout.safetensors"))
        .replace("__STEPS__", str(STEPS))
        .replace("__BATCH__", str(BATCH))
        .replace("__DIM__", str(DIM))
        .replace("__CLASSES__", str(CLASSES))
        .replace("__MOM__", str(MOM))
        .replace("__BUILD_STAMP__", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__TINYGRAD_VERSION__", importlib.metadata.version("tinygrad"))
        # the Pyodide engine installs THIS tree's wheel: name it from the
        # installed package version, never a hardcoded one (a stale 0.1.0
        # name silently resolved to the PyPI wheel without webgpu.py)
        .replace("__WHEEL__", f"keras_tinygrad-{importlib.metadata.version('keras_tinygrad')}-py3-none-any.whl")
    )
    path = os.path.join(BUILD, "hub.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
