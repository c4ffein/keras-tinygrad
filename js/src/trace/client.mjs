// keras-tinygrad/trace — the in-tab tracer half of the package: real Keras
// (keras + this repo's Python wheel) runs under Pyodide in a module Web
// Worker, traces a model described by a JSON config on the GPU-less
// NULL:WGSL device, and hands back the same {runnerJs, weights} bundle
// shape the Python-side exporter produces — which loadBundle (the runner
// half, ../index.mjs) turns into a training step(x, y).
//
// This entry lives at "keras-tinygrad/trace" (not in index.mjs) because the
// runner half must stay dependency-free and embeddable as a single blob —
// while the tracer inherently needs real URLs: a Worker to spawn, driver.py
// and shims/ to fetch (assets shipped in this package, siblings of this
// file), and Pyodide + keras + tinygrad from CDNs (~25 MB, first boot).
//
//   import { traceModel } from "keras-tinygrad/trace";
//   const { step, meta } = await traceModel(
//     { input_dim: 784, classes: 10, batch: 32,
//       layers: [{ type: "dense", units: 128, activation: "relu" },
//                { type: "dropout", rate: 0.3 }],
//       optimizer: { lr: 0.05, momentum: 0.9 } },
//     { versions: { ...DEFAULT_VERSIONS, wheel: "/build/wheels/<name>.whl" } },
//   );
//   while (training) loss = (await step(x, y))[0][0];

import { loadBundle } from "../index.mjs";

// The versions the worker installs. tinygradVersion must match the pin the
// wheel's vendored exporter targets (a 0.13 install against the 0.14
// exporter died on a missing ContextVar — keep these moving together).
// `wheel` has no default: it names THIS repo's Python wheel, whose URL is
// deployment-specific (the demo hub bakes it; consumers point at their own
// copy). It resolves against assetsBase, absolute paths/URLs win.
export const DEFAULT_VERSIONS = {
  pyodideVersion: "0.28.3",
  tinygradVersion: "0.14.0",
  kerasVersion: "3.15.1",
};

/** Low-level engine: one booted Pyodide worker, many traces.
 * `assetsBase` is where worker.js, driver.py and shims/ are served from —
 * defaults to this module's own directory, which is right whenever the
 * package is imported from a real URL (bundler, CDN, or a server exposing
 * node_modules); pass it explicitly when this module was inlined/blobbed. */
export function createTraceEngine({ assetsBase, versions, onStage } = {}) {
  const base = assetsBase ? new URL(assetsBase, globalThis.location?.href).href : new URL("./", import.meta.url).href;
  const v = { ...DEFAULT_VERSIONS, ...versions };
  if (!v.wheel) throw new Error("versions.wheel is required: the URL of the keras-tinygrad Python wheel to install");
  const timings = {};
  let worker = null,
    booted = null;
  const rpc = (msg) =>
    new Promise((resolve, reject) => {
      const handler = (e) => {
        const m = e.data;
        if (m.type === "stage") {
          timings[m.key] = m.ms;
          onStage?.(m);
          return;
        }
        worker.removeEventListener("message", handler);
        if (m.type === "error") reject(new Error(`[pyodide worker ${m.op}] ${m.error}`));
        else resolve(m);
      };
      worker.addEventListener("message", handler);
      worker.postMessage(msg);
    });
  return {
    timings,
    boot() {
      if (!booted) {
        worker = new Worker(new URL("worker.js", base), { type: "module" });
        worker.onerror = (e) => console.error("pyodide worker error", e);
        booted = rpc({ type: "boot", assetsBase: base, ...v }).then((m) => {
          timings.versions = m.versions;
          return m;
        });
      }
      return booted;
    },
    async trace(config) {
      await this.boot();
      return rpc({ type: "trace", config });
    },
  };
}

/** High-level one-shot: boot (cached per engine you pass, fresh otherwise),
 * trace `config`, and load the result into a WebGPU training step. */
export async function traceModel(config, { engine, assetsBase, versions, onStage, device } = {}) {
  const eng = engine ?? createTraceEngine({ assetsBase, versions, onStage });
  const { js, weights, meta, traceMs } = await eng.trace(config);
  const { step, device: dev } = await loadBundle({ runnerJs: js, weights, device });
  return { step, device: dev, meta, traceMs, timings: eng.timings, engine: eng };
}
