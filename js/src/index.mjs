/**
 * keras-tinygrad (JS) — load and drive training-step bundles exported by
 * the Python package (a Keras model traced through the tinygrad backend
 * into WGSL kernels + a safetensors weight file).
 *
 * API status: 0.0.x — UNSTABLE, tracks the export format of the Python
 * side's experiments. The roadmap (in-tab tracing of arbitrary Keras
 * models via Pyodide) will grow here; today this is the runner half.
 */

/** Import an exported runner module from its source text or bytes,
 * without needing it hosted anywhere (blob URL import). */
export async function importRunner(jsSource) {
  const text = typeof jsSource === "string" ? jsSource : new TextDecoder().decode(jsSource);
  const mod = await import(
    /* @vite-ignore */ URL.createObjectURL(new Blob([text], { type: "text/javascript" }))
  );
  return mod.default;
}

/** Load a bundle into a step function: step(...typedArrays) -> outputs.
 * `runnerJs` is the runner's SOURCE (a string); `weights` must be BINARY —
 * a Uint8Array or ArrayBuffer (`res.arrayBuffer()`, never `res.text()`:
 * `new Uint8Array(string)` coerces the string to a LENGTH and yields
 * zeros, crashing deep inside the runner's safetensors parse).
 * `device` defaults to a fresh WebGPU device. */
export async function loadBundle({ runnerJs, weights, device } = {}) {
  if (!device) {
    if (!navigator.gpu) throw new Error("WebGPU unavailable (secure context + Chrome/Edge needed)");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  }
  const runner = await importRunner(runnerJs);
  let bytes;
  if (weights instanceof Uint8Array) bytes = weights;
  else if (weights instanceof ArrayBuffer) bytes = new Uint8Array(weights);
  else {
    // Strings become a LENGTH; other TypedArrays copy element VALUES as
    // bytes. Both are silent corruption — reject at the boundary.
    throw new TypeError(
      `weights must be a Uint8Array or ArrayBuffer, got ${typeof weights === "string" ? "a string (use res.arrayBuffer(), not res.text())" : Object.prototype.toString.call(weights)}`,
    );
  }
  const step = await runner.setupNet(device, bytes);
  return { step, device };
}

/** Fetch-and-load convenience for hosted bundles. */
export async function fetchBundle(jsUrl, weightsUrl, { device } = {}) {
  const [runnerJs, weights] = await Promise.all([
    fetch(jsUrl).then((r) => r.text()),
    fetch(weightsUrl).then((r) => r.arrayBuffer()),
  ]);
  return loadBundle({ runnerJs, weights, device });
}

/** Deterministic 32-bit LCG (exact integer semantics via Math.imul) —
 * the batch/mask stream helper used by the reference demos, exported so
 * JS and Python sides can train on identical bytes. */
export function lcg32(seed = 1234567) {
  let s = seed >>> 0;
  let spare = null;
  const rand = () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 4294967296);
  const gauss = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { rand, gauss };
}
