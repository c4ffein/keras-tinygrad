// Module Web Worker: owns the Pyodide runtime. Boots once (runtime, packages,
// wheels, shims, `import keras`), then traces Keras models from JSON configs
// on request. Runs off the main thread so the page never freezes during the
// ~4 s `import keras` or a ~5 s trace. The emitted weights ArrayBuffer is
// TRANSFERRED to the caller, the JS runner source is copied.
//
// protocol (postMessage):
//   -> {type:"boot", assetsBase, pyodideVersion, tinygradVersion, kerasVersion, wheel}
//   <- {type:"stage", key, label, ms} ...   <- {type:"booted", versions}
//   -> {type:"trace", config}               <- {type:"traced", js, weights, meta, traceMs}
//   <- {type:"error", op, error} on failure
let py = null;
const post = (m, transfer) => self.postMessage(m, transfer || []);
const stage = async (label, key, fn) => {
  const t0 = performance.now();
  const out = await fn();
  post({ type: "stage", key, label, ms: +(performance.now() - t0).toFixed(0) });
  return out;
};

async function boot({ assetsBase, pyodideVersion, tinygradVersion, kerasVersion, wheel }) {
  if (py) return;
  const { loadPyodide } = await stage("runtime module", "moduleMs",
    () => import(`https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/pyodide.mjs`));
  const inst = await stage("runtime boot", "bootMs",
    () => loadPyodide({ indexURL: `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/` }));
  await stage("pyodide packages (numpy h5py rich packaging sqlite3 micropip)", "pkgsMs",
    () => inst.loadPackage(["micropip", "sqlite3", "numpy", "h5py", "rich", "packaging"]));
  await stage(`wheels (tinygrad ${tinygradVersion}, keras ${kerasVersion}, absl-py, namex, keras-tinygrad from this tree)`, "wheelsMs", () =>
    inst.runPythonAsync(`
import micropip
await micropip.install(["absl-py", "namex", "tinygrad==${tinygradVersion}"])
await micropip.install("keras==${kerasVersion}", deps=False)   # optree/ml-dtypes: no wasm wheels -> shims
await micropip.install("${new URL(wheel, assetsBase).href}", deps=False)
`));
  await stage("shims + driver into the Pyodide FS", "shipMs", async () => {
    inst.FS.mkdirTree("/work/shims/optree");
    for (const [src, dst] of [
      ["shims/optree/__init__.py", "/work/shims/optree/__init__.py"], ["shims/optree/utils.py", "/work/shims/optree/utils.py"],
      ["shims/ml_dtypes.py", "/work/shims/ml_dtypes.py"], ["driver.py", "/work/driver.py"],
    ]) {
      const r = await fetch(new URL(src, assetsBase));
      if (!r.ok) throw new Error(`fetch ${src}: ${r.status}`);
      inst.FS.writeFile(dst, await r.text());
    }
  });
  await stage("import keras_tinygrad + keras (WASM)", "importKerasMs", () => inst.runPythonAsync(`
import os, sys
os.environ["DEV"] = "NULL:WGSL"; os.environ["NULL_ALLOW_COPYOUT"] = "1"
sys.path.insert(0, "/work/shims"); sys.path.insert(0, "/work")
import driver
`));
  py = inst;
}

async function trace(config) {
  if (!py) throw new Error("worker not booted");
  const t0 = performance.now();
  py.globals.set("config_json", JSON.stringify(config));
  const res = await py.runPythonAsync(`
import json, driver
out = driver.build(config_json)
{"js": out["js"], "weights": out["weights"], "meta": json.dumps(out["meta"])}
`);
  const js = res.get("js");
  const weights = res.get("weights").toJs();  // bytes -> Uint8Array (fresh buffer)
  const meta = JSON.parse(res.get("meta"));
  res.destroy();
  return { js, weights, meta, traceMs: +(performance.now() - t0).toFixed(0) };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "boot") {
      await boot(msg);
      const versions = await py.runPythonAsync(`import keras, sys, optree; f"python {sys.version.split()[0]} keras {keras.__version__} backend {keras.backend.backend()} optree {optree.__version__}"`);
      post({ type: "booted", versions });
    } else if (msg.type === "trace") {
      const r = await trace(msg.config);
      post({ type: "traced", ...r }, [r.weights.buffer]);
    } else throw new Error(`unknown message ${msg.type}`);
  } catch (err) {
    post({ type: "error", op: msg.type, error: String((err && err.stack) || err) });
  }
};
