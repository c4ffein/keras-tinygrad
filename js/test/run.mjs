// Node-land tests for the npm package — framework-free (runs under bun or
// node >= 20: `bun test/run.mjs` / `node test/run.mjs`). What can be proven
// without a GPU is proven here: export surface, the LCG's bit-parity with
// the Python twin, and loadBundle's input contract through a stub runner.
// The real end-to-end — these same functions driving WebGPU training in a
// browser — is tests/test_webgpu_export.py at the repo root: the demo hub imports THIS
// module (embedded by gen_hub.py) for all bundle loading and batch streams.
import { lcg32, importRunner, loadBundle, fetchBundle } from "../src/index.mjs";
import { createTraceEngine, traceModel, DEFAULT_VERSIONS } from "../src/trace/client.mjs";

let failed = 0;
const check = (name, fn) =>
  Promise.resolve()
    .then(fn)
    .then(() => console.log("ok -", name))
    .catch((e) => { failed++; console.error("FAIL -", name, "\n   ", e && e.message ? e.message : e); });
const assert = (cond, msg) => { if (!cond) throw new Error(msg); };

await check("exports are functions", () => {
  for (const [n, f] of Object.entries({ lcg32, importRunner, loadBundle, fetchBundle }))
    assert(typeof f === "function", `${n} is not a function`);
});

await check("lcg32 rand matches the Python twin bit-for-bit", () => {
  // Baked from the Python twin (js/demo/gen_ref_curve.py BatchGen), full repr():
  //   g = BatchGen(); [g.rand() for _ in range(4)]
  const want = [0.9718729080632329, 0.25026380200870335, 0.7882686448283494, 0.7235767922829837];
  const g = lcg32();
  want.forEach((w, i) => { const v = g.rand(); assert(v === w, `draw ${i}: ${v} != ${w}`); });
});

await check("lcg32 gauss matches the Python twin bit-for-bit", () => {
  //   g = BatchGen(); [g.gauss() for _ in range(4)]  (Box-Muller incl. spare)
  const want = [-0.0003959364166141248, 0.23887301107141348, -0.113997570554157, -0.6803214010233163];
  const g = lcg32();
  want.forEach((w, i) => { const v = g.gauss(); assert(v === w, `draw ${i}: ${v} != ${w}`); });
});

await check("lcg32 seeds are honored and independent", () => {
  const a = lcg32(42), b = lcg32(42), c = lcg32(43);
  const va = a.rand(), vb = b.rand(), vc = c.rand();
  assert(va === vb, "same seed diverged");
  assert(va !== vc, "different seeds collided");
});

const STUB_RUNNER = "export default { setupNet: async (device, bytes) => async () => [[bytes.length]] };";

await check("importRunner round-trips a runner module from source text", async () => {
  const runner = await importRunner(STUB_RUNNER);
  assert(typeof runner.setupNet === "function", "setupNet missing after blob import");
});

await check("loadBundle accepts Uint8Array and ArrayBuffer weights", async () => {
  const device = {}; // stub: setupNet is ours, no WebGPU involved
  for (const weights of [new Uint8Array(7), new ArrayBuffer(7)]) {
    const { step } = await loadBundle({ runnerJs: STUB_RUNNER, weights, device });
    assert((await step())[0][0] === 7, "weights bytes did not reach setupNet intact");
  }
});

await check("loadBundle rejects string weights loudly (the res.text() trap)", async () => {
  let threw = null;
  await loadBundle({ runnerJs: STUB_RUNNER, weights: "not bytes", device: {} }).catch((e) => (threw = e));
  assert(threw instanceof TypeError, "string weights must raise TypeError, not corrupt silently");
  assert(String(threw.message).includes("arrayBuffer"), "the error should point at res.arrayBuffer()");
});

await check("trace entry exports and versions are sane", () => {
  for (const [n, f] of Object.entries({ createTraceEngine, traceModel }))
    assert(typeof f === "function", `${n} is not a function`);
  for (const k of ["pyodideVersion", "tinygradVersion", "kerasVersion"])
    assert(/^\d+\.\d+\.\d+$/.test(DEFAULT_VERSIONS[k]), `DEFAULT_VERSIONS.${k} not a version: ${DEFAULT_VERSIONS[k]}`);
});

await check("createTraceEngine refuses a missing wheel URL loudly", () => {
  let threw = null;
  try { createTraceEngine({ versions: {} }); } catch (e) { threw = e; }
  assert(threw && String(threw.message).includes("wheel"), "must demand versions.wheel up front");
});

await check("tracer's tinygrad default matches the Python pin (repo context)", async () => {
  // The 0.13-vs-0.14 skew bug, as a test: DEFAULT_VERSIONS.tinygradVersion
  // must satisfy pyproject's tinygrad floor. Skipped outside the repo
  // (published tarball has no pyproject above it).
  let toml;
  try { toml = await (await import("node:fs/promises")).readFile(new URL("../../pyproject.toml", import.meta.url), "utf8"); }
  catch { console.log("   (no pyproject.toml above — standalone tarball, skipped)"); return; }
  const pin = toml.match(/tinygrad\s*>=\s*(\d+)\.(\d+)/);
  assert(pin, "no tinygrad>= pin found in pyproject.toml");
  const [maj, min] = DEFAULT_VERSIONS.tinygradVersion.split(".").map(Number);
  assert(maj === +pin[1] && min === +pin[2],
    `DEFAULT_VERSIONS.tinygradVersion ${DEFAULT_VERSIONS.tinygradVersion} != pyproject pin >=${pin[1]}.${pin[2]} — keep them moving together`);
});

if (failed) { console.error(`\n${failed} test(s) FAILED`); process.exit(1); }
console.log("\nall js tests passed");
