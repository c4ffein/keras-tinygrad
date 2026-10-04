"""The WebGPU export chain, proven end-to-end in a browser — five flows,
asserted from the hub's own results JSON. Subject-named like its siblings:
what these tests indict on failure is the EXPORT CHAIN (webgpu.py, the
vendored exporter, the backend's traced train step, the npm runner/tracer),
with the browser as the required runtime, not the subject.

What runs: `js/demo/server.mjs` (bun) serves the generated hub on a
free port; headless Chromium with SwiftShader WebGPU opens
`hub.html?report=1&...` per flow; the page posts its results JSON to
/report and this suite asserts the page's `checks`/`pass` fields. Because
gen_hub.py embeds js/src/index.mjs and routes ALL bundle loading and batch
streams through it, every one of these runs is also an end-to-end of the
npm package's API surface on real (software) WebGPU.

Prerequisites — loud, not skipped (a missing prerequisite would silently
hollow out the suite): generated assets (`make browser-assets`; `make e2e`
chains it), bun, and a Chromium binary (playwright's cache, or $CHROME).
The exception is the pyodide flow: it needs the network (Pyodide + PyPI
CDNs), so it is opt-in via KERAS_TINYGRAD_E2E_PYODIDE=1 and SKIPS
otherwise — CI sets it, a plane ride doesn't.

Caps (tests/_limits.py, the rule since the 2026-09-21 stall): the bun
server runs under the full child limits; Chromium under the CPU-time cap
and nice only — it cannot start under any address-space limit (see
`child_limits`). A hung page therefore ends as SIGXCPU, not as a box
pinned until someone notices. `make browser-assets` (chained by `make
e2e`) runs outside this file and is uncapped: it is the same trace the
loader receipts already cap, run once, not a test.
"""

import glob
import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
import urllib.request

import pytest
from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent
DEMO_DIR = REPO / "js" / "demo"
CHROME_FLAGS = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-unsafe-webgpu",
    "--use-webgpu-adapter=swiftshader",
    "--enable-unsafe-swiftshader",
]


def _chrome():
    if path := os.environ.get("CHROME"):
        return path
    hits = glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome"))
    assert hits, "no Chromium: `playwright install chromium` or set CHROME=/path/to/chrome"
    return sorted(hits)[-1]


@pytest.fixture(scope="module")
def server():
    for name in ("hub.html", "out.js", "dropout.js"):
        assert (DEMO_DIR / "build" / name).exists(), (
            f"js/demo/build/{name} missing — run `make browser-assets` (or `make e2e`)"
        )
    # the pyodide flow imports the npm package's tracer half through the server
    assert (REPO / "js" / "src" / "trace" / "client.mjs").exists(), "js/src/trace/client.mjs missing"
    assert shutil.which("bun"), "bun is required to serve the hub (https://bun.sh)"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        ["bun", "server.mjs"],
        cwd=DEMO_DIR,
        env=dict(os.environ, PORT=str(port)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=child_limits,
    )
    try:
        deadline = time.time() + 10
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/results", timeout=1)
                break
            except OSError:
                assert time.time() < deadline, "demo server never came up"
                time.sleep(0.2)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def run_hub(port, timeout=300, **params):
    """Open the hub headless with ?report=1&..., return the posted results.

    Headless Chromium never exits on its own after the page settles, so
    /results is polled WHILE it runs and the process is killed as soon as
    the report lands (or the deadline passes — the loud path)."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    mark = time.time() * 1000
    proc = subprocess.Popen(
        [_chrome(), *CHROME_FLAGS, f"http://127.0.0.1:{port}/hub.html?report=1&{query}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: child_limits(address_space=False),
    )
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/results", timeout=5) as res:
                results = json.load(res)
            if fresh := {k: v for k, v in results.items() if float(k) >= mark}:
                report = fresh[max(fresh, key=float)]
                assert report.get("ok"), f"page failed for {query}: {report.get('error')}"
                return report
            assert proc.poll() is None, f"chromium exited (code {proc.returncode}) before reporting for {query}"
            time.sleep(2)
        raise AssertionError(f"no report within {timeout}s for {query} — WebGPU flags / page hang")
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_dense_train_and_tfjs_equivalence(server):
    """One run, two proofs: the bundle TRAINS (loss falls, all finite), and
    its first two steps match tf.js on identical init + batches — the
    pre-update computation-identity window (post-update float32 chaos is
    expected to diverge; both tails must still fall)."""
    r = run_hub(server, model="dense", task="easy", mode="train", steps=60, engines="bundle,tfjs")
    for name in ("bundle", "tfjs"):
        eng = r["engines"][name]
        assert eng["allFinite"], f"{name}: NaN/Inf in the loss curve"
        assert eng["last10"] < eng["first10"] * 0.5, f"{name}: loss did not fall ({eng['first10']} -> {eng['last10']})"
    assert r["checks"]["bundleVsTfjsSteps01"] is True, "bundle and tf.js disagree on steps 0-1"


def test_dense_probe_is_deterministic(server):
    """Negative control: no RNG in the dense bundle, lr=0 — the same batch
    three times must produce three IDENTICAL losses."""
    r = run_hub(server, model="dense", task="easy", mode="probe", engines="bundle")
    eng = r["engines"]["bundle"]
    assert eng["distinct"] == 1 and eng["pass"], f"dense probe not deterministic: {eng['losses']}"


def test_dropout_probe_draws_fresh_masks(server):
    """The device-RNG proof (docs/device-rng.md): lr=0 freezes the weights,
    so three DISTINCT losses on the same batch can only come from fresh
    on-device threefry dropout masks."""
    r = run_hub(server, model="dropout", task="easy", mode="probe", engines="bundle")
    eng = r["engines"]["bundle"]
    assert eng["distinct"] == 3 and eng["pass"], f"dropout masks not fresh: {eng['losses']}"


def test_dense_train_matches_local_python(server):
    """The seal against LOCAL keras-tinygrad: js/demo/build/ref_curve.json is
    the same model (same exported init bytes) trained on the same lcg32
    batch stream by the Python backend on CPU (gen_ref_curve.py, run at
    asset-build time). Steps 0-1 must match tightly — the pre-update
    computation-identity window — and both tails must fall; per-step
    identity beyond that is not expected (float32 op-order chaos
    compounds after the first weight update)."""
    ref_path = DEMO_DIR / "build" / "ref_curve.json"
    assert ref_path.exists(), "ref_curve.json missing — run `make browser-assets`"
    ref = json.loads(ref_path.read_text())["losses"]
    r = run_hub(server, model="dense", task="easy", mode="train", steps=60, engines="bundle")
    browser = r["engines"]["bundle"]["losses"]
    for i in (0, 1):
        assert abs(browser[i] - ref[i]) < 1e-3, (
            f"step {i}: browser {browser[i]} vs local python {ref[i]} — pre-update identity broken"
        )
    assert sum(browser[-10:]) / 10 < 0.05 and sum(ref[50:60]) / 10 < 0.05, (
        f"tails did not both converge: browser last10 {sum(browser[-10:]) / 10}, ref steps 50-60 {sum(ref[50:60]) / 10}"
    )


def test_pyodide_trace_matches_bundle(server):
    """The in-tab engine (keras + this wheel under Pyodide) traces the same
    model live and must produce EVERY loss identical to the prebuilt
    bundle — same kernels, same zeroed rng state. Needs the network."""
    if os.environ.get("KERAS_TINYGRAD_E2E_PYODIDE") != "1":
        pytest.skip("needs network (Pyodide/PyPI CDNs) — set KERAS_TINYGRAD_E2E_PYODIDE=1")
    r = run_hub(server, model="dense", task="easy", mode="train", steps=20, engines="bundle,pyodide", timeout=560)
    assert r["checks"]["pyodideVsBundleIdentical"] is True, (
        f"in-tab trace diverged from the bundle: max |diff| {r['checks']['pyodideVsBundleMaxDiff']}"
    )
