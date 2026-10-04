"""Receipts for the 2026-09-22 export-chain fixes (commit 4742f0e, HANDOFF
"tinygrad 0.13 -> 0.14"), each the check that would have caught its bug:

- the vendored exporter's `compile_net` keyed kernels by tinygrad's
  shape-derived names; 0.14 reuses one name for DISTINCT programs in a single
  trace, so later bodies silently replaced earlier ones and the WebGPU bundle
  died at setup (bind-group layout vs shader binding count). Only the browser
  e2e suite saw it. Here the m0 dense model is traced on NULL:WGSL in a
  subprocess (env must be set before tinygrad is imported) and the raw
  per-call (name, body) pairs are compared to what compile_net emits.
- the Pyodide tracer must zero `PARALLEL` before tinygrad is imported (0.14
  compiles through a multiprocessing pool; Pyodide has no _multiprocessing).
- the hub bakes the tinygrad version from the build venv, never a literal
  (a 0.13 pin against the 0.14 exporter was the second Pyodide failure).
"""

import ast
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent

_TRACE = """
import os, json
os.environ["DEV"] = "NULL:WGSL"
os.environ["NULL_ALLOW_COPYOUT"] = "1"
os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"
import keras_tinygrad
from keras_tinygrad._vendor import export_model as _em
from keras_tinygrad.webgpu import export_train_step
import keras
from tinygrad import Device

seen = {}

def spying_compile_net(linear, output_bufs):
    # The raw pairs, exactly as compile_net derives them before its rename.
    raw = []
    for call in _em.iter_kernel_calls(linear):
        arg_uops = [b for b in call.src[1:] if not b.is_bound_var]
        prg = _em.to_program(call.src[0], Device[arg_uops[0].device].renderer)
        raw.append((prg.arg.function_name, prg.src[2].arg))
    functions, statements, bufs, bufs_to_save = real_compile_net(linear, output_bufs)
    seen["raw_calls"] = len(raw)
    seen["raw_distinct_names"] = len({n for n, _ in raw})
    seen["raw_distinct_bodies"] = len({b for _, b in raw})
    seen["emitted_names"] = list(functions)
    seen["emitted_distinct_bodies"] = len(set(functions.values()))
    seen["statement_names"] = [s[0] for s in statements]
    return functions, statements, bufs, bufs_to_save

real_compile_net, _em.compile_net = _em.compile_net, spying_compile_net

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.layers.Input((784,)), keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10)])
out = export_train_step(model, keras.optimizers.SGD(0.05, momentum=0.9),
                        keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None),
                        batch_size=32, input_shape=(784,), learning_rate=0.05)
seen["kernels_in_js"] = out["meta"]["kernels"]
print(json.dumps(seen))
"""


def test_compile_net_keeps_every_distinct_kernel_under_name_collisions():
    """Names in the emitted table are unique, every statement resolves to a
    body, and no body was dropped: distinct emitted bodies == distinct raw
    bodies. The precondition (tinygrad reused a name for distinct programs
    in this trace) is asserted too, so a reverted fix cannot pass by luck —
    if a future tinygrad stops colliding, the assertion message says so."""
    env = dict(os.environ, KERAS_BACKEND="tinygrad", KERAS_TINYGRAD_NO_VERSION_WARNING="1")
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_TRACE)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=600,
        preexec_fn=child_limits,
    )
    assert out.returncode == 0, f"trace subprocess failed:\n{out.stdout}\n{out.stderr}"
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen["raw_distinct_names"] < seen["raw_distinct_bodies"], (
        "no kernel-name collision in this trace any more — the fix is untested here; "
        f"raw names {seen['raw_distinct_names']} vs bodies {seen['raw_distinct_bodies']} "
        "(tinygrad changed its naming: find a colliding trace or retire this receipt)"
    )
    names = seen["emitted_names"]
    assert len(names) == len(set(names)), f"duplicate kernel names emitted: {names}"
    assert seen["emitted_distinct_bodies"] == seen["raw_distinct_bodies"], (
        f"compile_net dropped kernel bodies: {seen['emitted_distinct_bodies']} kept of "
        f"{seen['raw_distinct_bodies']} distinct"
    )
    assert len(names) == seen["raw_distinct_bodies"], "one name per distinct body, no more"
    missing = [s for s in seen["statement_names"] if s not in names]
    assert not missing, f"statements reference kernels with no body: {missing}"
    assert len(seen["statement_names"]) == seen["raw_calls"], "one statement per kernel call"
    assert seen["kernels_in_js"] == seen["raw_distinct_bodies"], "the runner JS must carry every kernel"
    assert any(re.search(r"_ktg\d+$", n) for n in names), "the collision rename never fired"


def test_pyodide_driver_zeroes_parallel_before_tinygrad_import():
    """Static check of js/src/trace/driver.py (Pyodide cannot run in CI):
    `PARALLEL` is set in os.environ, and every statement that touches
    keras/tinygrad comes AFTER it. PARALLEL's default is computed at tinygrad
    import time, so a later Context(...) is too late."""
    src = (REPO / "js" / "src" / "trace" / "driver.py").read_text()
    tree = ast.parse(src)
    parallel_line = None
    first_import_line = None
    for node in tree.body:
        text = ast.get_source_segment(src, node) or ""
        if parallel_line is None and "os.environ" in text and '"PARALLEL"' in text:
            parallel_line = node.lineno
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            if any(n.split(".")[0] in ("keras", "keras_tinygrad", "tinygrad") for n in names):
                first_import_line = first_import_line or node.lineno
    assert parallel_line is not None, "driver.py no longer sets PARALLEL in os.environ"
    assert first_import_line is not None, "driver.py imports neither keras nor tinygrad?"
    assert parallel_line < first_import_line, (
        f"PARALLEL is set on line {parallel_line}, after the first keras/tinygrad import on line "
        f"{first_import_line}; it must precede it"
    )
    assert re.search(r'setdefault\("PARALLEL",\s*"0"\)|\["PARALLEL"\]\s*=\s*"0"', src), "PARALLEL must be zeroed"


def test_hub_bakes_tinygrad_version_from_the_build_environment():
    """gen_hub.py's template must carry a placeholder for the tinygrad version
    that is replaced with importlib.metadata.version('tinygrad') at generation
    time — never a version literal next to PYODIDE/KERAS."""
    src = (REPO / "js" / "demo" / "gen_hub.py").read_text()
    decl = re.search(r"TINYGRAD_VERSION\s*=\s*\"([^\"]*)\"", src)
    assert decl, "gen_hub.py no longer declares TINYGRAD_VERSION in the page"
    assert not re.fullmatch(r"\d+\.\d+(\.\d+)?", decl.group(1)), (
        f"TINYGRAD_VERSION is a literal {decl.group(1)!r} — must be a placeholder baked at generation time"
    )
    placeholder = decl.group(1)
    assert re.search(
        rf"\.replace\(\s*\"{re.escape(placeholder)}\"\s*,\s*importlib\.metadata\.version\(\s*['\"]tinygrad['\"]\s*\)",
        src,
    ), f"{placeholder} is not replaced from importlib.metadata.version('tinygrad')"
