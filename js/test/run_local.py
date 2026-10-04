"""CPython harness for driver.py — the fast dev loop, and the SHIM test:
by default the pure-Python optree/ml_dtypes shims are put FIRST on sys.path
so they shadow the real extensions, proving keras imports and traces on the
shims alone before any browser is involved.

  python run_local.py                  # default config, shims forced
  python run_local.py --real-deps      # same with the real optree/ml_dtypes
  python run_local.py '{"layers":[{"type":"dense","units":64,"activation":"relu"},{"type":"dropout","rate":0.3}]}'

Writes ../demo/build/run-local/kerasstep.{js,safetensors} + meta.json and, for
the default config, checks the emitted kernel set against m0's demo/build/out.js.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The engine (driver + shims) is the npm package's tracer half now.
TRACE = os.path.normpath(os.path.join(HERE, "..", "src", "trace"))
OUT = os.path.join(HERE, "..", "demo", "build", "run-local")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--real-deps" not in sys.argv:
    sys.path.insert(0, os.path.join(TRACE, "shims"))
    # Emulate the Pyodide environment: no jax / tensorflow / torch exist
    # there. A None entry makes `import jax` raise ImportError, which keras'
    # LazyModule/`available` probes handle — the dev venv HAS jax, and keras
    # would otherwise import it eagerly and hit the ml_dtypes shim's limits.
    for absent in ("jax", "jaxlib", "tensorflow", "torch", "optree_real"):
        sys.modules[absent] = None
sys.path.insert(0, TRACE)
os.environ["DEV"] = "NULL:WGSL"

import driver  # noqa: E402
import optree  # noqa: E402

print("optree in use:", getattr(optree, "__version__", "?"), "->", os.path.dirname(optree.__file__))

out = driver.build(args[0] if args else None)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "kerasstep.js"), "w") as f:
    f.write(out["js"])
with open(os.path.join(OUT, "kerasstep.safetensors"), "wb") as f:
    f.write(out["weights"])
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(out["meta"], f, indent=1)
m = out["meta"]
print(f"kernels {m['kernels']}  js {m['jsBytes']}B  weights {m['weightBytes']}B  timings {m['timings']}")
print("state:", m["stateEntries"])

if not args:  # default config == m0's dense model: kernels must match
    kern = lambda js: sorted(re.findall(r"const \w+ = `([^`]*@compute[^`]*)`", js))  # noqa: E731
    with open(os.path.join(HERE, "..", "demo", "build", "out.js")) as f:
        m0 = kern(f.read())
    ours = kern(out["js"])
    same = ours == m0
    print(f"kernel sources identical to m0's out.js: {same} ({len(ours)} vs {len(m0)})")
    assert len(ours) > 0, "kernel regex matched nothing — vacuous comparison"
    assert same, "default-config trace diverged from m0's proven dense export"
print("OK")
