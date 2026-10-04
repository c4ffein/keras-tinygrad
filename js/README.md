# keras-tinygrad (JS)

---

*WARNING — this is a vibe-engineering experiment - not affiliated with the
Keras team or the tiny corp.*
The JS half of [keras-tinygrad on PyPI](https://pypi.org/project/keras-tinygrad/):
a Keras 3 backend for tinygrad whose training steps can be traced and
exported as self-contained WebGPU bundles.

---

**Two halves (0.0.x, unstable):**

**The runner** (`keras-tinygrad`): load an exported bundle and train in
the browser. A real Keras model (layers, loss, optimizer) is traced once
in Python on a GPU-less device; this loads the resulting WGSL kernels +
weights and gives you `step(x, y)`, whose in-place weight updates make
looping it SGD training. Dependency-free, inlineable as a single blob.
Verified against a CPU reference on identical bytes (see the Python
repo's `js/demo/` and its `tests/test_webgpu_export.py`).

```js
import { fetchBundle } from "keras-tinygrad";

const { step } = await fetchBundle("./model.js", "./model.safetensors");
for (const [x, y] of batches) {
  const [loss] = await step(x, y); // weights update in place on the GPU
}
```

**The tracer** (`keras-tinygrad/trace`): the REAL keras-tinygrad running
in the tab — keras 3.15 + this repo's Python wheel under Pyodide in a
module Web Worker — traces a model described by a JSON config and hands
the result to the runner. Needs the network on first boot (Pyodide +
keras + tinygrad from CDNs, ~25 MB) and a URL for the Python wheel;
`driver.py` and the pure-Python `shims/` for the two C extensions
Pyodide lacks (optree, ml_dtypes) ship inside this package. In-tab
traces are loss-IDENTICAL to Python-side exports (asserted by the repo's
e2e suite on every CI run).

```js
import { traceModel } from "keras-tinygrad/trace";

const { step } = await traceModel(
  { input_dim: 784, classes: 10, batch: 32,
    layers: [{ type: "dense", units: 128, activation: "relu" },
             { type: "dropout", rate: 0.3 }],
    optimizer: { lr: 0.05, momentum: 0.9 } },
  { versions: { wheel: "/wheels/keras_tinygrad-X.Y.Z-py3-none-any.whl" } },
);
```

Requires WebGPU (Chrome/Edge; secure context — https or localhost).
Apache-2.0.
