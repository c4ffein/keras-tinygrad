# keras-tinygrad (JS)

---

*WARNING — this is a vibe-engineering experiment - not affiliated with the
Keras team or the tiny corp.*
The JS half of [keras-tinygrad on PyPI](https://pypi.org/project/keras-tinygrad/):
a Keras 3 backend for tinygrad whose training steps can be traced and
exported as self-contained WebGPU bundles.

---

**Today (0.0.x, unstable):** load an exported bundle and train in the
browser — the runner half. A real Keras model (layers, loss, optimizer)
is traced once in Python on a GPU-less device; this package loads the
resulting WGSL kernels + weights and gives you `step(x, y)`, whose
in-place weight updates make looping it SGD training. Verified against a
CPU reference on identical bytes (see the Python repo's
`experiments/m0-keras-trainstep/`).

```js
import { fetchBundle } from "keras-tinygrad";

const { step } = await fetchBundle("./model.js", "./model.safetensors");
for (const [x, y] of batches) {
  const [loss] = await step(x, y); // weights update in place on the GPU
}
```

**Roadmap:** in-tab tracing of arbitrary Keras models (Pyodide +
keras + keras-tinygrad wheels) — define the model in the page, trace it
there, train it there. The Python side's
[docs/browser-training.md](https://github.com/c4ffein/keras-tinygrad/blob/main/docs/browser-training.md)
is the plan of record.

Requires WebGPU (Chrome/Edge; secure context — https or localhost).
Apache-2.0.
