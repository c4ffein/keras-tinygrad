"""Runs INSIDE Pyodide (and identically under CPython via run_local.py):
builds a Keras model FROM A JSON CONFIG and exports one training step via
`keras_tinygrad.webgpu.export_train_step` — THE shipped implementation (the
recipe used to be hand-copied here; the wheel micropip-installed into the
tab now carries it, so in-tab tracing runs the exact code every other
consumer runs).

This is the in-tab retrace that gives "select options" parity with a JS
framework: every knob (layers, units, activation, dropout rate, lr,
momentum, batch size) is a config field, and changing one costs a retrace
(seconds), never a redeploy.

IMPORTANT: DEV=NULL:WGSL must be in os.environ BEFORE tinygrad is imported.
"""

import json
import os
import time

os.environ.setdefault("DEV", "NULL:WGSL")
assert os.environ["DEV"] == "NULL:WGSL", "driver.py traces on NULL:WGSL only"
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")
# tinygrad 0.14 compiles kernels through a multiprocessing worker pool by
# default; Pyodide has no _multiprocessing. PARALLEL's default is computed at
# tinygrad import time, so it must be zeroed here, not in a later Context.
os.environ.setdefault("PARALLEL", "0")
os.environ["KERAS_BACKEND"] = "tinygrad"
os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"
os.environ.setdefault("KERAS_TINYGRAD_NO_VERSION_WARNING", "1")

TIMINGS = {}
_t = time.time()
from keras_tinygrad.webgpu import export_train_step  # noqa: E402

import keras  # noqa: E402

TIMINGS["import_keras_s"] = round(time.time() - _t, 2)

DEFAULT_CONFIG = {
    "input_dim": 784,
    "classes": 10,
    "batch": 32,
    "layers": [{"type": "dense", "units": 128, "activation": "relu"}],
    "optimizer": {"lr": 0.05, "momentum": 0.9},
}


def build_model(cfg):
    layers = [keras.layers.Input((int(cfg["input_dim"]),))]
    for spec in cfg["layers"]:
        if spec["type"] == "dense":
            layers.append(keras.layers.Dense(int(spec["units"]), activation=spec.get("activation")))
        elif spec["type"] == "dropout":
            layers.append(keras.layers.Dropout(float(spec["rate"])))
        else:
            raise ValueError(f"unsupported layer type {spec['type']!r} (dense, dropout)")
    layers.append(keras.layers.Dense(int(cfg["classes"])))  # logits
    return keras.Sequential(layers)


def build(config_json=None):
    cfg = dict(DEFAULT_CONFIG)
    if config_json:
        cfg.update(json.loads(config_json) if isinstance(config_json, str) else config_json)
    t0 = time.time()
    opt_cfg = cfg["optimizer"]
    out = export_train_step(
        build_model(cfg),
        keras.optimizers.SGD(learning_rate=float(opt_cfg["lr"]), momentum=float(opt_cfg["momentum"])),
        keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None),
        batch_size=int(cfg["batch"]),
        input_shape=(int(cfg["input_dim"]),),
        learning_rate=float(opt_cfg["lr"]),
    )
    TIMINGS["trace_export_s"] = round(time.time() - t0, 2)
    meta = dict(out["meta"], config=cfg, timings=dict(TIMINGS), keras=keras.__version__)
    return {"js": out["js"], "weights": out["weights"], "meta": meta}
