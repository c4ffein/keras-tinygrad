"""Dropout-through-export probe (docs/device-rng.md criterion #2).

With KERAS_TINYGRAD_DEVICE_RNG=1, dropout's mask is threefry computed
on-device and the counter advance is part of the graph. This probe answers:
does that survive `export_model`? I.e. does an exported WebGPU bundle of a
Dropout model draw a FRESH mask every `step()` call?

  python export_dropout_probe.py   # NULL:WGSL trace -> build/dropout.{js,safetensors,meta.json}

(The old `page` mode's standalone probe page was superseded by the hub's
probe mode + tests/test_webgpu_export.py's dropout test, 2026-09-23.)

Browser assertion (same batch fed 3x): losses must be pairwise different
(fresh masks) and finite. The step/pinning/export recipe lives in
`keras_tinygrad.webgpu` (the shipped implementation); this file only picks
the model and writes the artifacts. m0.py keeps its own copy on purpose —
it is the owner's pristine original proof.
"""

import os

os.environ["KERAS_TINYGRAD_DEVICE_RNG"] = "1"
os.environ.setdefault("DEV", "NULL:WGSL")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")
os.environ.setdefault("KERAS_BACKEND", "tinygrad")
os.environ["KERAS_TINYGRAD_TRAINER_JIT"] = "0"

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
BATCH, DIM, CLASSES = 32, 784, 10
LR = 0.05


def run_export():
    import keras_tinygrad  # noqa: F401

    # THE implementation lives in the package now (keras_tinygrad.webgpu) —
    # this probe is a thin caller; the recipe used to be hand-copied here.
    from keras_tinygrad.webgpu import export_train_step

    import keras

    model = keras.Sequential(
        [
            keras.layers.Input((DIM,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(CLASSES),
        ]
    )
    out = export_train_step(
        model,
        keras.optimizers.SGD(learning_rate=LR, momentum=0.9),
        keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None),
        batch_size=BATCH,
        input_shape=(DIM,),
        learning_rate=LR,
        model_name="dropoutstep",
    )
    with open(os.path.join(BUILD, "dropout.js"), "w") as f:
        f.write(out["js"])
    with open(os.path.join(BUILD, "dropout.safetensors"), "wb") as f:
        f.write(out["weights"])
    meta = out["meta"]
    assert meta["learningRate"] == LR, f"baked lr {meta['learningRate']} != {LR}"
    rng_in_state = any("rng/" in k for k in meta["stateEntries"])
    print("state entries:", meta["stateEntries"])
    print(f"kernels {meta['kernels']}, js {meta['jsBytes']}B, rng buffers in state: {rng_in_state}")
    assert rng_in_state, "dropout model must carry rng/seed + rng/counter state"
    print("OK — dropout step exported (validated inside export_train_step)")


if __name__ == "__main__":
    run_export()
