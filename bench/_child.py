"""One backend x one model, in its own process (a Keras process is locked to
one backend at import). Prints exactly one `BENCH_JSON {...}` line on stdout;
everything else the backend logs is noise the orchestrator ignores.

Fairness lives here: identical initial weights (numpy, fixed seed, applied
with `model.set_weights` — every backend's own initializer RNG differs),
identical synthetic data in the same batch order, the same optimizer with
the same hyper-parameters, no dropout anywhere. What is timed is
`train_on_batch` (first step = compile/capture; steady state = median of
the steps after five warm-up steps) and `predict` on 1,024 samples after
one warm-up call. Peak RSS comes from getrusage.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback

MODELS = {
    "mlp": {"batch": 128, "steps": 100},
    "cnn": {"batch": 64, "steps": 100},
    "lstm": {"batch": 32, "steps": 30},
}
CHECKPOINTS = (1, 10, 25, 50, 100)
SEQ_LEN = 25
VOCAB = 10000


def box_info():
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    ram_gb = round(int(line.split()[1]) / 1e6, 1)
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cores_total": os.cpu_count(),
        "cores_pinned": len(os.sched_getaffinity(0)),
        "ram_gb": ram_gb,
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


def make_data(name, rng, batch, steps):
    n = batch * steps
    if name == "mlp":
        x = rng.random((n, 784), dtype="float32")
        y = rng.integers(0, 10, size=(n,)).astype("int32")
    elif name == "cnn":
        x = rng.random((n, 28, 28, 1), dtype="float32")
        y = rng.integers(0, 10, size=(n,)).astype("int32")
    elif name == "lstm":
        x = rng.integers(1, VOCAB, size=(n, SEQ_LEN)).astype("int32")
        y = rng.integers(0, 2, size=(n, 1)).astype("float32")
    else:
        raise ValueError(name)
    return x, y


def build(name, keras):
    layers = keras.layers
    if name == "mlp":
        model = keras.Sequential(
            [layers.Input((784,)), layers.Dense(128, activation="relu"), layers.Dense(10, activation="softmax")]
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    elif name == "cnn":
        model = keras.Sequential(
            [
                layers.Input((28, 28, 1)),
                layers.Conv2D(16, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(10),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
    elif name == "lstm":
        model = keras.Sequential(
            [
                layers.Input((SEQ_LEN,)),
                layers.Embedding(VOCAB, 64),
                layers.LSTM(32, dropout=0.0, recurrent_dropout=0.0),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer=keras.optimizers.SGD(0.05, momentum=0.9), loss="binary_crossentropy")
    else:
        raise ValueError(name)
    return model


def deterministic_weights(model, rng):
    """Same bytes on every backend: normal(0.05) for anything 2-D or more,
    zeros for 1-D (biases), in `get_weights` order (fixed by keras)."""
    out = []
    for w in model.get_weights():
        shape = tuple(w.shape)
        if len(shape) >= 2:
            out.append(rng.normal(0.0, 0.05, size=shape).astype("float32"))
        else:
            out.append(__import__("numpy").zeros(shape, "float32"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    os.environ["KERAS_BACKEND"] = args.backend
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    result = {"backend": args.backend, "model": args.model, "box": box_info()}
    try:
        import numpy as np

        if args.backend == "tinygrad":
            import keras_tinygrad  # noqa: F401  (must precede keras)
        import keras

        assert keras.backend.backend() == args.backend, keras.backend.backend()
        spec = MODELS[args.model]
        steps = args.steps or spec["steps"]
        batch = spec["batch"]
        versions = {"keras": keras.__version__}
        try:
            import importlib.metadata as md

            lib = {"tinygrad": "tinygrad", "tensorflow": "tensorflow-cpu", "jax": "jax", "torch": "torch"}[args.backend]
            try:
                versions[args.backend] = md.version(lib)
            except md.PackageNotFoundError:
                versions[args.backend] = md.version("tensorflow" if lib == "tensorflow-cpu" else lib)
        except Exception as e:  # noqa: BLE001 - versions are informational
            versions[args.backend] = f"? ({e})"
        result["versions"] = versions
        result["steps"] = steps
        result["batch"] = batch

        rng = np.random.default_rng(1234)
        x, y = make_data(args.model, rng, batch, steps)
        model = build(args.model, keras)
        model.set_weights(deterministic_weights(model, np.random.default_rng(42)))

        losses, step_times = [], []
        t_start = time.perf_counter()
        for i in range(steps):
            xb, yb = x[i * batch : (i + 1) * batch], y[i * batch : (i + 1) * batch]
            t0 = time.perf_counter()
            loss = model.train_on_batch(xb, yb)
            dt = time.perf_counter() - t0
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            losses.append(float(loss))
            step_times.append(dt)
        total_train = time.perf_counter() - t_start
        assert all(np.isfinite(losses)), losses

        warm = step_times[5:] if len(step_times) > 8 else step_times[1:]
        median_step = float(np.median(warm))
        target = 0.5 * losses[0]
        t_to_target = None
        cum = 0.0
        for dt, loss in zip(step_times, losses):
            cum += dt
            if loss <= target:
                t_to_target = cum
                break

        xp = x[:1024] if len(x) >= 1024 else np.concatenate([x] * (1024 // len(x) + 1))[:1024]
        model.predict(xp, batch_size=128, verbose=0)
        t0 = time.perf_counter()
        model.predict(xp, batch_size=128, verbose=0)
        predict_dt = time.perf_counter() - t0

        result.update(
            {
                "first_step_s": step_times[0],
                "steady_step_s": median_step,
                "steady_steps_per_s": 1.0 / median_step,
                "train_total_s": total_train,
                "loss_at": {str(k): losses[k - 1] for k in CHECKPOINTS if k <= steps},
                "losses": losses,
                "step_times": step_times,
                "target_loss": target,
                "time_to_target_s": t_to_target,
                "predict_samples_per_s": 1024 / predict_dt,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        )
    except Exception:  # noqa: BLE001 - the error IS the result
        result["error"] = traceback.format_exc().strip().splitlines()[-1]
        result["traceback"] = traceback.format_exc()[-3000:]
    sys.stdout.flush()
    print("BENCH_JSON " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
