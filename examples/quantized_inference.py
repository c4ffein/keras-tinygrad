"""Post-training int8 quantization: train float32, quantize, measure drift.

Keras 3 quantizes in place: model.quantize("int8") swaps each Dense kernel
for int8 weights plus per-channel scales, no calibration dataset needed.
int8 works on this backend today. We train a small regressor, snapshot its
float32 predictions, quantize, and report the mean squared drift between the
two — small relative to the signal is what "still predicts" means.
"""

import argparse

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np


def make_data(n: int, rng):
    # Mildly nonlinear regression target so the network has to use its hidden
    # layers — quantizing a model that only learned a bias would prove nothing.
    x = rng.normal(size=(n, 8)).astype("float32")
    w = rng.normal(size=(8, 1)).astype("float32")
    y = np.tanh(x @ w) + 0.05 * rng.normal(size=(n, 1))
    return x, y.astype("float32")


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ],
        name="quantize_me",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("keras", keras.__version__, "| backend:", keras.backend.backend())

    rng = np.random.default_rng(0)
    x_train, y_train = make_data(1024, rng)
    x_test, y_test = make_data(256, rng)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # Snapshot float32 behavior BEFORE quantizing — quantize() mutates the
    # model in place, so this is our only chance at a reference.
    ref = np.asarray(model.predict(x_test, verbose=0))
    loss_f32 = model.evaluate(x_test, y_test, verbose=0)

    model.quantize("int8")

    quant = np.asarray(model.predict(x_test, verbose=0))
    loss_int8 = model.evaluate(x_test, y_test, verbose=0)

    drift = float(np.mean((ref - quant) ** 2))
    signal = float(np.var(ref))
    print(f"float32 test loss: {loss_f32:.5f} | int8 test loss: {loss_int8:.5f}")
    print(f"mean squared drift (f32 -> int8): {drift:.6f}")
    print(f"float32 prediction variance:      {signal:.6f}")
    print(f"drift / signal variance:          {drift / max(signal, 1e-12):.4%}")

    # "Still predicts" made precise: quantization noise is a small fraction of
    # the prediction signal, not comparable to it.
    assert drift < 0.05 * signal, "int8 drift unexpectedly large vs signal"
    print("first 4 predictions f32 :", np.round(ref[:4].ravel(), 4).tolist())
    print("first 4 predictions int8:", np.round(quant[:4].ravel(), 4).tolist())
    print("OK: int8 model tracks the float32 model")


if __name__ == "__main__":
    main()
