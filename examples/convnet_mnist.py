"""Small convnet on MNIST-shaped data.

Synthetic data by default so the example runs fully offline; pass --real to
download the actual MNIST digits through keras.datasets (needs network access
on the first run, cached under ~/.keras afterwards).
"""

import argparse

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np


def load_data(real: bool):
    if real:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        # Synthetic stand-in with MNIST's exact shapes and dtypes. The point of
        # this example is exercising the backend's conv/pool/softmax kernels,
        # not digit accuracy — random pixels do that just as well, offline.
        rng = np.random.default_rng(0)
        x_train = rng.integers(0, 256, size=(512, 28, 28), dtype=np.uint8)
        y_train = rng.integers(0, 10, size=(512,), dtype=np.int64)
        x_test = rng.integers(0, 256, size=(128, 28, 28), dtype=np.uint8)
        y_test = rng.integers(0, 10, size=(128,), dtype=np.int64)
    # Scale to [0, 1] and add the trailing channel axis Conv2D expects.
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model() -> keras.Model:
    # Deliberately narrow (8/16 filters): big enough to learn real MNIST to
    # ~97% in a few epochs, small enough that a CPU tinygrad device stays
    # comfortable.
    return keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(8, kernel_size=3, activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_convnet",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--real",
        action="store_true",
        help="fetch real MNIST via keras.datasets instead of synthetic data",
    )
    args = parser.parse_args()

    print("keras", keras.__version__, "| backend:", keras.backend.backend())

    (x_train, y_train), (x_test, y_test) = load_data(args.real)
    model = build_model()
    model.summary()

    # Integer labels + sparse loss: no need to one-hot 60k labels host-side.
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test loss {loss:.4f} | test accuracy {acc:.4f}")
    if not args.real:
        print("(synthetic labels are random — accuracy near 0.10 is expected)")

    # Sanity: predicted class distribution over a handful of test digits.
    probs = model.predict(x_test[:8], verbose=0)
    print("predicted classes:", np.argmax(probs, axis=-1).tolist())


if __name__ == "__main__":
    main()
