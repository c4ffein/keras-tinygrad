"""Dense autoencoder on synthetic data: train, predict, reconstruction error.

The data is generated to genuinely live on a 4-dimensional manifold embedded
in 32 dimensions, so the 4-unit bottleneck has something real to compress.
After training we show that reconstruction error separates in-distribution
samples from pure noise — the classic autoencoder-as-anomaly-score reading.
"""

import argparse

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np

DATA_DIM = 32
LATENT_DIM = 4


def make_data(n: int, rng) -> np.ndarray:
    # Draw 4 latent factors, lift them through a fixed random linear map, and
    # squash — plus a little observation noise. Random 32-D noise would be
    # incompressible; this is not.
    factors = rng.normal(size=(n, LATENT_DIM)).astype("float32")
    lift = rng.normal(size=(LATENT_DIM, DATA_DIM)).astype("float32")
    x = np.tanh(factors @ lift) + 0.05 * rng.normal(size=(n, DATA_DIM))
    return x.astype("float32")


def build_model() -> keras.Model:
    # Symmetric hourglass. A single Model (not separate encoder/decoder) keeps
    # the example minimal; slice out sub-models later if you need embeddings.
    return keras.Sequential(
        [
            keras.layers.Input(shape=(DATA_DIM,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(LATENT_DIM, activation="relu"),  # bottleneck
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(DATA_DIM),  # linear output: targets are unbounded
        ],
        name="dense_autoencoder",
    )


def reconstruction_mse(model: keras.Model, x: np.ndarray) -> np.ndarray:
    recon = model.predict(x, verbose=0)
    return np.mean((np.asarray(recon) - x) ** 2, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("keras", keras.__version__, "| backend:", keras.backend.backend())

    rng = np.random.default_rng(0)
    x_train = make_data(1024, rng)
    x_test = make_data(256, rng)

    model = build_model()
    # Autoencoder = supervised learning where the target is the input itself.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(
        x_train,
        x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, x_test),
        verbose=2,
    )

    # Per-sample reconstruction error on held-out data vs structureless noise.
    err_test = reconstruction_mse(model, x_test)
    noise = rng.normal(size=x_test.shape).astype("float32")
    err_noise = reconstruction_mse(model, noise)

    print(f"reconstruction MSE  in-distribution: {err_test.mean():.4f}")
    print(f"reconstruction MSE  random noise:    {err_noise.mean():.4f}")
    print("first 5 per-sample errors:", [round(float(e), 4) for e in err_test[:5]])
    assert err_test.mean() < err_noise.mean(), "model failed to learn the manifold"
    print("OK: in-distribution data reconstructs better than noise")


if __name__ == "__main__":
    main()
