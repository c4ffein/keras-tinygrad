"""MLP compile+fit+predict smoke test against stock keras + keras_tinygrad."""

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np

print("keras", keras.__version__, "| backend:", keras.backend.backend())
print("keras module file:", keras.__file__)

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
w_true = rng.normal(size=(8, 1)).astype("float32")
y = (x @ w_true + 0.1 * rng.normal(size=(256, 1))).astype("float32")

model = keras.Sequential(
    [
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
hist = model.fit(x, y, epochs=5, batch_size=32, verbose=2)
pred = model.predict(x[:4], verbose=0)
print("losses:", [round(float(v), 4) for v in hist.history["loss"]])
print("predict sample:", np.asarray(pred).ravel().tolist())
assert hist.history["loss"][-1] < hist.history["loss"][0], "loss did not go down"
print("SMOKE OK")
