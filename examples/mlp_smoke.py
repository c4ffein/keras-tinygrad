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
losses = [float(v) for v in hist.history["loss"]]
# One assert per failure this script has actually let through or could:
# - `last < first` alone stayed green on 2026-09-21 while the jitted step
#   froze the loss tracker's count and fit REPORTED 0.0 from epoch 2 — a
#   loss of exactly 0.0 on noisy targets is impossible, so it is asserted
#   away, as is anything non-finite.
# - the reported curve must agree with a fresh evaluation: `evaluate`
#   rebuilds the metric state from scratch, so a frozen or drifting
#   tracker shows up as a mismatch against the last epoch's mean.
# - the last loss must beat the first by a real margin (the target noise
#   floor is 0.01; 5 Adam epochs reach ~0.1-0.6 from ~6-13), not by an ulp.
assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
assert all(v > 0.0 for v in losses), f"a loss of exactly 0.0 is a frozen tracker, not a fit: {losses}"
assert losses[-1] < 0.5 * losses[0], f"loss did not go down enough: {losses}"
evaluated = float(model.evaluate(x, y, batch_size=32, verbose=0))
print("evaluate:", round(evaluated, 4))
assert evaluated < losses[0] and abs(evaluated - losses[-1]) < 0.5 * losses[-1] + 0.05, (
    f"evaluate {evaluated} disagrees with the reported curve {losses}"
)
assert np.asarray(pred).shape == (4, 1), f"predict shape {np.asarray(pred).shape}"
print("SMOKE OK")
