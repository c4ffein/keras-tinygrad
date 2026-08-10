"""Character-level language model with SimpleRNN on an embedded tiny corpus.

Trains next-character prediction on a few pangrams baked into this file (no
downloads), then samples some text. With the repetition below, a tiny RNN
visibly overfits its corpus within a handful of epochs — which is exactly the
signal we want from an example: the recurrent kernels are doing their job.
"""

import argparse

import keras_tinygrad  # noqa: F401  (must precede keras)

import keras
import numpy as np

# A tiny corpus, repeated: repetition lets a small model latch onto the
# patterns quickly, so the sampled text becomes recognizably corpus-like
# after very little training.
CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
) * 8

SEQ_LEN = 24  # context window fed to the RNN for each next-char prediction


def build_dataset():
    vocab = sorted(set(CORPUS))
    char_to_ix = {c: i for i, c in enumerate(vocab)}
    encoded = np.array([char_to_ix[c] for c in CORPUS], dtype=np.int64)
    # Every window of SEQ_LEN chars predicts the char that follows it.
    n = len(encoded) - SEQ_LEN
    x = np.stack([encoded[i : i + SEQ_LEN] for i in range(n)])
    y = encoded[SEQ_LEN:]
    return x, y, vocab


def build_model(vocab_size: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN,)),
            keras.layers.Embedding(vocab_size, 16),
            # Default recurrent initializer (Orthogonal, QR-based) on
            # purpose: this doubles as a regression test for linalg's
            # Householder QR.
            keras.layers.SimpleRNN(64),
            keras.layers.Dense(vocab_size, activation="softmax"),
        ],
        name="char_rnn",
    )


def sample_index(probs: np.ndarray, temperature: float, rng) -> int:
    # Temperature-rescale in log space, renormalize, draw. temperature < 1
    # sharpens toward the argmax, > 1 flattens toward uniform.
    logits = np.log(np.clip(probs.astype("float64"), 1e-8, 1.0)) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model: keras.Model, vocab, seed: str, length: int, temperature: float) -> str:
    char_to_ix = {c: i for i, c in enumerate(vocab)}
    window = [char_to_ix[c] for c in seed]
    rng = np.random.default_rng(0)
    out = list(seed)
    for _ in range(length):
        # One predict per character: each call is an independent forward pass
        # over the current window — simple and stateless, no cached RNN state.
        probs = model.predict(np.array([window], dtype=np.int64), verbose=0)[0]
        ix = sample_index(probs, temperature, rng)
        out.append(vocab[ix])
        window = window[1:] + [ix]
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    print("keras", keras.__version__, "| backend:", keras.backend.backend())

    x, y, vocab = build_dataset()
    print(f"corpus: {len(CORPUS)} chars, vocab: {len(vocab)}, windows: {len(x)}")

    model = build_model(len(vocab))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    seed = CORPUS[:SEQ_LEN]
    text = generate(model, vocab, seed, length=120, temperature=args.temperature)
    print(f"--- sample (seed={seed!r}, temperature={args.temperature}) ---")
    print(text)


if __name__ == "__main__":
    main()
