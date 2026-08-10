#!/usr/bin/env python3
"""Child worker for parity_fuzz.py — runs one case batch under ONE backend.

Spawned as a fresh subprocess per (backend, batch) because a Keras process
is locked to a single backend at import time.  Reads case specs (JSON) and
tensors (.npz), executes every case under ``KERAS_BACKEND``, and writes
outputs (.npz) plus a per-case status manifest (JSON).

Statuses per case:
  ok           outputs (and requested gradients) computed
  unsupported  the backend raised NotImplementedError anywhere in the case
  error        anything else went wrong (message carries the traceback tail)

Gradient handling:
  * analytic gradients of ``loss = sum of all outputs`` w.r.t. the inputs
    listed in ``grad.wrt``, via the backend's own autograd (tensorflow,
    torch, jax, tinygrad).  Backends without autograd report
    ``grads_status: unsupported``.
  * when ``grad.fd`` is set, central finite differences of the same loss
    are ALSO computed here, forward-evals only — no autograd and no second
    backend needed.  Saved as ``<id>__fd<i>``.

Only this module imports keras; the parent stays keras-free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable

import numpy as np

Forward = Callable[..., list[Any]]


def _as_list(out: Any) -> list[Any]:
    return list(out) if isinstance(out, (list, tuple)) else [out]


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def case_inputs(case: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    cid = case["id"]
    return [arrays[f"{cid}__in{i}"] for i in range(case["n_inputs"])]


def case_weights(case: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    cid = case["id"]
    return [arrays[f"{cid}__w{i}"] for i in range(case["n_weights"])]


def build_op_forward(keras: Any, case: dict[str, Any]) -> Forward:
    fn = getattr(keras.ops, case["name"])
    kwargs = case["kwargs"]
    if case.get("list_input"):
        return lambda *xs: _as_list(fn(list(xs), **kwargs))
    return lambda *xs: _as_list(fn(*xs, **kwargs))


def build_layer_forward(keras: Any, case: dict[str, Any], weights: list[np.ndarray], sample: np.ndarray) -> Forward:
    cls = getattr(keras.layers, case["name"])
    layer = cls(**case["kwargs"])
    layer(sample)  # build so set_weights sees the final shapes
    if weights:
        layer.set_weights(weights)  # identical weights on every backend
    return lambda x: [layer(x, training=False)]


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def analytic_grads(backend: str, forward: Forward, inputs: list[np.ndarray], wrt: list[int]) -> list[np.ndarray]:
    """Backend-native autograd of loss = sum of all outputs, w.r.t. inputs[wrt]."""
    if backend == "tensorflow":
        import tensorflow as tf

        ts = [tf.constant(x) for x in inputs]
        with tf.GradientTape() as tape:
            for i in wrt:
                tape.watch(ts[i])
            outs = forward(*ts)
            loss = tf.add_n([tf.reduce_sum(o) for o in outs])
        grads = tape.gradient(loss, [ts[i] for i in wrt])
        return [np.asarray(g) for g in grads]
    if backend == "torch":
        import torch

        ts = [torch.from_numpy(x.copy()) for x in inputs]
        for i in wrt:
            ts[i].requires_grad_(True)
        outs = forward(*ts)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        return [ts[i].grad.detach().cpu().numpy() for i in wrt]
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        def loss_fn(*xs: Any) -> Any:
            return sum(jnp.sum(o) for o in forward(*xs))

        grads = jax.grad(loss_fn, argnums=tuple(wrt))(*[jnp.asarray(x) for x in inputs])
        return [np.asarray(g) for g in grads]
    if backend == "tinygrad":
        from tinygrad import Tensor

        # tinygrad 0.13: no requires_grad constructor kwarg; gradients come
        # from the explicit `loss.gradient(*targets)` API.
        ts = [Tensor(x.copy()) for x in inputs]
        outs = forward(*ts)
        sums = [o.sum() for o in outs]
        loss = sums[0]
        for s in sums[1:]:
            loss = loss + s
        grads = loss.gradient(*[ts[i] for i in wrt])
        return [g.numpy() for g in grads]
    raise NotImplementedError(f"backend '{backend}' has no autograd")


def fd_grads(
    forward: Forward,
    to_numpy: Callable[[Any], np.ndarray],
    inputs: list[np.ndarray],
    wrt: list[int],
    eps_scale: float,
) -> list[np.ndarray]:
    """Central finite differences of loss = sum of all outputs. Forward-only."""

    def loss_of(xs: list[np.ndarray]) -> float:
        return float(sum(np.asarray(to_numpy(o), dtype=np.float64).sum() for o in forward(*xs)))

    grads: list[np.ndarray] = []
    for i in wrt:
        base = inputs[i]
        grad = np.zeros(base.size, dtype=np.float64)
        flat = base.astype(np.float64).reshape(-1)
        for j in range(flat.size):
            eps = eps_scale * (1.0 + abs(flat[j]))
            for sign in (1.0, -1.0):
                bumped = flat.copy()
                bumped[j] += sign * eps
                xs = list(inputs)
                xs[i] = bumped.reshape(base.shape).astype(base.dtype)
                grad[j] += sign * loss_of(xs)
            grad[j] /= 2.0 * eps
        grads.append(grad.reshape(base.shape).astype(np.float32))
    return grads


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


def run_case(
    keras: Any,
    backend: str,
    case: dict[str, Any],
    arrays: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> dict[str, Any]:
    cid = case["id"]
    to_numpy = keras.ops.convert_to_numpy
    inputs = case_inputs(case, arrays)
    if case["kind"] == "layer":
        forward = build_layer_forward(keras, case, case_weights(case, arrays), inputs[0])
    else:
        forward = build_op_forward(keras, case)

    outputs = [np.asarray(to_numpy(o)) for o in forward(*inputs)]
    for i, o in enumerate(outputs):
        out[f"{cid}__out{i}"] = o
    manifest: dict[str, Any] = {"status": "ok", "n_outputs": len(outputs)}

    grad = case.get("grad")
    if grad:
        try:
            grads = analytic_grads(backend, forward, inputs, grad["wrt"])
            for i, g in enumerate(grads):
                out[f"{cid}__grad{i}"] = np.asarray(g)
            manifest["grads_status"] = "ok"
            manifest["n_grads"] = len(grads)
        except NotImplementedError as exc:
            manifest["grads_status"] = "unsupported"
            manifest["grads_message"] = str(exc)
        if grad.get("fd") and manifest["grads_status"] == "ok":
            fds = fd_grads(forward, to_numpy, inputs, grad["wrt"], grad["eps_scale"])
            for i, g in enumerate(fds):
                out[f"{cid}__fd{i}"] = g
            manifest["fd"] = True
    return manifest


def run_batch(
    keras: Any, backend: str, cases: list[dict[str, Any]], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest: dict[str, Any] = {}
    out: dict[str, np.ndarray] = {}
    for case in cases:
        try:
            manifest[case["id"]] = run_case(keras, backend, case, arrays, out)
        except NotImplementedError as exc:
            manifest[case["id"]] = {"status": "unsupported", "message": str(exc)}
        except Exception:
            tail = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
            manifest[case["id"]] = {"status": "error", "message": tail.strip()}
    return manifest, out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def import_keras(backend: str) -> Any:
    os.environ["KERAS_BACKEND"] = backend
    if backend == "tinygrad":
        import keras_tinygrad  # noqa: F401  (must precede keras: installs the import hook)
    import keras

    actual = keras.backend.backend()
    if actual != backend:
        print(
            f"fatal: requested backend '{backend}' but keras loaded '{actual}'",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return keras


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="_parity_child.py")
    p.add_argument("--backend", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--arrays", required=True)
    p.add_argument("--out-arrays", required=True)
    p.add_argument("--out-manifest", required=True)
    args = p.parse_args(argv)

    keras = import_keras(args.backend)
    with open(args.cases, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    arrays = dict(np.load(args.arrays))

    manifest, out = run_batch(keras, args.backend, cases, arrays)

    np.savez_compressed(args.out_arrays, **out)
    with open(args.out_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
