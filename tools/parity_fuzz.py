#!/usr/bin/env python3
"""Property-based cross-backend parity fuzzer for the tinygrad Keras backend.

Keras' own tests are example-based; this tool hunts numerical drift the
examples miss, by comparing the backend under test (default: tinygrad)
against a reference backend (default: numpy) on *randomized* op, layer and
gradient configurations.

Architecture: a Keras process is locked to one backend at import time, so
this parent orchestrator never imports keras.  It generates seeded cases
(specs as JSON, tensors as .npz), spawns one fresh subprocess per
(backend, case-batch) running ``_parity_child.py``, collects each child's
outputs as .npz + a JSON manifest, and compares everything here with
numpy only.

The parent depends on the stdlib + numpy exclusively.

Usage synopsis::

    parity_fuzz.py [--seed N] [--cases N] [--ops a,b] [--layers A,B]
                   [--kinds op,layer,grad]
                   [--backend-under-test tinygrad] [--reference numpy]
                   [--slow] [--top-k K] [--json PATH] [--repro CASE_ID]
                   [--batch-size N] [--python PATH] [--tol-scale F]
                   [--child-timeout SEC] [--keep-artifacts DIR]

Exit code 0 when every compared case is within tolerance; 1 when any case
fails tolerance or errors on the backend under test; 2 on usage/infra
problems — including a run where no case was compared at all (an all-skipped
run is not a pass).  ``NotImplementedError`` raised by a backend is counted
as "unsupported", never as failure.

One dtype mismatch is deliberately benign: Keras demotes implicit float64 to
float32 on every backend except tensorflow (the 64->32 entries of the
promotion table in ``keras/src/backend/common/dtypes.py``); the numpy
reference keeps float64 only because its raw ops skip the conversion step.
When the reference output is float64 and the test output is float32 — that
exact direction, float only — values are compared under float32 tolerances
and the comparison is ok with a visible "keras 64->32 demotion" note.  The
reverse direction and every other dtype mismatch still fail.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

Rng = np.random.Generator

CHILD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_child.py")

AUTOGRAD_BACKENDS = frozenset({"tensorflow", "torch", "jax", "tinygrad"})

# Per-dtype (rtol, atol) for forward outputs.  Gradients compared across
# backends use these scaled by GRAD_TOL_SCALE; analytic-vs-finite-difference
# uses FD_TOLS (central differences in float32 are inherently noisy).
FLOAT_TOLS: dict[str, tuple[float, float]] = {
    "float64": (1e-6, 1e-9),
    "float32": (1e-4, 1e-6),
    "float16": (1e-2, 1e-3),
}
GRAD_TOL_SCALE = 10.0
FD_TOLS = (2e-2, 1e-3)
FD_EPS_SCALE = 1e-3  # eps = scale * (1 + |x|), central difference

DEFAULT_OPS = (
    "matmul",
    "add",
    "subtract",
    "multiply",
    "divide",
    "softmax",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "exp",
    "log",
    "sum",
    "mean",
    "max",
    "min",
    "var",
    "std",
    "transpose",
    "reshape",
    "concatenate",
    "conv",
    "argmax",
)
GRAD_OPS = (
    "matmul",
    "multiply",
    "softmax",
    "tanh",
    "sigmoid",
    "sum",
    "mean",
    "exp",
    "conv",
)
DEFAULT_LAYERS = (
    "Dense",
    "Conv2D",
    "BatchNormalization",
    "LayerNormalization",
    "Embedding",
    "SimpleRNN",
)

PASS = "PASS"
FAIL = "FAIL"
UNSUPPORTED = "UNSUPPORTED"
SKIPPED = "SKIPPED"
ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass
class Case:
    cid: str
    kind: str  # "op" | "layer" | "grad"
    name: str
    kwargs: dict[str, Any]
    inputs: list[np.ndarray]
    weights: list[np.ndarray] = field(default_factory=list)
    list_input: bool = False
    grad_wrt: list[int] = field(default_factory=list)

    def spec(self, fd: bool) -> dict[str, Any]:
        """JSON-safe description sent to a child (arrays travel via npz)."""
        d: dict[str, Any] = {
            "id": self.cid,
            "kind": self.kind,
            "name": self.name,
            "kwargs": self.kwargs,
            "n_inputs": len(self.inputs),
            "n_weights": len(self.weights),
            "list_input": self.list_input,
        }
        if self.grad_wrt:
            d["grad"] = {"wrt": self.grad_wrt, "fd": fd, "eps_scale": FD_EPS_SCALE}
        return d

    def arrays(self) -> dict[str, np.ndarray]:
        out = {f"{self.cid}__in{i}": a for i, a in enumerate(self.inputs)}
        out.update({f"{self.cid}__w{i}": a for i, a in enumerate(self.weights)})
        return out


@dataclass
class ArrayCmp:
    label: str
    ok: bool
    max_abs: float
    ratio: float  # max |diff| / (atol + rtol * |ref|); > 1 fails
    note: str = ""


@dataclass
class CaseResult:
    cid: str
    kind: str
    name: str
    status: str
    worst_ratio: float = 0.0
    max_abs: float = 0.0
    message: str = ""
    comparisons: list[ArrayCmp] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.cid,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "worst_ratio": self.worst_ratio,
            "max_abs": self.max_abs,
            "message": self.message,
            "comparisons": [dataclasses.asdict(c) for c in self.comparisons],
        }


# ---------------------------------------------------------------------------
# Random sampling helpers (pure functions of an rng)
# ---------------------------------------------------------------------------


def _ri(rng: Rng, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _shape(rng: Rng, tiny: bool, rank_lo: int = 1, rank_hi: int = 3) -> tuple[int, ...]:
    dim_hi = 3 if tiny else 8
    return tuple(_ri(rng, 1, dim_hi) for _ in range(_ri(rng, rank_lo, rank_hi)))


def _rand(rng: Rng, shape: tuple[int, ...], dtype: str, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(dtype)


def _pick_dtype(rng: Rng, tiny: bool) -> str:
    if tiny:
        return "float32"  # gradient cases stay float32
    return str(rng.choice(["float32", "float16", "float64"], p=[0.7, 0.15, 0.15]))


def _away_from_zero(a: np.ndarray) -> np.ndarray:
    return np.where(a >= 0, a + 0.5, a - 0.5).astype(a.dtype)


Sampled = tuple[list[np.ndarray], dict[str, Any], bool]  # inputs, kwargs, list_input
OpSampler = Callable[[Rng, str, bool], Sampled]


def _s_matmul(rng: Rng, dt: str, tiny: bool) -> Sampled:
    hi = 3 if tiny else 8
    m, k, n = _ri(rng, 1, hi), _ri(rng, 1, hi), _ri(rng, 1, hi)
    if not tiny and rng.random() < 0.4:
        b = _ri(rng, 1, 3)
        return [_rand(rng, (b, m, k), dt), _rand(rng, (b, k, n), dt)], {}, False
    return [_rand(rng, (m, k), dt), _rand(rng, (k, n), dt)], {}, False


def _s_binary(post: Callable[[np.ndarray], np.ndarray] | None = None) -> OpSampler:
    def sampler(rng: Rng, dt: str, tiny: bool) -> Sampled:
        shape = _shape(rng, tiny)
        bshape = tuple(1 if rng.random() < 0.3 else d for d in shape)
        a, b = _rand(rng, shape, dt), _rand(rng, bshape, dt)
        if post is not None:
            b = post(b)
        return [a, b], {}, False

    return sampler


def _s_unary(pre: Callable[[np.ndarray], np.ndarray] | None = None, scale: float = 1.0) -> OpSampler:
    def sampler(rng: Rng, dt: str, tiny: bool) -> Sampled:
        x = _rand(rng, _shape(rng, tiny), dt, scale)
        if pre is not None:
            x = pre(x).astype(dt)
        return [x], {}, False

    return sampler


def _s_softmax(rng: Rng, dt: str, tiny: bool) -> Sampled:
    x = _rand(rng, _shape(rng, tiny), dt)
    axis = _ri(rng, -x.ndim, x.ndim - 1)
    return [x], {"axis": axis}, False


def _s_reduction(rng: Rng, dt: str, tiny: bool) -> Sampled:
    x = _rand(rng, _shape(rng, tiny), dt)
    axis: int | None = None if rng.random() < 0.3 else _ri(rng, -x.ndim, x.ndim - 1)
    return [x], {"axis": axis, "keepdims": bool(rng.random() < 0.5)}, False


def _s_transpose(rng: Rng, dt: str, tiny: bool) -> Sampled:
    x = _rand(rng, _shape(rng, tiny, rank_lo=2), dt)
    axes = [int(i) for i in rng.permutation(x.ndim)]
    return [x], {"axes": axes}, False


def _s_reshape(rng: Rng, dt: str, tiny: bool) -> Sampled:
    x = _rand(rng, _shape(rng, tiny), dt)
    total = int(x.size)
    divisors = [d for d in range(1, total + 1) if total % d == 0]
    d = divisors[_ri(rng, 0, len(divisors) - 1)]
    newshape = [(total,), (d, total // d), (total // d, d)][_ri(rng, 0, 2)]
    return [x], {"newshape": list(newshape)}, False


def _s_concatenate(rng: Rng, dt: str, tiny: bool) -> Sampled:
    shape = list(_shape(rng, tiny, rank_lo=1))
    axis = _ri(rng, 0, len(shape) - 1)
    other = list(shape)
    other[axis] = _ri(rng, 1, 3 if tiny else 8)
    return (
        [_rand(rng, tuple(shape), dt), _rand(rng, tuple(other), dt)],
        {"axis": axis},
        True,
    )


def _s_conv(rng: Rng, dt: str, tiny: bool) -> Sampled:
    kh, kw = (_ri(rng, 1, 2), _ri(rng, 1, 2)) if tiny else (_ri(rng, 1, 3), _ri(rng, 1, 3))
    h = _ri(rng, kh, 4 if tiny else 7)
    w = _ri(rng, kw, 4 if tiny else 7)
    cin = 1 if tiny else _ri(rng, 1, 3)
    cout = 1 if tiny else _ri(rng, 1, 4)
    b = 1 if tiny else _ri(rng, 1, 2)
    x = _rand(rng, (b, h, w, cin), dt)
    kernel = _rand(rng, (kh, kw, cin, cout), dt, 0.5)
    strides = 1 if tiny else _ri(rng, 1, 2)
    padding = str(rng.choice(["valid", "same"]))
    return [x, kernel], {"strides": strides, "padding": padding}, False


def _s_argmax(rng: Rng, dt: str, tiny: bool) -> Sampled:
    x = _rand(rng, _shape(rng, tiny), dt)
    return [x], {"axis": _ri(rng, -x.ndim, x.ndim - 1)}, False


OP_SAMPLERS: dict[str, OpSampler] = {
    "matmul": _s_matmul,
    "add": _s_binary(),
    "subtract": _s_binary(),
    "multiply": _s_binary(),
    "divide": _s_binary(post=_away_from_zero),
    "softmax": _s_softmax,
    "relu": _s_unary(),
    "sigmoid": _s_unary(),
    "tanh": _s_unary(),
    "gelu": _s_unary(),
    "exp": _s_unary(scale=0.5),
    "log": _s_unary(pre=lambda a: np.abs(a) + 0.1),
    "sum": _s_reduction,
    "mean": _s_reduction,
    "max": _s_reduction,
    "min": _s_reduction,
    "var": _s_reduction,
    "std": _s_reduction,
    "transpose": _s_transpose,
    "reshape": _s_reshape,
    "concatenate": _s_concatenate,
    "conv": _s_conv,
    "argmax": _s_argmax,
}


# ---------------------------------------------------------------------------
# Layer sampling — kwargs + input + weights, weight shapes derived from the
# stable Keras contracts so the parent never needs to import keras.
# ---------------------------------------------------------------------------

LayerSampler = Callable[[Rng], tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]]


def _l_dense(rng: Rng) -> tuple[dict, list, list]:
    b, fin, units = _ri(rng, 1, 4), _ri(rng, 1, 8), _ri(rng, 1, 8)
    use_bias = bool(rng.random() < 0.8)
    kwargs = {
        "units": units,
        "use_bias": use_bias,
        "activation": str(rng.choice(["linear", "relu", "tanh"])),
    }
    weights = [_rand(rng, (fin, units), "float32", 0.5)]
    if use_bias:
        weights.append(_rand(rng, (units,), "float32", 0.5))
    return kwargs, [_rand(rng, (b, fin), "float32")], weights


def _l_conv2d(rng: Rng) -> tuple[dict, list, list]:
    k = _ri(rng, 1, 3)
    h, w = _ri(rng, k, 6), _ri(rng, k, 6)
    cin, filters, b = _ri(rng, 1, 3), _ri(rng, 1, 4), _ri(rng, 1, 2)
    use_bias = bool(rng.random() < 0.8)
    kwargs = {
        "filters": filters,
        "kernel_size": k,
        "padding": str(rng.choice(["valid", "same"])),
        "use_bias": use_bias,
    }
    weights = [_rand(rng, (k, k, cin, filters), "float32", 0.5)]
    if use_bias:
        weights.append(_rand(rng, (filters,), "float32", 0.5))
    return kwargs, [_rand(rng, (b, h, w, cin), "float32")], weights


def _l_batchnorm(rng: Rng) -> tuple[dict, list, list]:
    b, feat = _ri(rng, 1, 4), _ri(rng, 2, 8)
    weights = [
        _rand(rng, (feat,), "float32", 0.5) + 1.0,  # gamma
        _rand(rng, (feat,), "float32", 0.5),  # beta
        _rand(rng, (feat,), "float32", 0.5),  # moving_mean
        rng.uniform(0.5, 1.5, (feat,)).astype("float32"),  # moving_variance
    ]
    return {}, [_rand(rng, (b, feat), "float32")], weights


def _l_layernorm(rng: Rng) -> tuple[dict, list, list]:
    b, feat = _ri(rng, 1, 4), _ri(rng, 2, 8)
    weights = [
        _rand(rng, (feat,), "float32", 0.5) + 1.0,
        _rand(rng, (feat,), "float32", 0.5),
    ]
    return {}, [_rand(rng, (b, feat), "float32")], weights


def _l_embedding(rng: Rng) -> tuple[dict, list, list]:
    input_dim, output_dim = _ri(rng, 3, 12), _ri(rng, 2, 6)
    b, t = _ri(rng, 1, 3), _ri(rng, 1, 5)
    x = rng.integers(0, input_dim, (b, t)).astype("int32")
    return (
        {"input_dim": input_dim, "output_dim": output_dim},
        [x],
        [_rand(rng, (input_dim, output_dim), "float32", 0.5)],
    )


def _l_simplernn(rng: Rng) -> tuple[dict, list, list]:
    units, feat = _ri(rng, 1, 6), _ri(rng, 1, 4)
    b, t = _ri(rng, 1, 3), _ri(rng, 2, 5)
    kwargs = {"units": units, "return_sequences": bool(rng.random() < 0.5)}
    weights = [
        _rand(rng, (feat, units), "float32", 0.5),
        _rand(rng, (units, units), "float32", 0.5),
        _rand(rng, (units,), "float32", 0.5),
    ]
    return kwargs, [_rand(rng, (b, t, feat), "float32")], weights


LAYER_SAMPLERS: dict[str, LayerSampler] = {
    "Dense": _l_dense,
    "Conv2D": _l_conv2d,
    "BatchNormalization": _l_batchnorm,
    "LayerNormalization": _l_layernorm,
    "Embedding": _l_embedding,
    "SimpleRNN": _l_simplernn,
}


# ---------------------------------------------------------------------------
# Case generation (deterministic per (seed, index))
# ---------------------------------------------------------------------------


def generate_case(
    index: int,
    seed: int,
    kinds: tuple[str, ...],
    ops: tuple[str, ...],
    layers: tuple[str, ...],
    grad_ops: tuple[str, ...],
) -> Case:
    rng = np.random.default_rng([seed, index])
    kind = kinds[_ri(rng, 0, len(kinds) - 1)]
    if kind == "layer":
        name = layers[_ri(rng, 0, len(layers) - 1)]
        kwargs, inputs, weights = LAYER_SAMPLERS[name](rng)
        cid = f"{index:04d}-layer-{name.lower()}"
        return Case(cid, "layer", name, kwargs, inputs, weights)
    if kind == "grad":
        name = grad_ops[_ri(rng, 0, len(grad_ops) - 1)]
        inputs, kwargs, list_input = OP_SAMPLERS[name](rng, "float32", tiny=True)
        wrt = [i for i, a in enumerate(inputs) if a.dtype.kind == "f"]
        cid = f"{index:04d}-grad-{name}"
        return Case(cid, "grad", name, kwargs, inputs, list_input=list_input, grad_wrt=wrt)
    name = ops[_ri(rng, 0, len(ops) - 1)]
    inputs, kwargs, list_input = OP_SAMPLERS[name](rng, _pick_dtype(rng, tiny=False), tiny=False)
    cid = f"{index:04d}-op-{name}"
    return Case(cid, "op", name, kwargs, inputs, list_input=list_input)


def generate_cases(cfg: "Config") -> list[Case]:
    grad_ops = tuple(o for o in GRAD_OPS if o in cfg.ops) or GRAD_OPS
    return [generate_case(i, cfg.seed, cfg.kinds, cfg.ops, cfg.layers, grad_ops) for i in range(cfg.cases)]


# ---------------------------------------------------------------------------
# Child orchestration
# ---------------------------------------------------------------------------


@dataclass
class ChildRun:
    manifest: dict[str, dict[str, Any]]
    arrays: dict[str, np.ndarray]


class ChildFailure(RuntimeError):
    pass


def run_child(cfg: "Config", backend: str, batch: list[Case], fd: bool, workdir: str, tag: str) -> ChildRun:
    cases_path = os.path.join(workdir, f"{tag}-cases.json")
    arrays_path = os.path.join(workdir, f"{tag}-arrays.npz")
    out_path = os.path.join(workdir, f"{tag}-out.npz")
    manifest_path = os.path.join(workdir, f"{tag}-manifest.json")

    with open(cases_path, "w", encoding="utf-8") as fh:
        json.dump({"cases": [c.spec(fd and bool(c.grad_wrt)) for c in batch]}, fh)
    arrays: dict[str, np.ndarray] = {}
    for c in batch:
        arrays.update(c.arrays())
    np.savez_compressed(arrays_path, **arrays)

    env = dict(os.environ)
    env["KERAS_BACKEND"] = backend
    cmd = [
        cfg.python,
        CHILD_PATH,
        "--backend",
        backend,
        "--cases",
        cases_path,
        "--arrays",
        arrays_path,
        "--out-arrays",
        out_path,
        "--out-manifest",
        manifest_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=cfg.child_timeout)
    if proc.returncode != 0 or not os.path.exists(manifest_path):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise ChildFailure(f"child [{backend}] rc={proc.returncode}:\n" + "\n".join(tail))
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    out_arrays = dict(np.load(out_path)) if os.path.exists(out_path) else {}
    return ChildRun(manifest, out_arrays)


def run_child_safe(cfg: "Config", backend: str, batch: list[Case], fd: bool, workdir: str, tag: str) -> ChildRun:
    try:
        return run_child(cfg, backend, batch, fd, workdir, tag)
    except (ChildFailure, subprocess.TimeoutExpired) as exc:
        manifest = {c.cid: {"status": "error", "message": str(exc)} for c in batch}
        return ChildRun(manifest, {})


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def tol_for(dtype: str, scale: float) -> tuple[float, float]:
    rtol, atol = FLOAT_TOLS.get(dtype, FLOAT_TOLS["float32"])
    return rtol * scale, atol * scale


def is_keras_demotion(ref_dtype: np.dtype, test_dtype: np.dtype) -> bool:
    """The one benign dtype mismatch: ref float64 vs test float32, exactly.

    Keras demotes implicit float64 to float32 on every backend except
    tensorflow (the 64->32 entries in keras/src/backend/common/dtypes.py);
    the numpy reference keeps float64 only because its raw ops skip the
    conversion step.  The reverse direction, and any other pair, is real.
    """
    return ref_dtype == np.float64 and test_dtype == np.float32


def compare_arrays(label: str, ref: np.ndarray, test: np.ndarray, rtol: float, atol: float) -> ArrayCmp:
    if ref.shape != test.shape:
        return ArrayCmp(
            label,
            False,
            float("inf"),
            float("inf"),
            f"shape mismatch: ref {ref.shape} vs test {test.shape}",
        )
    if ref.dtype.kind in "iub":
        if test.dtype.kind not in "iub":
            return ArrayCmp(
                label,
                False,
                float("inf"),
                float("inf"),
                f"dtype kind mismatch: ref {ref.dtype} vs test {test.dtype}",
            )
        same = np.array_equal(ref.astype(np.int64), test.astype(np.int64))
        max_abs = 0.0 if same or ref.size == 0 else float(np.abs(ref.astype(np.int64) - test.astype(np.int64)).max())
        return ArrayCmp(
            label,
            same,
            max_abs,
            0.0 if same else float("inf"),
            "" if same else "integer outputs differ",
        )
    note, benign = "", True
    if is_keras_demotion(ref.dtype, test.dtype):
        note = f"keras 64->32 demotion (ref {ref.dtype}, test {test.dtype})"
    elif ref.dtype != test.dtype:
        note = f"dtype mismatch: ref {ref.dtype} vs test {test.dtype}"
        benign = False
    a = ref.astype(np.float64)
    b = test.astype(np.float64)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return ArrayCmp(label, False, float("inf"), float("inf"), "NaN placement differs")
    finite = np.isfinite(a) & np.isfinite(b)
    nonfinite = ~finite & ~nan_a
    if nonfinite.any() and not np.array_equal(a[nonfinite], b[nonfinite]):
        return ArrayCmp(label, False, float("inf"), float("inf"), "inf mismatch")
    if not finite.any():
        return ArrayCmp(label, benign, 0.0, 0.0, note)
    diff = np.abs(a[finite] - b[finite])
    denom = atol + rtol * np.abs(a[finite])
    ratio = float((diff / denom).max()) if diff.size else 0.0
    max_abs = float(diff.max()) if diff.size else 0.0
    ok = ratio <= 1.0 and benign
    return ArrayCmp(label, ok, max_abs, ratio, note)


def _case_dtype(case: Case) -> str:
    for a in case.inputs:
        if a.dtype.kind == "f":
            return str(a.dtype)
    return "float32"


def compare_case(case: Case, ref: ChildRun, test: ChildRun, cfg: "Config") -> CaseResult:
    rm = ref.manifest.get(case.cid)
    tm = test.manifest.get(case.cid)
    if rm is None or tm is None:
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            ERROR,
            message="no result from child (batch crash?)",
        )
    if tm["status"] == "unsupported":
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            UNSUPPORTED,
            message=f"under-test: {tm.get('message', '')}",
        )
    if tm["status"] == "error":
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            ERROR,
            message=f"under-test: {tm.get('message', '')}",
        )
    if rm["status"] == "unsupported":
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            SKIPPED,
            message=f"reference unsupported: {rm.get('message', '')}",
        )
    if rm["status"] == "error":
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            SKIPPED,
            message=f"reference error: {rm.get('message', '')}",
        )

    comparisons: list[ArrayCmp] = []
    notes: list[str] = []
    dtype = _case_dtype(case)
    fw_rtol, fw_atol = tol_for(dtype, cfg.tol_scale)

    if rm.get("n_outputs", 0) != tm.get("n_outputs", 0):
        return CaseResult(
            case.cid,
            case.kind,
            case.name,
            FAIL,
            worst_ratio=float("inf"),
            message=f"output arity differs: ref {rm.get('n_outputs')} vs test {tm.get('n_outputs')}",
        )
    for i in range(int(rm.get("n_outputs", 0))):
        key = f"{case.cid}__out{i}"
        ra, ta = ref.arrays[key], test.arrays[key]
        rtol, atol = fw_rtol, fw_atol
        if ra.dtype == ta.dtype and ra.dtype.kind == "f" and str(ra.dtype) in FLOAT_TOLS:
            # Both backends produced the same float dtype: that dtype judges.
            # This covers the BOTH-demoted case (float64 inputs, keras'
            # 64->32 policy applied by reference and test alike — see
            # docs/float64-promotion.md): two float32 computations must not
            # be judged under the float64 tolerances of the case's inputs.
            rtol, atol = tol_for(str(ra.dtype), cfg.tol_scale)
        elif is_keras_demotion(ra.dtype, ta.dtype):
            rtol, atol = tol_for("float32", cfg.tol_scale)  # the weaker dtype judges
        comparisons.append(compare_arrays(f"out{i}", ra, ta, rtol, atol))

    if case.grad_wrt:
        g_rtol, g_atol = fw_rtol * GRAD_TOL_SCALE, fw_atol * GRAD_TOL_SCALE
        ref_grads_ok = rm.get("grads_status") == "ok"
        test_grads_ok = tm.get("grads_status") == "ok"
        if not test_grads_ok:
            return CaseResult(
                case.cid,
                case.kind,
                case.name,
                UNSUPPORTED,
                comparisons=comparisons,
                message=f"under-test grads: {tm.get('grads_message', 'unsupported')}",
            )
        if ref_grads_ok:
            for gi in range(len(case.grad_wrt)):
                key = f"{case.cid}__grad{gi}"
                comparisons.append(compare_arrays(f"grad{gi}", ref.arrays[key], test.arrays[key], g_rtol, g_atol))
        else:
            notes.append("reference lacks autograd; cross-backend grad check skipped")
        if tm.get("fd"):
            fd_rtol, fd_atol = FD_TOLS[0] * cfg.tol_scale, FD_TOLS[1] * cfg.tol_scale
            for gi in range(len(case.grad_wrt)):
                comparisons.append(
                    compare_arrays(
                        f"fd{gi} (analytic vs central-diff)",
                        test.arrays[f"{case.cid}__fd{gi}"],
                        test.arrays[f"{case.cid}__grad{gi}"],
                        fd_rtol,
                        fd_atol,
                    )
                )
        elif not ref_grads_ok:
            notes.append("no grad check performed (pass --slow for finite differences)")

    worst = max((c.ratio for c in comparisons), default=0.0)
    max_abs = max((c.max_abs for c in comparisons), default=0.0)
    failed = [c for c in comparisons if not c.ok]
    notes.extend(f"{c.label}: {c.note}" for c in comparisons if c.ok and c.note)
    status = FAIL if failed else PASS
    message = "; ".join([f"{c.label}: {c.note or f'ratio {c.ratio:.2f}x tol'}" for c in failed] + notes)
    return CaseResult(
        case.cid,
        case.kind,
        case.name,
        status,
        worst_ratio=worst,
        max_abs=max_abs,
        message=message,
        comparisons=comparisons,
    )


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    seed: int
    cases: int
    ops: tuple[str, ...]
    layers: tuple[str, ...]
    kinds: tuple[str, ...]
    backend_under_test: str
    reference: str
    slow: bool
    top_k: int
    json_path: str | None
    repro: str | None
    batch_size: int
    python: str
    tol_scale: float
    child_timeout: float
    keep_artifacts: str | None


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def resolve_kinds(args: argparse.Namespace) -> tuple[tuple[str, ...], str | None]:
    """Pick enabled case kinds; returns (kinds, note-or-None)."""
    if args.kinds:
        kinds = _csv(args.kinds)
    else:
        kinds = ("op", "layer", "grad")
        if args.ops and not args.layers:
            kinds = ("op", "grad")
        elif args.layers and not args.ops:
            kinds = ("layer",)
    note = None
    grads_possible = args.slow or args.reference in AUTOGRAD_BACKENDS
    if "grad" in kinds and not grads_possible:
        kinds = tuple(k for k in kinds if k != "grad")
        note = (
            f"gradient cases skipped: reference '{args.reference}' has no autograd "
            f"and --slow (finite differences) was not given"
        )
    bad = set(kinds) - {"op", "layer", "grad"}
    if bad:
        raise SystemExit(f"unknown kinds: {sorted(bad)}")
    if not kinds:
        raise SystemExit("no case kinds enabled")
    return kinds, note


def parse_args(argv: list[str]) -> tuple[Config, str | None]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], prog="parity_fuzz.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", type=int, default=100)
    p.add_argument(
        "--ops",
        type=_csv,
        default=None,
        help=f"comma list; default: {','.join(DEFAULT_OPS)}",
    )
    p.add_argument(
        "--layers",
        type=_csv,
        default=None,
        help=f"comma list; default: {','.join(DEFAULT_LAYERS)}",
    )
    p.add_argument("--kinds", default=None, help="comma list of op,layer,grad")
    p.add_argument("--backend-under-test", default="tinygrad")
    p.add_argument("--reference", default="numpy", help="any KERAS_BACKEND value (numpy default)")
    p.add_argument(
        "--slow",
        action="store_true",
        help="finite-difference gradient checking on the backend under test",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json", dest="json_path", default=None)
    p.add_argument(
        "--repro",
        default=None,
        metavar="CASE_ID",
        help="re-run one case verbosely (same seed/filters required)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--tol-scale", type=float, default=1.0)
    p.add_argument("--child-timeout", type=float, default=600.0)
    p.add_argument("--keep-artifacts", default=None, metavar="DIR")
    args = p.parse_args(argv)

    for op in args.ops or ():
        if op not in OP_SAMPLERS:
            raise SystemExit(f"unknown op '{op}' (known: {', '.join(sorted(OP_SAMPLERS))})")
    for layer in args.layers or ():
        if layer not in LAYER_SAMPLERS:
            raise SystemExit(f"unknown layer '{layer}' (known: {', '.join(sorted(LAYER_SAMPLERS))})")

    kinds, note = resolve_kinds(args)
    cfg = Config(
        seed=args.seed,
        cases=args.cases,
        ops=args.ops or DEFAULT_OPS,
        layers=args.layers or DEFAULT_LAYERS,
        kinds=kinds,
        backend_under_test=args.backend_under_test,
        reference=args.reference,
        slow=args.slow,
        top_k=args.top_k,
        json_path=args.json_path,
        repro=args.repro,
        batch_size=args.batch_size,
        python=args.python,
        tol_scale=args.tol_scale,
        child_timeout=args.child_timeout,
        keep_artifacts=args.keep_artifacts,
    )
    return cfg, note


def repro_command(cfg: Config, cid: str) -> str:
    """Command that regenerates exactly one case AND its verdict.

    Case identity depends on (seed, cases, ops, layers, kinds); the verdict
    additionally depends on tol-scale and slow.  All of them travel, always —
    an omitted filter silently regenerates a DIFFERENT case (or re-judges the
    same case under default tolerance and prints PASS for a real failure).
    Resolved kinds are passed explicitly so the repro run does not re-derive
    them from flag heuristics.
    """
    parts = [
        cfg.python,
        os.path.abspath(sys.argv[0]),
        "--seed",
        str(cfg.seed),
        "--cases",
        str(cfg.cases),
    ]
    if cfg.ops != DEFAULT_OPS:
        parts += ["--ops", ",".join(cfg.ops)]
    if cfg.layers != DEFAULT_LAYERS:
        parts += ["--layers", ",".join(cfg.layers)]
    parts += ["--kinds", ",".join(cfg.kinds)]
    parts += [
        "--backend-under-test",
        cfg.backend_under_test,
        "--reference",
        cfg.reference,
    ]
    if cfg.slow:
        parts.append("--slow")
    if cfg.tol_scale != 1.0:
        parts += ["--tol-scale", repr(cfg.tol_scale)]
    parts += ["--repro", cid]
    return shlex.join(parts)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _array_stats(name: str, a: np.ndarray) -> str:
    if a.dtype.kind == "f" and a.size:
        return f"  {name}: shape={a.shape} dtype={a.dtype} min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}"
    return f"  {name}: shape={a.shape} dtype={a.dtype}"


def print_repro(case: Case, result: CaseResult) -> None:
    print(f"case {case.cid}  kind={case.kind} name={case.name}")
    print(f"  kwargs: {json.dumps(case.kwargs)}")
    for i, a in enumerate(case.inputs):
        print(_array_stats(f"in{i}", a))
    for i, a in enumerate(case.weights):
        print(_array_stats(f"w{i}", a))
    for c in result.comparisons:
        verdict = "ok" if c.ok else "FAIL"
        print(f"  {c.label}: {verdict}  max_abs={c.max_abs:.4g} ratio={c.ratio:.4g}x tol  {c.note}")
    print(f"  status: {result.status}  {result.message}")


def print_summary(cfg: Config, results: list[CaseResult]) -> None:
    counts = {s: 0 for s in (PASS, FAIL, UNSUPPORTED, SKIPPED, ERROR)}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"parity_fuzz  seed={cfg.seed}  cases={len(results)}  test={cfg.backend_under_test}  ref={cfg.reference}")
    print("  " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    offenders = sorted((r for r in results if r.status in (FAIL, ERROR)), key=lambda r: -r.worst_ratio)[: cfg.top_k]
    if offenders:
        print(f"top {len(offenders)} offender(s):")
        for i, r in enumerate(offenders, 1):
            print(f"  {i}. {r.cid} [{r.status}] worst={r.worst_ratio:.3g}x tol max_abs={r.max_abs:.3g}  {r.message}")
            print(f"     repro: {repro_command(cfg, r.cid)}")


def write_json(cfg: Config, results: list[CaseResult], path: str) -> None:
    def _finite(x: float) -> float | str:
        return x if np.isfinite(x) else "inf"

    report = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()},
        "tolerances": {
            "float": FLOAT_TOLS,
            "grad_scale": GRAD_TOL_SCALE,
            "fd": FD_TOLS,
        },
        "summary": {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, UNSUPPORTED, SKIPPED, ERROR)},
        "results": [
            {
                **r.to_json(),
                "worst_ratio": _finite(r.worst_ratio),
                "comparisons": [
                    {
                        **dataclasses.asdict(c),
                        "ratio": _finite(c.ratio),
                        "max_abs": _finite(c.max_abs),
                    }
                    for c in r.comparisons
                ],
                "repro": repro_command(cfg, r.cid),
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _chunks(items: list[Case], size: int) -> list[list[Case]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main(argv: list[str]) -> int:
    cfg, note = parse_args(argv)
    if note:
        print(f"note: {note}")
    cases = generate_cases(cfg)
    if cfg.repro is not None:
        cases = [c for c in cases if c.cid == cfg.repro]
        if not cases:
            print(
                f"error: case '{cfg.repro}' not found — repro requires the same "
                f"--seed/--cases/--ops/--layers/--kinds as the original run",
                file=sys.stderr,
            )
            return 2

    workdir = cfg.keep_artifacts or tempfile.mkdtemp(prefix="parity-fuzz-")
    os.makedirs(workdir, exist_ok=True)
    results: list[CaseResult] = []
    for bi, batch in enumerate(_chunks(cases, cfg.batch_size)):
        ref = run_child_safe(cfg, cfg.reference, batch, fd=False, workdir=workdir, tag=f"b{bi:03d}-ref")
        test = run_child_safe(
            cfg,
            cfg.backend_under_test,
            batch,
            fd=cfg.slow,
            workdir=workdir,
            tag=f"b{bi:03d}-test",
        )
        results.extend(compare_case(c, ref, test, cfg) for c in batch)

    if cfg.repro is not None:
        print_repro(cases[0], results[0])
    print_summary(cfg, results)
    if cfg.json_path:
        write_json(cfg, results, cfg.json_path)
        print(f"json report: {cfg.json_path}")
    if cfg.keep_artifacts:
        print(f"artifacts kept in: {workdir}")
    if any(r.status in (FAIL, ERROR) for r in results):
        return 1
    if not any(r.status == PASS for r in results):
        # Nothing was actually compared (reference dead, everything skipped or
        # unsupported).  Exit 0 here would be a silent green with zero evidence.
        print(
            "error: no case was compared successfully — treating as an infrastructure failure, not a pass",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
