"""Receipts for ops keras 3.15.x's own suite cannot referee.

`copysign`, `float_power`, `cov` (numpy), `lgamma`, `gammainc` (math) are
keras MASTER ops (3.16-dev) — the pinned 3.15.1 tree has no tests for
them, so the master test cases are carried here, against the same numpy /
scipy references, until the pin moves. Also the `MissingOpError` contract
keras master relies on (`hasattr(backend.ops.numpy, "x")` probes).

One subprocess (a Keras process is locked to one backend at import) runs
every check and reports per-check results; each test reads its own.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from _limits import child_limits

REPO = pathlib.Path(__file__).resolve().parent.parent

_PROBE = """
import json
import math

import keras_tinygrad  # noqa: F401  (must precede keras)
import numpy as np
import scipy.special as sp
from tinygrad import Tensor

import keras
from keras_tinygrad.src.ops import math as tmath
from keras_tinygrad.src.ops import numpy as tnp
from keras_tinygrad.src.ops.core import MissingOpError
from keras_tinygrad.src.ops.core import convert_to_numpy as cn


results = {}


def close(name, got, exp, **kw):
    g = cn(got)
    results[name] = bool(np.allclose(g, exp, equal_nan=True, **kw))


# --- copysign: the master test, plus every float width's sign bits -------
x = np.array([[1, -2, 3], [-3, 2, -1]])
y = np.array([[-4, 5, -6], [3, -2, 1]])
close("copysign_int", tnp.copysign(x, y), np.copysign(x, y))
for dt in ("float32", "float16"):
    x = np.array([1.0, -1.0, 0.0, -0.0, np.inf, -np.inf], dt)
    y = np.array([-0.0, 0.0, -np.inf, np.inf, 1.0, -1.0], dt)
    g = cn(tnp.copysign(x, y))
    e = np.copysign(x, y)
    results[f"copysign_signed_zeros_{dt}"] = bool(
        np.allclose(g, e) and np.array_equal(np.signbit(g), np.signbit(e))
    )
results["copysign_dtype"] = str(cn(tnp.copysign(x, y)).dtype) == "float16"

# --- float_power: the master test (negative exponents on int inputs) -----
x = np.array([[1, 2, 3], [3, 2, 1]])
y = np.array([[4, 5, 6], [3, 2, 1]])
close("float_power", tnp.float_power(x, y), np.float_power(x, y))
y = np.array([[-1, -2, -3], [-1, -2, -3]])
close("float_power_negative", tnp.float_power(x, y), np.float_power(x, y))
results["float_power_dtype"] = str(cn(tnp.float_power(x, y)).dtype) == "float32"

# --- cov: the master test ------------------------------------------------
x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
close("cov_2d", tnp.cov(x), np.cov(x))
x = np.array([1.0, 4.0, 2.0, 8.0])
close("cov_1d", tnp.cov(x), np.cov(x))
close("cov_single_variable", tnp.cov(x[None, :]), np.cov(x[None, :]))
results["cov_scalar_is_nan"] = bool(np.isnan(cn(tnp.cov(3.0))))
try:
    tnp.cov(np.ones((2, 3, 4)))
    results["cov_3d_raises"] = False
except ValueError:
    results["cov_3d_raises"] = True

# --- lgamma: the master tests (atol 1e-4 vs scipy.special.gammaln) -------
sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 10.0], "float32")
close("lgamma_basic", tmath.lgamma(sample), sp.gammaln(sample), atol=1e-4)
edge = np.array([-0.5, -1.5, -2.5, 1e-5, 50.0, 100.0, float("inf")])
close("lgamma_edges", tmath.lgamma(edge), sp.gammaln(edge), atol=1e-4)
poles = np.array([0.0, -1.0, -2.0, -np.inf, 1e6, -7.7], "float32")
close("lgamma_poles", tmath.lgamma(poles), sp.gammaln(poles), rtol=1e-5, atol=1e-4)
t = Tensor([0.7, 3.0, -0.3])
(g,) = tmath.lgamma(t).sum().gradient(t)
results["lgamma_grad_is_digamma"] = bool(
    np.allclose(g.numpy(), sp.digamma([0.7, 3.0, -0.3]), rtol=1e-4, atol=1e-4)
)

# --- gammainc: the master test, a sweep across both branches, the edges --
x1 = np.array([[1.0, 2.0], [3.0, 4.0]], "float32")
x2 = np.array([[0.5, 1.0], [2.0, 5.0]], "float32")
close("gammainc_master", tmath.gammainc(x1, x2), sp.gammainc(x1, x2))
a = np.array([0.1, 0.5, 1.0, 2.5, 10.0, 50.0, 200.0], "float32")
sweep = True
for xv in (0.05, 0.5, 2.0, 9.0, 11.0, 49.0, 60.0, 200.0):
    xs = np.full_like(a, xv)
    # float32 cancellation in the log prefactor costs ~1e-5 relative for
    # a >~ 50 (documented next to the op).
    sweep &= bool(
        np.allclose(cn(tmath.gammainc(a, xs)), sp.gammainc(a, xs), rtol=5e-5, atol=2e-6)
    )
results["gammainc_sweep"] = sweep
ea = np.array([0.0, -1.0, 2.0, 2.0, 2.0, 1e-3, 0.0], "float32")
ex = np.array([1.0, 1.0, -1.0, 0.0, np.inf, 0.0, 0.0], "float32")
close("gammainc_edges", tmath.gammainc(ea, ex), sp.gammainc(ea, ex))
close(
    "gammainc_broadcast",
    tmath.gammainc(np.array([[2.0], [7.5]], "float32"), np.array([3.0, 6.0], "float32")),
    sp.gammainc([[2.0], [7.5]], [3.0, 6.0]),
)
at, xt = Tensor([2.0, 5.0]), Tensor([1.5, 7.0])
ga, gx = tmath.gammainc(at, xt).sum().gradient(at, xt)
h = 1e-4
fd_x = (sp.gammainc([2.0, 5.0], [1.5 + h, 7.0 + h]) - sp.gammainc([2.0, 5.0], [1.5 - h, 7.0 - h])) / (2 * h)
fd_a = (sp.gammainc([2.0 + h, 5.0 + h], [1.5, 7.0]) - sp.gammainc([2.0 - h, 5.0 - h], [1.5, 7.0])) / (2 * h)
results["gammainc_grad_x"] = bool(np.allclose(gx.numpy(), fd_x, rtol=1e-3, atol=1e-5))
results["gammainc_grad_a"] = bool(np.allclose(ga.numpy(), fd_a, rtol=1e-3, atol=1e-5))
# Edge elements (x = 0, x = inf) must not poison the gradient: not their
# own, and not the other elements' through a broadcast `a`.
at, xt = Tensor([2.0]), Tensor([1.5, 0.0, float("inf")])
ga, gx = tmath.gammainc(at, xt).sum().gradient(at, xt)
results["gammainc_grad_edges_finite"] = bool(
    np.allclose(ga.numpy(), fd_a[:1], rtol=1e-3, atol=1e-5)
    and np.allclose(gx.numpy(), [fd_x[0], 0.0, 0.0], rtol=1e-3, atol=1e-5)
)

# --- the loud-stub contract ---------------------------------------------
results["missing_op_absent_for_hasattr"] = not hasattr(tnp, "definitely_not_an_op")
results["missing_op_getattr_default"] = getattr(tmath, "definitely_not_an_op", None) is None
try:
    tnp.definitely_not_an_op(1)
    results["missing_op_call_is_loud"] = False
except MissingOpError as e:
    results["missing_op_call_is_loud"] = isinstance(e, NotImplementedError) and isinstance(
        e, AttributeError
    )
# keras.ops itself: the public op still errors loudly, not silently.
try:
    keras.ops.unique(np.array([1, 1, 2]))
    results["public_missing_op_is_loud"] = False
except NotImplementedError:
    results["public_missing_op_is_loud"] = True

print(json.dumps(results))
"""

CHECKS = (
    "copysign_int",
    "copysign_signed_zeros_float32",
    "copysign_signed_zeros_float16",
    "copysign_dtype",
    "float_power",
    "float_power_negative",
    "float_power_dtype",
    "cov_2d",
    "cov_1d",
    "cov_single_variable",
    "cov_scalar_is_nan",
    "cov_3d_raises",
    "lgamma_basic",
    "lgamma_edges",
    "lgamma_poles",
    "lgamma_grad_is_digamma",
    "gammainc_master",
    "gammainc_sweep",
    "gammainc_edges",
    "gammainc_broadcast",
    "gammainc_grad_x",
    "gammainc_grad_a",
    "gammainc_grad_edges_finite",
    "missing_op_absent_for_hasattr",
    "missing_op_getattr_default",
    "missing_op_call_is_loud",
    "public_missing_op_is_loud",
)


@pytest.fixture(scope="module")
def results():
    env = dict(os.environ, KERAS_BACKEND="tinygrad", KERAS_TINYGRAD_NO_VERSION_WARNING="1")
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=900,
        preexec_fn=child_limits,
    )
    assert out.returncode == 0, f"probe subprocess failed:\n{out.stdout}\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("check", CHECKS)
def test_beyond_pin(results, check):
    assert check in results, f"probe reported no result for {check}"
    assert results[check], f"{check} failed (see the probe in {__file__})"
