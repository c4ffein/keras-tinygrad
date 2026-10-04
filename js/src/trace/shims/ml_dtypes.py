"""Stand-in for the `ml_dtypes` C extension under Pyodide (no wasm wheel).

keras 3.15.1 + the tinygrad backend touch it in three ways: `finfo`/`iinfo`
on ordinary dtypes (delegated to numpy), `finfo` on float8 names inside the
float8-quantization paths, and `bfloat16` when SAVING bf16 weights. The
narrow dtypes cannot exist without the extension, so asking for them raises
loudly instead of returning a wrong answer.
"""

import numpy as np

_NARROW = (
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e4m3b11fnuz",
    "int4",
    "uint4",
    "int2",
    "uint2",
)


class _Unavailable:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<ml_dtypes shim: {self.name} unavailable under Pyodide>"

    def __call__(self, *a, **k):
        raise NotImplementedError(f"ml_dtypes shim: dtype {self.name} is unavailable under Pyodide")


for _n in _NARROW:
    globals()[_n] = _Unavailable(_n)


def _name(dtype):
    return dtype.name if isinstance(dtype, _Unavailable) else str(getattr(dtype, "name", dtype))


def finfo(dtype):
    if _name(dtype) in _NARROW:
        raise NotImplementedError(f"ml_dtypes shim: finfo({_name(dtype)}) needs the real ml_dtypes (no wasm wheel)")
    return np.finfo(dtype)


def iinfo(dtype):
    if _name(dtype) in _NARROW:
        raise NotImplementedError(f"ml_dtypes shim: iinfo({_name(dtype)}) needs the real ml_dtypes (no wasm wheel)")
    return np.iinfo(dtype)
