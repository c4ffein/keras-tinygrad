"""Vendored third-party code shipped inside the package.

`export_model.py` is tinygrad's exporter from the `extra/` tree of tag
v0.14.0 (not in the PyPI wheel; ported from the v0.13.0 copy 2026-09-21 by
applying upstream's eight-hunk v0.13.0..v0.14.0 diff), reformatted to this repo's ruff style —
semantically identical to upstream, not byte-identical. This is THE copy:
`keras_tinygrad.webgpu` and the live experiments import it from here
(`experiments/m0-keras-trainstep/m0.py`). The archived raw-tinygrad
experiment `experiments/pyodide-tinygrad/` ships its own frozen copy into
Pyodide, where this package is not installed; it is not kept in sync on
purpose. The tinygrad `>=0.14,<0.15` pin is load-bearing for this file: the
exporter follows tinygrad's internals (PARAM / AddrSpace / ContextVar
names) and breaks on every minor.
"""
