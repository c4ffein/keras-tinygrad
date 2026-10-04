#!/usr/bin/env bash
# Run a command on tinygrad's OpenCL device with NO GPU: the PyPI wheel
# `pocl-binary-distribution` ships PoCL, a CPU OpenCL implementation, as a
# shared library plus its ICD file. tinygrad's CL runtime then goes through
# the system ICD loader (`libOpenCL.so.1`, package ocl-icd-libopencl1 on
# Debian/Ubuntu) to PoCL, and every kernel is rendered by tinygrad's OpenCL
# renderer — the same codegen a real GPU would get, executed on the CPU.
# Not a speed test (2026-09-28, this box: the referee-quick slice takes
# 3:06 on CL vs ~1 min on CPU) — a second renderer for the referee.
#
#   scripts/with_cl.sh bash scripts/referee.sh keras/src/layers/core/dense_test.py
#   scripts/with_cl.sh uv run python examples/mlp_smoke.py
#
# The wheel lands in .cl/ (gitignored) on first use. Tinygrad's other
# no-hardware option is DEV=PYTHON::sm_80 / ::METAL / ::gfx1100 — its pure
# python UOp interpreter with that GPU's renderer rules (tensor cores
# included); it runs a Keras fit but ~100x slower than CPU, so it is a
# spot-check tool, not a referee host.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLDIR="$ROOT/.cl"
if ! ls "$CLDIR"/lib/python*/site-packages/pyopencl/.libs/pocl.icd >/dev/null 2>&1; then
  echo "with_cl: installing pocl-binary-distribution into $CLDIR (once)"
  uv venv -q "$CLDIR" --python 3.12
  uv pip install -q --python "$CLDIR/bin/python" pocl-binary-distribution
fi
ICD_DIR=$(dirname "$(ls "$CLDIR"/lib/python*/site-packages/pyopencl/.libs/pocl.icd | head -1)")
if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so\.1'; then
  echo "with_cl: no system OpenCL ICD loader (libOpenCL.so.1) — install ocl-icd-libopencl1" >&2; exit 1
fi
export OCL_ICD_VENDORS="$ICD_DIR" LD_LIBRARY_PATH="$ICD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" DEV=CL
echo "with_cl: DEV=CL via PoCL at $ICD_DIR"
exec "$@"
