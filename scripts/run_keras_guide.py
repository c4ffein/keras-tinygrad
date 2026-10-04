"""Run one keras.io guide (a `guides/*.py` from keras-team/keras-io) under the
tinygrad backend, capped, in a subprocess — one guide per invocation.

    uv run python scripts/run_keras_guide.py <path/to/guide.py> [--timeout S]

Prints the outcome line the caller greps for:
    GUIDE <name> PASS|FAIL|TIMEOUT wall=<s> exit=<code> [first error line]

Caps mirror tests/_limits.py (RLIMIT_AS, RLIMIT_CPU, nice, no core files):
a guide that builds a runaway graph is a red line here, not a dead box.
`import keras_tinygrad` happens BEFORE the guide's `import keras` (the
import hook must be installed first); KERAS_BACKEND is forced to tinygrad.
Guides are run with `runpy` from their own directory (relative asset paths).
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

MEM_MB = int(os.environ.get("KERAS_TINYGRAD_TEST_MEM_MB", "8192"))
CPU_SECONDS = int(os.environ.get("KERAS_TINYGRAD_TEST_CPU_S", "1800"))

CHILD = """
import os, sys, runpy
import keras_tinygrad  # noqa: F401  (must precede keras)
guide = sys.argv[1]
if os.environ.get("KTG_GUIDE_NO_PLOT_MODEL") == "1":
    # `keras.utils.plot_model` raises ImportError outside a notebook when
    # pydot or the graphviz `dot` binary is missing; the picture is not what
    # a guide run proves. The harness replaces it, the clone stays untouched.
    import keras
    import keras.src.utils.model_visualization as mv
    def _no_plot(model, to_file="model.png", *a, **k):
        print(f"[run_keras_guide] plot_model({to_file}) skipped: --no-plot-model")
    keras.utils.plot_model = mv.plot_model = _no_plot
os.chdir(os.path.dirname(os.path.abspath(guide)))
runpy.run_path(guide, run_name="__main__")
"""


def child_limits():
    import resource

    for limit, value in ((resource.RLIMIT_AS, MEM_MB << 20), (resource.RLIMIT_CPU, CPU_SECONDS)):
        if value > 0:
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    os.nice(10)


def first_error(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):  # the exception line is the last non-empty one of a traceback
        if ln[:1] not in (" ", "\t") and ("Error" in ln or "Exception" in ln or "error" in ln):
            return ln.strip()[:200]
    return lines[-1].strip()[:200] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("guide")
    ap.add_argument("--timeout", type=int, default=900, help="wall-clock seconds (default 900)")
    ap.add_argument("--log", help="write the child's stdout+stderr here")
    ap.add_argument(
        "--no-plot-model",
        action="store_true",
        help="replace keras.utils.plot_model with a no-op (no graphviz on this box); the rest of the guide runs as is",
    )
    args = ap.parse_args()
    guide = pathlib.Path(args.guide).resolve()
    name = guide.stem
    env = dict(os.environ, KERAS_BACKEND="tinygrad", PYTHONUNBUFFERED="1")
    if args.no_plot_model:
        env["KTG_GUIDE_NO_PLOT_MODEL"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(guide)],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            preexec_fn=child_limits,
        )
        wall = time.time() - t0
        out, err, code = proc.stdout, proc.stderr, proc.returncode
        status = "PASS" if code == 0 else "FAIL"
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, status = -1, "TIMEOUT"
    if args.log:
        pathlib.Path(args.log).write_text(out + "\n--- stderr ---\n" + err)
    tail = "" if status == "PASS" else " " + first_error(err)
    print(f"GUIDE {name} {status} wall={wall:.0f}s exit={code}{tail}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
