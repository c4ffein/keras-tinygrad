"""Resource limits for every subprocess the tests launch.

All backend work in this suite runs in child processes (a Keras process is
locked to one backend at import). A backend bug that builds a runaway lazy
graph does not fail there — it eats the machine: on 2026-09-21 an unbounded
Jacobi graph pinned an 8-core / 31 GB box (load average 174) until the
process was killed from the host. With these limits the same bug is a
MemoryError / SIGXCPU in the child, i.e. a red test with a traceback.

Numbers: a bare `import keras_tinygrad, keras` peaks at 0.2 GB RSS / 0.7 GB
of address space; the heaviest test (test_ops_beyond_pin's probe) at
2.95 GB RSS / 4.05 GB of address space (measured 2026-09-22, tinygrad
0.14.0, CPU device, zig cc — up from 0.7 / 1.9 GB on the 2026-09-21
measurement: 0.14's `fold_where_closure` rewrite builds per-node
`bool_slice` frozensets over the graph, and the probe's gammainc graphs
are where-heavy; results stay correct, the pass is just hungrier).
RLIMIT_AS caps ADDRESS SPACE, not RSS, hence 8 GB for a 4.05 GB peak and
not something that looks tighter — 4 GB was exactly the old cliff: CI
passed under it by a few pages, this box MemoryError'd. The limits are
inherited by the compiler processes tinygrad spawns (checked: zig cc
compiles under a 3 GB cap).
"""

import os

try:
    import resource
except ImportError:  # not a POSIX box: no limits, the tests still run
    resource = None

# 0 disables a limit (e.g. a GPU device that maps a large address range).
MEM_MB = int(os.environ.get("KERAS_TINYGRAD_TEST_MEM_MB", "8192"))
CPU_SECONDS = int(os.environ.get("KERAS_TINYGRAD_TEST_CPU_S", "1800"))


def child_limits():
    """`preexec_fn` for subprocess.run: runs in the child, before exec."""
    if resource is None:
        return
    for limit, value in ((resource.RLIMIT_AS, MEM_MB << 20), (resource.RLIMIT_CPU, CPU_SECONDS)):
        if value > 0:
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass  # a lower hard limit is already in force, or the OS ignores this one
    os.nice(10)
