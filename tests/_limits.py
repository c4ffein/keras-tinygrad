"""Resource limits for every subprocess the tests launch.

All backend work in this suite runs in child processes (a Keras process is
locked to one backend at import). A backend bug that builds a runaway lazy
graph does not fail there — it eats the machine: on 2026-09-21 an unbounded
Jacobi graph pinned an 8-core / 31 GB box (load average 174) until the
process was killed from the host. With these limits the same bug is a
MemoryError / SIGXCPU in the child, i.e. a red test with a traceback.

Numbers: a bare `import keras_tinygrad, keras` peaks at 0.2 GB RSS / 0.7 GB
of address space; the heaviest test (test_ops_beyond_pin's probe) at 0.7 GB
/ 1.9 GB (measured 2026-09-21, CPU device, zig cc). RLIMIT_AS caps ADDRESS
SPACE, not RSS, hence 4 GB and not something that looks tighter. The limits
are inherited by the compiler processes tinygrad spawns (checked: zig cc
compiles under a 3 GB cap).
"""

import os

try:
    import resource
except ImportError:  # not a POSIX box: no limits, the tests still run
    resource = None

# 0 disables a limit (e.g. a GPU device that maps a large address range).
MEM_MB = int(os.environ.get("KERAS_TINYGRAD_TEST_MEM_MB", "4096"))
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
