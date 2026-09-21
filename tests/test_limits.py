"""The limiter itself: a runaway child must die loudly, not take the box."""

import subprocess
import sys

import pytest
from _limits import MEM_MB, child_limits, resource


@pytest.mark.skipif(resource is None or MEM_MB <= 0, reason="no RLIMIT_AS on this platform / limit disabled")
def test_memory_cap_turns_a_runaway_child_into_a_failure():
    code = f"x = bytearray({(MEM_MB + 512) << 20}); print('allocated')"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, preexec_fn=child_limits
    )
    assert out.returncode != 0 and "MemoryError" in out.stderr, (out.returncode, out.stdout, out.stderr[-300:])


def test_limits_do_not_break_a_normal_child():
    out = subprocess.run(
        [sys.executable, "-c", "print('ok')"], capture_output=True, text=True, timeout=60, preexec_fn=child_limits
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
