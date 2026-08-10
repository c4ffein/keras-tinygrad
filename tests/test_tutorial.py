"""The tutorial is executable: every ```python block in TUTORIAL.md runs,
in order, in one fresh subprocess. If the tutorial drifts from the code,
this test goes red — the page cannot rot.

Kept out of `make tests-fast` (it trains small models); run via
`make tutorial` or plain pytest on this file.
"""

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIAL = os.path.join(REPO_ROOT, "TUTORIAL.md")

FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_tutorial_blocks_run():
    with open(TUTORIAL, encoding="utf-8") as fh:
        blocks = FENCE.findall(fh.read())
    assert len(blocks) >= 5, "tutorial lost its code blocks?"
    program = "\n\n".join(blocks)

    env = dict(os.environ)
    env.pop("KERAS_BACKEND", None)  # the tutorial exercises the hook default
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, f"tutorial code failed:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
