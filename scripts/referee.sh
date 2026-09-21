#!/usr/bin/env bash
# The referee: Keras' OWN test suite run against this package.
#
# No sibling checkout, nothing edited by hand. This script
#   1. clones keras-team/keras at the tag matching the installed pin into
#      .referee/keras-v<ver> (shallow; reused on later runs),
#   2. builds .referee/venv-<ver> (python 3.12 — tensorflow-cpu + grain are
#      needed merely to COLLECT the preprocessing tests; jax for a few
#      numerics comparisons) with this package installed editable,
#   3. runs pytest from INSIDE the keras tree with the import hook active
#      (`-p keras_tinygrad`): the tree's stock sources get the six dispatch
#      patches at import, exactly as an end user's keras does,
#   4. compares the FAILED set against scripts/referee-baseline.txt —
#      any difference (new failure OR newly-green test) exits 1, so the
#      baseline is only ever moved deliberately, with a real tally.
#
#   scripts/referee.sh                       # full tree of record: keras/src/layers
#   scripts/referee.sh keras/src/ops/nn_test.py   # any paths: known failures stay green, any other is red
#   KERAS_VERSION=3.15.0 scripts/referee.sh  # pin override
#   PYTEST_ARGS="-x -k dense" scripts/referee.sh  # a subset run, judged like custom paths
#
# Guard rails (2026-09-21: a runaway lazy graph in one linalg test pinned the
# whole box — load average 174 — until the process was killed from the
# host). The run is niced, its address space is capped (REFEREE_MEM_MB,
# default 8192: a TF-collecting run peaks at 3.9 GB of address space / 0.7 GB
# RSS; 0 disables), and every test has a wall-clock budget
# (REFEREE_TEST_TIMEOUT seconds, default 900; 0 disables) — so a runaway is
# ONE red test with a traceback, and the rest of the tree still runs.
#
# Runtime: the layers tree is ~25 min single-process. CI runs it weekly and
# on demand (.github/workflows/referee.yml); locally, `make referee`.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
REFEREE_DIR=${REFEREE_DIR:-$REPO/.referee}
REFEREE_PYTHON=${REFEREE_PYTHON:-3.12}
BASELINE=$REPO/scripts/referee-baseline.txt
DEFAULT_PATHS=(keras/src/layers)

KERAS_VERSION=${KERAS_VERSION:-$(uv run --project "$REPO" python -c \
  'import importlib.metadata as m; print(m.version("keras"))')}
# The DEPENDENCY string, i.e. one with a version operator: the bare
# `"tinygrad"` in pyproject's keywords comes first in the file, and until
# 2026-09-21 that is what this matched — the venv got the latest tinygrad
# (0.14.0), not the pin.
TINYGRAD_PIN=$(grep -oE '"tinygrad[<>=!~][^"]*"' "$REPO/pyproject.toml" | head -1 | tr -d '"')
[ -n "$TINYGRAD_PIN" ] || { echo "referee: no tinygrad version pin found in pyproject.toml"; exit 2; }
TREE=$REFEREE_DIR/keras-v$KERAS_VERSION
VENV=$REFEREE_DIR/venv-$KERAS_VERSION

# tinygrad's CPU jit shells out to clang; the ziglang shim stands in.
if ! command -v clang >/dev/null 2>&1 && [ -x "$HOME/.local/bin/zigcc" ]; then
  export CC=$HOME/.local/bin/zigcc
fi

if [ ! -d "$TREE" ]; then
  echo "referee: cloning keras v$KERAS_VERSION -> $TREE"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "v$KERAS_VERSION" \
    https://github.com/keras-team/keras.git "$TREE"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "referee: building $VENV (python $REFEREE_PYTHON)"
  uv venv --quiet --python "$REFEREE_PYTHON" "$VENV"
  uv pip install --quiet --python "$VENV/bin/python" \
    "keras==$KERAS_VERSION" "$TINYGRAD_PIN" pytest numpy scipy pillow pandas \
    tensorflow-cpu grain "jax[cpu]" pytest-timeout
  uv pip install --quiet --python "$VENV/bin/python" --no-deps -e "$REPO"
fi

# Cheap and idempotent: an existing venv may hold a tinygrad outside the pin
# (see TINYGRAD_PIN above).
uv pip install --quiet --python "$VENV/bin/python" "$TINYGRAD_PIN"
# A venv built before the guard rails existed has no pytest-timeout.
if ! "$VENV/bin/python" -c "import pytest_timeout" 2>/dev/null; then
  uv pip install --quiet --python "$VENV/bin/python" pytest-timeout
fi

REFEREE_MEM_MB=${REFEREE_MEM_MB:-8192}
REFEREE_TEST_TIMEOUT=${REFEREE_TEST_TIMEOUT:-900}
[ "$REFEREE_MEM_MB" -gt 0 ] && ulimit -v $((REFEREE_MEM_MB * 1024))
ulimit -c 0  # no core files (see the shutdown-abort note below: ~1 GB each)

cd "$TREE"
# Preflight: the TREE's keras must win over the venv's installed wheel
# (cwd is sys.path[0] under `python -m pytest`), and the hook must be live.
KERAS_BACKEND=tinygrad "$VENV/bin/python" - <<'PY'
import keras_tinygrad, keras, os
assert keras.__file__.startswith(os.getcwd()), f"wrong keras: {keras.__file__}"
print(f"referee: keras {keras.__version__} from {keras.__file__}")
print(f"referee: backend {keras.backend.backend()}")
import importlib.metadata as m
print(f"referee: tinygrad {m.version('tinygrad')}")
PY

PATHS=("$@"); [ ${#PATHS[@]} -eq 0 ] && PATHS=("${DEFAULT_PATHS[@]}")
LOG=$REFEREE_DIR/last-run.log
set +e
# shellcheck disable=SC2086
KERAS_BACKEND=tinygrad nice -n 10 "$VENV/bin/python" -m pytest -p keras_tinygrad \
  "${PATHS[@]}" -q --no-header -p no:cacheprovider --timeout="$REFEREE_TEST_TIMEOUT" \
  ${PYTEST_ARGS:-} 2>&1 | tee "$LOG" \
  | grep -E "^FAILED |^ERROR |^=+ .*(passed|failed|error)" | tail -12
PYTEST_STATUS=${PIPESTATUS[0]}
set -e

# Guarded greps: a fully GREEN run has zero FAILED lines and grep exits 1 —
# under `set -euo pipefail` the unguarded version killed the script at the
# exact moment everything passed.
TALLY=$(grep -E "^=+ .*(passed|failed|error)" "$LOG" | tail -1 || true)
echo
echo "referee: keras v$KERAS_VERSION  ${PATHS[*]}"
echo "referee: ${TALLY:-NO PYTEST SUMMARY LINE — crash? see $LOG}"

# A test file that imports tensorflow (ops/image_test, the preprocessing
# tree) made the interpreter abort AT EXIT with tinygrad 0.13 in the same
# process — `free(): invalid pointer`, exit 134, after pytest has printed
# its complete summary; same on the unmodified tree, no test affected
# (2026-09-21). Not seen on 0.14, the pin since that day; kept because it
# costs nothing and a future tinygrad may bring it back. Only THAT shape is
# tolerated, loudly: any abort without a
# final summary line is still a crash.
if [ "$PYTEST_STATUS" -ge 2 ]; then
  if [ "$PYTEST_STATUS" -eq 134 ] && [ -n "$TALLY" ] && tail -3 "$LOG" | grep -q "free(): invalid pointer"; then
    echo "referee: WARNING — interpreter aborted at exit (134, free(): invalid pointer) AFTER the complete summary; judging the run by its results"
  else
    echo "referee: pytest crashed/aborted (exit $PYTEST_STATUS) — see $LOG"; exit "$PYTEST_STATUS"
  fi
fi
if grep -qE "^ERROR " "$LOG"; then
  echo "referee: COLLECTION/SETUP ERRORS — see $LOG"; exit 1
fi

{ grep -E "^FAILED " "$LOG" || true; } | sed 's/^FAILED //; s/ - .*//' | sort -u > "$REFEREE_DIR/failed.txt"
grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | sort -u > "$REFEREE_DIR/baseline.txt"
NEW=$(comm -23 "$REFEREE_DIR/failed.txt" "$REFEREE_DIR/baseline.txt")

# Trailing slashes would make the tree of record look like a custom path.
PATHS=("${PATHS[@]%/}")
# A run that cannot have executed the whole tree of record — custom paths
# (referee-quick, a dispatch paths input) or any extra pytest args (`-k`,
# `-x` deselect/stop early) — must not report the unexecuted baseline
# entries as NOW PASSING.
if [ "${PATHS[*]}" != "${DEFAULT_PATHS[*]}" ] || [ -n "${PYTEST_ARGS:-}" ]; then
  # Custom paths (referee-quick, dispatch with a paths input) are judged
  # against the baseline too: known failures inside the subset stay green,
  # ANY other failure is red. Raw pytest status cannot be used here —
  # dense_test's known float8 failure would make referee-quick permanently
  # red — and the old unconditional `exit 0` made it permanently green.
  if [ -n "$NEW" ]; then
    echo "referee: NEW FAILURES (not in baseline):"; echo "$NEW" | sed 's/^/  /'; exit 1
  fi
  echo "referee: OK — failures (if any) are baseline-known; the full-baseline check needs the tree of record"
  exit 0
fi

FIXED=$(comm -13 "$REFEREE_DIR/failed.txt" "$REFEREE_DIR/baseline.txt")
if [ -z "$NEW" ] && [ -z "$FIXED" ]; then
  echo "referee: OK — failed set == baseline ($(wc -l < "$REFEREE_DIR/baseline.txt") known)"; exit 0
fi
[ -n "$NEW" ]   && { echo "referee: NEW FAILURES (regression):"; echo "$NEW" | sed 's/^/  /'; }
[ -n "$FIXED" ] && { echo "referee: NOW PASSING (move the baseline, with this tally as the receipt):"; echo "$FIXED" | sed 's/^/  /'; }
exit 1
