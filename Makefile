# Dev entry points. Everything runs through uv (https://docs.astral.sh/uv/):
# `uv run` resolves the project + dev dependency group into .venv on first use.
UV ?= uv

# tinygrad's CPU jit shells out to clang; on clang-less boxes the ziglang
# shim is a drop-in (see README "No clang? Use zig"). Unconditional `:=`,
# not `?=`: make's built-in default CC=cc would defeat `?=`.
ifeq ($(shell command -v clang 2>/dev/null),)
export CC := $(HOME)/.local/bin/zigcc
endif

.PHONY: verify lint-check format-check format tests-fast tutorial test \
        smoke fuzz fuzz-grad vendor-check readme-check referee referee-quick \
        browser-assets

## verify: the pre-review gate — lint + format + fast tests
verify: lint-check format-check tests-fast

lint-check:
	$(UV) run --group dev ruff check .

format-check:
	$(UV) run --group dev ruff format --check .

## format: apply formatting + safe lint fixes (never touches _backend/)
format:
	$(UV) run --group dev ruff format .
	$(UV) run --group dev ruff check --fix .

## tests-fast: loader test suite (~seconds; no model training)
tests-fast:
	$(UV) run --group dev pytest tests -q --no-header --ignore=tests/test_tutorial.py

## tutorial: execute every python block in TUTORIAL.md (the page cannot rot)
tutorial:
	$(UV) run --group dev pytest tests/test_tutorial.py -q --no-header

test: tests-fast tutorial

## smoke: compile+fit+predict against stock keras (must print SMOKE OK)
smoke:
	$(UV) run python examples/mlp_smoke.py

## fuzz: randomized cross-backend parity hunt vs the numpy reference
## (forward ops only — grad cases need --slow and live in fuzz-grad)
fuzz:
	$(UV) run python tools/parity_fuzz.py --seed 0 --cases 100

## fuzz-grad: finite-difference gradient checks (slower; the numpy
## reference has no autograd, so these only run with --slow)
fuzz-grad:
	$(UV) run python tools/parity_fuzz.py --seed 0 --cases 60 --kinds grad --slow

## vendor-check: loader patch anchors match the installed keras exactly once
vendor-check:
	$(UV) run python scripts/sync_vendor.py --self-check

## referee: Keras' OWN layers tree against this package (the tally of
## record, ~25 min). Clones the pinned keras tag into .referee/ itself and
## compares the FAILED set to scripts/referee-baseline.txt.
referee:
	scripts/referee.sh

## referee-quick: ~1 min slice of the same suite (backend/optimizer/core ops
## + Dense) — catches a broken convert_to_tensor/Variable/SGD in seconds.
## Not a tally: baseline-known failures stay green, any other failure is red.
referee-quick:
	scripts/referee.sh keras/src/backend/tests keras/src/optimizers/sgd_test.py \
	  keras/src/ops/core_test.py keras/src/layers/core/dense_test.py

## readme-check: the README's tally/matrix numbers are internally consistent
readme-check:
	$(UV) run python scripts/check_readme_numbers.py

## browser-assets: regenerate EVERY generated browser artifact (none are in
## git): tf.js (pinned fetch), the two Keras-traced WebGPU bundles (NULL:WGSL
## traces, deterministic), and the three pages. ~1 min. The hub server
## (experiments/m0-keras-trainstep/demo-server.mjs) serves the result.
M0 := experiments/m0-keras-trainstep
PYODIDE_WHEELS := experiments/pyodide-keras/wheels
browser-assets:
	scripts/fetch_tfjs.sh
	# the wheel the Pyodide tab installs is THIS tree's, named by its version
	# (wheels/latest.txt is how main.js finds it; gen_hub.py asks the package)
	$(UV) build --wheel -o $(PYODIDE_WHEELS)
	cd $(PYODIDE_WHEELS) && ls -t keras_tinygrad-*.whl | head -1 > latest.txt
	cd $(M0) && $(UV) run --project $(CURDIR) python m0.py export out
	cd $(M0) && $(UV) run --project $(CURDIR) python export_dropout_probe.py export
	cd $(M0) && $(UV) run --project $(CURDIR) python gen_demo.py
	cd $(M0) && $(UV) run --project $(CURDIR) python export_dropout_probe.py page
	cd $(M0) && $(UV) run --project $(CURDIR) python gen_hub.py
