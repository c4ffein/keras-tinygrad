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
        smoke examples fuzz fuzz-grad vendor-check readme-check referee referee-quick \
        browser-assets e2e referee-cl bench

## verify: the pre-review gate — lint + format + fast tests
verify: lint-check format-check tests-fast

# The backend sources are excluded from lint/format (keras style, kept
# byte-stable), which also hides a dropped import until the op is called:
# undefined names only, and --isolated so the exclusion does not apply.
lint-check:
	$(UV) run --group dev ruff check .
	$(UV) run --group dev ruff check --isolated --select F821 src/keras_tinygrad/src

format-check:
	$(UV) run --group dev ruff format --check .

## format: apply formatting + safe lint fixes (never touches keras_tinygrad/src/)
format:
	$(UV) run --group dev ruff format .
	$(UV) run --group dev ruff check --fix .

## tests-fast: loader test suite (~seconds; no model training)
tests-fast:
	$(UV) run --group dev pytest tests -q --no-header --ignore=tests/test_tutorial.py --ignore=tests/test_webgpu_export.py

## tutorial: execute every python block in TUTORIAL.md (the page cannot rot)
tutorial:
	$(UV) run --group dev pytest tests/test_tutorial.py -q --no-header

test: tests-fast tutorial

## smoke: compile+fit+predict against stock keras (must print SMOKE OK)
smoke:
	$(UV) run python examples/mlp_smoke.py

## examples: run every other example at its smallest useful size (1 epoch,
## synthetic data, no network). Each script ends in its own assertion, so a
## rotten example is a red run, not a quiet one. ~3-4 min on a CPU device
## (char_rnn dominates: every batch-shape signature is a JIT capture and
## sampling is one predict per character — hence its larger batch).
EXAMPLES := autoencoder:32 char_rnn:128 convnet_mnist:32 quantized_inference:32
examples:
	@for spec in $(EXAMPLES); do \
	  ex=$${spec%%:*}; bs=$${spec##*:}; \
	  echo "=== examples/$$ex.py --epochs 1 --batch-size $$bs"; \
	  nice -n 10 $(UV) run python examples/$$ex.py --epochs 1 --batch-size $$bs || exit 1; \
	done

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
	bash scripts/referee.sh

## referee-quick: ~1 min slice of the same suite (backend/optimizer/core ops
## + Dense) — catches a broken convert_to_tensor/Variable/SGD in seconds.
## Not a tally: baseline-known failures stay green, any other failure is red.
referee-quick:
	bash scripts/referee.sh keras/src/backend/tests keras/src/optimizers/sgd_test.py \
	  keras/src/ops/core_test.py keras/src/layers/core/dense_test.py

## referee-cl: the referee-quick slice on tinygrad's OpenCL device with no
## GPU — PoCL (a CPU OpenCL implementation from PyPI) behind the system ICD
## loader, so every kernel goes through tinygrad's OpenCL renderer. ~3 min.
## Any keras path: `scripts/with_cl.sh bash scripts/referee.sh <paths>`.
referee-cl:
	scripts/with_cl.sh bash scripts/referee.sh keras/src/backend/tests keras/src/optimizers/sgd_test.py \
	  keras/src/ops/core_test.py keras/src/layers/core/dense_test.py

## readme-check: the README's tally/matrix numbers are internally consistent
readme-check:
	$(UV) run python scripts/check_readme_numbers.py

## browser-assets: regenerate EVERY generated browser artifact (sources live
## in js/demo/, outputs land in js/demo/build/ — gitignored): tf.js (pinned
## fetch), the two Keras-traced WebGPU bundles (NULL:WGSL traces,
## deterministic), the CPU reference curve, and the hub page. ~1-2 min.
## `bun js/demo/server.mjs` serves the result.
DEMO := js/demo
BUILD := js/demo/build
browser-assets:
	mkdir -p $(BUILD)/wheels
	scripts/fetch_tfjs.sh
	# the wheel the Pyodide tab installs is THIS tree's, named by its version
	# (wheels/latest.txt is how consumers find it; gen_hub.py asks the package)
	$(UV) build --wheel -o $(BUILD)/wheels
	cd $(BUILD)/wheels && ls -t keras_tinygrad-*.whl | head -1 > latest.txt
	cd $(DEMO) && $(UV) run --project $(CURDIR) python m0.py export build/out
	cd $(DEMO) && $(UV) run --project $(CURDIR) python export_dropout_probe.py
	cd $(DEMO) && $(UV) run --project $(CURDIR) python gen_ref_curve.py
	cd $(DEMO) && $(UV) run --project $(CURDIR) python gen_hub.py

## e2e: the browser end-to-end suite — drives the hub in headless Chromium
## (SwiftShader WebGPU) through the four proof flows and asserts the page's
## own checks JSON. Needs: fresh browser-assets, bun, and a chromium binary
## (playwright's cache or $$CHROME). The pyodide flow additionally needs
## network (CDN) — opt in with KERAS_TINYGRAD_E2E_PYODIDE=1.
e2e: browser-assets
	$(UV) run --group dev pytest tests/test_webgpu_export.py -q --no-header

## bench: the cross-backend CPU benchmark of record (bench/README.md): MLP,
## small CNN, LSTM on tinygrad / tensorflow / jax / torch, same init, same
## data, same cores, one process per run. First use installs .bench/
## (tf-cpu, jax, torch CPU wheels, this backend); ~10-20 min per run.
## Results land in bench/results/<date>-cpu-<host>.{md,json} (tracked).
bench:
	python3 bench/run.py
