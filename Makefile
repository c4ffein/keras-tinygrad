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
        smoke fuzz fuzz-grad vendor-check

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
