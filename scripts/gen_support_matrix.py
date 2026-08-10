#!/usr/bin/env python3
"""Generate the keras-tinygrad backend support matrix from pytest results.

Runs (or re-reads) the Keras test suites a backend cares about, collects the
per-suite pass/fail/skip tallies, and renders them as a Markdown table, JSON,
or a shields.io endpoint badge — optionally injecting the table into README.md
between ``<!-- SUPPORT_MATRIX -->`` markers.

"Coverage" is passed / (passed + failed); skipped tests are excluded.

Usage:
    # Run the built-in suite manifest against a Keras checkout
    python scripts/gen_support_matrix.py --run --keras-root ~/src/keras

    # Same, worst suites first, with a 10-minute cap per suite
    python scripts/gen_support_matrix.py --run --keras-root ~/src/keras \\
        --sort --timeout 600

    # Parse a saved log: `=== <suite>` header lines, each followed by
    # that suite's pytest output (extra noise between tallies is fine)
    python scripts/gen_support_matrix.py --from-log results.log

    # Machine-readable outputs
    python scripts/gen_support_matrix.py --from-log results.log --emit json
    python scripts/gen_support_matrix.py --from-log results.log --emit badge

    # Rewrite the README's support-matrix region in place (idempotent)
    python scripts/gen_support_matrix.py --from-log results.log --inject README.md

Environment (used with --run only; passed through, with defaults applied
when unset):
    KERAS_BACKEND   backend under test           (default: tinygrad)
    CC              C compiler for tinygrad JIT  (default: clang)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Suite manifest — name -> pytest path relative to --keras-root.
# Paths drift between Keras versions; a missing path is reported as "absent".
# --------------------------------------------------------------------------- #

SUITES: tuple[tuple[str, str], ...] = (
    ("activations", "keras/src/activations/activations_test.py"),
    ("Dense", "keras/src/layers/core/dense_test.py"),
    ("EinsumDense", "keras/src/layers/core/einsum_dense_test.py"),
    ("Embedding", "keras/src/layers/core/embedding_test.py"),
    (
        "BatchNormalization",
        "keras/src/layers/normalization/batch_normalization_test.py",
    ),
    ("Dropout", "keras/src/layers/regularization/dropout_test.py"),
    ("Conv", "keras/src/layers/convolutional/conv_test.py"),
    ("pooling", "keras/src/layers/pooling"),
    ("SimpleRNN", "keras/src/layers/rnn/simple_rnn_test.py"),
    ("MultiHeadAttention", "keras/src/layers/attention/multi_head_attention_test.py"),
    ("losses", "keras/src/losses/losses_test.py"),
    ("Adam", "keras/src/optimizers/adam_test.py"),
    ("SGD", "keras/src/optimizers/sgd_test.py"),
    ("accuracy metrics", "keras/src/metrics/accuracy_metrics_test.py"),
    ("ops/core", "keras/src/ops/core_test.py"),
    ("ops/image", "keras/src/ops/image_test.py"),
    ("ops/math", "keras/src/ops/math_test.py"),
)

PYTEST_ARGS = ("-q", "--no-header", "-p", "no:cacheprovider")
ENV_DEFAULTS = {"KERAS_BACKEND": "tinygrad", "CC": "clang"}

MARKER_OPEN = "<!-- SUPPORT_MATRIX -->"
MARKER_CLOSE = "<!-- /SUPPORT_MATRIX -->"

# Statuses beyond a normally tallied run.
OK = "ok"
TIMEOUT = "timeout"
BLOCKED = "collection-blocked"
ABSENT = "absent"


@dataclass(frozen=True)
class SuiteResult:
    """Outcome of one test suite."""

    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    status: str = OK

    @property
    def ran(self) -> bool:
        return self.status == OK

    @property
    def coverage(self) -> float | None:
        """Percentage of non-skipped tests that passed, or None if unknown."""
        total = self.passed + self.failed
        return 100.0 * self.passed / total if self.ran and total else None


# --------------------------------------------------------------------------- #
# Parsing pytest output
# --------------------------------------------------------------------------- #

_TALLY_LINE = re.compile(
    r"\b\d+\s+(?:passed|failed|skipped|errors?|xfailed|xpassed|warnings?)\b"
    r".*\bin\s+[\d.]+s"
)
_TALLY_TOKEN = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?)\b")


def parse_tally(output: str) -> dict[str, int] | None:
    """Extract counts from the last pytest tally line, if any.

    Matches lines like ``2 failed, 40 passed, 3 skipped in 12.34s`` (with or
    without ``=`` banners around them) and returns e.g.
    ``{"passed": 40, "failed": 2, "skipped": 3, "errors": 0}``.
    """
    tallies = [line for line in output.splitlines() if _TALLY_LINE.search(line)]
    if not tallies:
        return None
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for number, word in _TALLY_TOKEN.findall(tallies[-1]):
        counts["errors" if word.startswith("error") else word] = int(number)
    return counts


def result_from_output(name: str, output: str) -> SuiteResult:
    """Classify one suite's pytest output into a SuiteResult.

    Errors alongside real test results count as failures; errors with no
    test results at all mean collection never succeeded.
    """
    counts = parse_tally(output)
    if counts is None:
        return SuiteResult(name, status=BLOCKED)
    if counts["errors"] and not (counts["passed"] or counts["failed"]):
        return SuiteResult(name, skipped=counts["skipped"], status=BLOCKED)
    return SuiteResult(
        name,
        passed=counts["passed"],
        failed=counts["failed"] + counts["errors"],
        skipped=counts["skipped"],
    )


# --------------------------------------------------------------------------- #
# Mode 1: run the suites
# --------------------------------------------------------------------------- #


def build_env() -> dict[str, str]:
    """Current environment plus defaults for unset backend variables."""
    return {**ENV_DEFAULTS, **os.environ}


def run_suite(name: str, rel_path: str, keras_root: Path, timeout: float) -> SuiteResult:
    """Run one suite via pytest and tally its output."""
    path = keras_root / rel_path
    if not path.exists():
        return SuiteResult(name, status=ABSENT)
    command = [sys.executable, "-m", "pytest", str(path), *PYTEST_ARGS]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=build_env(),
            cwd=keras_root,
        )
    except subprocess.TimeoutExpired:
        return SuiteResult(name, status=TIMEOUT)
    return result_from_output(name, proc.stdout + "\n" + proc.stderr)


def run_all(keras_root: Path, timeout: float) -> list[SuiteResult]:
    """Run every suite in the manifest, reporting progress on stderr."""
    results = []
    for name, rel_path in SUITES:
        print(f"[{name}] {rel_path} ...", file=sys.stderr)
        result = run_suite(name, rel_path, keras_root, timeout)
        print(f"[{name}] {describe(result)}", file=sys.stderr)
        results.append(result)
    return results


def describe(result: SuiteResult) -> str:
    if not result.ran:
        return result.status
    return f"{result.passed} passed, {result.failed} failed, {result.skipped} skipped"


# --------------------------------------------------------------------------- #
# Mode 2: parse a saved log
# --------------------------------------------------------------------------- #


def _is_header(line: str) -> bool:
    # Exactly "=== <name>"; pytest's own banners use 4+ equals signs.
    return line.startswith("=== ") and not line.startswith("====")


def parse_log(text: str) -> list[SuiteResult]:
    """Parse ``=== <suite>`` sections of saved pytest output into results."""
    results: list[SuiteResult] = []
    name: str | None = None
    body: list[str] = []

    def flush() -> None:
        if name is not None:
            results.append(result_from_output(name, "\n".join(body)))

    for line in text.splitlines():
        if _is_header(line):
            flush()
            name = line.removeprefix("=== ").strip().rstrip("=").strip()
            body = []
        else:
            body.append(line)
    flush()
    return results


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #


def totals(results: list[SuiteResult]) -> SuiteResult:
    """Sum of every suite that ran to a tally."""
    ran = [r for r in results if r.ran]
    return SuiteResult(
        "TOTAL",
        passed=sum(r.passed for r in ran),
        failed=sum(r.failed for r in ran),
        skipped=sum(r.skipped for r in ran),
    )


def sorted_worst_first(results: list[SuiteResult]) -> list[SuiteResult]:
    """Broken suites first, then ascending coverage, then name."""

    def key(r: SuiteResult) -> tuple[int, float, str]:
        cov = r.coverage
        return (0 if not r.ran else 1, cov if cov is not None else 100.0, r.name)

    return sorted(results, key=key)


def _coverage_cell(result: SuiteResult) -> str:
    if not result.ran:
        return result.status
    cov = result.coverage
    return f"{cov:.1f}%" if cov is not None else "—"


def emit_markdown(results: list[SuiteResult]) -> str:
    def row(r: SuiteResult, bold: bool = False) -> str:
        cells = [r.name] + [str(n) if r.ran else "—" for n in (r.passed, r.failed, r.skipped)] + [_coverage_cell(r)]
        if bold:
            cells = [f"**{cell}**" for cell in cells]
        return "| " + " | ".join(cells) + " |"

    lines = [
        "| Suite | ✅ passed | ❌ failed | skipped | coverage |",
        "|---|---:|---:|---:|---:|",
        *(row(r) for r in results),
        row(totals(results), bold=True),
    ]
    return "\n".join(lines)


def emit_json(results: list[SuiteResult]) -> str:
    def as_dict(r: SuiteResult) -> dict[str, object]:
        return {
            "name": r.name,
            "passed": r.passed,
            "failed": r.failed,
            "skipped": r.skipped,
            "status": r.status,
            "coverage": r.coverage,
        }

    payload = {
        "suites": [as_dict(r) for r in results],
        "total": as_dict(totals(results)),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def emit_badge(results: list[SuiteResult]) -> str:
    """shields.io endpoint JSON (https://shields.io/badges/endpoint-badge)."""
    total = totals(results)
    denominator = total.passed + total.failed
    if denominator == 0:
        color = "lightgrey"
    else:
        percent = 100.0 * total.passed / denominator
        color = "green" if percent >= 95 else "yellow" if percent >= 80 else "red"
    badge = {
        "schemaVersion": 1,
        "label": "keras test suite",
        "message": f"{total.passed}/{denominator}",
        "color": color,
    }
    return json.dumps(badge, indent=2)


# --------------------------------------------------------------------------- #
# README injection
# --------------------------------------------------------------------------- #


def inject_readme(readme: Path, table: str) -> None:
    """Replace the region between the SUPPORT_MATRIX markers with `table`.

    If only the opening marker exists, the closing marker is created right
    after the injected table. Repeated runs with identical input are no-ops.
    """
    text = readme.read_text(encoding="utf-8")
    if MARKER_OPEN not in text:
        raise SystemExit(f"error: {readme} has no {MARKER_OPEN} marker")
    region = f"{MARKER_OPEN}\n{table}\n{MARKER_CLOSE}"
    if MARKER_CLOSE in text:
        pattern = re.escape(MARKER_OPEN) + r".*?" + re.escape(MARKER_CLOSE)
        new_text = re.sub(pattern, lambda _: region, text, count=1, flags=re.DOTALL)
    else:
        new_text = text.replace(MARKER_OPEN, region, 1)
    if new_text != text:
        readme.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1],
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="store_true", help="run the suite manifest via pytest")
    source.add_argument("--from-log", metavar="FILE", help="parse tallies from a saved log")
    parser.add_argument(
        "--keras-root",
        metavar="DIR",
        help="Keras checkout to run suites in (required with --run)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        metavar="SECONDS",
        help="per-suite time limit; overruns are marked 'timeout' (default: 1800)",
    )
    parser.add_argument(
        "--emit",
        choices=("markdown", "json", "badge"),
        default="markdown",
        help="output format (default: markdown)",
    )
    parser.add_argument("--sort", action="store_true", help="order suites worst-first")
    parser.add_argument(
        "--inject",
        metavar="README",
        help="rewrite this file's SUPPORT_MATRIX region with the markdown table",
    )
    args = parser.parse_args(argv)
    if args.run and not args.keras_root:
        parser.error("--run requires --keras-root")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run:
        results = run_all(Path(args.keras_root).expanduser(), args.timeout)
    else:
        results = parse_log(Path(args.from_log).read_text(encoding="utf-8"))
    if args.sort:
        results = sorted_worst_first(results)

    if args.inject:
        inject_readme(Path(args.inject), emit_markdown(results))
        print(f"updated {args.inject}", file=sys.stderr)
        return
    emitter = {"markdown": emit_markdown, "json": emit_json, "badge": emit_badge}[args.emit]
    print(emitter(results))


if __name__ == "__main__":
    main()
