"""Cross-backend CPU benchmark of record — orchestrator.

Runs bench/_child.py once per (backend, model) in `.bench/` (a py3.12 venv
with keras 3.15.1 + tensorflow-cpu + jax[cpu] + torch (CPU wheels) + this
repo editable), every child pinned to the SAME cores (`taskset -c 0-3`),
one at a time, under resource caps (address space, CPU seconds, no core
files). Writes bench/results/<date>-cpu-<host>.{json,md}.

The like-with-like check: the step-1 loss must agree across backends to
1e-3 relative (same init, same data, same optimizer → same first
gradient), else the run fails loudly; step-10 divergence beyond 5% is a
finding printed and recorded, not hidden.

    python3 bench/run.py                      # everything (~10-20 min; first run installs .bench/)
    python3 bench/run.py --backends tinygrad,jax --models mlp --steps 20
"""

import argparse
import datetime as dt
import json
import os
import pathlib
import resource
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
VENV = REPO / ".bench"
PY = VENV / "bin" / "python"
RESULTS = REPO / "bench" / "results"
LOGS = VENV / "logs"
BACKENDS = ["tinygrad", "tensorflow", "jax", "torch"]
MODELS = ["mlp", "cnn", "lstm"]
CORES = os.environ.get("BENCH_CORES", "0-3")
MEM_MB = int(os.environ.get("BENCH_MEM_MB", "12288"))
CPU_S = int(os.environ.get("BENCH_CPU_S", "3600"))


def child_limits():
    for limit, value in ((resource.RLIMIT_AS, MEM_MB << 20), (resource.RLIMIT_CPU, CPU_S), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def sh(*cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ensure_venv(reinstall=False):
    if PY.exists() and not reinstall:
        return
    sh("uv", "venv", "-q", VENV, "--python", "3.12")
    sh("uv", "pip", "install", "-q", "--python", PY, "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    sh("uv", "pip", "install", "-q", "--python", PY, "keras==3.15.1", "tensorflow-cpu", "jax[cpu]")
    sh("uv", "pip", "install", "-q", "--python", PY, "-e", REPO)


def run_child(backend, model, steps):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{backend}-{model}.log"
    cmd = ["taskset", "-c", CORES, str(PY), str(REPO / "bench" / "_child.py"), "--backend", backend, "--model", model]
    if steps:
        cmd += ["--steps", str(steps)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=child_limits)
    wall = time.perf_counter() - t0
    text = log.read_text(encoding="utf-8")
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith("BENCH_JSON ")), None)
    if line is None:
        return {
            "backend": backend,
            "model": model,
            "error": f"no result (exit {proc.returncode}); see {log}",
            "traceback": text[-2000:],
            "wall_s": wall,
        }
    out = json.loads(line[len("BENCH_JSON ") :])
    out["wall_s"] = wall
    out["exit"] = proc.returncode
    return out


def fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec)


def agreement(results, models):
    notes = []
    for model in models:
        rows = [r for r in results if r["model"] == model and "error" not in r]
        if len(rows) < 2:
            continue
        s1 = {r["backend"]: r["loss_at"]["1"] for r in rows}
        ref = rows[0]["loss_at"]["1"]
        bad = {b: v for b, v in s1.items() if abs(v - ref) > 1e-3 * max(abs(ref), 1e-6)}
        if bad:
            raise SystemExit(f"step-1 loss disagrees for {model}: {s1} — the runs are NOT like-with-like")
        n = min(10, min(len(r["losses"]) for r in rows))
        worst = 0.0
        worst_at = None
        for i in range(n):
            ref_i = rows[0]["losses"][i]
            for r in rows[1:]:
                rel = abs(r["losses"][i] - ref_i) / max(abs(ref_i), 1e-6)
                if rel > worst:
                    worst, worst_at = rel, (i + 1, r["backend"])
        if worst > 1e-3:
            msg = (
                f"WARNING {model}: per-step losses diverge across backends within the first {n} steps"
                f" (worst {worst:.2e} relative at step {worst_at[0]}, {worst_at[1]} vs {rows[0]['backend']})"
                " — same weights, same data: a trajectory difference, see the per-step table"
            )
            print(msg, flush=True)
            notes.append(msg)
        else:
            notes.append(
                f"{model}: losses agree across backends on every one of the first {n} steps"
                f" (worst {worst:.1e} relative); step-1 {s1}"
            )
    return notes


def per_step_table(results, model, n=10):
    rows = [r for r in results if r["model"] == model and "error" not in r]
    if not rows:
        return ""
    head = "| step | " + " | ".join(r["backend"] for r in rows) + " |\n|---|" + "---|" * len(rows) + "\n"
    body = ""
    for i in range(min(n, min(len(r["losses"]) for r in rows))):
        body += f"| {i + 1} | " + " | ".join(f"{r['losses'][i]:.5f}" for r in rows) + " |\n"
    return head + body


def write_md(path, results, notes, models, total_wall, cmdline):
    box = next((r["box"] for r in results if "box" in r), {})
    versions = {}
    for r in results:
        versions.update(r.get("versions", {}))
    lines = [
        f"# Cross-backend CPU benchmark — {path.stem}",
        "",
        f"Box: {box.get('cpu')} · {box.get('cores_pinned')} of {box.get('cores_total')} cores pinned"
        f" (`taskset -c {CORES}`) · {box.get('ram_gb')} GB RAM · Linux {box.get('kernel')}"
        f" · Python {box.get('python')}",
        "Versions: " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items())),
        f"Run: `{cmdline}` · total wall {total_wall / 60:.1f} min · one process per (backend, model), sequential,"
        f" RLIMIT_AS {MEM_MB} MB, RLIMIT_CPU {CPU_S} s",
        "",
        "Same initial weights (numpy, seed 42), same data (seed 1234), same batch order, same optimizer; no dropout."
        " CPU only, single box, software as listed. The numbers are what they are; read bench/README.md"
        " before quoting.",
        "",
        "| model | backend | first step (s) | steady step (s) | steps/s | loss @1 | @10 | @25 | @50 | @100 |"
        " t→½·loss@1 (s) | predict samples/s | peak RSS (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['backend']} | ERROR: {r['error']} |" + " |" * 11)
            continue
        la = r["loss_at"]
        lines.append(
            f"| {r['model']} | {r['backend']} | {fmt(r['first_step_s'])} | {fmt(r['steady_step_s'], '.4f')} |"
            f" {fmt(r['steady_steps_per_s'], '.1f')} | {fmt(la.get('1'), '.4f')} | {fmt(la.get('10'), '.4f')} |"
            f" {fmt(la.get('25'), '.4f')} | {fmt(la.get('50'), '.4f')} | {fmt(la.get('100'), '.4f')} |"
            f" {fmt(r['time_to_target_s'])} | {fmt(r['predict_samples_per_s'], '.0f')} |"
            f" {fmt(r['peak_rss_mb'], '.0f')} |"
        )
    lines += ["", "## Like-with-like check", ""] + [f"- {n}" for n in notes]
    for model in models:
        lines += ["", f"## Loss per step — {model} (first 10 steps)", "", per_step_table(results, model)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--steps", type=int, default=None, help="override every model's step count")
    ap.add_argument("--reinstall", action="store_true", help="rebuild .bench/")
    ap.add_argument("--tag", default="cpu", help="results file infix (default cpu)")
    ap.add_argument(
        "--merge",
        help="an existing results .json: rows for the (backend, model) pairs run now replace theirs, the rest stay",
    )
    args = ap.parse_args()
    backends, models = args.backends.split(","), args.models.split(",")
    ensure_venv(args.reinstall)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    results = []
    previous = None
    if args.merge:
        previous = json.loads(pathlib.Path(args.merge).read_text(encoding="utf-8"))
    for model in models:
        for backend in backends:
            print(f"=== {model} / {backend}", flush=True)
            r = run_child(backend, model, args.steps)
            if "error" in r:
                print(f"    ERROR {r['error']}", flush=True)
            else:
                print(
                    f"    first {r['first_step_s']:.2f}s steady {r['steady_step_s']:.4f}s/step"
                    f" loss@1 {r['loss_at'].get('1'):.4f} predict {r['predict_samples_per_s']:.0f}/s"
                    f" rss {r['peak_rss_mb']:.0f}MB wall {r['wall_s']:.0f}s",
                    flush=True,
                )
            results.append(r)
    total_wall = time.perf_counter() - t_all
    stamp = dt.date.today().isoformat()
    base = RESULTS / f"{stamp}-{args.tag}-{socket.gethostname()}"
    cmdline = "python3 bench/run.py " + " ".join(sys.argv[1:])
    if previous is not None:
        redone = {(r["backend"], r["model"]) for r in results}
        kept = [r for r in previous["results"] if (r["backend"], r["model"]) not in redone]
        results = kept + results
        order = {(m, b): i for i, (m, b) in enumerate((m, b) for m in MODELS for b in BACKENDS)}
        results.sort(key=lambda r: order.get((r["model"], r["backend"]), 999))
        models = [m for m in MODELS if any(r["model"] == m for r in results)]
        total_wall += previous.get("total_wall_s", 0.0)
        cmdline = previous.get("cmd", "") + "  ++  " + cmdline
        base = pathlib.Path(args.merge).with_suffix("")
    notes = agreement(results, models)
    base.with_suffix(".json").write_text(
        json.dumps({"cmd": cmdline, "total_wall_s": total_wall, "cores": CORES, "results": results}, indent=1),
        encoding="utf-8",
    )
    write_md(base.with_suffix(".md"), results, notes, models, total_wall, cmdline)
    print(f"wrote {base}.md and .json ({total_wall / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
