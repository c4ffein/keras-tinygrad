# bench/ — cross-backend CPU benchmark of record

Three small Keras models trained through four backends — **tinygrad** (this
package), **tensorflow**, **jax**, **torch** — on one box, one process per
(backend, model), same cores, same initial weights, same data. Output: a
Markdown table and its JSON under `results/`, named `<date>-cpu-<host>`.

```sh
make bench                                   # everything; first use builds .bench/ (~1 GB of wheels)
python3 bench/run.py --backends tinygrad,jax --models mlp --steps 20   # a partial run
```

## What is measured

| column | meaning |
|---|---|
| first step (s) | wall time of the first `train_on_batch`: compile / trace / JIT capture included — for tinygrad that is kernel compilation through the C compiler (`CC`) |
| steady step (s), steps/s | median `train_on_batch` wall time over the steps after five warm-up steps |
| loss @1 … @100 | the loss reported at those steps; @1 is the like-with-like check (below) |
| t→½·loss@1 (s) | cumulative wall time (first step included) until the loss first drops to half of the step-1 loss; `—` = not within the run |
| predict samples/s | `model.predict` on 1,024 samples, batch 128, after one warm-up call |
| peak RSS (MB) | `getrusage` max RSS of the child process |

Models (`bench/_child.py`): MLP 784→128 relu→10 softmax, batch 128, Adam
1e-3, 100 steps · CNN Conv(16,3)→pool→Conv(32,3)→pool→Dense(10), batch 64,
Adam 1e-3, 100 steps · LSTM Embedding(10000,64)→LSTM(32)→Dense(1) sigmoid,
sequence 25, batch 32, SGD 0.05 momentum 0.9, 30 steps. Synthetic data
(uniform pixels / random labels / random token ids), so the losses only
show that the optimizer moves, not that anything is learned.

## Fairness rules (all enforced by the code)

- Same initial weights on every backend: generated in numpy from seed 42
  and loaded with `model.set_weights` (each backend's own initializer RNG
  differs). Same data from seed 1234, same batch order. No dropout.
- **Like-with-like check:** the step-1 loss must agree across backends to
  1e-3 relative or the run fails — same weights, same batch, same loss
  function give the same number, and any backend that disagrees is broken,
  not fast. A step-10 divergence above 5% is printed as a WARNING and kept
  in the results (a finding, never hidden); the per-step losses of the first
  ten steps are in every results file.
- One process per (backend, model), run sequentially, all pinned to the
  same cores (`taskset -c 0-3`; `BENCH_CORES` overrides), at normal
  priority, under `RLIMIT_AS` 12 GB / `RLIMIT_CPU` 3600 s / no core files.
  Nothing else heavy should run on the box meanwhile.
- The versions and the box (CPU model, cores, RAM, kernel, python) are
  recorded by the child and printed at the top of each results file.

## Reading the numbers

- CPU only, single box, the software versions listed in the file. A
  different machine, a GPU, another tinygrad release or a different `CC`
  gives different numbers; rerun rather than extrapolate.
- The table is the claim. This README, the results files and anything that
  quotes them state ratios only as "on this run, on this box" and never as
  a property of the backend.
- tinygrad's first-step time is dominated by compiling every kernel of the
  train step through `CC`; its steady state is the train-step JIT replay
  (`KERAS_TINYGRAD_TRAINER_JIT`, default on). Set it to 0 to measure the
  eager path instead.
