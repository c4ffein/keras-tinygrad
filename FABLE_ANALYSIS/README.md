# FABLE_ANALYSIS — a thorough review of keras-tinygrad

*Produced 2026-08-03 by Claude Fable 5: five parallel deep-dive agents (each reading its subsystem in full, two of them empirically testing the live backend), synthesized here. Read this file first; each chapter stands alone with file:line evidence.*

| Chapter | Scope | One-line verdict |
|---|---|---|
| [01-core-engine.md](01-core-engine.md) | `core.py`, `trainer.py`, `rnn.py` — Variable, gradients, tape, TinyJit | Invariants genuinely enforced; JIT/eager bit-parity independently reproduced; **one verified silent-wrong gradient bug found** |
| [02-ops-surface.md](02-ops-surface.md) | `numpy/nn/math/linalg/image/random` (~7k lines) | Disciplined and clean; a handful of verified numeric edge-case defects; Jacobi's missing convergence check is the systemic risk |
| [03-packaging-loader.md](03-packaging-loader.md) | Import hook, vendor sync, pyproject, CI | Best-in-class loud-failure design; the gaps are one level up (textual-not-semantic anchors, a latent setuptools build bug) |
| [04-docs-tooling-dx.md](04-docs-tooling-dx.md) | README/TUTORIAL/HANDOFF/docs, fuzzer, Makefile | Honest and verifiable; the enemy is fact-duplication drift the repo's own machinery doesn't yet cover |
| [05-strategy-viability-risk.md](05-strategy-viability-risk.md) | Adoption, upstream PRs, maintenance economics, NNVP | Staff-level engineering pointed at a market of ~one; NNVP/WebGPU M0 is the valuation event — run it first |

---

## Overall verdict

**This is an exceptionally well-executed project — and the reviews prove it the hard way.** Two agents didn't just read the code; they attacked it. The core-engine reviewer wrote 8 adversarial gradient-composition tests and independently reproduced the JIT/eager bit-parity claim (`maxdiff: 0.0` across BatchNorm state, Adam slots, and a partial final batch). The ops reviewer reproduced every suspected defect against the live backend before reporting it. Under that scrutiny, the project's headline claims held: the invariants are enforced in code, not just documented; the test tallies are real numbers, not marketing; the loud-stub architecture genuinely prevents silent fallbacks.

What makes it unusual isn't the 9k lines of backend — most of that is competent, mechanical porting. It's the **methodology**: numpy backend as semantic reference, Keras' own suite as referee, exactly-once patch anchors that fail loud, decision memos that name the invariant being bent, an executable tutorial that already caught a real regression, and an architecture doc with a binding "update in the same diff" contract. That is discipline most funded teams don't maintain, on a project one person built.

## The findings that matter most

Across ~40 ranked findings in the five chapters, these change what you'd do next:

1. **[CORRECTNESS — verified] Duplicate custom-gradient blocks silently double-count.** tinygrad hash-conses UOps, so two identical `custom_gradient` calls on the same tensor share one detached proxy; `compute_gradients` hands each block the *total* upstream. Reproduced: 12.0 where 6.0 is correct. Reachable via a shared quantized layer applied twice to the same tensor. ~4-line fix (dedupe tape entries on `proxy.uop` identity). This is the only place the engine can be silently wrong — in the subsystem whose ethos is "loud over wrong." (01 §2.1)
2. **[CORRECTNESS — verified] Ops edge-case defects**: `logaddexp` NaN-poisons at infinities (the fix already exists ten lines below in `logaddexp2`); `expm1`/`log1p` return 0.0 for small inputs; `take` zero-fills out-of-range indices where numpy raises; `random.shuffle` round-trips *data* through host numpy, detaching gradients. All small, all cited, mostly cheap fixes. (02 §2)
3. **[SILENT-INACCURACY RISK] Fixed-sweep Jacobi eig/svd has no convergence or residual check** — fine at Keras-test scale, silently degraded on large/ill-conditioned matrices. One cheap post-loop residual restores the loud-over-wrong ethos. (02 §2.2)
4. **[EXISTENTIAL, NON-CODE] Everything is uncommitted.** The clone, this repo, the tinygrad patch. The project's entire value is currently one `rm -rf` from being partially unrecoverable. Commit and publish before touching anything else. (05 §2, §5)
5. **[MECHANISM GAP] The anchor system is textual, not semantic.** A keras release that adds a *seventh* dispatch site imports cleanly but incompletely patched; and invariant 11 ("every touchpoint has a patch-table anchor") is enforced by convention only — a git-diff check against the clone's upstream base would machine-close it. (03 findings 1–2)
6. **[LATENT BUILD BUG] `license = "Apache-2.0"` (SPDX, needs setuptools ≥77) vs build floor `setuptools>=68`** — fails to build in non-latest environments; plus `tinygrad>=0.13` unbounded while the trainer is coupled to 0.13 internals. (03 findings 3–4)
7. **[DOC DRIFT] Facts duplicated beyond the anti-rot machinery's reach**: three different headline percentages in circulation; README says Orthogonal init works while `examples/` says it doesn't; `make fuzz` doesn't run the gradient checks two documents attribute to it. Single root cause, and the repo already invented the cure (token-injected blocks) — it just stops at the README's edge. (04 findings 1–6)

## What's genuinely excellent (keep doing this)

- **The JitError-as-safety-net design**: JIT bugs structurally degrade to slowness, never wrongness — and the bit-parity claim survives independent reproduction. (01)
- **Hand-composed VJPs over `Tensor.gradient`** on a backend with no custom-VJP hook, with torch-convention parity verified against the torch backend's source. (01)
- **The exactly-once anchor rule + the patched/unpatched-state distinction** in the drift guards — including the honesty of documenting that the first version was wrong and what fixing it uncovered. (03)
- **Gems in the ops surface**: `nextafter` via two's-complement bitcast, parallel round-robin Jacobi with mathematically-exact disjoint rotation planes, `fold` as grouped conv-transpose against an identity kernel, fixed-shape CTC beam-search dedup. (02 §5)
- **The decision-record culture**: float64-promotion, complex-support, and unique-vectorize memos that state options *with blast radius* and name the invariant being bent. The upstream drafts keep dropped items with reasons instead of deleting them. (04 §5)

## Status update (2026-08-03, same session)

The fixable findings were applied and verified after this analysis was written
— gradient double-count (de-aliased proxies, repro now exact), the verified
ops defects (`logaddexp`/`expm1`/`log1p`/`take`/`power`/`bincount`/`shuffle`),
Jacobi convergence loudness, `Variable.__array__`, the setuptools/tinygrad
pins, `fuzz-grad`, the sync-vendor subdir guard, and the doc-drift items.
A fuzzer tolerance bug found during verification (both-backends-demoted
float64 cases judged under float64 tolerances) was also fixed — see
`docs/float64-promotion.md`'s addendum. Referee: 5,875/3/721 on
ops/numpy+math+core (the 3 = the documented pre-existing reds); `make
fuzz` 100/100, `fuzz-grad` 60/60; loader/tutorial/smoke green. Items 1
(commit/publish), 3 (M0), and 7 (upstream PRs) below remain owner calls.

## Unified priority list

1. **Commit everything; publish; claim the PyPI name.** (05)
2. **Fix the double-count gradient bug** + add the `y1=f(w); y2=f(w)` case to the fuzz suite. (01)
3. **Run the NNVP M0 spike** — it decides whether this is a platform or a very polished curiosity; `_TrainStepJit` has already shrunk the unknown. (05)
4. **Batch the small verified fixes**: `logaddexp` masks, `expm1`/`log1p`, `take`/`bincount` guards, `random.shuffle` permutation-gather, Jacobi residual check, `Variable.__array__` signature, setuptools floor, tinygrad upper bound. (01, 02, 03)
5. **One doc-consistency pass**: single token-injected headline number, fix the Orthogonal contradiction (make `char_rnn.py` use the default init — it doubles as a regression test), `fuzz-grad` Makefile target, reconcile invariant 6's wording with the actual `__bool__` override, strike HANDOFF's superseded numbers. (02, 04)
6. **Harden the triangle**: git-diff touchpoint check, recursive vendor sync, `keras.__version__` runtime gate, build-wheel-then-install CI step, CI canary against the latest keras wheel. (03)
7. **Submit the keras test-side PR** (or `test_cross` alone); tinygrad PRs after a genuine human rewrite with AI disclosure, one attempt each. (05)

## The honest bottom line

As engineering, this is top-percentile work: verified-correct where it claims to be, loud where it isn't, documented to a standard that survived adversarial review with one real bug and a handful of edge cases. As a product, its default fate — absent the WebGPU/NNVP story — is to be admired and unused, decaying one Keras release at a time. The project's own documents contain both the diagnosis and the cure. Commit the work, fix the one silent-wrong bug, run the M0 spike, and let its result decide how much of the rest deserves feeding.
