# Documentation, Tooling & Developer Experience

**Verdict:** This is an unusually honest, unusually verifiable documentation and tooling stack for a one-day-old project. The three load-bearing ideas — an executable tutorial enforced by CI, a support matrix generated from pytest tallies rather than written by hand, and decision memos that record options, blast radius, and the reason a rule was bent — are all genuinely good engineering and mostly delivered. The failures are the classic failure mode of a fast-moving multi-document repo: **truth is duplicated across README, HANDOFF, TUTORIAL, and examples, and the copies have already drifted apart** (three different headline percentages, one direct contradiction about Orthogonal-initializer support, and a `make fuzz` target that does not run the gradient checks two documents say it runs). Nothing found rises to dishonesty; several things rise to staleness that the project's own machinery was built to prevent but does not yet cover.

---

## 1. Documentation honesty: claims vs. reality

### The support matrix and tally are real numbers, not marketing

The README's numbers were checked against the generator and against each other:

- The per-suite arithmetic is correct under the stated metric. `scripts/gen_support_matrix.py:104-106` defines coverage as `passed / (passed + failed)`, skips excluded, and the docstring says so plainly (`gen_support_matrix.py:9`). Spot-checks hold: Dense 70/71 = 98.6% (`README.md:89`), ops/numpy 5502/5505 = 99.9% (`README.md:105`), TOTAL 7655/7664 = 99.9% (`README.md:107`).
- Every failure in the tally is individually named with a cause and a disposition (`README.md:63-69`, `README.md:109-117`) — upstream test-harness gaps, a tinygrad slicing limitation with an upstream draft, an FMA-precision residual. This is the opposite of hiding failures.
- The "Known gaps" section (`README.md:120-132`) is genuinely a gaps list: data-dependent-shape ops, fused RNN kernels, sparse/ragged, TF-string preprocessing, complex arithmetic. Each links to a memo explaining why.
- The project caught and corrected its own overclaim: HANDOFF records that the earlier "3.15 / 3.16" version-support claim was wrong and was fixed after checking PyPI (`HANDOFF.md:49-52`). That is the honesty culture working.

### Is the matrix cherry-picked?

**No, but it is not regenerable either.** The suite manifest in `gen_support_matrix.py:52-73` contains 17 suites — a curated set of representative layer/op files. Curation cuts *in the project's favor to a degree* (Dense, Conv, losses are the polished core), but two facts defuse the cherry-picking charge:

1. The `<!-- TALLY -->` block (`README.md:59-70`) reports the **full** layers test tree, preprocessing included — the un-curated referee number (1,989/5/215).
2. The two largest, ugliest suites — ops/numpy (5,502 tests, 708 skips) and the preprocessing tree — **are in the published table** (`README.md:105-106`) and drag the total down, not up.

The real problem is the inverse of cherry-picking: those two rows **cannot come from the script's `--run` mode**, because they are absent from `SUITES` (`gen_support_matrix.py:52-73`). They can only have been fed in via `--from-log` with hand-written `=== <suite>` headers. So the published table is a merge of generated and hand-assembled data, and a future `--run --inject` would silently *drop* the two biggest suites from the table. HANDOFF compounds the confusion by saying "18 suites" (`HANDOFF.md:32`) when the script has 17 and the README shows 19 rows.

One metric caveat worth stating in the README rather than only in a script docstring: with 762 skips excluded from the denominator (708 of them the sparse-gated ops/numpy skips, `docs/ops-numpy-triage.md:18-21`), "coverage 99.9%" measures *correctness of what runs*, not *fraction of Keras surface supported*. The skips are legitimately backend-gated and documented, but a hurried reader will read the wrong claim.

### Where the copies have drifted

Four documents state overlapping facts; three disagreements were found:

| Claim | Location A | Location B |
|---|---|---|
| Layers-tree pass rate | 99.7% — `README.md:61` | "currently 99.8%" — `TUTORIAL.md:137` |
| Matrix headline | 99.9% (19 rows) — `README.md:107` | "18 suites … 98.1%" — `HANDOFF.md:32-33` |
| Orthogonal initializer | works, "RNN layers (incl. default Orthogonal init)" — `README.md:76-77` | *unsupported*, "land[s] with the tinygrad linalg wave — until then pass an explicit alternative" — `examples/README.md:40-43`, enforced in code at `examples/char_rnn.py:45-48` |

The third is a direct user-facing contradiction: linalg's Householder QR exists (`docs/architecture.md:58-59`), the README says the default init works, and the examples page tells users it doesn't. HANDOFF also carries superseded numbers inside itself — ops/image "306/25/5" and preprocessing "679/14/29" in item 5 (`HANDOFF.md:134-137`) versus the README's 331/0/5 and 689/4/29, and the "39/40" fuzz line in the verified-state section (`HANDOFF.md:43-44`) versus item 1's "now 80/80" (`HANDOFF.md:104`). HANDOFF's header does say "verify with the commands given rather than trusting numbers blindly" (`HANDOFF.md:4-5`), which is the right disclaimer, but a handoff doc that disagrees with itself taxes exactly the newcomer it exists for.

**Overall honesty grade: high.** The claims that matter (test tallies, failure dispositions, known gaps) are accurate, sourced, and self-correcting. The drift is bookkeeping debt, not spin.

## 2. The executable tutorial

The pattern: `tests/test_tutorial.py` regex-extracts every ` ```python ` fence from TUTORIAL.md, concatenates them in order, and runs them in one fresh subprocess with `KERAS_BACKEND` deliberately popped so the import-hook default path is what gets exercised (`test_tutorial.py:17-34`). CI runs it on every push (`.github/workflows/ci.yml:18-19`).

**It works, and it has already earned its keep.** HANDOFF records a real catch: the tutorial's loud-stub demo used `rot90`; when `rot90` was implemented, the tutorial went red and had to be updated to use `unique` (`HANDOFF.md:121-122`). The tutorial even anticipates its own failure mode in prose: "If this block ever breaks the build, that's the tutorial working as intended" (`TUTORIAL.md:98-100`). The content choices are also right — the two most philosophy-defining behaviors (loud `NotImplementedError`, complex-arithmetic refusal) are demonstrated as *executable assertions* (`TUTORIAL.md:91-96`, `TUTORIAL.md:113-118`), not described.

Three limits, in descending order of importance:

1. **Prose is not enforced.** The "99.8%" at `TUTORIAL.md:137` is exactly the kind of claim the mechanism exists to keep fresh, and it has already rotted (README says 99.7%). Numbers in prose need either removal or a token-injection mechanism like the README's `<!-- TALLY -->` markers.
2. **Failure localization is poor.** A failing run dumps the whole stdout/stderr of the concatenated program (`test_tutorial.py:35`); nothing maps the failure back to *which markdown block* broke. Running blocks cumulatively (block 1, then 1+2, …) or injecting `print` sentinels between blocks would cost little.
3. `sh` fences (the `pip install` line) are not executed — acceptable, but worth knowing the install instructions are the one untested part of the page.

**Verdict on the pattern: adopt-worthy.** This is the cheapest anti-rot mechanism in the repo and the only one that covers user-facing prose+code together.

## 3. The parity fuzzer

### Design quality: strong

The architecture is correct for a hard constraint — a Keras process is locked to one backend at import, so the parent (`tools/parity_fuzz.py`) never imports keras and compares `.npz` outputs from per-backend child subprocesses (`tools/README.md:13-27`, `_parity_child.py:2-24`). Specific design decisions that deserve credit:

- **Determinism done right**: every case is a pure function of `(seed, index)` via `np.random.default_rng([seed, index])` (`parity_fuzz.py:451`), so case IDs are stable and reproducible.
- **The repro-command bug and its fix**: a printed repro command that omitted `--kinds`/`--tol-scale` could re-judge a FAIL as a PASS; the fix makes every generation-affecting flag travel, with a docstring explaining exactly why (`parity_fuzz.py:858-867`, `HANDOFF.md:96-104`).
- **The all-skipped guard**: a run where nothing was compared exits 2 with "silent green with zero evidence" named in a comment (`parity_fuzz.py:1015-1022`). This closes the classic fuzzer failure mode of a dead reference producing a green run.
- **Comparison rigor**: NaN-placement and inf-placement are checked separately from tolerance (`parity_fuzz.py:593-599`); integer outputs must match exactly, with width differences tolerated but values not (`parity_fuzz.py:567-584`); weights for layer cases are generated in the parent and installed via `set_weights` so both backends compute on byte-identical parameters (`parity_fuzz.py:346-349`, `_parity_child.py:67-73`).
- **Three-tier gradient checking**: cross-backend analytic when the reference has autograd, and — the workhorse against the numpy reference — central finite differences computed on the backend-under-test itself (`_parity_child.py:129-156`), with honest, documented tolerances for FD noise (`tools/README.md:63-66`).
- `NotImplementedError` counted as `UNSUPPORTED`, never FAIL — "the honest state of a partial backend" (`tools/README.md:81-84`).

### The 39/40 claim and the float64 decision record

The "39/40" (`HANDOFF.md:43-44`) was a single dated run whose one flag became `docs/float64-promotion.md` — and that memo is the best document in the repo. It pins the mechanism to a specific upstream line (`keras/src/backend/common/dtypes.py:240`, quoted at `float64-promotion.md:25-29`), presents a verified four-row behavior table (`float64-promotion.md:11-21`), lays out both options *with blast radius* (`float64-promotion.md:43-61`), and — critically — names the invariant being bent: "option A knowingly deviates from the reference's observed behavior… That deviation-with-a-named-reason is the owner's call" (`float64-promotion.md:64-67`). The implementation matches the record exactly: the benign direction is ref-float64/test-float32 only, judged under float32 tolerances, with a visible note (`parity_fuzz.py:547-555`, `parity_fuzz.py:679-680`); the reverse direction still fails (`parity_fuzz.py:588-589`). One cosmetic defect: the memo's status line says "decided & implemented" (`float64-promotion.md:3`) while its final section is still titled "Why this stays open" — a pre-decision draft heading that survived the decision.

### Real gaps

1. **`make fuzz` does not do what two documents say it does.** The Makefile runs `parity_fuzz.py --seed 0 --cases 100` (`Makefile:44-45`) — no `--slow`. Against the numpy reference, `resolve_kinds` then *drops all gradient cases* with a printed note (`parity_fuzz.py:775-779`). Yet `TUTORIAL.md:143-144` describes it as "randomized cross-backend parity fuzzing with finite-difference gradient checks (`make fuzz`)", and `README.md:67-69` headlines "incl. finite-difference gradient checks: green". The FD-gradient runs happened (they are how the float64 flag was found), but the *reproducible entry point everyone is told to use* is forwards-only. Either the Makefile should grow a `fuzz-grad` target (`--kinds grad --slow`) or the prose should stop attributing gradient checking to `make fuzz`.
2. **UNSUPPORTED is invisible to regression detection.** Exit code is 0 as long as one case passes and none fail (`parity_fuzz.py:1013-1023`). If a previously-implemented op regressed to a loud stub, every one of its fuzz cases would flip PASS→UNSUPPORTED and the fuzzer would stay green. The referee suite covers this, but a `--max-unsupported` threshold or a baseline count would make the fuzzer self-sufficient.
3. **Coverage is narrow relative to its billing**: 23 ops, 9 grad-ops, 6 layers (`parity_fuzz.py:76-119`), all layer cases inference-only (`_parity_child.py:73`), grad cases float32-only (`parity_fuzz.py:217-219`). Fine as a drift-hunter for the core; not the broad surface the phrase "cross-backend parity fuzz" suggests. The extension path is well documented (`tools/README.md:146-153`).
4. **Fuzz is not in CI** (`.github/workflows/ci.yml` runs verify/tutorial/smoke/vendor-check only). Defensible on runtime grounds, but combined with the fixed `--seed 0`, nothing ever explores new seeds automatically.

## 4. Onboarding

**A new contributor to the *packaged repo* gets productive fast.** The path is short and every step is executable: README quick start → TUTORIAL (CI-verified) → `CONTRIBUTING.md`'s numbered method (`CONTRIBUTING.md:5-32`) → `make verify tutorial smoke` (`Makefile:15-41`). The method section is exceptional as contributor documentation because it teaches *judgment*, not just mechanics: "Never port by intuition" (`CONTRIBUTING.md:13`), "Wrong answers are the one bug class this project refuses to ship" (`CONTRIBUTING.md:21`), "Widening an anchor to 'probably matches' is a rejected PR" (`CONTRIBUTING.md:31-32`). The PR checklist (`CONTRIBUTING.md:69-79`) and the update-numbers-only-with-a-tally rule (`CONTRIBUTING.md:64-67`) directly encode the honesty culture. The examples directory has a proper index with runtime expectations (`examples/README.md:26-35`).

**A contributor to the *backend itself* hits a wall.** The referee workflow — the heart of the method — depends on artifacts that exist only on the author's machine: the keras clone at `/home/dev/workspace/keras`, `ktg-venv`, `tf-venv`, `pkg-test-venv`, and the zigcc shim at `/home/dev/.local/bin/zigcc` (`HANDOFF.md:63-79`). CONTRIBUTING's version of the same instructions is one paragraph ("clone `keras` at the supported release tag… a root-level `conftest.py` containing `import keras_tinygrad` does it", `CONTRIBUTING.md:48-53`) — the conftest trick is a genuinely useful nugget, but there is no script that stands up a referee environment from scratch, and no documented recipe for the tensorflow-collection venv that preprocessing/ops-image suites require (`README.md:116-117`). Smaller nits: the zig-shim note appears twice in CONTRIBUTING (`CONTRIBUTING.md:44-46` and `:90-91`); `examples/README.md:6-8` tells users to `export KERAS_BACKEND=tinygrad` even though the hook defaults it (the hook's *entire point*, `TUTORIAL.md:18-22`), with only "on this box" as a hedge.

HANDOFF as a genre piece is very good — state, commands, invariant summary, a remainders list ordered smallest-first, and an explicit owner-only decision queue (`HANDOFF.md:156-183`) that cleanly separates "work anyone can do" from "calls only the owner may make." Its weakness is §1's: it is the most drift-prone document in the repo and has no enforcement mechanism.

## 5. The decision-record culture

The `docs/` directory is effectively an ADR log, and it is the repo's strongest asset:

- **Every bent rule has a memo.** float64 demotion (`float64-promotion.md`), the complex tier boundary (`complex-support.md`), unique/vectorize (`unique-vectorize.md`). Each carries a status line with a date, the mechanism with citations, options with blast radius, and — the distinctive move — the *invariant tension named out loud*: complex-support's "with the referee that will judge the new ops named" rule for future extensions (`complex-support.md:54-59`), unique-vectorize's insistence that option (a) requires "the invariant amendment written in the same diff" (`unique-vectorize.md:60-62`).
- **Architecture.md has a binding maintenance contract**: "update this file in the same diff whenever a boundary moves… If a statement in this file is wrong, that's a bug in the diff that made it wrong" (`architecture.md:10-14`), plus a "Known leaks and quirks (accepted, not aspirational)" section (`architecture.md:181`) — the rare architecture doc that documents its own compromises.
- **The triage memo is a model of the form**: all 1,196 failures bucketed with 100% accounting, silent-wrong-answer bugs (bucket c) explicitly prioritized above the 1,047-test mechanical bucket because "it's the only silent-wrong-answer surface" (`ops-numpy-triage.md:152-154`), and a re-verify command at the top instead of trust-me numbers (`ops-numpy-triage.md:9-13`).
- **The upstream drafts show the culture surviving contact with inconvenient facts**: item 3 of the keras draft was investigated, found stale, and *kept in the document as a dropped item with the reason* (`upstream-keras-draft.md:233-264`); the tinygrad draft carries the mandatory AI-disclosure constraint and the `__bool__`-ban history rather than hiding them (`upstream-tinygrad-draft.md:178-190`).

Weaknesses of the culture as practiced: (1) **no index** — nothing lists the decision records, their statuses, or their supersession relationships; a newcomer discovers them via scattered links; (2) **statuses rot in prose** (the "decided" memo with the "stays open" section, §3); (3) the culture covers *decisions* but not *facts* — the numbers duplicated across README/HANDOFF/TUTORIAL/examples have no single source of truth, which is exactly where all the drift in §1 lives. The README's token-injection blocks (`<!-- TALLY -->`, `<!-- SUPPORT_MATRIX -->`) are the right pattern; they just stop at the README's edge.

## 6. Ranked findings

1. **User-facing contradiction on Orthogonal-initializer support.** `README.md:76-77` says RNN layers work "incl. default Orthogonal init"; `examples/README.md:40-43` and `examples/char_rnn.py:45-48` say it is unsupported pending "the tinygrad linalg wave" (which has landed — `docs/architecture.md:58-59`). One of these is wrong; the examples appear stale. *Fix: update char_rnn.py to use the default initializer (it doubles as a regression test) and delete the note.*
2. **`make fuzz` does not run the gradient checks attributed to it.** `Makefile:44-45` omits `--slow`, so grad cases are dropped against the numpy reference (`parity_fuzz.py:775-779`), while `TUTORIAL.md:143-144` and `README.md:67-69` both attribute finite-difference gradient checking to the fuzz entry point. *Fix: add a `fuzz-grad` target or amend the prose.*
3. **The published support matrix is not regenerable from the generator.** The ops/numpy and preprocessing rows in `README.md:105-106` are absent from `SUITES` (`gen_support_matrix.py:52-73`); a future `--run --inject` silently drops the two largest suites. HANDOFF's "18 suites" (`HANDOFF.md:32`) matches neither the script (17) nor the table (19). *Fix: add the two suites to the manifest (with the tf-venv caveat noted) or document the log-merge procedure.*
4. **Three different headline percentages in circulation.** 99.7% (`README.md:61`), 99.8% (`TUTORIAL.md:137`), 99.9% (`README.md:107`), plus HANDOFF's stale 98.1% (`HANDOFF.md:33`). Each is defensible in its own frame; together they erode the credibility the tally machinery earns. *Fix: one number, token-injected, everywhere else linked.*
5. **Fuzzer blind spot: support regressions are invisible.** UNSUPPORTED never affects the exit code (`parity_fuzz.py:1013-1023`); an implemented-op → loud-stub regression keeps fuzz green. *Fix: an unsupported-count baseline or `--max-unsupported` flag.*
6. **HANDOFF disagrees with itself.** "Verified state (2026-07-27)" retains the superseded 39/40 fuzz line (`HANDOFF.md:43-44`) against item 1's 80/80 (`HANDOFF.md:104`), and item 5's ops/image 306/25/5 (`HANDOFF.md:134`) against the README's 331/0/5. *Fix: strike superseded numbers in the same diff that lands the newer run.*
7. **Backend-contributor onboarding depends on one machine.** Referee venvs, the keras clone, and the zigcc shim are absolute paths on the author's box (`HANDOFF.md:63-79`) with no bring-up script; CONTRIBUTING covers the referee in one paragraph (`CONTRIBUTING.md:48-53`). Acceptable for a solo handoff; the first blocker for anyone else.
8. **Tutorial enforcement gaps**: prose numbers unverified (`TUTORIAL.md:137`), failing block not localized in test output (`test_tutorial.py:35`).
9. **Decision-record hygiene**: stale "Why this stays open" heading in the decided float64 memo (`docs/float64-promotion.md:63`); no ADR index; minor duplication (zig note twice in `CONTRIBUTING.md:44-46,90-91`; redundant `KERAS_BACKEND` export in `examples/README.md:7`).

The pattern across findings 1, 3, 4, and 6 is a single root cause: facts are stated in more places than the anti-rot machinery covers. The repo already invented the cure — executable docs and token-injected tables — and needs only to apply it to its remaining prose.
