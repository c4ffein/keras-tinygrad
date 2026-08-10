# Packaging & Loader — the Import-Hook Mechanism

**Verdict.** This is an unusually disciplined implementation of an inherently fragile idea. The meta-path hook is small (215 lines), fails loud in every failure mode the authors thought of — and they thought of most of them: keras-imported-first, anchor drift, bytecode-only installs, the `DynamicBackend` silent-`None` trap. The exactly-once anchor rule is a genuinely sound safety invariant *for the six sites it covers*. The real risks live one level up: the anchor check is textual, not semantic, so a keras release that adds a **seventh** dispatch site sails through cleanly patched-but-incomplete; the drift-guard triangle machine-checks three of its four edges but leaves "every clone edit has a patch-table entry" as a convention; and the packaging metadata has a latent build-breaking bug (`setuptools>=68` floor vs. an SPDX license string that needs ≥77). Nothing here is careless — but several guarantees the docs state as absolute are actually "absolute within the covered set."

---

## 1. The import-hook mechanism

### Architecture (sound)

`src/keras_tinygrad/_loader.py` installs one `MetaPathFinder` at the front of `sys.meta_path` (`_loader.py:215`) doing two jobs:

1. **Serve** `keras.src.backend.tinygrad` from `_backend/` (`_loader.py:175-180`). Only the package name is intercepted; `submodule_search_locations=[_BACKEND_DIR]` lets stock `PathFinder` resolve submodules. Clean.
2. **Patch** six stock modules by exec'ing source rewritten with exact-string replacements (`_PATCHES`, `_loader.py:56-134`). Each anchor must occur **exactly once** (`_apply_patches`, `_loader.py:159-170`) or the import dies with an explicit "your keras version is not supported" `ImportError`.

Well-chosen details:

- `_PatchedLoader` locates the real module via `importlib.machinery.PathFinder.find_spec` directly (`_loader.py:184`), avoiding re-entering `sys.meta_path` (no infinite recursion, no dependence on the hook's own position).
- The patched spec keeps the real `origin` and real `submodule_search_locations` (`_loader.py:194-197`), so `keras.src.backend`'s siblings resolve from the stock install — the hook owns six files, not a subtree.
- `get_source` raises loudly on bytecode-only installs (`_loader.py:149-150`) instead of returning garbage.
- `install()` refuses to run if any patch target is already in `sys.modules` (`_loader.py:209-213`) — a half-patched Keras cannot exist. Tested end-to-end in a subprocess (`tests/test_loader.py:83-98`).

### Failure-mode audit

**Import order.** Guarded and tested. The guard covers the six targets plus the backend package; since any `keras.*` import runs `keras/__init__.py`, which pulls in `keras.src.backend`, partial-keras-first scenarios are caught in practice. The ruff isort custom section (`pyproject.toml:48-58`) that *forces* `keras_tinygrad` to sort before `keras` is a small, clever piece of tooling — the linter enforces a load-bearing import order instead of alphabetizing it away.

**Anchor mismatch / keras 3.16.** When an anchor drifts, the failure is exactly as loud as claimed: `ImportError` naming the module, the match count, and the diagnosis (`_loader.py:162-168`); both zero-match and double-match are unit-tested (`test_loader.py:142-149`). **But** loudness is conditional on the anchors actually drifting. Two quieter paths exist:

- *Anchors survive, semantics don't.* The anchors are `else:\n    raise ...` tails of dispatch chains — among the most stable strings in keras, which is good for compatibility but means a 3.16 that adds a **new** per-backend dispatch site (a seventh touchpoint) or changes behavior around the existing ones imports cleanly and fails later, deep in keras, or silently (the exact bug class the `DynamicBackend` patch exists for — its `__getattr__` returns `None` for unknown backends, `_loader.py:99-118`). There is no runtime `keras.__version__` gate in the loader; the only version enforcement is pip's resolver via the `<3.16` pin, which a user can override with a dependency-conflict warning. A cheap hardening: check `keras.__version__` against a tested list, overridable via env var for experimentation.
- *Whitespace churn.* Anchors embed exact indentation and line breaks. A keras-wide formatter change breaks all six at once — loudly, so this is safe, but it makes every keras patch release a potential support event even when nothing semantic changed.

**Fall-through under exotic finders.** If `PathFinder` can't locate a patch target, the finder returns `None` "to let the normal machinery produce its error" (`_loader.py:185-186`). Under a normal install that's correct. But under PyInstaller/other vendoring meta-path finders that serve keras themselves, `PathFinder` fails, our finder steps aside, and the *later* meta-path finder imports the module **unpatched** — silently, contradicting the "we never let Keras fall through" doctrine (`docs/how-it-works.md:43-46`). Narrow audience, but the one hole in the loudness story. Raising when `PathFinder` misses but `fullname` is a patch target would close it.

**Thread safety.** The finder is stateless after install; concurrent imports are serialized by Python's per-module import locks. `install()` itself is a check-then-act on `_FINDER` (`_loader.py:207-215`) — two threads racing it could insert two finders (harmless: first wins per lookup) — and in principle a thread could import keras between the `sys.modules` scan and the insert. In practice `install()` runs at `keras_tinygrad` module init under the import lock, so this is theoretical.

**.pyc caching.** Handled correctly by construction. The patched spec comes from `spec_from_loader` without `has_location`, so Python never writes bytecode for the six patched modules; they recompile per process (negligible — six files). Stale `__pycache__` entries for stock keras are irrelevant because `get_source` reads the `.py` from disk. No pip-cache interaction exists.

**Traceback fidelity (undocumented wart).** Patched source is compiled with the *real* file as `origin` (`_loader.py:155`), but the patches insert lines (the `keras.src.backend` patch adds five). Line numbers in the compiled code refer to the patched text while `linecache`/`inspect` read the on-disk unpatched file — so within the six patched modules, traceback source lines *after* a patch insertion point are off by the insertion size. `docs/how-it-works.md:48-50` says "tracebacks point at the real files," which is true but overstates fidelity. Low impact, worth a docs sentence.

**Import side effect.** `__init__.py:18` mutates `os.environ` (`setdefault("KERAS_BACKEND", "tinygrad")`) on import. Documented in the docstring and explicit-env-wins is tested (`test_loader.py:66-80`), but it means any incidental import of the package flips the backend for a subsequent `import keras`. A deliberate ergonomic choice; note it as one.

## 2. The dual-source-of-truth model and the drift triangle

Four artifacts must agree: the clone (patched-state source of truth), the vendored `_backend/`, the loader's replacement texts, and installed stock keras. The triangle:

- **`--check`** (`sync_vendor.py:146-161`): vendored ↔ clone byte-diff, plus clone contains each loader *replacement* exactly once.
- **`--self-check`** (`sync_vendor.py:188-193`): installed stock keras contains each *anchor* exactly once. Duplicated in-process by `test_loader.py:160-179` so plain `pytest` catches it.
- The patched-state / unpatched-state distinction (`check_anchors`, `sync_vendor.py:68-107`) is exactly right, and `HANDOFF.md:53-59` records that getting it right found real drift ("`--check` was crying wolf… checking anchors against the clone is wrong by construction: the standardize_dtype patch consumes its anchor"). The asymmetric diagnostics ("REPLACEMENT present, file appears patched in-tree" / "ANCHOR present unpatched", `sync_vendor.py:94-101`) show the failure modes were thought through, not just enumerated.

**Two real gaps:**

1. **Unregistered touchpoints are invisible.** `--check` iterates *the loader's patch table* (`sync_vendor.py:84`). A new keras-core edit in the clone that never got a patch-table entry touches a file the tool never opens — architecture invariant 11 ("every keras-core touchpoint has an anchor in the patch table," `docs/architecture.md:116-117, 178`) is enforced by convention and review only. The clone is a git repo; `git diff <upstream-tag> --name-only` minus the backend dir minus the patch-table files would machine-close this edge. This is the largest gap in the triangle: the failure it misses is precisely "works in-tree, silently broken in the package."
2. **Non-recursive vendoring.** `list_py` (`sync_vendor.py:37-41`) and `compare` see only top-level `.py` files, and the wheel's package-data glob is equally flat (`_backend/*.py`, `pyproject.toml:21`). Today the backend is flat (verified: 12 modules, no subdirs), but the day the clone grows a subpackage, `--check` stays green, `--sync` skips it, and the wheel ships without it — three silent failures from one structural change. Cheap fix: recurse, or assert no subdirectories exist.

Minor: `test_loader.py:167-174` reimplements `module_source_path` with a different candidate-probing strategy — two implementations of the same path logic can diverge; importing the script's helper would be tighter.

## 3. Packaging quality

**Metadata (`pyproject.toml`) is functional but thin, with one latent build bug.**

- `license = "Apache-2.0"` (line 9) is a PEP 639 SPDX string, which setuptools accepts only from **77.0**; the build floor is `setuptools>=68` (line 2). Any environment resolving setuptools 68–76 fails the build (or worse, older versions mis-set metadata). uv's fresh-latest habit hides this today. Fix the floor.
- Missing for a PyPI-bound package: `readme`, `authors`, `urls`, `classifiers`, keywords. `HANDOFF.md:160` lists "publishing + PyPI name claim" in the decision queue, so this is pre-publication state — but it's the checklist.
- Version is stated twice (`pyproject.toml:7`, `__init__.py:16`) with no single-sourcing; they will drift on the first bump someone makes in a hurry.
- **`tinygrad>=0.13` with no upper bound** (line 13) is the mirror-image risk of the carefully pinned keras: the trainer is built on tinygrad 0.13's `loss.gradient` API (`docs/how-it-works.md:68-70`), tinygrad is pre-1.0 and breaks APIs freely, and nothing fails loud at import when 0.14 changes semantics. The keras pin got a whole doctrine; tinygrad got a floor.

**The keras pin itself (`keras>=3.15,<3.16`) is the right strategy** for text-anchor coupling — honest, and consistent with README/docs (`README.md:37-40`, `docs/how-it-works.md:99-103`; HANDOFF records correcting an earlier false "3.16" claim). Caveat: 3.15.0 support is verified by *file-level anchor check on a downloaded wheel* (`HANDOFF.md:49-52`), while only 3.15.1 (the lockfile resolution) is runtime-tested. The claim and the evidence differ by a runtime.

**CI (`.github/workflows/ci.yml`)** covers lint, format, loader tests, executable tutorial, smoke train, `--self-check`, and byte-compiling the vendored sources (a nice cheap syntax gate on files ruff is forbidden to touch). Gaps:

- Single Python, single lockfile-pinned keras — no matrix across the claimed 3.15.x range or the `>=3.11` python floor.
- **No build-and-install-from-wheel step.** uv installs the project from the checkout; the `package-data` path that actually ships `_backend/` in a wheel is never exercised as an installed artifact. A package-data regression would ship.
- `--check` (clone diff) can't run in CI by design (needs the sibling clone) — acceptable since the vendored copy *is* the shipped artifact, but it means the triangle's clone edge is verified only on the owner's machine.

**Test quality is high** where it exists: every keras-importing scenario in a fresh subprocess with `KERAS_BACKEND` controlled (`test_loader.py:27-39`) — the correct discipline for process-global import state; both anchor-failure directions unit-tested; the executable tutorial (`test_tutorial.py`) is documentation that cannot rot, and `HANDOFF.md:121-122` records it catching a real doc regression. Untested: the `DynamicBackend` patch's behavior (the one patch whose failure mode is silent), and an end-to-end anchor-mismatch against a simulated drifted keras tree.

## 4. Maintainability — what breaks first

Ranked by expected time-to-breakage:

1. **tinygrad releases** — unpinned above, fast-moving, API-coupled trainer. Breaks at runtime, not import.
2. **keras 3.16** — pip resolver blocks co-install (good); a forced install fails loud *iff* anchors drifted; the incomplete-patch-set scenario (new dispatch site) fails late or quiet. Cost per keras release is otherwise mechanical and well-defined: run `--self-check` against the new wheel, re-run the referee suite, bump pin and docs — likely under a day when anchors hold, since the anchors are stable `else: raise` tails.
3. **Clone structural drift** — a backend subpackage or an unregistered seventh touchpoint defeats the flat, table-scoped guards (§2).
4. **Bus factor** — the whole system leans on a sibling `../keras` clone that is *entirely uncommitted* (`HANDOFF.md:13-15`). Until the owner commits, the source of truth is one `rm -rf` from being reconstructible only from the vendored snapshot (which, fortunately, `--check` keeps byte-identical).

## 5. Notable engineering

- **The exactly-once anchor rule** — converting "monkeypatching upstream source" from a hope into a checkable invariant with a designed failure message, unit tests for both failure directions, and a standalone verifier.
- **The patched-state/unpatched-state distinction** in `check_anchors` — and the honesty of `HANDOFF.md:53-59` documenting that the first version was wrong and what finding that bug turned up.
- **The `DynamicBackend` catch** (`_loader.py:99-103`): identifying that keras's one dispatch site that *doesn't* raise would produce downstream `NoneType` crashes, and patching it anyway. That's the difference between "made it import" and "understood the dispatch surface."
- **Linter-enforced import order** (`pyproject.toml:44-58`) and the **executable tutorial** — both turn prose invariants into mechanical ones, the same instinct throughout.
- `install()`'s all-or-nothing guard, and `get_source`'s loud bytecode-only refusal — small, complete failure-mode coverage.

## 6. Ranked findings

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| 1 | High | Anchor checks are textual, not semantic: a keras release adding a new per-backend dispatch site imports cleanly but incompletely patched; no runtime `keras.__version__` gate backs up the pip pin | `_loader.py:159-170`, `pyproject.toml:12`, `docs/how-it-works.md:99-103` |
| 2 | High | Drift triangle can't see clone keras-core edits absent from the patch table — invariant 11 is convention-only; a git-diff check against the clone's upstream base would close it | `scripts/sync_vendor.py:84`, `docs/architecture.md:116-117` |
| 3 | Med | `license = "Apache-2.0"` (SPDX string, needs setuptools ≥77) with declared floor `setuptools>=68` — latent build failure in non-latest environments | `pyproject.toml:2,9` |
| 4 | Med | `tinygrad>=0.13` unbounded upper pin despite 0.13-specific `loss.gradient` coupling; breaks at runtime, not loudly at import | `pyproject.toml:13`, `docs/how-it-works.md:68-70` |
| 5 | Med | Non-recursive vendor sync and wheel glob: a backend subpackage would be silently unsynced, unchecked, and unshipped | `scripts/sync_vendor.py:37-41,121-132`, `pyproject.toml:21` |
| 6 | Med | CI: no keras 3.15.0/3.15.x or python matrix (3.15.0 verified file-level only), and no build-wheel-then-install test — package-data shipping path unexercised | `.github/workflows/ci.yml`, `HANDOFF.md:49-52` |
| 7 | Low | Silent fall-through to unpatched keras when `PathFinder` misses but another meta-path finder (PyInstaller-style) serves the target | `_loader.py:184-186` |
| 8 | Low | Traceback line numbers within the six patched modules drift from on-disk source after patch insertion points; docs overstate fidelity | `_loader.py:155`, `docs/how-it-works.md:48-50` |
| 9 | Low | Version stated in two places with no single-sourcing; missing readme/urls/classifiers/authors ahead of the planned PyPI publish | `pyproject.toml:7`, `src/keras_tinygrad/__init__.py:16` |
| 10 | Low | `install()` check-then-act race (duplicate finders / concurrent-thread keras import); theoretical under module-init import lock | `_loader.py:207-215` |
| 11 | Nit | `test_loader.py` reimplements module-path resolution differently from `sync_vendor.module_source_path`; DynamicBackend patch behavior untested | `tests/test_loader.py:167-174`, `scripts/sync_vendor.py:58-65` |
