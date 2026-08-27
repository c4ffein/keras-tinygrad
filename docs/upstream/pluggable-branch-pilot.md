# Pilot run — keras-tinygrad on keras' `pluggable_backend` branch

Status: WORKING, 2026-08-21, against branch head `937c08e60`. This is the
evidence base for the engagement issue drafts in
`/home/dev/workspace/exports/07-ISSUE_DRAFTS.md`.

**RE-RUN 2026-08-27 against branch head `fd4be6b46` (the #23507
merge):** still working — training green, dtype hook fires, dense_test
**88/1/1** (unchanged; the 1 is still the float8 `train_one_step`
assumption). Friction item 1 (the `operation_fn` branch bug) was fixed
upstream by #23507 with the same fix shape — our local branch fix and
`exports/06` are retired; the generic-else mechanism patch
(`exports/05`) re-applies clean on the new head with zero conflicts.
Items 2–5 (dtype hook, protocol docs, test assumptions, DynamicBackend)
remain open upstream and current.

## Result

With ~80 generic lines added to the branch's six dispatch `else:` tails
(name-convention resolution of `keras_<name>.src` — no entry points, no
aliasing in keras, matching their idiom) plus a thin protocol shim in
this package, **`KERAS_BACKEND=tinygrad` + plain `import keras` works on
their branch**: MLP compiles/fits/predicts, tinygrad DTypes standardize
correctly, and `dense_test.py` scores **88 passed / 1 failed / 1
skipped** — the 1 fail being the known float8 `train_one_step`
test-side assumption (our old PR-1b fix #1), now reproduced on their own
branch.

## Friction found (all verified live)

1. **Branch bug (blocks everything eager):** `functional.py::call`
   references `operation_fn` — deleted by the #23397 refactor — and a
   `**kwargs` its signature doesn't have. Every eager
   Functional/Sequential call raises NameError. Local fix: master's
   `self._run_through_graph(inputs)` shape
   (`exports/06-branch-functional-bugfix.patch`). Sibling of the
   branch's own `backend.is_tensor` fix commit; merge fallout of
   `0b0639195`. → issue draft (a).
2. **dtype gap:** `standardize_dtype` consults `.name` first; tinygrad
   `DType.name` ("half"/"float") mis-standardizes before any string
   heuristic runs. Fixed via optional plugin hook
   `standardize_dtype_hook` consulted for non-string dtypes (in the
   mechanism patch); a per-backend branch would also work. MLX only
   passes by spelling luck. → issue draft (b) point 1.
3. **Protocol surface (undocumented, inferred from in-tree backends):**
   star surface must export `ops` (package: `core/image/linalg/math/
   nn/numpy`, `__init__` star-exports `core` — keras calls
   `backend.ops.is_tensor`, `backend.ops.numpy.where`, ...), `random`,
   `rnn`, `Variable`, `compute_output_spec`, `device_scope`,
   `name_scope`, `SUPPORTS_*`, `IS_THREAD_SAFE`, optional
   `distribution_lib`; plus `trainer.py`/`layer.py`/`export.py`
   submodules. Note the star-update only sees what the plugin
   `__init__` itself imports — the `ops` submodule must be imported
   there explicitly.
4. **Test-side assumptions:** float8 `train_one_step`
   (`UnboundLocalError`, the 1 red above) and `trainer_test.py`'s
   dispatch-with-raise — both still present on the branch. →
   `exports/01b-keras-backend-assumptions.patch`, tree TBD by
   maintainers.
5. **DynamicBackend:** unknown backends leave `module` unbound
   (UnboundLocalError) and `set_backend` hard-whitelists the six names —
   both generalized in the mechanism patch.

## Artifacts

- Mechanism (their branch, generic `else:` tails only, in-tree elifs
  untouched): `exports/05-pluggable-branch-generic-else.patch`
  (worktree `/home/dev/workspace/keras-pluggable-poc`, branch
  `pluggable-poc`, uncommitted).
- Branch bugfix one-liner: `exports/06-branch-functional-bugfix.patch`.
- Plugin-side shim (this repo, additive): `src/keras_tinygrad/src/`
  (`__init__` = alias-register `_backend` + protocol surface incl.
  `ops` aggregation; `trainer/layer/export` re-exports) and the
  `KERAS_TINYGRAD_NO_HOOK=1` escape in `keras_tinygrad/__init__.py`
  (needed because the hook's stand-down probe only knows OUR fork's
  marker; a proper probe replaces it when their design ships).
- Demo venv: `/home/dev/workspace/poc-venv` (branch keras installed
  non-editable + this package `--no-deps`).

## Re-run

```sh
cd /home/dev/workspace && CC=~/.local/bin/zigcc KERAS_BACKEND=tinygrad \
  KERAS_TINYGRAD_NO_HOOK=1 poc-venv/bin/python -c \
  "import keras; print(keras.config.backend())"
cd keras-pluggable-poc && CC=~/.local/bin/zigcc KERAS_BACKEND=tinygrad \
  KERAS_TINYGRAD_NO_HOOK=1 ../poc-venv/bin/python -m pytest \
  keras/src/layers/core/dense_test.py -q
```

The stock-keras hook path is unaffected by all of this (shim is
additive; `NO_HOOK` defaults off) — re-verify with `make verify smoke`.
