# PoC — keras backend plugins via entry points (the fork)

Status: WORKING PoC, 2026-08-10. Nothing submitted upstream.

**UPDATE 2026-08-21 — role changed from proposal to reference.** Upstream
recon found the keras team building pluggable backends themselves since
early August (PRs #23396/#23397/#23410 + follow-ups; official
`keras-team/keras-mlx` and `keras-team/keras-openvino` plugin repos;
mechanism = `keras_<name>` naming-convention imports at the same six
touchpoints; initial master landing rewound 2026-08-20, branches active).
The design issue this PoC was built to support is therefore obsolete —
never file it. The PoC remains valuable as (a) proof our backend loads
plugin-style with zero patches, (b) an experience report for engaging
with their effort (sys.modules aliasing, standardize_dtype hook,
stand-down probe), and (c) the basis for adapting to their protocol.
See `/home/dev/workspace/KERAS_COMMITS_AND_ORDER_GUIDE.md` §0/§3/§4.

## What it proves

On a keras fork carrying one new file and five generalized dispatch sites,
**`KERAS_BACKEND=tinygrad python -c "import keras"` just works** — no
`import keras_tinygrad` first, no import hook, no source patching. Keras
itself discovers and imports the backend through a setuptools entry point.
The same installed `keras-tinygrad` package serves both worlds: hook on
stock keras, entry point on the fork (the hook detects native plugin
support and stands down).

Verified 2026-08-10:

- Demo (fork venv, plain `import keras`): backend `tinygrad`, hook absent
  from `sys.meta_path`, MLP compiles/fits/predicts, tinygrad `DType`
  routes through `standardize_dtype`. Zero patches applied.
- Referee subset on the fork path: `dense_test.py` **70 passed / 1 failed
  / 1 skipped** — identical to the stock-path tally (the 1 fail is the
  known upstream float8 test-side issue, see docs/upstream/keras-pr/).
- Stock path unchanged: `make verify` (9 loader tests), `make tutorial`,
  `make smoke` all green with the hook installing exactly as before.

## Where things live

- **Fork**: `/home/dev/workspace/keras-plugin-fork` — a git worktree of
  the keras clone, branch `plugin-backends` off upstream master
  (`abd068b`). All changes uncommitted, owner reviews and commits.
  The fork contains ZERO occurrences of "tinygrad" — fully generic.
- **Demo venv**: `/home/dev/workspace/fork-venv` (fork keras installed
  non-editable — the repo-root `keras/__init__.py` import hack makes
  editable installs resolve an empty module; keras-tinygrad installed
  `--no-deps` because the fork's dev version is outside the `<3.16` pin).
- **zig shim note**: `~/.local/bin/zigcc` now execs through
  `/home/dev/workspace/zig-venv` (ziglang 0.16 wheel) after the old
  ktg-venv vanished in an environment reset.

## The fork diff (would-be upstream PR)

One new file, five sites generalized. Every `elif` for the in-tree
backends is untouched; only the `else: raise` tails change.

1. **`keras/src/backend/plugins.py`** (new, ~120 lines with docs): resolves
   the `keras.backends` entry-point group. `load(name)` imports the
   plugin package and registers it in
   `sys.modules["keras.src.backend.<name>"]` **before** executing it, so
   the plugin's internal absolute imports and keras' per-submodule
   dispatch imports resolve as if the backend were in-tree. (This is the
   same aliasing trick the keras-tinygrad hook uses — moved inside keras,
   it stops being a crime and becomes the mechanism.) Also
   `load_submodule`, `optional_hook`, `star_import` helpers.
2. **`keras/src/backend/__init__.py`**: `else:` → try plugin; star-import
   its surface, `BackendVariable = plugin.core.Variable`,
   `distribution_lib` optional; original `ValueError` if no plugin.
3. **`keras/src/models/model.py`**: `else:` →
   `plugins.load_submodule(name, "trainer").Trainer`; original
   `RuntimeError` if absent.
4. **`keras/src/layers/layer.py`**: same, `layer` / `Layer` mixin.
5. **`keras/src/export/saved_model.py`**: same, `export` /
   `ExportArchive`.
6. **`keras/src/backend/common/variables.py`**: `standardize_dtype` calls
   an optional plugin hook `standardize_dtype_hook(dtype)` for non-string
   dtypes before the generic `name`/`__name__` fallbacks.
7. **`keras/src/utils/backend_utils.py`**: `DynamicBackend.__getattr__`
   falls through to `plugins.load`; a plugin miss now raises `ValueError`
   (previously it silently returned None — the exact bug class the
   keras-tinygrad loader had to patch around).

## The plugin protocol (empirically = the six-touchpoint contract)

An entry point `name = "package.path"` in group `keras.backends`, where
the package mirrors the in-tree backend layout:

- star-import surface: the ops modules, `convert_to_tensor`, `SUPPORTS_*`
  flags, `core.Variable`, ... (what `keras.src.backend.<name>` exports
  in-tree);
- `trainer.Trainer`, `layer.Layer`, `export.ExportArchive` (standard
  names — the backend added one-line aliases:
  `Trainer = TinygradTrainer` etc.);
- optional `standardize_dtype_hook(dtype)` (returns a Keras dtype name or
  None to decline) — the backend's maps tinygrad `DType` objects through
  `to_keras_dtype`;
- optional `distribution_lib`.

Conformance = keras' own test suite under `KERAS_BACKEND=<name>`; a
conformant plugin behaves identically installed out-of-tree or copied
into the keras source tree.

## keras-tinygrad side changes

- `pyproject.toml`: `[project.entry-points."keras.backends"]
  tinygrad = "keras_tinygrad._backend"` — inert on stock keras.
- `__init__.py`: filesystem-only probe (`find_spec("keras")` +
  `.../src/backend/plugins.py` existence — no keras import, which would
  defeat the hook's install-before-keras requirement); hook installs only
  when the probe says stock.
- Backend (clone → synced to `_backend/`): the three protocol aliases +
  `standardize_dtype_hook` in `core.py`, exported in `__init__.py`.
  All additive; the in-tree dispatch and the hook path are unaffected.

## How to re-run

```sh
# plugin path (fork):
cd /home/dev/workspace && CC=~/.local/bin/zigcc KERAS_BACKEND=tinygrad \
  fork-venv/bin/python -c "import keras; print(keras.config.backend())"
# referee subset on the fork:
cd keras-plugin-fork && CC=~/.local/bin/zigcc KERAS_BACKEND=tinygrad \
  ../fork-venv/bin/python -m pytest keras/src/layers/core/dense_test.py -q
# stock path (hook):
cd /home/dev/workspace/keras-tinygrad && make verify tutorial smoke
```

## Open items before this becomes a PR

- A dummy in-repo test plugin + tests for `plugins.py` (keras CI must be
  able to exercise the mechanism without any real third-party backend).
- Decide entry-point group name with maintainers (`keras.backends` is the
  PoC's guess).
- The `DynamicBackend` silent-None → ValueError change is a behavior fix
  bundled with the mechanism; maintainers may want it split.
- Sequencing per docs/upstream-keras-draft.md: test-side PR first, then
  the design issue with this branch + the published package as exhibits.
