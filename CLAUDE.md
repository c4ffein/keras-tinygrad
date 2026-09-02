# keras-tinygrad — project instructions

- START by reading HANDOFF.md (state, how to run, decision queue), then
  docs/architecture.md (the 11 invariants — binding, update in the same diff
  that moves a boundary).
- The backend SOURCE OF TRUTH is src/keras_tinygrad/_backend/ in THIS
  repo. Nothing here depends on a sibling keras checkout: the referee
  (`make referee`, scripts/referee.sh) clones the pinned keras tag into
  .referee/ itself and runs Keras' tests with the import hook active.
  ../keras, if present, is a leftover of the in-tree experiment — never
  read from or write to it.
- Never commit or push anywhere unless the owner explicitly asks — he
  reviews and commits every diff himself.
- Keras' own test suite is the referee: baseline before, tally after, and a
  test that can't go green stays loudly failing — never skipped, never
  silently worked around.
- No silent numpy fallbacks in differentiable paths. Loud
  NotImplementedError over a wrong answer, always.
