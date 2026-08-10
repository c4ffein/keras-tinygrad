# keras-tinygrad — project instructions

- START by reading HANDOFF.md (state, how to run, decision queue), then
  docs/architecture.md (the 11 invariants — binding, update in the same diff
  that moves a boundary).
- The backend SOURCE OF TRUTH is the keras clone at ../keras
  (keras/src/backend/tinygrad/); this repo vendors it via
  scripts/sync_vendor.py. Never edit the vendored copy directly.
- Never commit or push anywhere unless the owner explicitly asks — he
  reviews and commits every diff himself.
- Keras' own test suite is the referee: baseline before, tally after, and a
  test that can't go green stays loudly failing — never skipped, never
  silently worked around.
- No silent numpy fallbacks in differentiable paths. Loud
  NotImplementedError over a wrong answer, always.
