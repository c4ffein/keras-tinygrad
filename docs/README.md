# docs/

Living reference — each file states how something works today and is kept
current in the diff that changes it:

- `architecture.md` — module boundaries, state ownership, the twelve invariants.
- `how-it-works.md` — the import hook: how stock keras gets a backend without a fork.
- `device-rng.md` — on-device RNG for dropout and what it means for reproducibility.
- `float64-promotion.md` — the 64→32 promotion policy and the fuzzer's tolerance rule.
- `complex-support.md` — complex-lite interop: what works, what raises.
- `unique-vectorize.md` — the open decision on data-dependent output shapes.
- `browser-training.md` — the WebGPU export and in-tab tracing story, limits, roadmap.
- `npm-publishing.md` — how the JS package publishes, and its status.

`history/` holds the dated record: reviews, triages, upstream drafts, the
full handoff log. Nothing there is maintained; read it for the why, not
the how.
