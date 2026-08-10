# Core Engine — Variable, Gradients, Trainer, JIT

**Verdict.** The engine is well-architected and unusually honest: the documented invariants (copy-on-convert, realize-on-assign, loud-over-wrong, tape-scoped custom gradients, conservative JIT gating) are genuinely enforced in code, not just described, and the headline claim that JIT and eager training produce bit-identical weights was independently reproduced (3 epochs, partial final batch, BatchNorm state, Adam slots — `maxdiff: 0.0`). The custom-gradient VJP composition in `compute_gradients` is mathematically sound and passed 7 of 8 adversarial composition tests written for this review. It failed one: **tinygrad's UOp hash-consing aliases the detached proxies of two identical `custom_gradient` calls, silently double-counting gradients** — the single silent-wrong bug found in the engine, and it directly violates the project's own invariant #2 ("loud over wrong"). It is narrow (requires the same block applied twice to the *same* tensor) but real and fixable in a few lines. Everything else in the risk list is either a scaling cliff (scatter), a protocol nit, or an accepted-and-documented quirk.

All file references are to `/home/dev/workspace/keras/keras/src/backend/tinygrad/` unless prefixed; tinygrad references are to the installed 0.13 tree at `/home/dev/workspace/ktg-venv/lib/python*/site-packages/tinygrad/`.

## 1. What was verified, and how

- **tinygrad `Tensor.gradient` semantics** underpin every assumption in `core.py`'s gradient code. Verified against source: disconnected targets return zeros, not raise (`tensor.py:848`, `y = x.const_like(0)`); the seed (`gradient=`) kwarg exists and a scalar root defaults to 1.0 (`tensor.py:841-843`); a target that *is* the root returns the seed itself (`gradient.py:97`, `grads = {root: root_grad}`). All three are load-bearing for `compute_gradients` (core.py:903-939) and all three hold.
- **VJP composition**: 8 adversarial cases (single block, chained blocks, block-output-feeds-block, loss using both block output and its input directly, unused block, no blocks, `None` grad for int arg per the dense int8 pattern, same block twice on the same tensor). 7/8 pass exactly. See §2.1 for the failure.
- **JIT/eager parity**: independent reproduction — MLP with BatchNorm, Adam, batch 20 over 64 samples (forcing a second capture for the partial batch of 4), `KERAS_TINYGRAD_TRAINER_JIT=0` vs `1`: identical per-epoch losses, bit-identical final weights including non-trainable moving statistics and optimizer state.
- **Convention parity**: the `grad_fn(*args, upstream=...)` calling convention and the "only positional args are recorded" behavior exactly match the torch backend (`torch/core.py:819` `ctx.save_for_backward(*args)`, `:839` `grad_fn(*args, upstream=grad_output)`), and match the real internal consumer (`layers/core/dense.py:782-813`, int8/int4 paths returning `(inputs_grad, None, None)`).
- **Touchpoint reality**: the `standardize_dtype` shim the dtype table depends on exists in keras-core (`keras/src/backend/common/variables.py:577-582`) and routes tinygrad `DType` through `to_keras_dtype`. `jit_compile="auto"` resolves to `True` for this backend (`trainers/trainer.py:256-275` falls through to `model_supports_jit`), so the trainer's `not trainer.jit_compile` gate (trainer.py:144) does not silently disable the JIT by default.
- **rnn.py** is a line-faithful port of `numpy/rnn.py` including the scan's flip/unflip/re-flip choreography (`rnn.py:258-267` + `:205-207` mirrors `numpy/rnn.py:236-245` + `:193`) — semantics identical to the reference by construction.

## 2. Correctness risks

### 2.1 HIGH — hash-consed proxies double-count duplicate custom-gradient blocks (silent)

`custom_gradient.__call__` records `(args, proxy, grad_fn)` where `proxy = outputs.detach()` (core.py:891-892). tinygrad hash-conses structurally identical UOps globally — verified directly:

```python
a = (w * 2.0).detach()
b = (w * 2.0).detach()
a.uop is b.uop  # True
```

So when the same block runs twice on the same input tensor, the tape holds **two entries sharing one proxy UOp**. `compute_gradients` (core.py:915-919) then computes the upstream at that UOp — which is the *total* accumulated gradient across both uses — and hands the full amount to *each* block's `grad_fn`. Empirical result: `y1 = f(w); y2 = f(w); loss = (y1+y2).sum()` with a custom VJP of `3*upstream` returns **12.0 where the correct answer is 6.0**. No warning, no error.

Reachability: a shared quantized layer (int8/int4 `Dense`/`EinsumDense`, the exact layers the tape exists for) applied twice to literally the same tensor in one forward — weight-sharing and siamese-on-identical-input topologies. The corrupted quantity is `inputs_grad`, i.e. the gradients of every layer *upstream* of the quantized block. Narrow, but it is the one place the engine can be silently wrong, in the subsystem whose comments promise the opposite.

Fix options, in order of preference:
1. **Dedupe in `compute_gradients`**: group recorded blocks by `proxy.uop` identity and process each unique proxy once. This is mathematically exact — VJPs are linear in the upstream, so one application with the summed upstream equals the sum of per-use applications (and duplicate entries are guaranteed-identical `(args, grad_fn)` by construction, since identical UOp graphs implied identical inputs).
2. Make each proxy unique at record time (e.g. a `.clone()`/unique-buffer copy in `custom_gradient.__call__`) — costs a copy per block per step, but keeps `compute_gradients` untouched.

### 2.2 MEDIUM — `scatter` materializes an O(slots × updates) one-hot matrix

`scatter` (core.py:691-710) builds `onehot` of shape `(m, num_updates)` where `m = prod(shape[:index_length])`. Elegant and differentiable w.r.t. `values`, but scattering 100k updates into a 1M-row table materializes a 10¹¹-element intermediate. Correctness is fine; this is a memory/latency cliff on embedding-scale workloads (`Embedding` gradient paths route through backend scatter in several ops). `scatter_update`'s O(updates) chained-`where` graph depth (core.py:733-746) is a documented quirk (architecture.md "Known leaks") — the `scatter` blow-up is *not* in that list and should be, or bounded (e.g. chunked one-hot matmul).

### 2.3 MEDIUM (performance, not correctness) — quantized training pays twice

With B recorded blocks, `compute_gradients` performs O(B²) `_vjp` calls (the per-block upstream loop iterates the growing `sources` list, core.py:915-918) plus B+1 full-graph backward constructions in the totals loop (core.py:934-938). Each is lazy graph construction, and UOp hash-consing dedupes repeated structure, so this is graph-build overhead rather than compute — but a fully-quantized deep model makes B ≈ layer count. Simultaneously, a non-empty tape permanently disables the train-step JIT (trainer.py:148-149, and the in-capture guard at trainer.py:217-228). Quantized training is therefore the slowest configuration on both axes at once. Acceptable for now; worth a line in the known-leaks list.

### 2.4 LOW — `Variable.__array__` drops the numpy protocol parameters

core.py:291-292 defines `def __array__(self)`, shadowing the base `KerasVariable.__array__(self, dtype=None)` (`common/variables.py:444-449`). numpy 2.x may invoke `__array__(dtype, copy=...)`; the Tensor-level monkeypatch handles this correctly (`numpy.py:56-58`, `def _tensor_array(self, dtype=None, copy=None)`) — the Variable override is the odd one out and will raise `TypeError` on `np.asarray(variable, dtype=...)`. Two-token fix; loud when it fires, so low severity.

### 2.5 LOW — assorted, verified-and-accepted

- **`_restore_binding` depends on `pin.uop.is_realized`** (trainer.py:298) — a tinygrad-internal attribute. Deliberate and defensive (it raises `RuntimeError` on the impossible state rather than proceeding), but it is a version-coupling point that will break loudly on a tinygrad upgrade.
- **`scan` dummy-y structure mismatch**: when `f` returns `y=None` and `init` is nested, `ys` mixes packed structures with the flat `dummy_y` list (core.py:533, 542) and `tree.map_structure` will choke. Inherited *verbatim* from the reference (`numpy/core.py:199, 208`) — a faithful port of a latent upstream wart, consistent with invariant #1, and loud when hit.
- **Gates evaluated once**: `_check_gates` runs only after the probe step (trainer.py:100-102). The dangerous late-arrival case — quantized layers appearing after the probe — *is* re-checked inside every capture (trainer.py:217-228, raising `JitError` so the fallback path engages). A layer acquiring a `SeedGenerator` post-probe without a recompile has no equivalent re-check; no realistic path to it was constructed (recompile resets `train_function`, which rebuilds `_TrainStepJit`, trainer.py:380-387), so this is a documented-conservatism gap, not a live bug.
- **`_loss_tracker.update_state(..., sample_weight=next(i for i in tree.flatten(x) if i is not None).shape[0])`** (trainer.py:337-341) raises bare `StopIteration` on an all-`None` x; `test_step` uses the different, less defensive `tree.flatten(x)[0].shape[0]` (trainer.py:368). Cosmetic inconsistency.
- **`requires_grad` mirroring** (core.py:280, 285) is inert: the `Tensor.gradient` API never consults it (`tensor.py:827-850`). Harmless — it exists for tinygrad's `backward()` path, which this backend doesn't use — but the comment-free mirror slightly overstates its role.
- **`custom_gradient` records only positional args**; a tensor passed by keyword gets no gradient and isn't passed to `grad_fn`. This is exact torch-backend parity (`torch/core.py:819`), so it is a *convention*, not a deviation — but nothing says so at core.py:892.

### 2.6 Non-risks specifically checked

- **Copy-on-convert holds on every path**: same-dtype arrays get `.copy()` (core.py:350), converts get `astype` (which copies, core.py:344, 355), and the complex path routes components back through `convert_to_tensor` precisely because `.real`/`.imag` are views (core.py:259-264). Tensors passed through unchanged are values already — correct to not copy.
- **Mixed-precision gradient targets**: `variable.value` in `train_step` (trainer.py:349) is read *outside* any autocast scope, so `_maybe_autocast` (`common/variables.py:245-249`) returns the raw `self._value` — the same object the forward pass's autocast casts descend from. Targets are in the loss graph; no detachment hazard.
- **Host syncs under JIT**: `cond` (core.py:409), `while_loop` (core.py:811), `_index_int` (core.py:668) all do `.item()`/`.numpy()` — every one of these raises `JitError` during capture, which the wrapper converts into a warned, permanent eager fallback (trainer.py:117-127). "Silent wrong results are structurally impossible on those paths" (trainer.py:76-77) checks out.
- **Replay-time weight staleness**: the pinning scheme (capture-time in-place `Tensor.assign` into the buffers the capture read, trainer.py:230-239; identity-checked re-sync of eager mutations before every jitted call, trainer.py:267-291; full capture reset on variable-set change, trainer.py:274-281) is the same pattern tinygrad's own optimizers use, and the bit-parity experiment exercises exactly the hard cases (BatchNorm non-trainable stats mutated during forward, Adam slots, metric resets between epochs, two capture signatures).

## 3. Design quality — are the invariants enforced or just documented?

Enforced, with one exception (§2.1). Specifics:

- **Invariant 4 (copy-on-convert)** is enforced at the single choke point `convert_to_tensor` with a comment explaining *why* at the exact line where omitting it would be tempting (core.py:346-350). The native-stacking branch for Tensor-bearing sequences (core.py:323-333) exists precisely to avoid the numpy round-trip that would detach gradients — the invariant is structural, not aspirational.
- **Invariant 5 (realize on assign)** is enforced in the only two mutation paths a Variable has (`_initialize`/`_direct_assign`, core.py:276-285), and the other realize points (per-batch predict conversion trainer.py:673-677, JIT pin batching trainer.py:238-239, 289-290) are each commented with the buffer-lifetime rationale.
- **Invariant 10 (tape-scoped custom gradients)** is structurally guaranteed: the tape is thread-local, the trainer is the only opener (trainer.py:319), and outside a tape the block is a passthrough (core.py:882-884) — matching the numpy backend's semantics for predict/evaluate while exceeding it for training.
- **The JIT safety story is a genuinely good design**: rather than trying to prove capture safety, it enumerates static disqualifiers (trainer.py:138-170), lets tinygrad's own `JitError` catch dynamic hazards, and exploits the fact that a failed capture executes nothing to make the fallback state-restorable (trainer.py:117-133). "Correct-but-slower" is always the failure mode. The `MAX_CAPTURES = 8` cap with eager overflow (trainer.py:108-109) closes the unbounded-capture hole for varying-shape generators.
- **ComplexTensor** is the invariant system in miniature: a closed op set sized exactly to its one consumer (keras-core's `view_as_complex`), a single canonical refusal message, dunders explicitly nailed shut so no protocol probe can leak into float math (core.py:185-213), and a documented tier boundary for extension.

The main architectural asymmetry: `_TrainStepJit` reaches into `Variable._value` directly (trainer.py:215, 232-237, 287-288, 302) — the trainer is now a second writer of a field `core.Variable` nominally owns. It is careful and correct, but the pinning contract ("`_value` identity change ⇒ eager assign happened") is enforced only by convention across two files. A short note at `Variable._direct_assign` pointing at the trainer's dependence on the rebind-on-assign behavior would make the coupling survivable through refactors.

## 4. Code quality

Strong overall: comments explain *why* at the exact point of temptation, names are honest (`_tape_was_nonempty`, `_restore_binding`), and the eager step is small enough to audit in one sitting (trainer.py:313-356). Hotspots:

- `compute_gradients` (core.py:903-939) is dense for what it does; the `sources` list serving double duty (VJP roots *and* pending upstream contributions) rewards careful reading. The comment block above it (core.py:834-854) is excellent and mostly compensates.
- `_TrainStepJit.__call__` (trainer.py:93-136) juggles five states (disabled/probe/eager-overflow/capture/replay) in 40 lines; the `capture_phase` / `_bound` interplay takes source-diving into tinygrad's `cnt` state machine (`jit.py:241, 257-300`) to validate. A one-line comment stating tinygrad's `cnt` semantics (0=run, 1=capture, ≥2=replay) would save the next reader that trip.
- `scan`'s comprehension shadows `init` (core.py:533) and the module shadows builtins `map`/`slice` (core.py:492, 749) — forced by the backend surface's naming contract, and consistently disambiguated via `builtins.`, but it puts a tax on every edit.
- Duplicated pin-sync logic between `step_fn` (trainer.py:230-239) and `_sync_binding` (trainer.py:282-290) could share a helper.

## 5. Notable engineering

- **The `JitError`-as-safety-net design** (§3) — turning tinygrad 0.13's loud-on-host-access capture behavior into a structural guarantee that JIT bugs degrade to slowness, never wrongness. The independent bit-parity reproduction backs it.
- **Hand-rolled VJP composition over `Tensor.gradient`** (core.py:834-939) — building torch-convention custom gradients on a backend with no custom-VJP hook, using only seeds + zeros-for-disconnected, with the reverse-creation-order argument stated and correct. Passed every composition test except the hash-consing alias.
- **The one-hot-matmul `scatter`** (core.py:691-710) — duplicate-accumulation semantics *and* differentiability w.r.t. values on a backend without item assignment, in ten lines.
- **`convert_to_tensor`'s dtype discipline** — the `_NUMPY_NATIVE_DTYPES` split with the bfloat16/float8 build-as-float32-cast-on-device dance (core.py:44-50, 351-358), each non-obvious step annotated with the tinygrad behavior that forces it (zero-copy wrap, ignored `dtype=` kwarg, `(1,)` scalar wrap).
- **Empirical claims that survive audit**: every documented behavior checked against code or experiment held (dtype touchpoint, jit-auto resolution, gate coverage, parity, rnn port fidelity). That is rare.

## 6. Recommendations, ranked

1. **Fix the duplicate-block double-count** (§2.1): dedupe tape entries on `proxy.uop` identity in `compute_gradients` (exact, ~4 lines), and add the `y1=f(w); y2=f(w)` case to the parity/fuzz suite. This is the only item that can silently corrupt training.
2. **Add `scatter`'s O(slots×updates) memory to the known-leaks list** (or chunk the one-hot matmul); note the quantized-training double penalty (O(B²) composition + JIT gate) alongside it.
3. **Restore the `dtype`/`copy` parameters on `Variable.__array__`** (core.py:291) to match the base class and the Tensor patch.
4. **Document the two cross-file contracts**: at `Variable._direct_assign`, the trainer's dependence on rebind-on-assign; at `custom_gradient.__call__`, the positional-args-only convention (torch parity) and the single-tensor-output restriction.
5. Minor: comment tinygrad's `cnt` state machine in `_TrainStepJit.__call__`; unify the train/test loss-tracker batch-size expressions; share the pin-sync helper.

---

**Method note:** all three engine files were read in full plus architecture.md/HANDOFF.md; assumptions were verified against tinygrad 0.13 source (`Tensor.gradient`, `compute_gradient`, `TinyJit`/`JitError`) and against the numpy/torch reference backends; 8 adversarial `compute_gradients` tests were run (finding one real silent-wrong bug: hash-consed proxy aliasing double-counts duplicate custom-gradient blocks, 12.0 vs correct 6.0), and bit-identical JIT/eager training was independently reproduced. No files in the repo or clone were modified; test scripts lived in the session scratchpad only.
