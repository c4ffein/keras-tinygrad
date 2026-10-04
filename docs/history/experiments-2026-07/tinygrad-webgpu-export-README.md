# tinygrad → WebGPU export experiments

> `out-inference/` and `out-trainstep/` are generated (gitignored since
> 2026-08-31, ~1.6 MB); regenerate with the two commands below (needs a
> tinygrad checkout on PYTHONPATH — this is the nnvp-era raw-tinygrad
> experiment, independent of the keras_tinygrad package).

Proof-of-concept from the 2026-07-13 investigation: can tinygrad power
in-browser **inference** and — the novel part — in-browser **training** for
NNVP, via static WGSL export? Both worked at the artifact level. Full context
and the resulting recommendation live in `docs/tasks.md`'s runtime notes and
the session summary below.

Tested against tinygrad @ `223c6d74c3ecc2fd096925b73cd089128d1f68ee`
(2026-07-12). tinygrad is dependency-light: a shallow clone + `PYTHONPATH`
suffices, no pip install needed.

## Reproduce

```bash
git clone --depth 1 https://github.com/tinygrad/tinygrad
# One-line patch: append "NULL" to EXPORT_SUPPORTED_DEVICE in
# tinygrad/extra/export_model.py (codegen needs no GPU via the NULL device).
PYTHONPATH=./tinygrad DEV="NULL:WGSL" NULL_ALLOW_COPYOUT=1 \
  python3 export_mlp_webgpu.py    # inference bundle -> out-inference/
PYTHONPATH=./tinygrad DEV="NULL:WGSL" NULL_ALLOW_COPYOUT=1 \
  python3 export_train_v2.py      # training-step bundle -> out-trainstep/
```

`check.ts` / `index.ts` (run with bun) parse every WGSL kernel with
`wgsl_reflect` (add it as a dev dependency where you run them) and import the
emitted JS modules — syntax-level validation only.

## Artifacts

- `out-inference/net.js` — MNIST-style MLP (flatten → 784→128 relu → 128→10
  softmax), 5 WGSL kernels + self-contained runner, ~13 KB, zero runtime deps.
- `out-trainstep/trainstep.js` — one full training STEP: forward →
  sparse-categorical-crossentropy → backward → SGD(momentum 0.9), batch 32,
  24 compute passes / 20 unique kernels. Weight + momentum buffers are updated
  IN PLACE, so looping the exported function from JS with fresh (x, y) batches
  IS an SGD training loop. No prior art found for this anywhere.
- `*.safetensors` — weight files. NB: exported on the NULL device, so the
  VALUES are zeros; kernel codegen is exact, but real initial weights need any
  real backend at export time (or JS-side init).

## Gotchas learned (the expensive ones)

- `Tensor.realize(*get_parameters(model))` BEFORE JIT capture, or the
  weight-init transform fuses into the matmul kernels.
- The step must end `Tensor.realize(loss, *opt.schedule_step())` — a naive
  `opt.step(); return loss` makes the scheduler RECOMPUTE the loss after the
  weight update (5 extra kernels, post-update loss).
- `Tensor.training = True` is gone; use `Context(TRAINING=1)`.

## Status / next steps

Unvalidated in a real browser (the box had no GPU): numerical correctness,
perf vs tfjs. Known one-liners still to do: `COPY_SRC` usage flag on
`createWeightBuf` for weight readback, `COPY_DST` on the lr buffer for
adjustable learning rate. Batch size is baked in (tinygrad's symbolic
Variables could lift that — tinychat uses them).

Recommendation on record: tfjs stays as the interactive in-browser training
engine (tinygrad cannot retrace in the browser — every architecture edit
would need a Python-side re-export); tinygrad export is a DEPLOYMENT feature
("download this model as a self-contained WebGPU bundle"), for which
`KerasGeneratorTinygradHelper` already emits exactly the Model shape
`export_model` consumes.
