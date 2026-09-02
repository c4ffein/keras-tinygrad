"""Trainer for the tinygrad backend.

Data plumbing follows the numpy backend (plain `EpochIterator`, batches
arrive as numpy arrays and are converted per step); the gradient step
follows the torch backend, expressed through tinygrad 0.13's explicit
`loss.gradient(*tensors)` API — no zero_grad bookkeeping, the gradients
are pure outputs. Keras' backend-agnostic optimizers then apply them via
`optimizer.apply(grads, variables)`.

The train step runs under tinygrad's `TinyJit` when (and only when) that is
provably safe — see `_TrainStepJit` for the gating rules and the
buffer-pinning scheme that keeps Variable updates visible across replays.
Everything that cannot be proven safe falls back to the eager step, which
is always correct.
"""

import os
import warnings

import numpy as np

from tinygrad import Tensor
from tinygrad.engine.jit import JitError
from tinygrad.engine.jit import TinyJit

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.tinygrad.core import compute_gradients
from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import custom_gradient_tape
from keras.src.backend.tinygrad.core import device_rng_enabled
from keras.src.backend.tinygrad.core import device_rng_scope
from keras.src.backend.tinygrad.core import is_tensor
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs


def _device_rng_covers(layer):
    """True iff every random draw this layer makes in its step uses the
    device-RNG path (and the path is enabled), making it safe inside a
    TinyJit capture. Covered: the regularization family whose draws are
    exactly `random.dropout` / `random.normal` / `random.uniform` through
    their SeedGenerators — the three ops with on-device threefry
    implementations. EXACT types only: a subclass (SpatialDropout, or a
    user class) may add host-side randomness in call() that a capture
    would bake and silently replay frozen — a class earns a place on this
    list only with its own A/B receipt (jit-disabled comparison, see
    docs/device-rng.md). The preprocessing Random* family stays uncovered
    (mixed op usage incl. host-only samplers) and keeps forcing eager —
    still correct there: covered ops sample on-device, host-only ops
    sample numpy, both fresh every eager step."""
    if not device_rng_enabled():
        return False
    try:
        from keras.src.layers.regularization.alpha_dropout import AlphaDropout
        from keras.src.layers.regularization.dropout import Dropout
        from keras.src.layers.regularization.gaussian_dropout import (
            GaussianDropout,
        )
        from keras.src.layers.regularization.gaussian_noise import (
            GaussianNoise,
        )
    except ImportError:
        # keras moved these private modules (this runs mid-fit on whatever
        # keras version the user has): fail SAFE — eager, never a frozen
        # capture, never an ImportError out of the second batch.
        return False

    return type(layer) in (Dropout, GaussianNoise, GaussianDropout, AlphaDropout)


class _TrainStepJit:
    """Shape-guarded `TinyJit` wrapper around the eager train step.

    tinygrad's `TinyJit` captures the kernels realized during one call and
    replays them afterwards, parametrized on the input buffers. Replay is
    only correct when everything except the batch tensors is step-invariant,
    so this wrapper is deliberately conservative:

    * The first call always runs the eager step ("probe"); afterwards the
      gates in `_check_gates` decide whether jitting is allowed at all.
      Custom `train_step` overrides, a non-empty custom-gradient tape
      (quantized layers), seed generators (host-side RNG would be frozen
      into the capture), and optimizers with data-dependent host branches
      (`ops.cond` reads tensor values) all force permanent eager mode.
    * One `TinyJit` per batch signature (tree structure + per-leaf
      shape/dtype + static leaf values), so Keras' partial final batch gets
      its own capture instead of a shape mismatch. A signature with no
      tensor leaves, or with leaves we cannot convert, runs eagerly.
    * Keras Variables rebind `_value` to a fresh tensor on every assign
      (invariant 5), which would leave a replayed graph reading stale
      buffers. The jitted step therefore pins every variable to the buffer
      the capture read (the "binding"): after the inner step it copies the
      newly assigned value back into the pinned tensor — in-place
      `Tensor.assign`, the same pattern tinygrad's own optimizers use — and
      repoints `_value` at it. Eager assigns made between calls (metric
      resets, LR-schedule callbacks, `set_weights`) are synced into the
      pinned buffers before every jitted call. If the variable *set*
      changes (recompile, re-quantize), all captures are reset.
    * tinygrad raises a loud `JitError` on any host data access during
      capture ("value would be baked in"), on replay signature mismatches,
      and on non-tensor returns. Capture executes nothing, so on `JitError`
      the pre-step state is fully restorable: restore the binding, run the
      batch eagerly, and disable jitting permanently. Silent wrong results
      are structurally impossible on those paths; eager fallback is always
      the safe answer.
    """

    # Distinct batch signatures worth capturing; datasets with ever-varying
    # batch shapes (custom generators) run eagerly past this rather than
    # accumulating captures without bound.
    MAX_CAPTURES = 8

    def __init__(self, trainer):
        self._trainer = trainer
        self._probed = False
        self._disabled = False
        self._jits = {}
        # (variables, pinned tensors) after the first jitted call.
        self._bound = None

    def __call__(self, data):
        trainer = self._trainer
        if self._disabled:
            return trainer.train_step(data)
        if not self._probed:
            # Probe: run eagerly once so the optimizer/metrics build, then
            # decide whether jitting is allowed.
            logs = trainer.train_step(data)
            self._probed = True
            self._check_gates()
            return logs
        key, tensors, slots, leaves = self._batch_signature(data)
        if key is None:
            return trainer.train_step(data)
        jit = self._jits.get(key)
        if jit is None and len(self._jits) >= self.MAX_CAPTURES:
            return trainer.train_step(data)
        variables = self._sync_binding()
        if jit is None:
            jit = TinyJit(self._make_step_fn(data, leaves, slots))
            self._jits[key] = jit
        capture_phase = jit.cnt == 1
        try:
            logs = jit(*tensors)
        except JitError as e:
            # Capture executes nothing and replay validates before running,
            # so pre-step state is intact; restore the variable pointers,
            # run this batch eagerly, and stop jitting.
            self._restore_binding()
            self._disabled = True
            warnings.warn(
                "Disabling TinyJit for the train step; falling back to "
                f"eager execution (correct, slower). Reason: {e}"
            )
            return trainer.train_step(data)
        except Exception:
            if capture_phase:
                # A failed capture executed nothing but may have rebound
                # `Variable._value` to never-executed tensors.
                self._restore_binding()
            raise
        if self._bound is None:
            self._bound = (variables, [v._value for v in variables])
        return logs

    def _check_gates(self):
        trainer = self._trainer
        reason = None
        if os.environ.get("KERAS_TINYGRAD_TRAINER_JIT", "1") == "0":
            # Escape hatch for debugging and A/B benchmarking.
            reason = "KERAS_TINYGRAD_TRAINER_JIT=0"
        elif trainer.run_eagerly or not trainer.jit_compile:
            reason = "run_eagerly/jit_compile settings"
        elif type(trainer).train_step is not TinygradTrainer.train_step:
            reason = "custom train_step override"
        elif getattr(trainer, "_tape_was_nonempty", True):
            reason = "custom-gradient tape in use (quantized training)"
        elif any(
            getattr(layer, "_seed_generators", None)
            and not _device_rng_covers(layer)
            for layer in trainer._flatten_layers()
        ):
            # Layers whose randomness draws ON DEVICE (threefry counter
            # advances in-graph — docs/device-rng.md) replay fresh under
            # capture and don't gate. Every OTHER seed-generator layer
            # still samples host-side numpy, which a capture would bake
            # and silently replay frozen — those keep forcing eager.
            reason = "seed generators present (host-side RNG in the step)"
        elif not trainer.trainable_weights or trainer.optimizer is None:
            reason = "no trainable weights"
        else:
            from keras.src.optimizers.loss_scale_optimizer import (
                LossScaleOptimizer,
            )

            optimizer = trainer.optimizer
            if isinstance(optimizer, LossScaleOptimizer):
                reason = "LossScaleOptimizer (data-dependent host branch)"
            elif getattr(optimizer, "gradient_accumulation_steps", None):
                reason = "gradient accumulation (data-dependent host branch)"
            elif getattr(optimizer, "use_ema", False):
                reason = "EMA overwrite (data-dependent host branch)"
        if reason is not None:
            self._disabled = True

    def _batch_signature(self, data):
        """Hashable signature for `data`, plus its tensor leaves.

        Returns `(key, tensors, slots, leaves)`; `key is None` means the
        batch cannot be safely jitted (no tensor leaves, or a leaf kind we
        do not understand) and must run eagerly.
        """
        leaves = tree.flatten(data)
        key = [repr(tree.map_structure(lambda x: 0, data))]
        tensors, slots = [], []
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, np.ndarray) or isinstance(leaf, np.generic):
                leaf = convert_to_tensor(np.asarray(leaf))
                leaves[i] = leaf
            if isinstance(leaf, Tensor):
                key.append(("tensor", tuple(leaf.shape), leaf.dtype))
                tensors.append(leaf)
                slots.append(i)
            elif leaf is None or isinstance(
                leaf, (bool, int, float, str, bytes)
            ):
                # Static python leaf: baked into the capture, so its VALUE
                # is part of the signature.
                key.append(("static", leaf))
            else:
                return None, None, None, None
        if not tensors:
            return None, None, None, None
        return tuple(key), tensors, slots, leaves

    def _make_step_fn(self, data, leaves, slots):
        structure = tree.map_structure(lambda x: None, data)
        template = list(leaves)
        for slot in slots:
            template[slot] = None
        trainer = self._trainer

        def step_fn(*tensors):
            batch_leaves = list(template)
            for slot, t in zip(slots, tensors):
                batch_leaves[slot] = t
            batch = tree.pack_sequence_as(structure, batch_leaves)
            variables = self._collect_variables()
            pinned = [v._value for v in variables]
            logs = trainer.train_step(batch)
            if getattr(trainer, "_tape_was_nonempty", True):
                from tinygrad.engine.realize import capturing

                if capturing:
                    # Quantized layers appeared after the probe (e.g.
                    # re-quantize between fits): a replayed tape recording
                    # would be frozen — refuse the capture. Nothing has
                    # executed, so the JitError path restores state and
                    # reruns this batch eagerly.
                    raise JitError(
                        "custom-gradient tape recorded during capture"
                    )
            # Pin: copy every reassigned variable value back into the
            # tensor the capture read, so replays see fresh state.
            changed = []
            for v, pin in zip(variables, pinned):
                if v._value is pin:
                    continue
                pin.assign(v._value.detach())
                v._value = pin
                changed.append(pin)
            if changed:
                Tensor.realize(*changed)
            return logs

        return step_fn

    def _collect_variables(self):
        """Every Variable the step may assign, deduplicated, stable order.

        Trainable and non-trainable model state (BatchNorm moving stats
        live in the latter), optimizer slots (iterations, learning rate,
        momenta), and metric accumulators.
        """
        trainer = self._trainer
        groups = [
            trainer.trainable_variables,
            trainer.non_trainable_variables,
            trainer.metrics_variables,
        ]
        if trainer.optimizer is not None:
            groups.append(trainer.optimizer.variables)
        out, seen = [], set()
        for group in groups:
            for v in group:
                if id(v) not in seen:
                    seen.add(id(v))
                    out.append(v)
        return out

    def _sync_binding(self):
        """Re-pin variables mutated eagerly since the last jitted call."""
        variables = self._collect_variables()
        if self._bound is None:
            return variables
        bound_vars, pinned = self._bound
        if len(bound_vars) != len(variables) or any(
            a is not b for a, b in zip(bound_vars, variables)
        ):
            # The variable set changed (recompile, re-quantize, new
            # metric): existing captures read retired buffers, recapture.
            for jit in self._jits.values():
                jit.reset()
            self._bound = None
            return variables
        changed = []
        for v, pin in zip(variables, pinned):
            if v._value is pin:
                continue
            pin.assign(v._value.detach())
            v._value = pin
            changed.append(pin)
        if changed:
            Tensor.realize(*changed)
        return variables

    def _restore_binding(self):
        if self._bound is None:
            return
        for v, pin in zip(*self._bound):
            if not pin.uop.is_realized:
                raise RuntimeError(
                    "TinyJit fallback found an unrealized pinned buffer; "
                    "variable state may be corrupt. This is a bug in the "
                    "tinygrad backend's train-step jit."
                )
            v._value = pin


class TinygradTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Record `custom_gradient` blocks (quantized layers) during the
        # forward pass so `compute_gradients` can honor their backward
        # functions; with no such blocks this is exactly `loss.gradient`.
        with custom_gradient_tape() as tape_blocks, device_rng_scope():
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)

            loss = self._compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
        # Read by `_TrainStepJit`: a non-empty tape means quantized layers
        # recorded custom-gradient blocks this step, which a jit replay
        # would freeze — that forces the eager path.
        self._tape_was_nonempty = bool(tape_blocks)
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

        if self.trainable_weights:
            trainable_weights = self.trainable_weights[:]
            gradients = compute_gradients(
                loss,
                [variable.value for variable in trainable_weights],
                tape_blocks,
            )
            self.optimizer.apply(gradients, trainable_weights)
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss, sample_weight=tree.flatten(x)[0].shape[0]
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        # A new train function is a new training run: the device RNG stream
        # re-seeds from the Keras seed state at its first draw, which lands
        # in the eager probe step below (never inside a capture — see
        # random.reset_device_stream for why that matters).
        from keras.src.backend.tinygrad import random as backend_random

        backend_random.reset_device_stream()

        # Rebuilt whenever the train function is (fresh compile / `force`),
        # so gates and captures never outlive the optimizer/metrics they
        # were probed against.
        jit_step = _TrainStepJit(self)

        def one_train_step(data):
            data = data[0]
            return jit_step(data)

        def multi_train_steps(data):
            for single_step_data in data:
                logs = one_train_step([single_step_data])
            return logs

        if self.steps_per_execution > 1:
            self.train_function = multi_train_steps
        else:
            self.train_function = one_train_step

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def one_test_step(data):
            data = data[0]
            return self.test_step(data)

        def multi_test_steps(data):
            for single_step_data in data:
                logs = one_test_step([single_step_data])
            return logs

        if self.steps_per_execution > 1:
            self.test_function = multi_test_steps
        else:
            self.test_function = one_test_step

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_predict_step(data):
            data = data[0]
            return self.predict_step(data)

        def multi_predict_steps(data):
            if len(data) == 1:
                return one_predict_step(data[:1])
            all_outputs = [one_predict_step([d]) for d in data]
            return tree.map_structure(
                lambda *args: np.concatenate(args, axis=0), *all_outputs
            )

        if self.steps_per_execution > 1:
            self.predict_function = multi_predict_steps
        else:
            self.predict_function = one_predict_step

    def _symbolic_build(self, data_batch):
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        compile_loss_unbuilt = (
            self._compile_loss is not None and not self._compile_loss.built
        )
        if model_unbuilt or compile_metrics_unbuilt or compile_loss_unbuilt:

            def to_symbolic(v):
                if isinstance(v, np.ndarray):
                    return KerasTensor(v.shape, standardize_dtype(v.dtype))
                if is_tensor(v):
                    from keras.src.backend.tinygrad.core import to_keras_dtype

                    return KerasTensor(tuple(v.shape), to_keras_dtype(v.dtype))
                return v

            data_batch = tree.map_structure(to_symbolic, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            try:
                y_pred = backend.compute_output_spec(self, x, training=False)
            except Exception as e:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it.\n"
                    f"Underlying error: {e}"
                )
            if compile_metrics_unbuilt:
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
            if compile_loss_unbuilt:
                backend.compute_output_spec(
                    self._compute_loss,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        self._post_build()

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self._assert_compile_called("fit")
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        epoch_iterator = EpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        if not all(layer.built for layer in self._flatten_layers()):
            for _, _, data in epoch_iterator:
                data_batch = data[0]
                self._symbolic_build(data_batch)
                break
            epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                callbacks.on_train_batch_begin(begin_step)
                logs = self.train_function(data)
                # Logs may carry raw backend tensors (e.g. compiled-metric
                # results); callbacks and History expect python scalars.
                callbacks.on_train_batch_end(end_step, pythonify_logs(logs))
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = EpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            epoch_logs = pythonify_logs(epoch_logs)
            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        epoch_iterator = EpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_predict_batch_begin(begin_step)
            batch_outputs = self.predict_function(data)
            # Realize per batch: holding lazy tensors across loop iterations
            # defers buffer reads past their validity.
            batch_outputs = tree.map_structure(
                backend.convert_to_numpy, batch_outputs
            )
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            epoch_iterator = EpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        if not all(layer.built for layer in self._flatten_layers()):
            for _, _, data in epoch_iterator:
                data_batch = data[0]
                self._symbolic_build(data_batch)
                break

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_test_batch_begin(begin_step)
            logs = self.test_function(data)
            callbacks.on_test_batch_end(end_step, logs)
            if self.stop_evaluating:
                break
        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        data = (x, y, sample_weight)
        self._symbolic_build(data)
        self.make_train_function()

        logs = self.train_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)
        self._symbolic_build(data)
        self.make_test_function()

        logs = self.test_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs


# Plugin-protocol name (see keras.src.backend.plugins): the generic
# dispatch imports `trainer.Trainer`.
Trainer = TinygradTrainer
