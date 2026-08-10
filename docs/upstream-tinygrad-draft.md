# Upstream draft — tinygrad additions (NOT submitted)

Status: local draft for the owner's review. Nothing submitted, no issue
opened, nothing pushed.

This extends the existing scalar-interop draft PR at
`/home/dev/workspace/tinygrad-upstream/DRAFT_PR.md` (the
`__bool__`/`__int__`/`__float__`/`__index__`/`__array__` dunders patch,
with test evidence, line budget, and the PR-A/PR-B split assessment). That
document stands as written; this one adds the two candidates discovered
2026-08-03 during the preprocessing/ops-image referee runs (HANDOFF item 5
and decision queue), plus the process constraints that govern any
submission. All tinygrad citations are against the installed **tinygrad
0.13.0** in `/home/dev/workspace/pkg-test-venv`
(`lib/python3.14/site-packages/tinygrad/`); keras-side citations are
against the backend source of truth in the clone,
`/home/dev/workspace/keras/keras/src/backend/tinygrad/`.

---

## A. `Tensor.__getitem__` rejects 0-d Tensor slice bounds

**Problem.** Slicing with 0-d integer Tensors as bounds raises, where
torch and numpy accept them:

```python
t = Tensor.ones(10)
t[Tensor(2) : Tensor(5)]
# TypeError: slice index=slice(<Tensor <UOp CPU () int> ...>, ...) is not supported
```

**Exact code path** (tinygrad 0.13.0):

1. `Tensor.__getitem__` — `tinygrad/tensor.py:968` → `self._getitem(indices)`
   (tensor.py:1006).
2. `Tensor._getitem` — `tensor.py:878`. The fast-path dispatch at
   tensor.py:880-881 checks `isinstance(i, (Tensor, list, tuple))` on each
   *top-level* index — a `slice` object carrying Tensor bounds is not
   itself a Tensor, so the whole index is routed to view-only indexing:
   `super().__getitem__` = `MovementMixin.__getitem__`
   (`tinygrad/mixin/movement.py:116`).
3. `MovementMixin._parse_view_index` — `movement.py:77`. The slice case at
   movement.py:87-89 requires every bound to be `None` or `sint`
   (`sint = int | UOp`, `tinygrad/uop/ops.py:1642`):

   ```python
   case slice():
     if not all(s is None or isinstance(s, sint) for s in (index.start, index.stop, index.step)):
       raise TypeError(f"slice {index=} is not supported")
   ```

   A 0-d Tensor is neither, so it raises. (Symbolic `UOp` bounds are
   already accepted — the machinery below the check handles non-int
   bounds; what's rejected is specifically the *device-scalar* spelling.)

**Minimal proposed change.** Normalize 0-d integer-Tensor slice bounds in
`Tensor._getitem` before dispatch (tensor.py, ~4 lines): map each `slice`
index through

```python
slice(
    *(
        int(b.item()) if isinstance(b, Tensor) and b.ndim == 0 and dtypes.is_int(b.dtype) else b
        for b in (index.start, index.stop, index.step)
    )
)
```

with non-int-dtype 0-d Tensors still raising. This is torch's semantics
exactly: torch resolves tensor slice bounds via `__index__`, which syncs.
It composes with the draft PR's `__index__` half — once that lands, the
normalization is literally `operator.index(b)`, and the dtype gate comes
for free (`__index__` refuses non-integer dtypes, see DRAFT_PR.md design
notes). Placing it in `tensor.py` rather than `movement.py` keeps the
mixin free of a circular `Tensor` import. Honest caveat for the PR
conversation: like `.item()`, this forces realization of the bound —
laziness-hostile in the same way the `__bool__` discussion is, so expect
the same scrutiny; unlike `__bool__` there is no prior ban, and slice
bounds have no lazy alternative short of symbolic shapes.

**What it blocks in keras.** `keras.layers.RandomCrop` slices images with
backend tensors produced by its RNG:
`keras/src/layers/preprocessing/image_preprocessing/random_crop.py:139-164`
(`images[:, crop_box_hstart : crop_box_hstart + crop_height, ...]`, where
the starts are 0-d int32 tensors from random.uniform, :103-113).

**Workaround today: none.** This is one of two places where we
deliberately have no backend-side workaround (no silent host fallback in
a differentiable path — the crop feeds training). The referee shows it
loudly:
`random_crop_test.py::RandomCropTest::test_random_crop` and
`::test_dict_input` — the "RandomCrop x2" in HANDOFF item 5's remaining
red. **Cost:** RandomCrop is unusable under training on tinygrad; 2
permanently red preprocessing tests until upstream accepts the change (or
we add a `getitem`-free crop via `dynamic_slice`-style gather, which
would cost a gather kernel where every other backend takes a view).

The adjacent-but-distinct case we *did* work around: 0-d Tensor **pad
widths** (shape metadata, not a differentiable path) are read out as host
ints in the backend's `pad` —
`keras/src/backend/tinygrad/numpy.py:1365-1372` (`_pad_amount`,
`int(v.item())`), matching the numpy backend where `np.pad` materializes
widths. That one stays ours regardless; it is semantics, not a tinygrad
limitation.

## B. `argfix` mis-parses list/tuple *subclasses*

**Problem.** `tinygrad/helpers.py:24-28`:

```python
def argfix(*x):
    if x and x[0].__class__ in (tuple, list):
        if len(x) != 1:
            raise ValueError(f"bad arg {x}")
        return tuple(x[0])
    return x
```

The exact-class check `x[0].__class__ in (tuple, list)` (helpers.py:25)
means a `list` subclass is treated as a *single scalar argument* instead
of a shape sequence:

```python
class TL(list):
    pass


Tensor.ones(6).reshape(TL([2, 3]))
# ValueError: size mismatch, can't reshape ((6,)) -> (([2, 3],))
```

`argfix` feeds every shape-taking movement op — `reshape`
(movement.py:167), `expand` (:153), `permute` (:220), `flip` (:241),
`shrink_to` (:251), `pad_to` (:254), `repeat` (:560) — and the creation
functions (`Tensor.empty` tensor.py:495, `Tensor.rand` :581, the random
and init family :718, :736-783). numpy and torch accept any sequence
here.

**Minimal proposed change.** One line, helpers.py:25:

```diff
-  if x and x[0].__class__ in (tuple, list):
+  if x and isinstance(x[0], (tuple, list)):
```

Behavior change beyond bugfix: subclass instances (including namedtuples)
are now unpacked as sequences — which is what numpy/torch do, and for a
shape argument is always the intended reading. Zero net lines; the kind
of diff their line-budget bot likes.

**What it hits in keras.** Keras passes `TrackedList` (a `list` subclass
that tracks variables/state) as shapes — e.g. `Normalization`'s
`_broadcast_shape` — so backend calls that forward keras-held shape
objects straight into tinygrad mis-parse.

**Workaround today.** Backend-side `tuple(...)` normalization at the two
call sites the referee caught (found as 2 of the 3 real preprocessing-run
bug fixes, HANDOFF item 5):

- `keras/src/backend/tinygrad/numpy.py:1046-1053` — `reshape`, plain
  `tuple(newshape)` with the comment naming argfix's exact-class check;
- `keras/src/backend/tinygrad/numpy.py:1117-1125` — `broadcast_to`, same.

(Vendored mirror: `src/keras_tinygrad/_backend/numpy.py`, byte-identical
by `sync_vendor.py --check`.)

**Cost:** a per-call-site vigilance tax — every current and future
backend op that forwards a caller-provided shape into tinygrad must
remember `tuple(...)`, and the failure when someone forgets is a
confusing size-mismatch `ValueError` (or a mis-shaped `expand`) surfacing
far from the cause. Both existing cases were only found by running the
full keras preprocessing referee; the workaround protects exactly the
call sites we've already been burned at, nothing else — keras-core code
that ever calls Tensor methods directly would still mis-parse.

## C. Process constraints (from HANDOFF and tinygrad's own rules)

Carried over verbatim as constraints on any actual submission:

- **AI-assistance disclosure is mandatory.** tinygrad's README states
  AI-written-looking code from new contributors gets closed and possibly
  banned, and that contributors must disclose what AI was used for. Every
  artifact in this bundle (the DRAFT_PR diff and both items above) is
  AI-assisted; none of it may be submitted without a human pass and an
  explicit disclosure line (DRAFT_PR.md says the same about itself).
- **The `__bool__` half needs an issue first.** It reverts a deliberate
  maintainer ban (PR #3632 "ban `__bool__` on Tensor", still pinned by
  `test_no_bool`), so per DRAFT_PR.md's assessment it should be raised as
  an issue/discussion before any PR — cold-PRing it invites a "we banned
  this on purpose" close (est. 15-25% as a cold PR).
- **Suggested sequencing** given the above: PR A
  (`__int__`/`__float__`/`__index__`/`__array__`, ~60-70% per DRAFT_PR.md)
  → item B here (1-line `argfix` fix, pure bugfix, no pinned behavior
  against it) → item A here (slice bounds; builds on `__index__` from
  PR A, torch-parity argument) → issue-then-PR-B for `__bool__`. Items A
  and B are independent of the `__bool__` outcome; if everything but
  `__bool__` lands, the backend keeps exactly one monkeypatch
  (`Tensor.__bool__`, installed at
  `keras/src/backend/tinygrad/numpy.py:48` — the one deliberate override
  of an existing tinygrad attribute, since stock `__bool__` raises)
  instead of five (numpy.py:48-87).

**What landing A+B removes on our side:** the two `tuple(...)`
normalizations (B), two red RandomCrop tests (A), and — with the dunders
PR — the five scalar-interop monkeypatches whose closed set is invariant
6 in `docs/architecture.md`.
