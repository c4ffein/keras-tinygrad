# Ops Surface — numpy, nn, math, linalg, image, random

**Verdict.** The ops surface is unusually disciplined for ~7k lines of fast-written backend code: numpy usage is confined almost everywhere to the three sanctioned roles, dtype promotion is handled with per-op referee rules that are commented when they diverge (e.g. `nanpercentile` vs `nanquantile`), and the loud-stub `__getattr__` pattern is applied uniformly across all five op modules. The verified defects are real but narrow: `logaddexp` NaN-poisons at infinities while `logaddexp2` ten lines below handles the same edge correctly; `expm1`/`log1p` lose all precision for small inputs; `take` silently zero-fills out-of-range indices where numpy raises; and `random.shuffle` is the one op that round-trips *data* (not samples) through host numpy, silently detaching gradients. The linalg module's fixed-sweep Jacobi (no convergence check) is the largest systemic accuracy risk — a silent-inaccuracy exposure in a codebase whose stated ethos is "loud over wrong." Code quality is high; the main debt is a ~40x-repeated cast-back idiom and a duplicated bit-table in `numpy.py`.

All paths below are under `/home/dev/workspace/keras/keras/src/backend/tinygrad/`.

## 1. Invariant compliance — the numpy audit

Every `np.` / host-read site across the six files was read and classified:

| Module | numpy / host-read appearances | Role | Verdict |
|---|---|---|---|
| `nn.py` | **none** — zero numpy imports | — | clean |
| `linalg.py` | **none** — zero numpy imports; `.item()` at `cholesky` (281), `eig` (421–422) | error/control paths only, as the module docstring promises | clean |
| `math.py` | DFT matrices (246–279), scipy windows (387–390), `segment_ids.max().numpy()` (162) | constant tables; structural read of an integer *index* tensor | clean |
| `numpy.py` | reflect-pad index table (1387), `np.eye`/`np.tri` (1664, 1673, 1803), window functions (2881–2900), `np.generic` unwrapping, structural host reads (`nonzero` 1752, `bincount` 871, `repeat` 1222, `split` 1194, `pad` widths 1371, `_host_q` 2467) | creation-time constants + shape metadata; every data-dependent realize is commented as such | clean |
| `image.py` | resize weight matrices (370–412), coordinate meshgrids (588, 936, 1151), homography `np.linalg.solve` (840), gaussian kernel (976) | all coordinate-side / static-shape constants; image data never round-trips | clean, with one nuance below |
| `random.py` | Generator sampling throughout; `shuffle` converts **x itself** (108); `categorical` converts **logits** (47) | sampling is sanctioned; `shuffle` is a data round-trip | **one violation-smell** |

Three compliance notes:

- **`random.shuffle` (`random.py:106–109`) converts the data tensor to numpy and back.** Unlike every other random op (where the *sample* is the constant), here the returned tensor *is* the input's values, permuted — so a model tensor passing through `keras.random.shuffle` silently loses its gradient path, exactly the failure mode invariant 2 exists to prevent. The fix is cheap and preserves bit-parity: draw the permutation from `rng.permuted(np.arange(n))` host-side (indices are sanctioned RNG output) and apply it via a differentiable `take`. `categorical` (`random.py:47`) also reads `logits` to host, but its output is integer samples with no gradient contract in any backend — acceptable, though it forces realization of a possibly-lazy graph.
- **Monkeypatch guard mismatch with the architecture doc.** `docs/architecture.md` invariant 6 says patches are "additive and guarded only — never override an existing tinygrad attribute," but `numpy.py:48` assigns `Tensor.__bool__` unconditionally, replacing tinygrad's existing (unconditionally-raising) implementation, and `numpy.py:61` assigns `__array__` unguarded, while `__float__`/`__int__`/`__index__` (82–87) get `hasattr` guards. The `__bool__` replacement is deliberate and well-commented in code, but the invariant's text doesn't match reality — per the doc's own maintenance rule, that's a bug in whichever diff made it wrong.
- `image.compute_homography_matrix` (`image.py:844–861`) runs `np.linalg.solve` on host. It's coordinate-side (no gradient contract, documented), but it is honestly a fourth numpy role — "coordinate-side *computation*", not a constant table. The module docstring declares it; the architecture doc's three-role list doesn't.

The PEP 562 loud-stub tail is present and identical in all five op modules (`numpy.py:3036`, `nn.py:1307`, `math.py:523`, `linalg.py:758`, `image.py:1236`) — missing ops genuinely cannot fall back silently.

## 2. Correctness risks (verified where marked)

Ranked by severity. Items marked **[verified]** were reproduced against the live backend (`ktg-venv`, zig-cc shim).

1. **`logaddexp` returns NaN whenever either operand is ±inf** — `numpy.py:363–368`. **[verified]**: `logaddexp(-inf, -inf) → nan` (numpy: `-inf`), `logaddexp(inf, -inf) → nan` (numpy: `inf`), `logaddexp(-inf, 1.0) → 1.0` (correct by accident of the max-shift). The bitter part: `logaddexp2` directly below (`numpy.py:371–386`) implements exactly the three fix-up masks needed. Port those masks up.

2. **Fixed-sweep Jacobi with no convergence or residual check** — `_jacobi_eigh(a, sweeps=12)` at `linalg.py:366`, `_svd_jacobi(a, want_uv, sweeps=10)` at `linalg.py:432`. Jacobi converges quadratically, so 10–12 sweeps is comfortably enough for the n ≤ ~32 matrices Keras tests exercise, but for larger or pathologically-conditioned inputs the routine returns *silently degraded* factors — the one place in the ops surface where "wrong answer over loud error" can actually happen. A cheap post-loop residual (off-diagonal norm for eigh, column-orthogonality for SVD) with a loud warning/raise would restore the ethos; it costs one extra reduction and can reuse the existing `.item()` control-path precedent (`cholesky` already realizes for its NaN check at `linalg.py:280–284`).

3. **`expm1` and `log1p` are naive compositions** — `numpy.py:343–344` (`exp(x) - 1`) and `359–360` (`log(x + 1)`). **[verified]**: both return exactly `0.0` for `x = 1e-8` float32 where numpy returns `1e-8`. Total precision loss below ~`eps`; these feed user losses and metrics. tinygrad lacks native variants, but a series/`where` split (or the classic `x * exp(x)/(exp(x)-1)` trick) would fix it in-kernel.

4. **`take` / `take_along_axis` silently zero-fill out-of-range positive indices** — `numpy.py:1695–1708`, `1711–1731`. **[verified]**: `take([1,2,3], [5]) → [0.]`; numpy raises `IndexError`. Negative indices are wrapped explicitly (1705), but positive OOB inherits tinygrad's gather zero-fill — a silent wrong answer with no comment naming the deviation (invariant 1 requires one). Same exposure in `bincount` for negative values (`numpy.py:874–876`: no one-hot row matches, so negatives are silently dropped; numpy raises `ValueError`).

5. **`power` with integer operands and negative exponent returns 0 silently** — `numpy.py:200–210`. **[verified]**: `power(2, -1) → 0`; numpy raises "Integers to negative integer powers are not allowed." Float pow then int cast truncates. Uncommented deviation.

6. **Empty-tensor edges in `roll` and `median`/`quantile`** — **[verified]**: `roll` on a zero-length axis raises `ZeroDivisionError` (`shift % x.shape[axis]`, `numpy.py:1340`; numpy returns the empty input), and `median` over an empty axis raises tinygrad's `IndexError` from the `(n-1)//2 = -1` index (`numpy.py:646`; numpy warns and returns NaN). Low impact, but the error types are misleading.

7. **DFT scale/accuracy ceiling** — `math.py:243–279`. Constants are built in float64 (good) but cast to the input dtype, so a float32 `fft` at n=2048 carries ~`sqrt(n)·eps` ≈ 1e-5 relative error per transform — fine for the declared STFT-scale contract, degrading for round-trips (`ifft(fft(x))` pays twice) and O(n²) memory means n=8192 full-FFT would allocate ~0.5 GB of constants. Also `_DFT_CACHE` (`math.py:243`) is unbounded and never evicts — each distinct `(kind, n, dtype)` pins its matrices for process lifetime. Acceptable per the documented contract; worth a size cap or an explicit raise above a sane n.

8. **`histogram` forces range bounds to float32** — `numpy.py:2163–2164` — even for float64 input, so edges come back float32-precision (numpy: float64). Cosmetic-to-minor.

9. **`arctan2` signed-zero cases** — `numpy.py:470–483`. `arctan2(-0.0, -1.0)` yields `+pi` (numpy: `-pi`) because `(y >= 0)` cannot see the sign bit; the `signbit` machinery at 2230 could. Only signed zeros are affected; the (0, 0) case is correct.

10. **SVD rank-deficient U columns** — `_svd_jacobi` at `linalg.py:472` produces zero columns for zero singular values (guarded divide), so reduced-form `U` is not orthonormal in the rank-deficient case, and `_complete_basis`'s sign-fix assumption ("R has a ±1 diagonal", `linalg.py:485–493`) breaks there. numpy returns an orthonormal basis. Edge case; the SVD-derived ops (`pinv`, `lstsq`) are insensitive because `sinv` zeroes those directions anyway (`linalg.py:533`).

Positive verification results worth recording: tinygrad's `//` and `%` are floor-division/floor-mod matching numpy exactly on negative ints and floats **[verified]**, `argsort` is stable **[verified]** (which `argpartition`'s full-sort strategy at `numpy.py:2838–2847` and `_ctc_unique_padded`'s dedup depend on), and `round` is half-to-even **[verified]**.

## 3. Dtype promotion and broadcasting

This is the strongest part of the surface. The `_pair` helper (`numpy.py:98–113`) implements Keras weak-typing correctly (python scalars stay unpromoted; tensor pairs go through `result_type`), and the harder per-op rules are individually researched and commented: the int8-matmul rule keyed on *original* operand dtypes (`numpy.py:176–189`, invariant 8), all-int8 einsum → int32 (`numpy.py:2059–2061`), bool contraction as any-of-products (`tensordot` 1993–1997, `vdot` 2008), small-int accumulation dtypes for `prod`/`nansum`/`trace` matching the reference exactly, and the deliberately *different* rules for `nanquantile` vs `nanpercentile` called out in a comment (`numpy.py:2630–2633`). The cast-back-after-compute idiom consistently prevents tinygrad's internal widening from leaking into result dtypes. `dot_product_attention` (`nn.py:670–701`) even matches `np.einsum`'s float32 accumulation of float16 operands to avoid one-ulp drift.

One inconsistency: `_norm_axis` (`numpy.py:124–129`) correctly passes positive out-of-range axes through to tinygrad's `IndexError` **[verified]**, but the inline `a % x.ndim` in `median` (632–634), `nanmedian` (673–676), `_quantile_impl` (2489–2491), `moveaxis` (1080–1081), and `expand_dims` (1093) silently *wraps* them — `median(x, axis=5)` on a 3-d tensor quietly reduces axis 2. Unify on `_norm_axis`.

## 4. Code quality and consistency

- **The cast-back idiom is duplicated ~40 times**: `return out.cast(tg_dtype) if out.dtype != tg_dtype else out` appears at the tail of nearly every dtype-managed op in `numpy.py`. A three-line `_cast(out, dtype)` helper would remove ~100 lines and one class of copy-paste risk.
- **Literal duplication**: `signbit` builds an inline float→int width table (`numpy.py:2238–2243`) and the identical `_FLOAT_BITS_INT` module constant is defined immediately after (`numpy.py:2249–2254`) for `nextafter`. `signbit` should use the constant.
- **Triplicated choreography**: `median` (629–655), `nanmedian` (671–701), and `_quantile_impl` (2485–2498) each re-derive the same permute-reduced-axes-to-tail-and-flatten dance. One helper would serve all three (and fix the empty-axis edge once).
- **Name collision**: `_pair` means "binary-op dtype promotion" in `numpy.py:98` and "int → 2-tuple" in `nn.py:593` — same private name, adjacent modules, unrelated semantics.
- **Minor**: `_pivot_onehot` computes `mag.argmax(-1)` twice (`linalg.py:154`; lazy-graph CSE probably dedups, but it reads as an oversight); `hstack` converts `xs[0]` and then `concatenate` re-converts everything (`numpy.py:1140–1142`); `image.py:26–29` interleaves a `draw_seed` import between two `core` imports; `erfc` sits between `erfinv`'s coefficient tables and `erfinv` itself (`math.py:64–65`).
- **No dead code found.** For 7k lines written fast, that's notable. Comment density and quality are exceptional — nearly every non-obvious decision names its referee behavior or the tinygrad quirk it works around (e.g. the TrackedList/argfix trap documented twice at `numpy.py:1050–1053` and `1119–1122`).
- Per-batch/per-channel Python loops in `image.perspective_transform` (944–962: B×C `map_coordinates` calls) and `elastic_transform` (1159–1177) replicate graphs O(B·C) — correct but will crawl on real batches; same for `_ctc_beam_search_decode`'s per-batch-per-timestep loop (`nn.py:1211–1250`, a faithful port of the torch backend's structure).

## 5. Notable engineering wins

- **`nextafter` via two's-complement bitcast arithmetic** (`numpy.py:2257–2282`): bit-exact ulp stepping with the inf edges falling out of IEEE sign-magnitude ordering for free — an elegant solution to an op most backends punt to a libm call.
- **Reflect/symmetric `pad` as a host-built index table + differentiable gather** (`numpy.py:1378–1393`): reproduces numpy's multi-bounce reflection exactly while keeping gradients flowing.
- **Shape-stable `gcd`** (`numpy.py:233–251`): Euclid with a worst-case iteration bound derived from Fibonacci pairs (`1.5 × bit-width + 2`) so the graph shape is data-independent.
- **Parallel round-robin Jacobi** (`linalg.py:328–397`): disjoint (p,q) rotation planes per round make the simultaneous rotations *mathematically exact* (rotations in disjoint planes commute and don't perturb each other's pivot entries), with precomputed constant masks and per-round realizes to bound graph depth. The same skeleton serves eigh and one-sided SVD.
- **`jvp` from double reverse-mode** (`linalg.py:709–750`), including the subtle CSE hazard note: dummy zero cotangents must be *distinct realized buffers* or tinygrad merges their gradients (734–736).
- **`fold` as a grouped `conv_transpose2d` against a constant identity kernel** (`nn.py:733–767`): col2im's overlap-add as a single differentiable op.
- **Fixed-shape row dedup for CTC beam search** (`nn.py:1051–1100`): hash + stable sort + collision check + scatter-to-compacted-slots, avoiding data-dependent shapes entirely.
- **Batched `bincount` that realizes only the scalar max** (`numpy.py:869–878`) while the counts stay lazy, and handles 2-D inputs via a one-hot matmul.
- **`irfft` synthesis weights with correct DC/Nyquist handling** (`math.py:267–273`): the 2× interior-bin doubling with unit weights at k=0 and n/2 matches `np.fft.irfft` exactly, constants built in float64.
- **`in_top_k` NaN masking** (`math.py:121–126`): explicitly works around tinygrad's non-IEEE NaN comparisons instead of inheriting them.

## 6. Recommendations, ranked

1. Port `logaddexp2`'s three inf/NaN fix-up masks into `logaddexp` (`numpy.py:363`). One-line-per-mask fix, verified defect.
2. Add a residual check (or at least an opt-in loud warning) after the Jacobi loops in `linalg.py:366/432` — the only silent-inaccuracy channel in the surface.
3. Make `random.shuffle` gather through a host-drawn permutation instead of round-tripping data (`random.py:106`), preserving both bit-parity and gradients.
4. Guard `take`/`take_along_axis`/`bincount` against positive OOB / negative indices with an explicit check or a comment naming the deviation (`numpy.py:1705, 1730, 874`).
5. Implement precision-safe `expm1`/`log1p` (`numpy.py:343, 359`).
6. Extract the cast-back helper and the reduce-axes-to-tail helper; point `signbit` at `_FLOAT_BITS_INT`; unify axis normalization on `_norm_axis`.
7. Reconcile invariant 6's "guarded only" wording with the unconditional `__bool__`/`__array__` assignments (`numpy.py:48, 61`) — doc fix, per the architecture file's own maintenance rule.
8. Raise loudly (or cap the cache) for DFT sizes beyond the intended STFT scale (`math.py:243`).

---

**Method note:** every claim marked [verified] was reproduced with the project venv (`/home/dev/workspace/ktg-venv/bin/python` with the zig-cc shim) against the live backend at `/home/dev/workspace/keras/keras/src/backend/tinygrad/`; tinygrad's floor-div/mod parity with numpy, argsort stability, and half-even rounding were also confirmed empirically, retiring three potential-risk hypotheses. No files were modified.
