"""`keras.ops.linalg` for the tinygrad backend — pure tinygrad Tensor ops.

Decompositions use classic shape-stable formulations (Householder QR,
Gauss-Jordan with partial pivoting, cyclic Jacobi rotations for eigen /
singular values) so the tensor shapes are identical on every step of the
python loops. Everything is computed internally in float64 (supported by the
CPU jit) and cast back to the input's float dtype at the end.

Host-side `.item()` reads appear ONLY on error-raising / control paths (the
non-positive-definite check in `cholesky`, the symmetry gate in `eig`) —
never on a value that flows into a returned tensor, so no gradient path is
detached. Loop bodies call `.contiguous()` (and `.realize()` in the pivoted
eliminations, see comments) to bound the lazy-graph size.

`jvp` is forward mode built from two reverse-mode passes (see its docstring).
"""

import math as _math
import warnings

from tinygrad import Tensor
from tinygrad import dtypes

from keras.src.backend.tinygrad.core import convert_to_tensor
from keras.src.backend.tinygrad.core import to_keras_dtype
from keras.src.backend.tinygrad.numpy import _float

_F64 = dtypes.float64
# Machine epsilons keyed by keras float dtype (used for default svd cutoffs).
_EPS = {
    "float16": 2.0**-10,
    "bfloat16": 2.0**-7,
    "float32": 2.0**-23,
    "float64": 2.0**-52,
}


def _prep(x):
    """Return (float64 working copy, tinygrad dtype to cast results back to)."""
    x = _float(convert_to_tensor(x))
    return x.cast(_F64), x.dtype


def _eps_of(tg_dtype):
    return _EPS.get(to_keras_dtype(tg_dtype), 2.0**-23)


def _broadcast_shape(s1, s2):
    s1, s2 = list(s1), list(s2)
    out = []
    for i in range(max(len(s1), len(s2))):
        d1 = s1[-1 - i] if i < len(s1) else 1
        d2 = s2[-1 - i] if i < len(s2) else 1
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Incompatible batch shapes: {s1} vs {s2}")
        out.append(max(d1, d2))
    return tuple(reversed(out))


def _broadcast_batches(a, b):
    """Broadcast the leading (batch) dims of two stacked-matrix tensors."""
    ba, bb = tuple(a.shape[:-2]), tuple(b.shape[:-2])
    if ba == bb:
        return a, b
    nb = _broadcast_shape(ba, bb)
    a = a.reshape((1,) * (len(nb) - len(ba)) + tuple(a.shape))
    a = a.expand(*nb, *a.shape[-2:])
    b = b.reshape((1,) * (len(nb) - len(bb)) + tuple(b.shape))
    b = b.expand(*nb, *b.shape[-2:])
    return a, b


def _sign01(x):
    """sign with sign(0) == +1, in float64 (matches LAPACK conventions)."""
    return (x >= 0).cast(_F64) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# QR (Householder, batched) — unlocks the Orthogonal initializer.
# ---------------------------------------------------------------------------


def _qr_householder(a):
    """Full QR of `a` (..., m, n) in float64: Q (..., m, m), R (..., m, n).

    Follows the LAPACK sign convention (reflector chosen so the produced
    diagonal entry is -sign(x_k)*||x||; a length-1 reflector is the
    identity), so results match `np.linalg.qr` including signs.
    """
    m, n = int(a.shape[-2]), int(a.shape[-1])
    rows = Tensor.arange(m).reshape(m, 1)
    R = a
    Q = Tensor.eye(m, dtype=_F64)
    for j in range(min(m, n)):
        if j == m - 1:
            # LAPACK dlarfg with no rows below the diagonal: H = I.
            continue
        ge = (rows >= j).cast(_F64)  # (m, 1)
        ej = (rows == j).cast(_F64)  # (m, 1)
        col = R[..., :, j : j + 1]  # (..., m, 1)
        x = col * ge
        xj = (col * ej).sum(-2, keepdim=True)  # (..., 1, 1)
        alpha = -_sign01(xj) * (x * x).sum(-2, keepdim=True).sqrt()
        v = x - alpha * ej
        vn2 = (v * v).sum(-2, keepdim=True)
        coef = (vn2 > 0).where(2.0 / (vn2 + (vn2 <= 0).cast(_F64)), 0.0)
        R = (R - coef * (v @ (v.transpose(-2, -1) @ R))).contiguous()
        Q = (Q - coef * ((Q @ v) @ v.transpose(-2, -1))).contiguous()
    if Q.ndim < a.ndim:  # loop may not have broadcast Q (e.g. m == 1)
        Q = Q.reshape((1,) * (a.ndim - 2) + (m, m)).expand(
            *a.shape[:-2], m, m
        )
    return Q, R.triu()


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    a, dt = _prep(x)
    m, n = int(a.shape[-2]), int(a.shape[-1])
    k = min(m, n)
    Q, R = _qr_householder(a)
    if mode == "reduced":
        Q = Q[..., :, :k]
        R = R[..., :k, :]
    return Q.cast(dt), R.cast(dt)


# ---------------------------------------------------------------------------
# Gauss-Jordan with partial pivoting: solve / inv.
# ---------------------------------------------------------------------------


def _swap_rows(M, i, ep):
    """Swap static row `i` with the dynamic one-hot row `ep` (batched)."""
    n = int(M.shape[-2])
    eic = (Tensor.arange(n).reshape(n, 1) == i).cast(_F64)  # (n, 1)
    epc = ep.unsqueeze(-1)  # (..., n, 1)
    Mi = M[..., i : i + 1, :]  # (..., 1, cols)
    Mp = ep.unsqueeze(-2) @ M  # (..., 1, cols)
    # When p == i the two corrections cancel exactly.
    return M + eic * (Mp - Mi) + epc * (Mi - Mp)


def _pivot_onehot(M, i):
    """One-hot of the partial-pivot row (max |col i| among rows >= i)."""
    n = int(M.shape[-2])
    rows = Tensor.arange(n)
    coli = M[..., :, i]
    mag = (rows >= i).where(coli.abs(), -1.0)
    return mag.argmax(-1), mag.argmax(-1).one_hot(n).cast(_F64)


def _gauss_jordan(a, b):
    """Solve a @ x = b for stacked square `a` (..., n, n), `b` (..., n, k)."""
    n = int(a.shape[-1])
    eic_all = Tensor.arange(n).reshape(n, 1)
    M = a.cat(b, dim=-1)
    for i in range(n):
        _, ep = _pivot_onehot(M, i)
        M = _swap_rows(M, i, ep)
        piv = M[..., i : i + 1, i : i + 1]  # (..., 1, 1)
        rowi = M[..., i : i + 1, :] / piv  # normalized pivot row
        colv = M[..., :, i : i + 1]  # (..., n, 1)
        eic = (eic_all == i).cast(_F64)
        # Eliminate column i from every row, then restore the normalized
        # pivot row. Realize each step to bound the lazy-graph size (this is
        # an O(n)-step loop of full-matrix updates).
        M = (M - colv @ rowi + eic * rowi).contiguous().realize()
    return M[..., :, n:]


def solve(a, b):
    a64, dt_a = _prep(a)
    b64, dt_b = _prep(b)
    dt = dt_a if dt_a.itemsize >= dt_b.itemsize else dt_b
    vector = b64.ndim == a64.ndim - 1
    if vector:
        b64 = b64.unsqueeze(-1)
    a64, b64 = _broadcast_batches(a64, b64)
    out = _gauss_jordan(a64, b64)
    if vector:
        out = out.squeeze(-1)
    return out.cast(dt)


def inv(a):
    a64, dt = _prep(a)
    n = int(a64.shape[-1])
    eye = Tensor.eye(n, dtype=_F64)
    eye = eye.reshape((1,) * (a64.ndim - 2) + (n, n)).expand(
        *a64.shape[:-2], n, n
    )
    return _gauss_jordan(a64, eye).cast(dt)


# ---------------------------------------------------------------------------
# det / lu_factor (Gaussian elimination with partial pivoting).
# ---------------------------------------------------------------------------


def det(a):
    a64, dt = _prep(a)
    n = int(a64.shape[-1])
    colr = Tensor.arange(n).reshape(n, 1)
    U = a64
    sgn = U[..., 0, 0] * 0.0 + 1.0  # ones with the batch shape
    for i in range(n - 1):
        p, ep = _pivot_onehot(U, i)
        U = _swap_rows(U, i, ep)
        sgn = sgn * ((p == i).cast(_F64) * 2.0 - 1.0)
        piv = U[..., i : i + 1, i : i + 1]
        zero = (piv.abs() <= 0).cast(_F64)
        # Guarded divide: an exactly-zero pivot leaves the (singular) column
        # untouched so the zero lands on the diagonal and the product is 0.
        f = (U[..., :, i : i + 1] / (piv + zero)) * (1.0 - zero)
        f = f * (colr > i).cast(_F64)
        # Row operations don't change the determinant; realize per step to
        # bound the lazy-graph size.
        U = (U - f @ U[..., i : i + 1, :]).contiguous().realize()
    diag = (U * Tensor.eye(n, dtype=_F64)).sum(-1)  # (..., n)
    return (sgn * diag.prod(-1)).cast(dt)


def lu_factor(a):
    """Doolittle LU with partial pivoting; scipy `lu_factor` conventions:

    returns (lu, piv) where row i was interchanged with row piv[i] >= i.
    """
    a64, dt = _prep(a)
    m, n = int(a64.shape[-2]), int(a64.shape[-1])
    k = min(m, n)
    colr = Tensor.arange(m).reshape(m, 1)
    ei_row_all = Tensor.arange(n)
    M = a64
    pivots = []
    for i in range(k):
        p, ep = _pivot_onehot(M, i)
        pivots.append(p.cast(dtypes.int32).unsqueeze(-1))
        M = _swap_rows(M, i, ep)
        piv = M[..., i : i + 1, i : i + 1]
        f = (M[..., :, i : i + 1] / piv) * (colr > i).cast(_F64)
        ei_row = (ei_row_all == i).cast(_F64)  # (n,)
        # U-update below row i restricted to columns >= i (columns < i hold
        # the already-stored L multipliers and must stay untouched), then
        # store this step's L multipliers in column i. Realize per step to
        # bound the lazy-graph size.
        mi = M[..., i : i + 1, :] * (ei_row_all >= i).cast(_F64)
        M = (M - f @ mi + f * ei_row).contiguous().realize()
    piv_out = pivots[0] if k == 1 else pivots[0].cat(*pivots[1:], dim=-1)
    return M.cast(dt), piv_out


# ---------------------------------------------------------------------------
# Cholesky.
# ---------------------------------------------------------------------------


def cholesky(a, upper=False):
    a64, dt = _prep(a)
    n = int(a64.shape[-1])
    colr = Tensor.arange(n).reshape(n, 1)
    ecols = Tensor.arange(n)
    L = a64 * 0.0
    for j in range(n):
        Lj = L[..., j : j + 1, :]  # row j: columns < j are filled
        c = a64[..., :, j : j + 1] - L @ Lj.transpose(-2, -1)
        djj = c[..., j : j + 1, :]  # (..., 1, 1)
        sq = djj.sqrt()  # NaN if the input is not positive definite
        ejc = (colr == j).cast(_F64)  # (n, 1)
        below = (colr > j).cast(_F64)
        newcol = (c / sq) * below + sq * ejc
        ejrow = (ecols == j).cast(_F64)  # (n,)
        L = (L + newcol * ejrow).contiguous()
    # Host-side validity check (error path only — raising has no gradient
    # path; sqrt of a negative diagonal produced NaN above).
    if bool(L.isnan().any().item()):
        raise ValueError(
            "Cholesky decomposition failed: the input may not be "
            "positive definite."
        )
    out = L.transpose(-2, -1) if upper else L
    return out.cast(dt)


def cholesky_inverse(a, upper=False):
    # Mirrors the numpy backend: invert via the triangular factor.
    a = _float(convert_to_tensor(a))
    n = int(a.shape[-1])
    identity = Tensor.eye(n, dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    if upper:
        return inv_chol @ inv_chol.transpose(-2, -1)
    return inv_chol.transpose(-2, -1) @ inv_chol


def solve_triangular(a, b, lower=False):
    a64, dt = _prep(a)
    b64, _ = _prep(b)
    vector = b64.ndim == a64.ndim - 1
    if vector:
        b64 = b64.unsqueeze(-1)
    a64, b64 = _broadcast_batches(a64, b64)
    n = int(a64.shape[-1])
    colr = Tensor.arange(n).reshape(n, 1)
    x = b64 * 0.0
    order = range(n) if lower else range(n - 1, -1, -1)
    for i in order:
        ai = a64[..., i : i + 1, :]  # (..., 1, n)
        xi = (b64[..., i : i + 1, :] - ai @ x) / a64[
            ..., i : i + 1, i : i + 1
        ]
        eic = (colr == i).cast(_F64)
        x = (x + eic * xi).contiguous()
    if vector:
        x = x.squeeze(-1)
    return x.cast(dt)


# ---------------------------------------------------------------------------
# Jacobi rotations: eigh / eig and one-sided-Jacobi SVD.
# ---------------------------------------------------------------------------


def _round_robin_pairs(n):
    """Rounds of disjoint (p, q) pairs covering all pairs (circle method)."""
    if n < 2:
        return []
    players = list(range(n)) + ([n] if n % 2 else [])  # `n` is a bye slot
    size = len(players)
    rounds = []
    for _ in range(size - 1):
        pairs = []
        for i in range(size // 2):
            p, q = players[i], players[size - 1 - i]
            if p < n and q < n:
                pairs.append((min(p, q), max(p, q)))
        rounds.append(pairs)
        players = [players[0]] + [players[-1]] + players[1:-1]
    return rounds


def _pair_masks(n, p, q):
    """Constant (n, n) masks for a Jacobi rotation in the (p, q) plane."""
    e = Tensor.arange(n)
    epv = (e == p).cast(_F64)
    eqv = (e == q).cast(_F64)
    cmask = epv.reshape(n, 1) * epv + eqv.reshape(n, 1) * eqv
    smask = epv.reshape(n, 1) * eqv - eqv.reshape(n, 1) * epv
    return cmask, smask


def _jacobi_cs(app, aqq, apq):
    """cos/sin of the Jacobi angle zeroing the (p, q) cross term."""
    nz = apq.abs() > 0
    tau = (aqq - app) / (nz.where(2.0 * apq, 1.0))
    t = _sign01(tau) / (tau.abs() + (1.0 + tau * tau).sqrt())
    c = 1.0 / (1.0 + t * t).sqrt()
    s = t * c
    return nz.where(c, 1.0), nz.where(s, 0.0)


def _jacobi_eigh(a, sweeps=12):
    """Cyclic two-sided Jacobi for symmetric `a` (..., n, n) in float64."""
    n = int(a.shape[-1])
    A = (a + a.transpose(-2, -1)) * 0.5  # enforce exact symmetry
    V = Tensor.eye(n, dtype=_F64)
    eye = Tensor.eye(n, dtype=_F64)
    rounds = _round_robin_pairs(n)
    masks = {pq: _pair_masks(n, *pq) for r in rounds for pq in r}
    for _ in range(sweeps):
        for pairs in rounds:
            jconst = eye
            for p, q in pairs:
                cm, sm = masks[(p, q)]
                jconst = jconst - cm
            J = jconst
            for p, q in pairs:
                cm, sm = masks[(p, q)]
                app = A[..., p : p + 1, p : p + 1]
                aqq = A[..., q : q + 1, q : q + 1]
                apq = A[..., p : p + 1, q : q + 1]
                c, s = _jacobi_cs(app, aqq, apq)
                J = J + c * cm + s * sm
            # Realize per rotation round to bound the lazy-graph size.
            A = (J.transpose(-2, -1) @ (A @ J)).contiguous().realize()
            V = (V @ J).contiguous().realize()
    # Convergence loudness: the fixed sweep count converges quadratically
    # at the matrix sizes Keras exercises, but larger or pathologically
    # conditioned inputs would otherwise return silently degraded factors.
    # Control-path scalar reads only (same precedent as cholesky's NaN gate).
    off = float((A * (1.0 - eye)).abs().max().item())
    scale = float(A.abs().max().item())
    if off > 1e-10 * max(scale, 1.0):
        warnings.warn(
            "tinygrad backend: Jacobi eigendecomposition did not fully "
            f"converge (relative off-diagonal residual {off / max(scale, 1.0):.2e}); "
            "results may be inaccurate for large or ill-conditioned "
            "matrices."
        )
    if V.ndim < a.ndim:
        V = V.reshape((1,) * (a.ndim - 2) + (n, n)).expand(
            *a.shape[:-2], n, n
        )
    w = (A * eye).sum(-1)  # (..., n)
    return w, V


def _sort_eig(w, V, descending=False):
    n = int(w.shape[-1])
    idx = w.argsort(-1, descending=descending)
    P = idx.one_hot(n).cast(_F64)  # (..., n, n): P[j, k] = [idx_j == k]
    w_s = (P @ w.unsqueeze(-1)).squeeze(-1)
    V_s = V @ P.transpose(-2, -1)
    return w_s, V_s


def eigh(a):
    a64, dt = _prep(a)
    w, V = _jacobi_eigh(a64)
    w, V = _sort_eig(w, V, descending=False)  # numpy eigh: ascending
    return w.cast(dt), V.cast(dt)


def eig(a):
    a64, dt = _prep(a)
    # Host-side symmetry gate (error path only; the values read here never
    # flow into a result). General non-symmetric eigendecomposition needs
    # complex arithmetic, which tinygrad does not provide.
    scale = float(a64.abs().max().item())
    asym = float((a64 - a64.transpose(-2, -1)).abs().max().item())
    if asym > 1e-9 * max(scale, 1.0):
        raise NotImplementedError(
            "tinygrad backend: `eig` is only implemented for symmetric "
            "matrices (general eigenvalues may be complex)."
        )
    w, V = _jacobi_eigh(a64)
    w, V = _sort_eig(w, V, descending=False)
    return w.cast(dt), V.cast(dt)


def _svd_jacobi(a, want_uv=True, sweeps=10):
    """One-sided Jacobi SVD of tall `a` (..., m, n), m >= n, in float64.

    Returns (U (..., m, n), s (..., n) descending, V (..., n, n)); the
    columns of `a` are orthogonalized by right rotations accumulated in V.
    """
    n = int(a.shape[-1])
    A = a
    V = Tensor.eye(n, dtype=_F64)
    eye = Tensor.eye(n, dtype=_F64)
    rounds = _round_robin_pairs(n)
    masks = {pq: _pair_masks(n, *pq) for r in rounds for pq in r}
    for _ in range(sweeps):
        for pairs in rounds:
            G = A.transpose(-2, -1) @ A  # (..., n, n) Gram matrix
            jconst = eye
            for p, q in pairs:
                cm, _ = masks[(p, q)]
                jconst = jconst - cm
            J = jconst
            for p, q in pairs:
                cm, sm = masks[(p, q)]
                app = G[..., p : p + 1, p : p + 1]
                aqq = G[..., q : q + 1, q : q + 1]
                apq = G[..., p : p + 1, q : q + 1]
                c, s = _jacobi_cs(app, aqq, apq)
                J = J + c * cm + s * sm
            # Realize per rotation round to bound the lazy-graph size.
            A = (A @ J).contiguous().realize()
            if want_uv:
                V = (V @ J).contiguous().realize()
    # Convergence loudness: see _jacobi_eigh. Columns are orthogonal at
    # convergence, so the Gram matrix's off-diagonal is the residual.
    G = A.transpose(-2, -1) @ A
    off = float((G * (1.0 - eye)).abs().max().item())
    scale = float(G.abs().max().item())
    if off > 1e-10 * max(scale, 1.0):
        warnings.warn(
            "tinygrad backend: Jacobi SVD did not fully converge "
            f"(relative off-diagonal residual {off / max(scale, 1.0):.2e}); "
            "results may be inaccurate for large or ill-conditioned "
            "matrices."
        )
    s = (A * A).sum(-2).sqrt()  # column norms = singular values
    idx = s.argsort(-1, descending=True)
    P = idx.one_hot(n).cast(_F64)
    s_sorted = (P @ s.unsqueeze(-1)).squeeze(-1)
    if not want_uv:
        return None, s_sorted, None
    A_sorted = A @ P.transpose(-2, -1)
    V_sorted = V @ P.transpose(-2, -1)
    denom = s_sorted.unsqueeze(-2)
    U = A_sorted / ((denom <= 0).cast(_F64) + denom)
    if V_sorted.ndim < a.ndim:
        V_sorted = V_sorted.reshape((1,) * (a.ndim - 2) + (n, n)).expand(
            *a.shape[:-2], n, n
        )
    return U, s_sorted, V_sorted


def _complete_basis(Ur):
    """Extend orthonormal columns `Ur` (..., m, k) to a full (..., m, m)
    orthonormal basis. Householder-QR of an orthonormal matrix returns its
    own columns up to sign (R has a ±1 diagonal), so fixing the signs keeps
    the first k columns equal to `Ur` and appends an orthonormal complement.
    """
    m, k = int(Ur.shape[-2]), int(Ur.shape[-1])
    Qf, Rf = _qr_householder(Ur)
    eyemk = Tensor.eye(m, dtype=_F64)[:, :k]  # (m, k) diagonal selector
    dvec = (Rf * eyemk).sum(-2)  # (..., k) diagonal of R
    sgn = _sign01(dvec)
    tail = (sgn[..., :1] * 0.0 + 1.0).expand(*sgn.shape[:-1], m - k)
    scol = sgn.cat(tail, dim=-1)  # (..., m)
    return Qf * scol.unsqueeze(-2)


def _svd(x64, full_matrices=True, want_uv=True):
    m, n = int(x64.shape[-2]), int(x64.shape[-1])
    if m >= n:
        U, s, V = _svd_jacobi(x64, want_uv=want_uv)
        if not want_uv:
            return None, s, None
        u, vh = U, V.transpose(-2, -1)
        if full_matrices and m > n:
            u = _complete_basis(u)
    else:
        U2, s, V2 = _svd_jacobi(x64.transpose(-2, -1), want_uv=want_uv)
        if not want_uv:
            return None, s, None
        # x^T = U2 S V2^T  =>  x = V2 S U2^T
        u, vh = V2, U2.transpose(-2, -1)
        if full_matrices:
            vh = _complete_basis(vh.transpose(-2, -1)).transpose(-2, -1)
    return u, s, vh


def svd(x, full_matrices=True, compute_uv=True):
    x64, dt = _prep(x)
    u, s, vh = _svd(x64, full_matrices=full_matrices, want_uv=compute_uv)
    if not compute_uv:
        return s.cast(dt)
    return u.cast(dt), s.cast(dt), vh.cast(dt)


# ---------------------------------------------------------------------------
# SVD-derived ops: pinv, lstsq, matrix_rank, matrix_power.
# ---------------------------------------------------------------------------


def _pinv_factors(x64, rcond):
    """Reduced svd + clipped inverse singular values (shared machinery)."""
    u, s, vh = _svd(x64, full_matrices=False, want_uv=True)
    cutoff = rcond * s.max(-1, keepdim=True)
    sinv = (s > cutoff).where(1.0 / ((s <= cutoff).cast(_F64) + s), 0.0)
    return u, sinv, vh


def pinv(x, rcond=None):
    x64, dt = _prep(x)
    rcond = 1e-15 if rcond is None else rcond  # numpy's default
    u, sinv, vh = _pinv_factors(x64, rcond)
    out = vh.transpose(-2, -1) @ (
        sinv.unsqueeze(-1) * (u.transpose(-2, -1))
    )
    return out.cast(dt)


def lstsq(a, b, rcond=None):
    a64, dt = _prep(a)
    b64, _ = _prep(b)
    m, n = int(a64.shape[-2]), int(a64.shape[-1])
    if rcond is None:
        # numpy lstsq default: machine precision (of the input dtype)
        # times max(M, N), relative to the largest singular value.
        rcond = _eps_of(dt) * max(m, n)
    vector = b64.ndim == a64.ndim - 1
    if vector:
        b64 = b64.unsqueeze(-1)
    u, sinv, vh = _pinv_factors(a64, rcond)
    x = vh.transpose(-2, -1) @ (
        sinv.unsqueeze(-1) * (u.transpose(-2, -1) @ b64)
    )
    if vector:
        x = x.squeeze(-1)
    return x.cast(dt)


def matrix_rank(x, tol=None):
    x64, dt = _prep(x)
    if x64.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    m, n = int(x64.shape[-2]), int(x64.shape[-1])
    _, s, _ = _svd(x64, want_uv=False)
    if tol is None:
        # numpy default: max(s) * max(M, N) * eps(input dtype).
        tol = s.max(-1, keepdim=True) * (max(m, n) * _eps_of(dt))
    return (s > tol).sum(-1).cast(dtypes.int32)


def matrix_power(a, n):
    a = convert_to_tensor(a)
    m = int(a.shape[-1])
    if n == 0:
        eye = Tensor.eye(m, dtype=a.dtype)
        return (
            eye.reshape((1,) * (a.ndim - 2) + (m, m))
            .expand(*a.shape[:-2], m, m)
            .contiguous()
        )
    if n < 0:
        a = inv(a)
        n = -n
    result = None
    base = a
    while n > 0:
        if n & 1:
            result = base if result is None else (result @ base).contiguous()
        n >>= 1
        if n:
            base = (base @ base).contiguous()
    return result


# ---------------------------------------------------------------------------
# norm (vector + matrix; the svd-based matrix ords use the Jacobi svd).
# ---------------------------------------------------------------------------


def _check_axis(axis, ndim):
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for tensor of dimension {ndim}"
        )
    return axis % ndim


def _vector_norm(x, ord, axis, keepdims):
    a = _check_axis(axis, x.ndim)
    if ord is None or ord == 2:
        return (x * x).sum(axis=a, keepdim=keepdims).sqrt()
    if ord == _math.inf:
        return x.abs().max(axis=a, keepdim=keepdims)
    if ord == -_math.inf:
        return x.abs().min(axis=a, keepdim=keepdims)
    if ord == 0:
        return (x != 0).cast(x.dtype).sum(axis=a, keepdim=keepdims)
    if ord == 1:
        return x.abs().sum(axis=a, keepdim=keepdims)
    if isinstance(ord, (int, float)):
        return (x.abs() ** ord).sum(axis=a, keepdim=keepdims) ** (1.0 / ord)
    raise ValueError(f"Invalid `ord` for vector norm: {ord}")


def _matrix_norm_svd(x, ord, r, c):
    """Singular-value matrix norms; returns keepdim-shaped output."""
    rest = [i for i in range(x.ndim) if i not in (r, c)]
    xp = x.permute(*rest, r, c).cast(_F64)
    _, s, _ = _svd(xp, want_uv=False)
    if ord == 2:
        red = s.max(-1)
    elif ord == -2:
        red = s.min(-1)
    else:  # "nuc"
        red = s.sum(-1)
    kd = list(x.shape)
    kd[r] = 1
    kd[c] = 1
    return red.reshape(kd).cast(x.dtype)


def _matrix_norm(x, ord, axis, keepdims):
    r, c = (_check_axis(a, x.ndim) for a in axis)
    if r == c:
        raise ValueError("Duplicate axes given for matrix norm")
    if ord is None or ord == "fro":
        out = (x * x).sum(axis=(r, c), keepdim=True).sqrt()
    elif ord == _math.inf:
        out = x.abs().sum(axis=c, keepdim=True).max(axis=r, keepdim=True)
    elif ord == -_math.inf:
        out = x.abs().sum(axis=c, keepdim=True).min(axis=r, keepdim=True)
    elif ord == 1:
        out = x.abs().sum(axis=r, keepdim=True).max(axis=c, keepdim=True)
    elif ord == -1:
        out = x.abs().sum(axis=r, keepdim=True).min(axis=c, keepdim=True)
    elif ord in (2, -2, "nuc"):
        out = _matrix_norm_svd(x, ord, r, c)
    else:
        raise ValueError(f"Invalid `ord` for matrix norm: {ord}")
    if keepdims:
        return out
    for a in sorted((r, c), reverse=True):
        out = out.squeeze(a)
    return out


def norm(x, ord=None, axis=None, keepdims=False):
    x = _float(x)
    if axis is None:
        if ord is None:
            # 2-norm of the flattened input.
            out = (x * x).sum().sqrt()
            return out.reshape((1,) * x.ndim) if keepdims else out
        if x.ndim == 1:
            axis = 0
        elif x.ndim == 2:
            axis = (0, 1)
        else:
            raise ValueError(
                "Improper number of dimensions to norm: `ord` requires a "
                "1-D or 2-D input when `axis` is None. "
                f"Received: ord={ord}, input rank={x.ndim}"
            )
    if isinstance(axis, int):
        return _vector_norm(x, ord, axis, keepdims)
    if isinstance(axis, (list, tuple)) and len(axis) == 2:
        return _matrix_norm(x, ord, tuple(axis), keepdims)
    raise ValueError(
        f"Invalid `axis` for norm: {axis}. Expected an int or a 2-tuple."
    )


# ---------------------------------------------------------------------------
# jvp — forward mode built from two reverse-mode passes.
# ---------------------------------------------------------------------------


def jvp(fun, primals, tangents, has_aux=False):
    """Forward-mode jvp via the classic jvp-from-vjp identity.

    With a dummy cotangent `u`, `g(u) = grad_p <f(p), u> = J^T u` is linear
    in `u`, so `grad_u <g(u), t> = J t`. tinygrad's `Tensor.gradient` builds
    the backward pass symbolically on the lazy graph, so differentiating
    through it a second time is exact — no numeric round-trips.
    """
    from keras.src import tree

    primals_c = tree.map_structure(convert_to_tensor, tuple(primals))
    tangents_c = tree.map_structure(convert_to_tensor, tuple(tangents))
    p_leaves = tree.flatten(primals_c)
    t_leaves = tree.flatten(tangents_c)
    if len(p_leaves) != len(t_leaves):
        raise ValueError(
            "`primals` and `tangents` must have the same structure. "
            f"Received: primals={primals}, tangents={tangents}"
        )
    outs_all = fun(*primals_c)
    if has_aux:
        outs, aux = outs_all
    else:
        outs = outs_all
    out_leaves = [convert_to_tensor(o) for o in tree.flatten(outs)]
    # Each dummy cotangent must be a DISTINCT realized buffer: identical
    # zero constants would be CSE'd into one UOp and their gradients merged.
    us = [Tensor.zeros_like(o).contiguous().realize() for o in out_leaves]
    l1 = None
    for o, u in zip(out_leaves, us):
        term = (o * u).sum()
        l1 = term if l1 is None else l1 + term
    gs = l1.gradient(*p_leaves)  # J^T u (as a graph linear in each u)
    l2 = None
    for g, t in zip(gs, t_leaves):
        term = (g * t).sum()
        l2 = term if l2 is None else l2 + term
    touts = l2.gradient(*us)  # J t
    tangents_out = tree.pack_sequence_as(outs, list(touts))
    if has_aux:
        return outs, tangents_out, aux
    return outs, tangents_out


# ---------------------------------------------------------------------------
# Not implemented (loud).
# ---------------------------------------------------------------------------


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"tinygrad backend: `keras.ops.linalg.{name}` is not implemented yet"
    )
