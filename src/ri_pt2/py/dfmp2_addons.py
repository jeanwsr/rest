"""
PyTorch implementation of the DF-MP2 pair-energy contraction.

References
----------
- GPU4PySCF: https://github.com/pyscf/gpu4pyscf/pull/662
  
  This code initialized from GPU4PySCF PR, adapted and optimized for
  pytorch usage with help of AI (glm-5.x).
"""

import contextlib
import os
import time

_t_import = time.perf_counter()
import torch
import numpy as np

_MP2_VERBOSE = os.environ.get("MP2_VERBOSE", "0") not in ("", "0")
if _MP2_VERBOSE:
    print(
        f"[dfmp2] torch+numpy import: {time.perf_counter() - _t_import:.3f}s",
        flush=True,
    )


def _as_torch(x, dtype, device):
    """Move a numpy / torch / array-like input onto a torch tensor of `dtype` on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@contextlib.contextmanager
def with_tf32(use_tf32):
    """Context manager that temporarily toggles TF32 for float32 CUDA matmul.

    TF32 only affects float32 matmul on CUDA; it is a no-op for float64 or for a CPU
    device. NOTE: ``torch.backends.cuda.matmul.allow_tf32`` is a *process-global* flag,
    so under multi-GPU threading the caller must set it once for the whole parallel
    region (the driver does this) rather than per-call -- the ``_gpu`` leaf functions
    below intentionally do NOT touch it.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ---------------------------------------------------------------------------
# intra: pair energies within one occ cluster (all occ on the same device)
# ---------------------------------------------------------------------------


def _intra_pair_gpu(cderi_ovl, occ_energy, vir_energy, ss_only=False, device=None):
    r"""Intra pair energies into on-device GPU tensors (no TF32 toggle, no host sync).

    Leaf contraction shared by :func:`get_dfmp2_energy_pair_intra` (single-device /
    direct calls) and the multi-GPU driver. The matmul runs in
    ``cderi_ovl.dtype``; the accumulation is always float64. Unlike the public wrapper
    it (a) does NOT toggle ``torch.backends.cuda.matmul.allow_tf32`` -- the caller owns
    the TF32 setting -- and (b) returns the pair-energy matrices as GPU tensors and
    performs NO host sync, so the work stays async on the device and the caller decides
    when to ``.cpu()`` (the driver defers this to a single assembly sync point).

    Parameters
    ----------
    cderi_ovl : torch.Tensor
        Cholesky-decomposed 3c-2e ERI in MO basis, of shape (nocc, nvir, naux). Must be
        a torch tensor; its dtype (float32 or float64) drives the matmul precision.
    occ_energy, vir_energy : np.ndarray | torch.Tensor
    ss_only : bool
    device : str | torch.device

    Returns
    -------
    (eng_pair_bi1, eng_pair_bi2) : torch.Tensor
        Pair-energy matrices of shape (nocc, nocc) on `device` (float64). When
        ``ss_only=True`` a single ``eng_pair_ss`` tensor is returned instead.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert isinstance(
        cderi_ovl, torch.Tensor
    ), f"cderi_ovl must be a torch.Tensor, got {type(cderi_ovl).__name__}"
    mat_dtype = cderi_ovl.dtype
    acc_dtype = torch.float64

    cderi_ovl = _as_torch(cderi_ovl, mat_dtype, device)
    occ_energy = _as_torch(occ_energy, acc_dtype, device)
    vir_energy = _as_torch(vir_energy, acc_dtype, device)

    nocc = occ_energy.shape[0]
    nvir = vir_energy.shape[0]
    naux = cderi_ovl.shape[2]
    assert tuple(cderi_ovl.shape) == (nocc, nvir, naux)

    # empty occ (e.g. a spin channel without electrons) or empty vir: no
    # excitations, all pair energies are zero by definition
    if nocc == 0 or nvir == 0:
        if ss_only:
            return torch.zeros([nocc, nocc], dtype=acc_dtype, device=device)
        return (
            torch.zeros([nocc, nocc], dtype=acc_dtype, device=device),
            torch.zeros([nocc, nocc], dtype=acc_dtype, device=device),
        )

    d_vv = -vir_energy[:, None] - vir_energy[None, :]  # (nvir, nvir), f64

    # fold precision for the f32 matmul path (env MP2_FOLD, default "f32acc"):
    #   f32acc : element-wise in mat_dtype, reduction accumulated in f64 (~1e-7 acc; default).
    #   f64    : upcast g to f64 then fold (conservative option; slower, more mem).
    #   f32    : full f32 fold incl. reduction (fastest, larger error - reference only).
    _fold = os.environ.get("MP2_FOLD", "f32acc").lower()
    # macro-batch size for the j<=i contraction (env MP2_BATCH, default 16). The full j<=i
    # batch per i would build an (nb,nvir,nvir) output up to ~nocc*nvir^2 elements, which is
    # large for big molecules; macro-batching bounds the output/intermediate tensors to
    # (B,nvir,nvir). The double loop stays over j<=i, but advances j in steps of B with one
    # batched GEMM per step. 0 => no chunking (full batch; benchmark only - large memory).
    _batch = int(os.environ.get("MP2_BATCH", "16"))
    _batch = _batch if _batch > 0 else nocc

    if _MP2_VERBOSE:
        # GEMM-only timing pass (macro-batched, async, single sync at end) to isolate
        # matmul throughput from the cast/elem/redu overhead of the full pair loop.
        torch.cuda.synchronize()
        _tgv0 = time.perf_counter()
        for _i in range(nocc):
            # ss pairs are strictly j < i (no i == j diagonal)
            _nb = _i if ss_only else _i + 1
            _at = (
                cderi_ovl[_i : _i + 1]
                .transpose(-1, -2)
                .contiguous()
                .reshape(naux, nvir)
            )
            for _j0 in range(0, _nb, _batch):
                _jb = min(_batch, _nb - _j0)
                _b2d = cderi_ovl[_j0 : _j0 + _jb].reshape(_jb * nvir, naux)
                _ = torch.matmul(_b2d, _at)
        torch.cuda.synchronize()
        _tgv = time.perf_counter() - _tgv0
        _npairs = nocc * (nocc - 1 if ss_only else nocc + 1) // 2
        _gflops = _npairs * 2 * nvir * nvir * naux / _tgv / 1e9
        print(
            f"[dfmp2] gemm-only: {_tgv:.3f}s  {_gflops:.0f} GF/s "
            f"({_npairs} pairs, macro-batch={_batch}, shape {nvir}x{naux}x{nvir})",
            flush=True,
        )

    if _fold != "f64":
        occ_m = occ_energy.to(mat_dtype)
        d_vv_m = d_vv.to(mat_dtype)

    if not ss_only:
        eng_pair_bi1 = torch.zeros([nocc, nocc], dtype=acc_dtype, device=device)
        eng_pair_bi2 = torch.zeros([nocc, nocc], dtype=acc_dtype, device=device)
        if _MP2_VERBOSE:
            torch.cuda.synchronize()
            _tc0 = time.perf_counter()
        # macro-batched over j<=i for fixed i: for each chunk [j0, j0+jb) of size jb<=B,
        # g (jb,nvir,nvir) = cderi[i] @ cderi[j0:j0+jb].mT (broadcasts the (1,nvir,naux) lhs
        # over the jb rhs slices -> one batched GEMM per chunk). Bounding jb bounds the
        # (jb,nvir,nvir) output/intermediate tensors while keeping the GEMM batched.
        for i in range(nocc):
            nb = i + 1
            a_t = (
                cderi_ovl[i : i + 1].transpose(-1, -2).contiguous().reshape(naux, nvir)
            )
            for j0 in range(0, nb, _batch):
                jb = min(_batch, nb - j0)
                b_2d = cderi_ovl[j0 : j0 + jb].reshape(jb * nvir, naux)
                g_ab = torch.matmul(b_2d, a_t).reshape(jb, nvir, nvir)
                if _fold == "f64":
                    g_ab = g_ab.to(acc_dtype)  # upcast for accumulation
                    d_ab = (
                        occ_energy[i]
                        + occ_energy[j0 : j0 + jb, None, None]
                        + d_vv[None]
                    )
                    t_ab = g_ab / d_ab
                    e_bi1 = (t_ab * g_ab).sum(dim=(1, 2))
                    e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2))
                else:
                    d_ab = occ_m[i] + occ_m[j0 : j0 + jb, None, None] + d_vv_m[None]
                    t_ab = g_ab / d_ab
                    if _fold == "f32acc":
                        e_bi1 = (t_ab * g_ab).sum(dim=(1, 2), dtype=acc_dtype)
                        e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2), dtype=acc_dtype)
                    else:  # "f32"
                        e_bi1 = (t_ab * g_ab).sum(dim=(1, 2)).to(acc_dtype)
                        e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2)).to(acc_dtype)
                eng_pair_bi1[i, j0 : j0 + jb] = e_bi1
                eng_pair_bi1[j0 : j0 + jb, i] = e_bi1
                eng_pair_bi2[i, j0 : j0 + jb] = e_bi2
                eng_pair_bi2[j0 : j0 + jb, i] = e_bi2
        if _MP2_VERBOSE:
            torch.cuda.synchronize()
            _tc = time.perf_counter() - _tc0
            _npairs = nocc * (nocc + 1) // 2
            _gflops = _npairs * 2 * nvir * nvir * naux / _tc / 1e9
            print(
                f"[dfmp2] compute: {_tc:.3f}s  {_gflops:.0f} GF/s "
                f"(full pair loop, {_npairs} pairs)",
                flush=True,
            )
        return eng_pair_bi1, eng_pair_bi2
    else:
        eng_pair_ss = torch.zeros([nocc, nocc], dtype=acc_dtype, device=device)
        if _MP2_VERBOSE:
            torch.cuda.synchronize()
            _tc0 = time.perf_counter()
        # macro-batched over j < i for fixed i, same chunked-GEMM scheme as the bi
        # branch above (one (jb*nvir, naux) @ (naux, nvir) GEMM per j chunk)
        for i in range(nocc):
            a_t = (
                cderi_ovl[i : i + 1].transpose(-1, -2).contiguous().reshape(naux, nvir)
            )
            for j0 in range(0, i, _batch):
                jb = min(_batch, i - j0)
                b_2d = cderi_ovl[j0 : j0 + jb].reshape(jb * nvir, naux)
                g_ab = torch.matmul(b_2d, a_t).reshape(jb, nvir, nvir)
                g_as = g_ab - g_ab.mT  # antisymmetrize in mat_dtype (no f64 copy)
                if _fold == "f64":
                    g_as = g_as.to(acc_dtype)  # upcast for accumulation
                    d_ab = (
                        occ_energy[i]
                        + occ_energy[j0 : j0 + jb, None, None]
                        + d_vv[None]
                    )
                    t_as = g_as / d_ab
                    e_ss = (t_as * g_as).sum(dim=(1, 2))
                else:
                    d_ab = occ_m[i] + occ_m[j0 : j0 + jb, None, None] + d_vv_m[None]
                    t_as = g_as / d_ab
                    if _fold == "f32acc":
                        e_ss = (t_as * g_as).sum(dim=(1, 2), dtype=acc_dtype)
                    else:  # "f32"
                        e_ss = (t_as * g_as).sum(dim=(1, 2)).to(acc_dtype)
                eng_pair_ss[i, j0 : j0 + jb] = e_ss
                eng_pair_ss[j0 : j0 + jb, i] = e_ss
        if _MP2_VERBOSE:
            torch.cuda.synchronize()
            _tc = time.perf_counter() - _tc0
            _npairs = nocc * (nocc - 1) // 2
            _gflops = _npairs * 2 * nvir * nvir * naux / _tc / 1e9
            print(
                f"[dfmp2] compute: {_tc:.3f}s  {_gflops:.0f} GF/s "
                f"(ss pair loop, {_npairs} pairs)",
                flush=True,
            )
        return eng_pair_ss


def get_dfmp2_energy_pair_intra(
    cderi_ovl, occ_energy, vir_energy, ss_only=False, device=None, use_tf32=False
):
    r"""Obtain MP2 occupied orbital pair energies (PyTorch, GPU).

    Torch implementation of the DF-MP2 occupied-orbital pair-energy contraction.
    The contraction runs on GPU via torch; the returned pair-energy matrices are
    always ``np.ndarray`` on CPU.

    Parameters
    ----------
    cderi_ovl : torch.Tensor
        Cholesky-decomposed 3c-2e ERI in MO basis, of shape (nocc, nvir, naux).
        Must be a torch tensor; its dtype (float32 or float64) drives the matmul
        precision.
    occ_energy : np.ndarray | torch.Tensor
        Occupied orbital energies, of shape (nocc,).
    vir_energy : np.ndarray | torch.Tensor
        Virtual orbital energies, of shape (nvir,).
    ss_only : bool, optional
        If True, only compute the same-spin pair energies. By default False.
    device : str | torch.device, optional
        Device to run on. Defaults to ``"cuda"`` when available, else ``"cpu"``.
    use_tf32 : bool, optional
        If True, enable TF32 for the float32 matmul on CUDA (``torch.backends
        .cuda.matmul.allow_tf32``); only takes effect when ``cderi_ovl`` is
        float32. By default False (plain FP32 matmul).

    Returns
    -------
    eng_pair_bi1 : np.ndarray
        Bi-orthogonal pair energies for the first term, of shape (nocc, nocc).

        .. math::
            E_{ij}^{\mathrm{bi1}} = \sum_{ab} t_{ij}^{ab}\, g_{ij}^{ab}
            \quad\mathrm{with}\quad t_{ij}^{ab} = g_{ij}^{ab} / D_{ij}^{ab}

    eng_pair_bi2 : np.ndarray
        Bi-orthogonal pair energies for the second term, of shape (nocc, nocc).

        .. math::
            E_{ij}^{\mathrm{bi2}} = \sum_{ab} t_{ij}^{ba}\, g_{ij}^{ab}

    When ``ss_only=True`` only ``eng_pair_ss`` (shape (nocc, nocc)) is returned.

    Notes
    -----
    ``cderi_ovl`` must be a torch tensor; its dtype (float32 or float64) drives
    the matmul precision. The occupied/virtual energies are used in float64 and
    the pair-energy *reduction* is always accumulated in float64. When
    ``cderi_ovl`` is float32, the element-wise fold (denominator / amplitude /
    contraction) runs in float32 with float64 accumulation by default
    (``MP2_FOLD=f32acc``); ``MP2_FOLD=f64`` upcasts ``g_ab`` to float64 before
    the fold instead (conservative option; slower, more memory).
    """
    with with_tf32(use_tf32):
        out = _intra_pair_gpu(
            cderi_ovl, occ_energy, vir_energy, ss_only=ss_only, device=device
        )
    if ss_only:
        return out.cpu().numpy()
    return out[0].cpu().numpy(), out[1].cpu().numpy()


# ---------------------------------------------------------------------------
# pure-CPU bookkeeping (no torch): occ splitting + the inter half-splitting tasks
# ---------------------------------------------------------------------------


def balanced_split(a, n):
    """Split integer `a` into `n` balanced integers.

    Examples
    --------
    >>> balanced_split(10, 3)
    [4, 3, 3]
    """
    v, r = divmod(a, n)
    lst = [v] * n
    for i in range(r):
        lst[i] += 1
    assert sum(lst) == a
    return lst


def build_inter_tasks(
    nocc_device, cderi_ovl_host_list, occ_energy_host_list, eval_mode_list
):
    r"""Build the per-task work lists for the off-diagonal (inter) pair contraction.

    Pure-CPU bookkeeping (no torch) implementing the half-splitting scheme of
    ``dfmp2_kernel_multi_gpu_cderi_cpu``: each host cluster is split into a first/second
    occupied half, and each half is paired with a half of the device's own cluster so
    that, across the two clusters' inter calls, every quarter-block of the off-diagonal
    block is computed exactly once (the caller's ``+= block`` / ``+= block.T`` assembly
    then neither double-counts nor omits a pair).

    Parameters
    ----------
    nocc_device : int
        Number of occupied orbitals in the device's own cluster.
    cderi_ovl_host_list : list of np.ndarray
        One CPU array per cluster (ALL clusters, including the device's own, which is
        skipped via ``eval_mode is None``), each shape (nocc_host, nvir, naux).
    occ_energy_host_list : list of np.ndarray
        One occ-energy array per cluster, each shape (nocc_host,).
    eval_mode_list : list of None | bool
        Per cluster: ``None`` for the device's own cluster, else ``True`` if the cluster
        precedes the device's cluster in global order, ``False`` if it follows.

    Returns
    -------
    cderi_ovl_host_view_list, occ_energy_host_view_list : list of np.ndarray
        Host cderi / occ-energy half-views, one per task.
    occ_idx_host_list : list of list[int]
        Global occupied indices for each task's host half.
    occ_idx_device_list : list of list[int]
        Local device-cluster occupied indices paired with each task's host half.
    nocc_full : int
        Total occupied count (column count of the resulting pair matrix).
    """
    cderi_ovl_host_view_list = []
    occ_energy_host_view_list = []
    occ_idx_host_list = []
    occ_idx_device_list = []
    nocc_device_split = nocc_device // 2
    nocc_full = 0
    for cderi_ovl_host, occ_energy_host, eval_mode in zip(
        cderi_ovl_host_list, occ_energy_host_list, eval_mode_list
    ):
        nocc_host = occ_energy_host.shape[0]
        if eval_mode is not None:
            nocc_split = nocc_host // 2
            # host first half
            cderi_ovl_host_view_list.append(np.asarray(cderi_ovl_host[:nocc_split]))
            occ_energy_host_view_list.append(np.asarray(occ_energy_host[:nocc_split]))
            occ_idx_host_list.append([nocc_full + i for i in range(nocc_split)])
            # host second half
            cderi_ovl_host_view_list.append(np.asarray(cderi_ovl_host[nocc_split:]))
            occ_energy_host_view_list.append(np.asarray(occ_energy_host[nocc_split:]))
            occ_idx_host_list.append(
                [nocc_full + i for i in range(nocc_split, nocc_host)]
            )
            if eval_mode is True:
                # host BEFORE device: host[:split] <-> device[:split], host[split:] <-> device[split:]
                occ_idx_device_list.append(list(range(nocc_device_split)))
                occ_idx_device_list.append(list(range(nocc_device_split, nocc_device)))
            elif eval_mode is False:
                # host AFTER device: host[:split] <-> device[split:], host[split:] <-> device[:split]
                occ_idx_device_list.append(list(range(nocc_device_split, nocc_device)))
                occ_idx_device_list.append(list(range(nocc_device_split)))
        nocc_full += nocc_host
    return (
        cderi_ovl_host_view_list,
        occ_energy_host_view_list,
        occ_idx_host_list,
        occ_idx_device_list,
        nocc_full,
    )


def nbatch_from_avail(nocc, nvir, naux, ndevice, nbytes, avail_bytes):
    """Deduce `nbatch` (batches per device) from an available-memory budget.

    Port of the heuristic in ``dfmp2_kernel_multi_gpu_cderi_cpu``: each batch's cderi
    slice must fit in 40% of the available memory; ``nsplit = ndevice * nbatch`` slices
    then cover all occupied orbitals.
    """
    nocc_batch_max = int(np.floor(avail_bytes * 0.4 / (nvir * naux * nbytes)))
    if nocc_batch_max < 8:
        raise RuntimeError(
            f"GPU memory seems insufficient. Available (bytes): {avail_bytes}. "
            f"Required (bytes): {8 * 2 * nvir * naux * nbytes}."
        )
    return max(int(np.ceil(nocc / nocc_batch_max / ndevice)), 1)


# ---------------------------------------------------------------------------
# inter: off-diagonal pair energies (device cluster vs host-cluster halves)
# ---------------------------------------------------------------------------


def _inter_contraction_gpu(
    cderi_dev,
    occ_energy_dev,
    vir_energy,
    cderi_host_half_list,
    occ_energy_host_half_list,
    occ_idx_device_list,
    occ_idx_host_list,
    nocc_full,
    ss_only=False,
    device=None,
):
    r"""Off-diagonal (inter) pair contraction into on-device GPU tensors.

    Leaf torch contraction shared by :func:`get_dfmp2_energy_pair_inter` and the
    multi-GPU driver. The task
    bookkeeping (``build_inter_tasks``) is done by the caller; this function streams
    the host half-views onto the device one task at a time and runs the pair loop:

    ``g_ab = cderi_dev[i] @ cderi_host_half[j].t()``

    With ``ss_only=True`` (the same-spin mode of the unrestricted driver) the fold is
    the antisymmetrized same-spin energy ``(g_ab - g_ab.mT)^2 / D`` per (i, j),
    mirroring the ss branch of :func:`_intra_pair_gpu` (same ``MP2_FOLD``
    accumulation scheme as the os fold: elementwise in mat_dtype, f64 upcast only
    for ``MP2_FOLD=f64``), and a single ``eng_pair_ss`` tensor is returned instead
    of the bi-orthogonal pair.

    Like :func:`_intra_pair_gpu` it does NOT toggle TF32 and performs NO host sync --
    it returns GPU tensors so the caller can defer ``.cpu()`` to a single sync point.

    Per-task host->device upload (synchronous)
    ------------------------------------------
    Each host half-view is uploaded with ``_as_torch`` (``torch.as_tensor``) and
    immediately consumed by the default-stream pair loop - one task at a time, peak
    device memory ~1 cluster (cderi_dev) + 1 half-view, so the ``nbatch``
    auto-detection (1 cluster <= 0.4*avail) is valid. Each task's GPU half-view is
    freed (``del cderi_task``) at the end of its iteration. Very close to the memory
    wall, ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` can help if allocator
    fragmentation from the varying-size half-view uploads becomes noticeable.

    Uploads are not overlapped with compute via a non-blocking side stream: under
    torch, ``torch.as_tensor`` from the non-pinned host arrays is synchronous / host-blocking
    (verified by timing: the upload call itself blocks for the full H2D time, and a
    ``side.synchronize()`` then waits ~0), so a side stream is never actually used -
    uploads and computes both land on the default stream, serialized. Making it truly
    async would need pinned host buffers + ``non_blocking=True`` (verified:
    ``copy_(pinned, non_blocking=True)`` on a side stream is async), but the benefit is
    <1% for this compute-bound (fp64 GEMM) workload, so the complexity is not worth it.
    Default-stream serialization also means there is no reuse race to guard, so no
    cross-stream sync (``wait_event`` / ``record_stream``) is needed.

    Parameters
    ----------
    cderi_dev : torch.Tensor | np.ndarray
        Device's own cluster, shape (nocc_dev, nvir, naux). May be a CPU torch view
        (zero-copy from rust memory) or already on `device`; `_as_torch` uploads it once.
    occ_energy_dev : np.ndarray | torch.Tensor
        Occupied energies of the device's own cluster, shape (nocc_dev,).
    vir_energy : np.ndarray | torch.Tensor
        Virtual orbital energies, shape (nvir,).
    cderi_host_half_list : list of np.ndarray | torch.Tensor
        Per-task host half-views (built by ``build_inter_tasks``), each shape
        (nocc_half, nvir, naux), CPU-resident; uploaded one task at a time.
    occ_energy_host_half_list : list of np.ndarray | torch.Tensor
        Per-task host occ-energy half-views, each shape (nocc_half,).
    occ_idx_device_list : list of list[int]
        Per task: local device-cluster occupied indices paired with this host half.
    occ_idx_host_list : list of list[int]
        Per task: global occupied indices for this host half (output columns).
    nocc_full : int
        Total occupied count (column count of the resulting pair matrix).
    device : str | torch.device, optional

    Returns
    -------
    eng_pair_bi1, eng_pair_bi2 : torch.Tensor
        Pair energies of shape (nocc_dev, nocc_full), float64 on `device`. Only the
        off-diagonal columns addressed by the tasks are filled; the rest stay zero.
        When ``ss_only=True`` a single ``eng_pair_ss`` tensor is returned instead.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mat_dtype = torch.float64 if _is_f64(cderi_dev) else torch.float32
    acc_dtype = torch.float64

    cderi_dev = _as_torch(cderi_dev, mat_dtype, device)
    occ_energy_dev = _as_torch(occ_energy_dev, acc_dtype, device)
    vir_energy = _as_torch(vir_energy, acc_dtype, device)
    nocc_dev = occ_energy_dev.shape[0]

    ntask = len(cderi_host_half_list)
    if ss_only:
        eng_pair_ss = torch.zeros([nocc_dev, nocc_full], dtype=acc_dtype, device=device)
        if ntask == 0:
            return eng_pair_ss
    else:
        eng_pair_bi1 = torch.zeros([nocc_dev, nocc_full], dtype=acc_dtype, device=device)
        eng_pair_bi2 = torch.zeros([nocc_dev, nocc_full], dtype=acc_dtype, device=device)
        if ntask == 0:
            return eng_pair_bi1, eng_pair_bi2

    d_vv = -vir_energy[:, None] - vir_energy[None, :]  # (nvir, nvir), f64

    nvir = vir_energy.shape[0]
    naux = cderi_dev.shape[2]

    # fold precision / macro-batch (same env knobs as _intra_pair_gpu; default f32acc, B=16).
    _fold = os.environ.get("MP2_FOLD", "f32acc").lower()
    _batch = int(os.environ.get("MP2_BATCH", "16"))
    _batch = _batch if _batch > 0 else nocc_full
    if _fold != "f64":
        occ_energy_dev_m = occ_energy_dev.to(mat_dtype)
        d_vv_m = d_vv.to(mat_dtype)

    if _MP2_VERBOSE:
        torch.cuda.synchronize()
        _tc0 = time.perf_counter()

    # Synchronous per-task upload + compute, all on the default stream. H2D from
    # non-pinned host arrays is host-blocking, so uploads and GEMM alternate with the
    # GPU idle during each upload.  Pinned-memory overlap would hide this but
    # cudaHostAlloc of the ~3.5 GB half-views is ~3 s/GB (mlock), which is not worth
    # the one-off cost for the problem sizes here.
    for itask in range(ntask):
        cderi_task = _as_torch(cderi_host_half_list[itask], mat_dtype, device)
        occ_energy_task = _as_torch(occ_energy_host_half_list[itask], acc_dtype, device)
        occ_device_task = occ_idx_device_list[itask]
        occ_idx_task = occ_idx_host_list[itask]
        njj = len(occ_idx_task)
        idx_t = torch.as_tensor(occ_idx_task, dtype=torch.long, device=device)
        occ_energy_task_m = occ_energy_task.to(mat_dtype) if _fold != "f64" else None
        for i in occ_device_task:
            a_t = (
                cderi_dev[i : i + 1].transpose(-1, -2).contiguous().reshape(naux, nvir)
            )
            for j0 in range(0, njj, _batch):
                jb = min(_batch, njj - j0)
                # single GEMM (jb*nvir, naux) @ (naux, nvir) -> (jb*nvir, nvir);
                # reshape to (jb, nvir, nvir).  Same output size as a batched bmm but
                # one larger GEMM that saturates the MFMA unit far better (TF32 26->51%).
                b_2d = cderi_task[j0 : j0 + jb].reshape(jb * nvir, naux)
                g_ab = torch.matmul(b_2d, a_t).reshape(jb, nvir, nvir)
                jcols = idx_t[j0 : j0 + jb]
                if ss_only:
                    # antisymmetrized same-spin fold, mirroring the ss branch of
                    # _intra_pair_gpu (same MP2_FOLD accumulation scheme as the os
                    # fold: elementwise in mat_dtype, no f64 copy of g_ab); the
                    # double (a, b) sum is transpose-invariant, so the (host b,
                    # device a) GEMM orientation is immaterial here
                    g_as = g_ab - g_ab.mT
                    if _fold == "f64":
                        g_as = g_as.to(acc_dtype)  # upcast for accumulation
                        d_ab = (
                            occ_energy_dev[i]
                            + occ_energy_task[j0 : j0 + jb, None, None]
                            + d_vv[None]
                        )
                        t_as = g_as / d_ab
                        e_ss = (t_as * g_as).sum(dim=(1, 2))
                    else:
                        d_ab = (
                            occ_energy_dev_m[i]
                            + occ_energy_task_m[j0 : j0 + jb, None, None]
                            + d_vv_m[None]
                        )
                        t_as = g_as / d_ab
                        if _fold == "f32acc":
                            e_ss = (t_as * g_as).sum(dim=(1, 2), dtype=acc_dtype)
                        else:  # "f32"
                            e_ss = (t_as * g_as).sum(dim=(1, 2)).to(acc_dtype)
                    eng_pair_ss[i, jcols] = e_ss
                elif _fold == "f64":
                    g_ab = g_ab.to(acc_dtype)  # upcast for accumulation
                    d_ab = (
                        occ_energy_dev[i]
                        + occ_energy_task[j0 : j0 + jb, None, None]
                        + d_vv[None]
                    )
                    t_ab = g_ab / d_ab
                    e_bi1 = (t_ab * g_ab).sum(dim=(1, 2))
                    e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2))
                    eng_pair_bi1[i, jcols] = e_bi1
                    eng_pair_bi2[i, jcols] = e_bi2
                else:
                    d_ab = (
                        occ_energy_dev_m[i]
                        + occ_energy_task_m[j0 : j0 + jb, None, None]
                        + d_vv_m[None]
                    )
                    t_ab = g_ab / d_ab
                    if _fold == "f32acc":
                        e_bi1 = (t_ab * g_ab).sum(dim=(1, 2), dtype=acc_dtype)
                        e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2), dtype=acc_dtype)
                    else:  # "f32"
                        e_bi1 = (t_ab * g_ab).sum(dim=(1, 2)).to(acc_dtype)
                        e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2)).to(acc_dtype)
                    eng_pair_bi1[i, jcols] = e_bi1
                    eng_pair_bi2[i, jcols] = e_bi2

        # free this half-view before the next upload allocates (the loop-top rebind
        # would drop it only after the new tensor is already resident)
        del cderi_task, b_2d

    if _MP2_VERBOSE:
        torch.cuda.synchronize()
        _tc = time.perf_counter() - _tc0
        _ng = sum(
            len(d) * len(h) for d, h in zip(occ_idx_device_list, occ_idx_host_list)
        )
        _gflops = _ng * 2 * nvir * nvir * naux / _tc / 1e9
        _tag = "ss " if ss_only else ""
        print(
            f"[dfmp2] inter {_tag}compute: {_tc:.3f}s  {_gflops:.0f} GF/s "
            f"(synced pair loop, {_ng} pairs)",
            flush=True,
        )

    if ss_only:
        return eng_pair_ss
    return eng_pair_bi1, eng_pair_bi2


def _is_f64(x):
    """Whether `x` (torch tensor or numpy array) is float64."""
    if isinstance(x, torch.Tensor):
        return x.dtype == torch.float64
    return np.dtype(np.asarray(x).dtype) == np.float64


def get_dfmp2_energy_pair_inter(
    cderi_ovl,
    occ_energy,
    vir_energy,
    cderi_ovl_host_list,
    occ_energy_host_list,
    eval_mode_list,
    ss_only=False,
    device=None,
    use_tf32=False,
):
    r"""Obtain MP2 occupied orbital pair energies across occ clusters.

    Convenience wrapper: builds the inter tasks (``build_inter_tasks``) then delegates the torch
    contraction to :func:`_inter_contraction_gpu`, sets TF32, and pulls the result back
    to CPU as numpy. (The multi-GPU driver uses the leaner :func:`_inter_pair_gpu`
    leaf directly, which keeps results on-device and defers ``.cpu()`` to a single
    assembly sync point.)

    Index ``i`` runs over the *device's own* occ cluster (``cderi_ovl``, already on the
    device); index ``j`` runs over every *other* cluster, which lives in CPU memory
    (``cderi_ovl_host_list``) and is streamed onto the device one task at a time.

    With ``ss_only=True`` a single same-spin pair matrix ``eng_pair_ss`` is returned
    (see :func:`_inter_contraction_gpu`).

    Returns
    -------
    eng_pair_bi1, eng_pair_bi2 : np.ndarray
        Pair energies of shape (nocc, nocc_full), where nocc_full is the TOTAL occupied
        count. Only the off-diagonal columns (other clusters) are filled; the device's
        own columns stay zero (the diagonal block is filled by `intra`).
    """
    nocc = np.asarray(occ_energy).shape[0]
    (
        cderi_ovl_host_view_list,
        occ_energy_host_view_list,
        occ_idx_host_list,
        occ_idx_device_list,
        nocc_full,
    ) = build_inter_tasks(
        nocc, cderi_ovl_host_list, occ_energy_host_list, eval_mode_list
    )
    with with_tf32(use_tf32):
        out = _inter_contraction_gpu(
            cderi_ovl,
            occ_energy,
            vir_energy,
            cderi_ovl_host_view_list,
            occ_energy_host_view_list,
            occ_idx_device_list,
            occ_idx_host_list,
            nocc_full,
            ss_only=ss_only,
            device=device,
        )
    if ss_only:
        return out.cpu().numpy()
    return out[0].cpu().numpy(), out[1].cpu().numpy()


def _inter_pair_gpu(
    cderi_ovl,
    occ_energy,
    vir_energy,
    cderi_ovl_host_list,
    occ_energy_host_list,
    eval_mode_list,
    ss_only=False,
    device=None,
):
    r"""Inter pair energies into on-device GPU tensors (no TF32 toggle, no host sync).

    Driver-side leaf: builds the inter tasks (:func:`build_inter_tasks`) then delegates
    to :func:`_inter_contraction_gpu`. Returns GPU tensors so the driver can defer the
    ``.cpu()`` to the single assembly sync point.
    """
    nocc = np.asarray(occ_energy).shape[0]
    (
        cderi_ovl_host_view_list,
        occ_energy_host_view_list,
        occ_idx_host_list,
        occ_idx_device_list,
        nocc_full,
    ) = build_inter_tasks(
        nocc, cderi_ovl_host_list, occ_energy_host_list, eval_mode_list
    )
    return _inter_contraction_gpu(
        cderi_ovl,
        occ_energy,
        vir_energy,
        cderi_ovl_host_view_list,
        occ_energy_host_view_list,
        occ_idx_device_list,
        occ_idx_host_list,
        nocc_full,
        ss_only=ss_only,
        device=device,
    )


# ---------------------------------------------------------------------------
# unrestricted: alpha-beta opposite-spin pair energies + one-GPU UMP2 driver
# ---------------------------------------------------------------------------


def _uos_pair_gpu(
    cderi_ovl_a,
    cderi_ovl_b,
    occ_energy_a,
    occ_energy_b,
    vir_energy_a,
    vir_energy_b,
    device=None,
):
    r"""Alpha-beta opposite-spin pair energies into on-device GPU tensors.

    Leaf contraction of the unrestricted (UHF/UKS) block: for each alpha-occupied
    ``i`` and beta-occupied ``j`` over the *full* (non-triangular) occ range,

    ``g_ab = cderi_ovl_a[i] @ cderi_ovl_b[j].t()``

    with the cross-spin denominator
    ``D_ij^ab = e_i^a + e_j^b - e_a^a - e_b^b`` (no exchange term for the
    alpha-beta block, unlike the same-spin case). Like :func:`_intra_pair_gpu`
    it does NOT toggle TF32 (the caller owns the flag) and performs NO host
    sync; the matmul runs in ``cderi`` dtype, the accumulation is float64, and
    the j loop is macro-batched with one GEMM per chunk (same env knobs
    ``MP2_FOLD`` / ``MP2_BATCH``).

    Parameters
    ----------
    cderi_ovl_a, cderi_ovl_b : torch.Tensor | np.ndarray
        Spin-resolved Cholesky-decomposed 3c-2e ERI in MO basis, shapes
        (nocc_a, nvir_a, naux) / (nocc_b, nvir_b, naux); both must share dtype.
    occ_energy_a, occ_energy_b : np.ndarray | torch.Tensor
    vir_energy_a, vir_energy_b : np.ndarray | torch.Tensor
    device : str | torch.device, optional

    Returns
    -------
    eng_pair_os : torch.Tensor
        Opposite-spin pair energies of shape (nocc_a, nocc_b), float64 on
        `device` (rectangular: no i/j symmetry across spins).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mat_dtype = torch.float64 if _is_f64(cderi_ovl_a) else torch.float32
    assert _is_f64(cderi_ovl_b) == _is_f64(cderi_ovl_a), "cderi_ovl_a/b must share dtype"
    acc_dtype = torch.float64

    cderi_ovl_a = _as_torch(cderi_ovl_a, mat_dtype, device)
    cderi_ovl_b = _as_torch(cderi_ovl_b, mat_dtype, device)
    occ_energy_a = _as_torch(occ_energy_a, acc_dtype, device)
    occ_energy_b = _as_torch(occ_energy_b, acc_dtype, device)
    vir_energy_a = _as_torch(vir_energy_a, acc_dtype, device)
    vir_energy_b = _as_torch(vir_energy_b, acc_dtype, device)

    nocc_a = occ_energy_a.shape[0]
    nocc_b = occ_energy_b.shape[0]
    nvir_a = vir_energy_a.shape[0]
    nvir_b = vir_energy_b.shape[0]
    naux = cderi_ovl_a.shape[2]
    assert tuple(cderi_ovl_a.shape) == (nocc_a, nvir_a, naux)
    assert tuple(cderi_ovl_b.shape) == (nocc_b, nvir_b, naux)

    # empty occ on either spin or empty vir: no cross-spin excitations
    if nocc_a == 0 or nocc_b == 0 or nvir_a == 0 or nvir_b == 0:
        return torch.zeros([nocc_a, nocc_b], dtype=acc_dtype, device=device)

    # (nvir_b, nvir_a): row index b runs over beta virtuals, column a over alpha
    d_vv = -vir_energy_a[None, :] - vir_energy_b[:, None]

    # fold precision / macro-batch over j (beta occ): same env knobs as the intra
    # kernel; the (jb, nvir_b, nvir_a) GEMM output is bounded by the batch size.
    _fold = os.environ.get("MP2_FOLD", "f32acc").lower()
    _batch = int(os.environ.get("MP2_BATCH", "16"))
    _batch = _batch if _batch > 0 else nocc_b
    if _fold != "f64":
        occ_a_m = occ_energy_a.to(mat_dtype)
        occ_b_m = occ_energy_b.to(mat_dtype)
        d_vv_m = d_vv.to(mat_dtype)

    eng_pair_os = torch.zeros([nocc_a, nocc_b], dtype=acc_dtype, device=device)
    if _MP2_VERBOSE:
        torch.cuda.synchronize()
        _tc0 = time.perf_counter()
    for i in range(nocc_a):
        a_t = (
            cderi_ovl_a[i : i + 1].transpose(-1, -2).contiguous().reshape(naux, nvir_a)
        )
        for j0 in range(0, nocc_b, _batch):
            jb = min(_batch, nocc_b - j0)
            # g_ab (jb, nvir_b, nvir_a) = cderi_b[j0:j0+jb] @ cderi_a[i].mT; the
            # (jb*nvir_b, naux) @ (naux, nvir_a) single GEMM saturates better than
            # a per-j bmm (same trick as the intra/inter loops).
            b_2d = cderi_ovl_b[j0 : j0 + jb].reshape(jb * nvir_b, naux)
            g_ab = torch.matmul(b_2d, a_t).reshape(jb, nvir_b, nvir_a)
            if _fold == "f64":
                g_ab = g_ab.to(acc_dtype)  # upcast for accumulation
                d_ab = (
                    occ_energy_a[i]
                    + occ_energy_b[j0 : j0 + jb, None, None]
                    + d_vv[None]
                )
                t_ab = g_ab / d_ab
                e_os = (t_ab * g_ab).sum(dim=(1, 2))
            else:
                d_ab = (
                    occ_a_m[i] + occ_b_m[j0 : j0 + jb, None, None] + d_vv_m[None]
                )
                t_ab = g_ab / d_ab
                if _fold == "f32acc":
                    e_os = (t_ab * g_ab).sum(dim=(1, 2), dtype=acc_dtype)
                else:  # "f32"
                    e_os = (t_ab * g_ab).sum(dim=(1, 2)).to(acc_dtype)
            eng_pair_os[i, j0 : j0 + jb] = e_os
    if _MP2_VERBOSE:
        torch.cuda.synchronize()
        _tc = time.perf_counter() - _tc0
        _gflops = nocc_a * nocc_b * 2 * nvir_a * nvir_b * naux / _tc / 1e9
        print(
            f"[dfmp2] os compute: {_tc:.3f}s  {_gflops:.0f} GF/s "
            f"(uos pair loop, {nocc_a * nocc_b} pairs)",
            flush=True,
        )
    return eng_pair_os


def get_dfump2_energy_pair_intra(
    cderi_ovl_a,
    cderi_ovl_b,
    occ_energy_a,
    occ_energy_b,
    vir_energy_a,
    vir_energy_b,
    device=None,
    use_tf32=False,
):
    r"""Obtain the alpha-beta MP2 pair energies (PyTorch, GPU).

    Public wrapper around :func:`_uos_pair_gpu`: toggles TF32 for the float32
    matmul, runs the contraction and pulls the result back to CPU as numpy.
    Returns ``eng_pair_os`` of shape (nocc_a, nocc_b).
    """
    with with_tf32(use_tf32):
        out = _uos_pair_gpu(
            cderi_ovl_a,
            cderi_ovl_b,
            occ_energy_a,
            occ_energy_b,
            vir_energy_a,
            vir_energy_b,
            device=device,
        )
    return out.cpu().numpy()


def dfump2_kernel_one_gpu(
    cderi_ovl_a,
    cderi_ovl_b,
    occ_energy_a,
    occ_energy_b,
    vir_energy_a,
    vir_energy_b,
    device=None,
    use_tf32=False,
    verbose=False,
):
    r"""Single-GPU DF-UMP2 pair-energy kernel (aa/bb same-spin + ab opposite-spin).

    ``cderi_ovl_a``/``cderi_ovl_b`` are supplied directly (integral / ao2mo /
    cholesky front-end lives in the caller), shapes (nocc_s, nvir_s, naux) per
    spin. Both spins' cderi are uploaded to `device` once and stay resident
    while the three contractions run: aa and bb through the ss-only intra
    kernel (:func:`_intra_pair_gpu`), ab through :func:`_uos_pair_gpu`. The
    same-spin pair matrices follow the full-matrix (symmetric, all ab) sum
    convention, so the physical same-spin energy carries the 0.25 factor:

    ``e_corr_ss = 0.25 * (sum(pair_aa) + sum(pair_bb))``,
    ``e_corr_os = sum(pair_ab)``.

    Returns
    -------
    dict with the pair matrices ``e_corr_pair_aa`` (nocc_a, nocc_a),
    ``e_corr_pair_bb`` (nocc_b, nocc_b), ``e_corr_pair_ab`` (nocc_a, nocc_b)
    and the scalars ``e_corr_aa``/``e_corr_bb``/``e_corr_ab``/``e_corr_os``/
    ``e_corr_ss``/``e_corr``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mat_dtype = torch.float64 if _is_f64(cderi_ovl_a) else torch.float32
    assert _is_f64(cderi_ovl_b) == _is_f64(cderi_ovl_a), "cderi_ovl_a/b must share dtype"

    with with_tf32(use_tf32):
        # upload both spins' cderi once; the leaves' _as_torch on the resident
        # tensors is then a no-op, so each cderi crosses H2D exactly once
        cderi_a = _as_torch(cderi_ovl_a, mat_dtype, device)
        cderi_b = _as_torch(cderi_ovl_b, mat_dtype, device)
        if verbose:
            print(
                f"[dfump2] cderi upload: a {tuple(cderi_a.shape)}, "
                f"b {tuple(cderi_b.shape)}, dtype={mat_dtype}",
                flush=True,
            )
        pair_aa = _intra_pair_gpu(
            cderi_a, occ_energy_a, vir_energy_a, ss_only=True, device=device
        )
        pair_bb = _intra_pair_gpu(
            cderi_b, occ_energy_b, vir_energy_b, ss_only=True, device=device
        )
        pair_ab = _uos_pair_gpu(
            cderi_a,
            cderi_b,
            occ_energy_a,
            occ_energy_b,
            vir_energy_a,
            vir_energy_b,
            device=device,
        )

    e_corr_pair_aa = pair_aa.cpu().numpy()
    e_corr_pair_bb = pair_bb.cpu().numpy()
    e_corr_pair_ab = pair_ab.cpu().numpy()
    e_corr_aa = 0.25 * float(e_corr_pair_aa.sum())
    e_corr_bb = 0.25 * float(e_corr_pair_bb.sum())
    e_corr_ab = float(e_corr_pair_ab.sum())
    e_corr_os = e_corr_ab
    e_corr_ss = e_corr_aa + e_corr_bb
    return {
        "e_corr_pair_aa": e_corr_pair_aa,
        "e_corr_pair_bb": e_corr_pair_bb,
        "e_corr_pair_ab": e_corr_pair_ab,
        "e_corr_aa": e_corr_aa,
        "e_corr_bb": e_corr_bb,
        "e_corr_ab": e_corr_ab,
        "e_corr_os": e_corr_os,
        "e_corr_ss": e_corr_ss,
        "e_corr": e_corr_os + e_corr_ss,
    }


# ---------------------------------------------------------------------------
# multi-GPU UMP2 driver (no mol / aux; per-spin cderi supplied directly)
# ---------------------------------------------------------------------------


def nbatch_from_avail_ump2(
    nocc_a, nvir_a, nocc_b, nvir_b, naux, ndevice, nbytes, avail_bytes
):
    """Deduce `nbatch` for the unrestricted multi-GPU driver from a memory budget.

    The per-device peak is ~``max(1.5*bytes_a, 1.5*bytes_b, bytes_a + bytes_b) / nsplit``
    (same-spin own cluster + inter half-view, or the opposite-spin alpha + beta cluster
    pair), budgeted to 60% of the available memory -- the same effective share the RHF
    heuristic of :func:`nbatch_from_avail` leaves (own cluster 40% + half-view ~20%).
    ``nsplit = ndevice * nbatch`` then covers both spins' occupied orbitals.
    """
    for nocc, nvir in ((nocc_a, nvir_a), (nocc_b, nvir_b)):
        if nocc == 0:
            continue
        nocc_batch_max = int(np.floor(avail_bytes * 0.6 / (nvir * naux * nbytes)))
        if nocc_batch_max < 8:
            raise RuntimeError(
                f"GPU memory seems insufficient. Available (bytes): {avail_bytes}. "
                f"Required (bytes): {8 * nvir * naux * nbytes}."
            )
    bytes_a = nocc_a * nvir_a * naux * nbytes
    bytes_b = nocc_b * nvir_b * naux * nbytes
    peak_per_split = max(1.5 * bytes_a, 1.5 * bytes_b, bytes_a + bytes_b)
    nsplit_min = int(np.ceil(peak_per_split / (avail_bytes * 0.6)))
    return max(int(np.ceil(nsplit_min / ndevice)), 1)


def dfump2_kernel_multi_gpu_cderi_cpu(
    cderi_ovl_a,
    occ_energy_a,
    vir_energy_a,
    cderi_ovl_b,
    occ_energy_b,
    vir_energy_b,
    ndevice=None,
    nbatch=None,
    max_memory=None,
    use_tf32=False,
    verbose=False,
):
    r"""Multi-GPU DF-UMP2 pair-energy kernel from CPU-resident per-spin `cderi_ovl`.

    Unrestricted counterpart of :func:`dfmp2_kernel_multi_gpu_cderi_cpu`; returns the
    same dict as :func:`dfump2_kernel_one_gpu`. Each spin's occupied space is split
    into clusters (per-spin counts, see below) and the three contractions become:

    - aa / bb (same spin): the ss pair matrix is symmetric under i <-> j, so the RHF
      cluster machinery applies unchanged -- diagonal blocks through the ss-only intra
      kernel (:func:`_intra_pair_gpu`), off-diagonal blocks through the ss inter kernel
      (:func:`_inter_pair_gpu` with the half-splitting scheme that assigns each
      quarter-block to exactly one cluster), assembled with ``+= block`` /
      ``+= block.T``.
    - ab (opposite spin): a rectangular (nocc_a, nocc_b) pair matrix with NO i/j
      symmetry, so every (alpha cluster, beta cluster) block is computed exactly once
      by the alpha cluster's owner through :func:`_uos_pair_gpu` -- no half-splitting,
      no transpose assembly; simpler than the RHF inter scheme. The alpha cluster stays
      resident while the beta clusters are streamed one at a time (each beta upload is
      call-local to `_uos_pair_gpu`, so it self-frees at return).

    Per-spin cluster counts: ``nsplit_s = max(1, min(ndevice * nbatch, nocc_s // 4))``
    for a nonempty spin (clusters of >= 4 occupied orbitals, matching the RHF driver's
    balanced-split assert); an empty spin channel (fully spin-polarized reference) is
    skipped entirely. Unlike the RHF driver, small channels are CLAMPED to a single
    cluster rather than rejected -- UHF beta channels are commonly tiny (high-spin
    references), and a tiny channel's whole cderi is itself small. ``nbatch=None``
    auto-detects through :func:`nbatch_from_avail_ump2` (min free VRAM across devices,
    or `max_memory` MB, or host memory on CPU).

    Peak device memory is ~``max(1.5*|A|, 1.5*|B|, |A| + |B|) / nsplit`` + O(nvir^2)
    transients per device. Near the memory wall,
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (or a larger ``nbatch``) helps
    against caching-allocator fragmentation from the varying-size uploads.

    Multi-GPU dispatch mirrors the RHF driver: one worker thread per entry of
    ``ndevice`` (:func:`_par_map`), each pinned to its own GPU and owning the clusters
    whose position is ``pos mod ndevice``; per-cluster results stay on their device and
    are only pulled to CPU at the single assembly sync point. TF32 is process-global,
    so it is set once for the whole parallel region. On a single physical GPU the
    dispatch is serial (no point in two threads on one device); set
    ``DFMP2_FORCE_THREADS=1`` to force the threaded path for testing on one GPU (e.g.
    with ``ndevice=[0, 0]``).

    Returns
    -------
    dict with ``e_corr_pair_aa`` (nocc_a, nocc_a), ``e_corr_pair_bb`` (nocc_b, nocc_b),
    ``e_corr_pair_ab`` (nocc_a, nocc_b) and the scalars ``e_corr_aa``/``e_corr_bb``/
    ``e_corr_ab``/``e_corr_os``/``e_corr_ss``/``e_corr`` (full-matrix ss convention,
    0.25 same-spin factor), same as :func:`dfump2_kernel_one_gpu`.
    """
    cderi_np = [np.asarray(cderi_ovl_a), np.asarray(cderi_ovl_b)]
    occ = [np.asarray(occ_energy_a), np.asarray(occ_energy_b)]
    vir = [np.asarray(vir_energy_a), np.asarray(vir_energy_b)]
    nocc = [occ[0].shape[0], occ[1].shape[0]]
    nvir = [vir[0].shape[0], vir[1].shape[0]]
    naux = cderi_np[0].shape[2]
    assert cderi_np[0].shape == (nocc[0], nvir[0], naux), cderi_np[0].shape
    assert cderi_np[1].shape == (nocc[1], nvir[1], naux), cderi_np[1].shape
    mat_dtype = (
        torch.float64 if np.dtype(cderi_np[0].dtype) == np.float64 else torch.float32
    )
    assert (
        np.dtype(cderi_np[1].dtype) == np.dtype(cderi_np[0].dtype)
    ), "cderi_ovl_a/b must share dtype"

    # ---- device list (same handling as the RHF driver) ----
    if ndevice is None:
        ndevice = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if isinstance(ndevice, (list, tuple)):
        device_list = list(ndevice)
        ndevice = len(device_list)
    else:
        assert isinstance(ndevice, int) and ndevice >= 1
        device_list = list(range(ndevice))

    if nbatch is None:
        nbytes = 4 if mat_dtype == torch.float32 else 8
        if max_memory is not None:
            avail_bytes = int(max_memory * 1024 * 1024)
        elif torch.cuda.is_available():
            avail_bytes = min(torch.cuda.mem_get_info(d)[0] for d in device_list)
        else:
            avail_bytes = _cpu_avail_bytes()
        nbatch = nbatch_from_avail_ump2(
            nocc[0], nvir[0], nocc[1], nvir[1], naux, ndevice, nbytes, avail_bytes
        )

    nsplit = ndevice * nbatch
    # per-spin cluster counts (see docstring): >= 4 orbitals per cluster when possible;
    # empty or tiny channels collapse to 0 / 1 clusters instead of being rejected
    nsplit_s = [max(1, min(nsplit, nocc[s] // 4)) if nocc[s] > 0 else 0 for s in range(2)]
    if verbose:
        distinct_devs = len(set(device_list))
        force_threads = bool(os.environ.get("DFMP2_FORCE_THREADS"))
        print(
            f"[info] ump2 ndevice={ndevice} nbatch={nbatch} nsplit={nsplit} "
            f"nsplit_a={nsplit_s[0]} nsplit_b={nsplit_s[1]} nocc={nocc} dtype={mat_dtype} "
            f"distinct_devs={distinct_devs} force_threads={force_threads}",
            flush=True,
        )

    # ---- split each spin's cderi / occ energy into per-cluster CPU slices ----
    cderi_cpu_list = [[], []]
    occ_split = [[], []]
    occ_balanced = [
        balanced_split(nocc[s], nsplit_s[s]) if nocc[s] > 0 else [] for s in range(2)
    ]
    for s in range(2):
        i0 = 0
        for nb in occ_balanced[s]:
            i1 = i0 + nb
            cderi_cpu_list[s].append(np.ascontiguousarray(cderi_np[s][i0:i1]))
            occ_split[s].append(np.ascontiguousarray(occ[s][i0:i1]))
            i0 = i1

    def _ss_blocks(spin, idx, cderi_dev, dev):
        # same-spin diagonal + off-diagonal blocks of spin `spin`'s cluster `idx`
        # (own cluster `cderi_dev` already on `dev`)
        return (
            _intra_pair_gpu(
                cderi_dev, occ_split[spin][idx], vir[spin], ss_only=True, device=dev
            ),
            _inter_pair_gpu(
                cderi_dev,
                occ_split[spin][idx],
                vir[spin],
                cderi_cpu_list[spin],
                occ_split[spin],
                [None if i == idx else (i < idx) for i in range(nsplit_s[spin])],
                ss_only=True,
                device=dev,
            ),
        )

    def _worker(pos_dev):
        pos, dev_id = pos_dev
        dev = _device_str(dev_id)
        ctx = (
            torch.cuda.device(dev_id)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        out_aa, out_bb, out_ab = {}, {}, {}
        with ctx:
            # alpha clusters: own cderi uploaded once, shared by the aa blocks and the
            # ab row block (beta clusters streamed inside the loop)
            for p in range(pos, nsplit_s[0], ndevice):
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    _w0 = time.perf_counter()
                cderi_a_dev = _as_torch(cderi_cpu_list[0][p], mat_dtype, dev)
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    print(
                        f"[dev{dev_id}] alpha cluster {p} upload: "
                        f"{time.perf_counter() - _w0:.2f}s "
                        f"({cderi_cpu_list[0][p].nbytes / 1e9:.2f} GB)",
                        flush=True,
                    )
                    _w1 = time.perf_counter()
                out_aa[p] = _ss_blocks(0, p, cderi_a_dev, dev)
                if nsplit_s[1] > 0:
                    # rectangular ab row block: no i/j symmetry, each (p, q) block
                    # computed exactly once by the alpha cluster's owner
                    block = torch.zeros(
                        [occ_balanced[0][p], nocc[1]], dtype=torch.float64, device=dev
                    )
                    j0 = 0
                    for q in range(nsplit_s[1]):
                        j1 = j0 + occ_balanced[1][q]
                        block[:, j0:j1] = _uos_pair_gpu(
                            cderi_a_dev,
                            cderi_cpu_list[1][q],
                            occ_split[0][p],
                            occ_split[1][q],
                            vir[0],
                            vir[1],
                            device=dev,
                        )
                        j0 = j1
                    out_ab[p] = block
                # free before the next cluster's upload allocates (the loop-top rebind
                # would drop it only after the new tensor is already resident)
                del cderi_a_dev
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    print(
                        f"[dev{dev_id}] alpha cluster {p} aa+ab: "
                        f"{time.perf_counter() - _w1:.2f}s",
                        flush=True,
                    )
            # beta clusters: bb blocks only (the ab work is done from the alpha side)
            for q in range(pos, nsplit_s[1], ndevice):
                cderi_b_dev = _as_torch(cderi_cpu_list[1][q], mat_dtype, dev)
                out_bb[q] = _ss_blocks(1, q, cderi_b_dev, dev)
                del cderi_b_dev
        return out_aa, out_bb, out_ab

    # TF32 is process-global: set it once for the whole parallel region (the `_gpu`
    # leaves do not toggle it), so concurrent workers never race on the flag.
    with with_tf32(use_tf32):
        results = _par_map(_worker, enumerate(device_list))

    res_aa = [None] * nsplit_s[0]
    res_bb = [None] * nsplit_s[1]
    res_ab = [None] * nsplit_s[0]
    for out_aa, out_bb, out_ab in results:
        for p, v in out_aa.items():
            res_aa[p] = v
        for q, v in out_bb.items():
            res_bb[q] = v
        for p, v in out_ab.items():
            res_ab[p] = v

    # ---- assemble: pull to CPU here (the single host sync point per cluster) ----
    if verbose:
        _asm0 = time.perf_counter()
    e_corr_pair_aa = np.zeros([nocc[0], nocc[0]], dtype=np.float64)
    e_corr_pair_bb = np.zeros([nocc[1], nocc[1]], dtype=np.float64)
    e_corr_pair_ab = np.zeros([nocc[0], nocc[1]], dtype=np.float64)
    for s, pair in ((0, e_corr_pair_aa), (1, e_corr_pair_bb)):
        res = (res_aa, res_bb)[s]
        i0 = 0
        for idx, nocc_batch in enumerate(occ_balanced[s]):
            i1 = i0 + nocc_batch
            slc = slice(i0, i1)
            intra, inter = res[idx]
            pair[slc, slc] = intra.cpu().numpy()
            inter_np = inter.cpu().numpy()
            pair[slc, :] += inter_np
            pair[:, slc] += inter_np.T
            i0 = i1
    i0 = 0
    for p, nocc_batch in enumerate(occ_balanced[0]):
        i1 = i0 + nocc_batch
        if res_ab[p] is not None:
            e_corr_pair_ab[i0:i1, :] = res_ab[p].cpu().numpy()
        i0 = i1
    if verbose:
        print(
            f"[asm] ump2 assembly (.cpu + numpy): {time.perf_counter() - _asm0:.2f}s",
            flush=True,
        )

    e_corr_aa = 0.25 * float(e_corr_pair_aa.sum())
    e_corr_bb = 0.25 * float(e_corr_pair_bb.sum())
    e_corr_ab = float(e_corr_pair_ab.sum())
    e_corr_os = e_corr_ab
    e_corr_ss = e_corr_aa + e_corr_bb
    return {
        "e_corr_pair_aa": e_corr_pair_aa,
        "e_corr_pair_bb": e_corr_pair_bb,
        "e_corr_pair_ab": e_corr_pair_ab,
        "e_corr_aa": e_corr_aa,
        "e_corr_bb": e_corr_bb,
        "e_corr_ab": e_corr_ab,
        "e_corr_os": e_corr_os,
        "e_corr_ss": e_corr_ss,
        "e_corr": e_corr_os + e_corr_ss,
    }


# ---------------------------------------------------------------------------
# multi-GPU driver (no mol / aux; cderi_ovl supplied directly)
# ---------------------------------------------------------------------------


def _device_str(idx_device):
    return f"cuda:{idx_device}" if torch.cuda.is_available() else "cpu"


def _cpu_avail_bytes():
    """Available host (CPU) memory in bytes.

    Uses `psutil` if importable, else reads `MemAvailable` from `/proc/meminfo`
    (Linux). Raises `RuntimeError` if neither is available - no silent 8 GB pretense
    (the caller should pass an explicit `max_memory` if host-memory detection is not
    possible on this platform).
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    raise RuntimeError(
        "cannot detect host memory: install `psutil`, or run on Linux with /proc/meminfo "
        "(or pass an explicit `max_memory`)."
    )


def _default_nbatch(nocc, nvir, naux, ndevice, mat_dtype, device_list):
    """Deduce `nbatch` from the least-free device's available memory.

    On CUDA this is the min free memory across the devices (`torch.cuda.mem_get_info`);
    on CPU it is the real host available memory ([`_cpu_avail_bytes`]) - no 8 GB pretense.
    """
    nbytes = 4 if mat_dtype == torch.float32 else 8
    if torch.cuda.is_available():
        avail_bytes = min(torch.cuda.mem_get_info(d)[0] for d in device_list)
    else:
        avail_bytes = _cpu_avail_bytes()
    return nbatch_from_avail(nocc, nvir, naux, ndevice, nbytes, avail_bytes)


def _par_map(func, items):
    """Run ``func(item)`` for each item, in parallel across threads when worthwhile.

    One worker thread per item, each pinned to
    its own CUDA device by the caller (via ``torch.cuda.device(dev_id)`` inside `func`).
    The GIL is released around torch's CUDA kernel launches (matmul / elementwise /
    reductions), so distinct devices genuinely compute concurrently.

    Falls back to a plain serial loop when there is a single item or CUDA is
    unavailable, so the single-GPU / ``ndevice=[0,0]``-fake paths stay deterministic
    and pay no thread overhead.
    """
    items = list(items)
    if len(items) <= 1 or not torch.cuda.is_available():
        return [func(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        return list(ex.map(func, items))


def dfmp2_kernel_multi_gpu_cderi_cpu(
    cderi_ovl,
    occ_energy,
    vir_energy,
    ndevice=None,
    nbatch=None,
    max_memory=None,
    use_tf32=False,
    verbose=False,
):
    r"""Multi-GPU DF-MP2 pair-energy kernel from a CPU-resident `cderi_ovl`.

    ``cderi_ovl`` is supplied directly: the integral / ao2mo / cholesky front-end
    lives in the caller, so no ``mol`` / ``aux`` is needed inside the kernel.

    The occupied orbitals are split into ``nsplit = ndevice * nbatch`` balanced clusters.
    Each cluster's diagonal pair block is computed by :func:`get_dfmp2_energy_pair_intra`;
    its off-diagonal blocks against every other cluster are computed by
    :func:`get_dfmp2_energy_pair_inter` (with the half-splitting scheme that assigns each
    quarter-block to exactly one cluster). Results are assembled into the full
    (nocc, nocc) bi1 / bi2 pair matrices.

    Peak device memory is ~1 cluster upload + 1 inter half-view per batch. Near the
    memory wall, ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (or a larger
    ``nbatch``) helps against caching-allocator fragmentation from the varying-size
    half-view uploads.

    Multi-GPU dispatch: the per-device work
    (upload this cluster's cderi slice once, run intra then inter) is run concurrently
    on one worker thread per distinct device (:func:`_par_map`), each pinned to its own
    GPU. Pair-energy results are kept on the device and only pulled to CPU at the
    single assembly sync point, so a device that finishes early does not stall the
    others. TF32 is a process-global flag, so it is set once for the whole parallel
    region (the ``_gpu`` leaves do not toggle it). On a single physical GPU the dispatch
    is serial (no point in two threads on one device); set ``DFMP2_FORCE_THREADS=1`` to
    force the threaded path for testing on one GPU.

    Returns
    -------
    dict with ``e_corr_pair_bi1`` (nocc, nocc), ``e_corr_pair_bi2`` (nocc, nocc),
    ``e_corr_os``, ``e_corr_ss``, ``e_corr``.  With
    ``e_corr_os = sum(bi1)``, ``e_corr_ss = sum(bi1) - sum(bi2)``, ``e_corr = os + ss``.
    """
    cderi_ovl = np.asarray(cderi_ovl)
    occ_energy = np.asarray(occ_energy)
    vir_energy = np.asarray(vir_energy)
    nocc = occ_energy.shape[0]
    nvir = vir_energy.shape[0]
    naux = cderi_ovl.shape[2]
    assert cderi_ovl.shape == (nocc, nvir, naux), cderi_ovl.shape
    mat_dtype = (
        torch.float64 if np.dtype(cderi_ovl.dtype) == np.float64 else torch.float32
    )

    # ---- device list ----
    if ndevice is None:
        ndevice = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if isinstance(ndevice, (list, tuple)):
        device_list = list(ndevice)
        ndevice = len(device_list)
    else:
        assert isinstance(ndevice, int) and ndevice >= 1
        device_list = list(range(ndevice))

    # original guard: each occupied batch should hold >= 8 orbitals
    if nocc < 8 * ndevice:
        if nocc < 8 * 2:
            raise RuntimeError(
                f"nocc={nocc} too small for multi-GPU MP2; run the single-device kernel instead."
            )
        ndevice = nocc // 8
        device_list = device_list[:ndevice]
        if verbose:
            print(f"[warn] lowered ndevice to {ndevice} (nocc={nocc})")

    if nbatch is None:
        nbytes = 4 if mat_dtype == torch.float32 else 8
        if max_memory is not None:
            # explicit memory budget (MB) -> nbatch via the same heuristic
            nbatch = nbatch_from_avail(
                nocc, nvir, naux, ndevice, nbytes, int(max_memory * 1024 * 1024)
            )
        else:
            nbatch = _default_nbatch(nocc, nvir, naux, ndevice, mat_dtype, device_list)

    nsplit = ndevice * nbatch
    occ_balanced_split = balanced_split(nocc, nsplit)
    assert (
        min(occ_balanced_split) >= 4
    ), f"smallest occ batch {min(occ_balanced_split)} < 4; reduce nbatch (nsplit={nsplit}, nocc={nocc})"

    # ---- decide serial vs threaded dispatch ----
    # Thread only across DISTINCT physical devices: `ndevice=[0,0]` (fake multi on one
    # GPU) stays serial -- two threads on one device would only contend. `DFMP2_FORCE_THREADS`
    # overrides this so the threaded path can be exercised on a single-GPU test machine.
    distinct_devs = len(set(device_list))
    force_threads = bool(os.environ.get("DFMP2_FORCE_THREADS"))
    use_threads = torch.cuda.is_available() and (distinct_devs > 1 or force_threads)
    if verbose:
        print(
            f"[info] ndevice={ndevice} nbatch={nbatch} nsplit={nsplit} "
            f"occ_balanced_split={occ_balanced_split} dtype={mat_dtype} "
            f"dispatch={'threads' if use_threads else 'serial'} "
            f"(distinct_devs={distinct_devs}, force_threads={force_threads})"
        )

    # ---- split cderi / occ energy into per-cluster CPU slices ----
    cderi_ovl_cpu_list = []
    occ_energy_split = []
    i0 = 0
    for nb in occ_balanced_split:
        i1 = i0 + nb
        cderi_ovl_cpu_list.append(np.ascontiguousarray(cderi_ovl[i0:i1]))
        occ_energy_split.append(np.ascontiguousarray(occ_energy[i0:i1]))
        i0 = i1

    result_intra = [None] * nsplit
    result_inter = [None] * nsplit

    # `idx_split` is keyed by the POSITION in device_list (the chunk-assignment formula
    # `pos + idx_batch * ndevice` assumes pos in 0..ndevice-1); the actual device id only
    # selects which GPU runs the work. This lets a fake `device_list=[0, 0]` exercise
    # the ndevice>1 path on a single-GPU machine.
    def _worker(pos_dev):
        pos, dev_id = pos_dev
        dev = _device_str(dev_id)
        ctx = (
            torch.cuda.device(dev_id)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        out = {}
        with ctx:
            for idx_batch in range(nbatch):
                idx_split = pos + idx_batch * ndevice
                # move this cluster's cderi onto the device once; reused by intra + inter
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    _w0 = time.perf_counter()
                cderi_dev = _as_torch(cderi_ovl_cpu_list[idx_split], mat_dtype, dev)
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    print(
                        f"[dev{dev_id}] own cderi upload: {time.perf_counter() - _w0:.2f}s "
                        f"({cderi_ovl_cpu_list[idx_split].nbytes / 1e9:.2f} GB)",
                        flush=True,
                    )
                    _w1 = time.perf_counter()
                out[idx_split] = (
                    _intra_pair_gpu(
                        cderi_dev,
                        occ_energy_split[idx_split],
                        vir_energy,
                        ss_only=False,
                        device=dev,
                    ),
                    _inter_pair_gpu(
                        cderi_dev,
                        occ_energy_split[idx_split],
                        vir_energy,
                        cderi_ovl_cpu_list,
                        occ_energy_split,
                        [
                            None if i == idx_split else (i < idx_split)
                            for i in range(nsplit)
                        ],
                        device=dev,
                    ),
                )
                if verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev_id)
                    print(
                        f"[dev{dev_id}] intra+inter: {time.perf_counter() - _w1:.2f}s",
                        flush=True,
                    )
        return out

    # TF32 is process-global: set it once for the whole parallel region (the `_gpu`
    # leaves do not toggle it), so concurrent workers never race on the flag.
    with with_tf32(use_tf32):
        if use_threads:
            results = _par_map(_worker, enumerate(device_list))
        else:
            results = [_worker(it) for it in enumerate(device_list)]

    # collect per-cluster GPU tensors (still async on their devices)
    for out in results:
        for idx_split, (intra, inter) in out.items():
            result_intra[idx_split] = intra
            result_inter[idx_split] = inter

    # ---- assemble: pull to CPU here (the single host sync point per cluster) ----
    if verbose:
        _asm0 = time.perf_counter()
    e_corr_pair_bi1 = np.zeros([nocc, nocc], dtype=np.float64)
    e_corr_pair_bi2 = np.zeros([nocc, nocc], dtype=np.float64)
    i0 = 0
    for idx_split, nocc_batch in enumerate(occ_balanced_split):
        i1 = i0 + nocc_batch
        slc = slice(i0, i1)
        intra_bi1, intra_bi2 = result_intra[idx_split]
        inter_bi1, inter_bi2 = result_inter[idx_split]
        e_corr_pair_bi1[slc, slc] = intra_bi1.cpu().numpy()
        e_corr_pair_bi2[slc, slc] = intra_bi2.cpu().numpy()
        inter_bi1_np = inter_bi1.cpu().numpy()
        inter_bi2_np = inter_bi2.cpu().numpy()
        e_corr_pair_bi1[slc, :] += inter_bi1_np
        e_corr_pair_bi1[:, slc] += inter_bi1_np.T
        e_corr_pair_bi2[slc, :] += inter_bi2_np
        e_corr_pair_bi2[:, slc] += inter_bi2_np.T
        i0 = i1
    if verbose:
        print(
            f"[asm] assembly (.cpu + numpy): {time.perf_counter() - _asm0:.2f}s",
            flush=True,
        )

    e_corr_bi1 = float(e_corr_pair_bi1.sum())
    e_corr_bi2 = float(e_corr_pair_bi2.sum())
    e_corr_os = e_corr_bi1
    e_corr_ss = e_corr_bi1 - e_corr_bi2
    return {
        "e_corr_pair_bi1": e_corr_pair_bi1,
        "e_corr_pair_bi2": e_corr_pair_bi2,
        "e_corr_os": e_corr_os,
        "e_corr_ss": e_corr_ss,
        "e_corr": e_corr_os + e_corr_ss,
    }
