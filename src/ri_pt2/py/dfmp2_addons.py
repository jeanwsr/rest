import contextlib
import os
import time

_t_import = time.perf_counter()
import torch
import numpy as np

_MP2_VERBOSE = bool(os.environ.get("MP2_VERBOSE"))
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
            _nb = _i + 1
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
        _npairs = nocc * (nocc + 1) // 2
        _gflops = _npairs * 2 * nvir * nvir * naux / _tgv / 1e9
        print(
            f"[dfmp2] gemm-only: {_tgv:.3f}s  {_gflops:.0f} GF/s "
            f"({_npairs} pairs, macro-batch={_batch}, shape {nvir}x{naux}x{nvir})",
            flush=True,
        )

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
        if _fold != "f64":
            occ_m = occ_energy.to(mat_dtype)
            d_vv_m = d_vv.to(mat_dtype)
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
        for i in range(nocc):
            for j in range(i):
                g_ab = cderi_ovl[i] @ cderi_ovl[j].t()  # matmul in mat_dtype
                g_ab = g_ab.to(acc_dtype)
                g_ab = g_ab - g_ab.t()  # antisymmetrize
                d_ab = occ_energy[i] + occ_energy[j] + d_vv
                t_ab = g_ab / d_ab
                e_ab = torch.dot(t_ab.reshape(-1), g_ab.reshape(-1))
                eng_pair_ss[i, j] = e_ab
                eng_pair_ss[j, i] = e_ab
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
    the pair-energy accumulation is *always* done in float64. So when
    ``cderi_ovl`` is float32, only the matmul itself runs in reduced precision;
    ``g_ab`` is upcast to float64 before the denominator / amplitude / contraction.
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
    device=None,
):
    r"""Off-diagonal (inter) pair contraction into on-device GPU tensors.

    Leaf torch contraction shared by :func:`get_dfmp2_energy_pair_inter` and the
    multi-GPU driver. The task
    bookkeeping (``build_inter_tasks``) is done by the caller; this function streams
    the host half-views onto the device one task at a time and runs the pair loop:

    ``g_ab = cderi_dev[i] @ cderi_host_half[j].t()``

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
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mat_dtype = torch.float64 if _is_f64(cderi_dev) else torch.float32
    acc_dtype = torch.float64

    cderi_dev = _as_torch(cderi_dev, mat_dtype, device)
    occ_energy_dev = _as_torch(occ_energy_dev, acc_dtype, device)
    vir_energy = _as_torch(vir_energy, acc_dtype, device)
    nocc_dev = occ_energy_dev.shape[0]

    eng_pair_bi1 = torch.zeros([nocc_dev, nocc_full], dtype=acc_dtype, device=device)
    eng_pair_bi2 = torch.zeros([nocc_dev, nocc_full], dtype=acc_dtype, device=device)
    ntask = len(cderi_host_half_list)
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
                if _fold == "f64":
                    g_ab = g_ab.to(acc_dtype)  # upcast for accumulation
                    d_ab = (
                        occ_energy_dev[i]
                        + occ_energy_task[j0 : j0 + jb, None, None]
                        + d_vv[None]
                    )
                    t_ab = g_ab / d_ab
                    e_bi1 = (t_ab * g_ab).sum(dim=(1, 2))
                    e_bi2 = (t_ab.mT * g_ab).sum(dim=(1, 2))
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
        del cderi_task

    if _MP2_VERBOSE:
        torch.cuda.synchronize()
        _tc = time.perf_counter() - _tc0
        _ng = sum(
            len(d) * len(h) for d, h in zip(occ_idx_device_list, occ_idx_host_list)
        )
        _gflops = _ng * 2 * nvir * nvir * naux / _tc / 1e9
        print(
            f"[dfmp2] inter compute: {_tc:.3f}s  {_gflops:.0f} GF/s "
            f"(synced pair loop, {_ng} pairs)",
            flush=True,
        )

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
        bi1, bi2 = _inter_contraction_gpu(
            cderi_ovl,
            occ_energy,
            vir_energy,
            cderi_ovl_host_view_list,
            occ_energy_host_view_list,
            occ_idx_device_list,
            occ_idx_host_list,
            nocc_full,
            device=device,
        )
    return bi1.cpu().numpy(), bi2.cpu().numpy()


def _inter_pair_gpu(
    cderi_ovl,
    occ_energy,
    vir_energy,
    cderi_ovl_host_list,
    occ_energy_host_list,
    eval_mode_list,
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
        device=device,
    )


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
