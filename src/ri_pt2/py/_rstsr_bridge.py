"""Zero-copy bridge from RSTSR (rust) tensors to torch tensors via raw C pointers.

A rust column-major tensor of shape ``(naux, nvir, nocc)`` and a row-major numpy/torch array
of shape ``(nocc, nvir, naux)`` share the *same* memory layout (the axis order is simply
reversed). So a foreign C pointer + the ``(nocc, nvir, naux)`` shape can be wrapped into a
read-only torch tensor with no copy: the only later copy is the unavoidable ``.to('cuda')``
GPU upload performed by ``get_dfmp2_energy_pair_intra``.

The caller (rust) must keep the backing memory alive for the lifetime of the returned tensor.
"""

import time

_t0 = time.perf_counter()
import ctypes

import numpy as np
import torch

print(f"[bridge] torch+numpy import: {time.perf_counter() - _t0:.3f}s", flush=True)

# Pre-warm the CUDA/HIP runtime + per-device contexts so the lazy init is not charged to the
# pair-energy kernel wall-clock. `torch.cuda.is_available()` is the heavyweight call here (it
# initializes the runtime + first context, ~10 s on this HIP build); the subsequent
# device_count() / per-device mem_get_info() are then near-free. Each step is timed so the cost
# can be attributed.
_ta = time.perf_counter()
_avail = torch.cuda.is_available()
print(f"[bridge] torch.cuda.is_available() = {_avail}: {time.perf_counter() - _ta:.3f}s", flush=True)
if _avail:
    _ndev = torch.cuda.device_count()
    _tw = time.perf_counter()
    for _d in range(_ndev):
        torch.cuda.mem_get_info(_d)
    print(
        f"[bridge] CUDA context warmup (mem_get_info x{_ndev}): {time.perf_counter() - _tw:.3f}s",
        flush=True,
    )

_CTYPE = {"f32": ctypes.c_float, "f64": ctypes.c_double}
_NPDTYPE = {"f32": np.float32, "f64": np.float64}


def torch_from_ptr(addr, shape, dtype):
    """Build a read-only torch CPU tensor viewing foreign (rust RSTSR) memory.

    Parameters
    ----------
    addr : int
        C data pointer of the rust tensor's contiguous storage (column-major).
    shape : sequence of int
        Row-major shape, e.g. ``(nocc, nvir, naux)``. This interprets the rust
        col-major ``(naux, nvir, nocc)`` buffer unchanged (reversed axis order).
    dtype : {"f32", "f64"}
        Element type of the foreign buffer; drives the matmul precision downstream.

    Returns
    -------
    torch.Tensor
        CPU tensor sharing memory with the rust buffer (no copy). Non-writable; only
        read by the pair-energy driver.
    """
    ct = _CTYPE[dtype]
    npdt = _NPDTYPE[dtype]
    ptr = ctypes.cast(int(addr), ctypes.POINTER(ct))
    arr = np.ctypeslib.as_array(ptr, shape=tuple(int(s) for s in shape))
    # check that the array is C-contiguous and has the correct dtype
    # if not, print a warning message
    if not arr.flags["C_CONTIGUOUS"] or arr.dtype != npdt:
        print(
            f"[WARNING] foreign array is not C-contiguous or has wrong dtype: "
            f"[WARNING] shape={arr.shape}, dtype={arr.dtype}, flags={arr.flags}",
            flush=True,
        )
    # already C-contiguous + matching dtype -> no-op; otherwise a contiguous, dtype-correct view
    arr = np.ascontiguousarray(arr, dtype=npdt)
    return torch.from_numpy(arr)
