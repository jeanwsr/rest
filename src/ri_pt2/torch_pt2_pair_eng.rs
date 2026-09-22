#![warn(unused)]
//! PyTorch-backed PT2 pair-energy engine (pyo3-embedded CPython).
//!
//! The pair-energy contraction is delegated to the torch kernel `get_dfmp2_energy_pair_intra`
//! / `dfmp2_kernel_multi_gpu_cderi_cpu` (restricted) or `dfump2_kernel_one_gpu` (unrestricted,
//! single device) in the vendored standalone python copy
//! [`py/dfmp2_addons.py`], running on a CUDA device
//! through an embedded CPython interpreter (pyo3). Everything before the contraction
//! (integral generation, j2c decomposition, ao2mo) stays in rust on the CPU, exactly as in
//! the [`pt2_pair_eng`](super::pt2_pair_eng) new-driver path.
//!
//! The two python sources are embedded into the binary at compile time
//! (`include_str!`) and materialized as python modules on first use
//! (`PyModule::from_code`, registered in `sys.modules`), so no runtime
//! file-system layout is assumed. The interpreter itself is lazily initialized
//! (`prepare_freethreaded_python`, idempotent); runs that never select
//! `engine = "torch"` never pay the torch import / CUDA-context warmup cost.
//!
//! `py/dfmp2_addons.py` and `py/_rstsr_bridge.py` are vendored from
//! `showcase-torch-mp2-pyo3@4f80e50` (`src/py/`; docstrings reworded locally,
//! code unchanged except a local `del cderi_task` in
//! `dfmp2_addons._inter_contraction_gpu`, freeing each task's GPU half-view
//! before the next upload allocates, and empty-occ/vir early returns in
//! `_intra_pair_gpu` mirroring the totality of the CPU pair kernels); they
//! keep both the single-device intra kernel and the multi-device intra+inter
//! driver, so future multi-GPU wiring needs no python-side changes. The
//! trailing *unrestricted* section of
//! `dfmp2_addons.py` (`_uos_pair_gpu` / `get_dfump2_energy_pair_intra` /
//! `dfump2_kernel_one_gpu`) is written locally for the UHF path of
//! [`evaluate_riupt2_eng_torch`], following the file's leaf/wrapper/driver
//! conventions.

use std::any::TypeId;

use num::FromPrimitive;
use pyo3::exceptions::PyRuntimeError;
// `Bound` is imported explicitly: both `pyo3::prelude` and `rstsr::prelude` glob-export
// a `Bound` name, and the pyo3 one is meant here.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rstsr::prelude::*;
use pyo3::Bound;

use rt::blas::{BlasFloat, LapackDriverAPI};

use crate::ri_jk;
use crate::ri_pt2::{PT2TorchFoldMode, PT2FPMode};
use crate::scf_io::SCF;
use crate::utilities::TimeRecords;
use crate::utilities::rstsr_util::*;

const RSTSR_BRIDGE_PY: &str = include_str!("py/_rstsr_bridge.py");
const DFMP2_ADDONS_PY: &str = include_str!("py/dfmp2_addons.py");

/// Same contract as [`super::pt2_pair_eng::evaluate_ript2_eng`], but the pair-energy
/// contraction runs on a CUDA device through the embedded-CPython torch kernel.
///
/// `occidx`/`viridx` follow the caller's frozen-core filtering (applied in
/// [`xdh_calculations`](super::xdh_calculations), same as the new-driver path). The
/// occupation weights are required to be canonical closed-shell (occ = 1, vir = 0
/// after the `/2` restricted-convention rescale): the python kernel does not
/// implement the fractional-occupation weights of the CPU kernel, so a violation is
/// a hard error rather than a silently different energy.
///
/// The `[ctrl.ri_pt2]` options `fp_mode` (`FP64`/`FP32`/`TF32`), `torch_devices`,
/// `torch_nbatch`, `torch_fold` and `torch_batch` are all honored; `torch_devices` with 2+ entries routes
/// through the multi-device intra+inter driver.
///
/// Returns `[eng_tot, eng_os, eng_ss]` with `eng_tot = eng_os + eng_ss`, where
/// `eng_os = sum(bi1)` and `eng_ss = sum(bi1) - sum(bi2)` (bi-orthogonal pair
/// decomposition), matching the pure-rust driver.
pub fn evaluate_ript2_eng_torch<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    occidx: [Option<&[usize]>; 2],
    viridx: [Option<&[usize]>; 2],
) -> [f64; 3]
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = DeviceBLAS::default();

    let mo_energy: Tsr<f64> = (&scf_data.eigenvalues[0]).to_rstsr(&device);
    let mo_occupation: Tsr<f64> = (&scf_data.occupation[0]).to_rstsr(&device);

    // apply occ and vir orbital indices (same convention as the CPU new-driver)
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo[0];
    let num_mo = mo_energy.size();

    let occ_list: Vec<usize> = occidx[0]
        .map(|x| x.to_vec())
        .unwrap_or_else(|| (idx_core..idx_lumo).collect());
    let vir_list: Vec<usize> = viridx[0]
        .map(|x| x.to_vec())
        .unwrap_or_else(|| (idx_lumo..num_mo).collect());

    if occ_list.is_empty() || vir_list.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let occ_energy = mo_energy.index_select(-1, &occ_list);
    let vir_energy = mo_energy.index_select(-1, &vir_list);
    // note occupation in restricted case should be divided by 2
    let occ_occupation = mo_occupation.index_select(-1, &occ_list) / 2;
    let vir_occupation = mo_occupation.index_select(-1, &vir_list) / 2;

    // The python kernel assumes canonical closed-shell occupations (its pair-energy
    // fold has no n_ij * n_ab weight factor). Anything else (smearing, fractional
    // occupations) would give a silently different energy than the CPU kernel.
    let occ_canonical = occ_occupation.to_vec().iter().all(|&x| (x - 1.0).abs() < 1e-8);
    let vir_canonical = vir_occupation.to_vec().iter().all(|&x| x.abs() < 1e-8);
    assert!(
        occ_canonical && vir_canonical,
        "ri_pt2 engine = \"torch\": the python DF-MP2 kernel supports canonical closed-shell \
         occupations only (occ = 1, vir = 0); fractional occupations (smearing/thermal) are \
         not implemented yet. Use engine = \"cpu\" for this system."
    );

    // copy the engine options out of the card (scf_data stays free for the ao2mo call)
    let ri_pt2_opt = &scf_data.mol.ctrl.ri_pt2;
    let devices = ri_pt2_opt.torch_devices.clone();
    let force_batch_inter = ri_pt2_opt.torch_force_batch_inter;
    let nbatch = ri_pt2_opt.torch_nbatch;
    let use_tf32 = matches!(ri_pt2_opt.fp_mode, PT2FPMode::TF32);
    let fold_str = match ri_pt2_opt.torch_fold {
        PT2TorchFoldMode::F32Acc => "f32acc",
        PT2TorchFoldMode::F64 => "f64",
        PT2TorchFoldMode::F32 => "f32",
    };
    let batch = ri_pt2_opt.torch_batch;
    let verbose = std::env::var("MP2_VERBOSE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);

    // device ids must be pairwise distinct: the list assigns physical GPUs, not work
    // slots. To split the occ space over one GPU (GPU-OOM remedy), use
    // torch_force_batch_inter instead of a duplicated id.
    {
        let mut seen = std::collections::HashSet::new();
        assert!(
            devices.iter().all(|d| seen.insert(d)),
            "ri_pt2 engine = \"torch\": torch_devices entries must be pairwise distinct \
             (logical CUDA device ids); got {devices:?}. To run the batched intra+inter \
             evaluation on a single GPU (e.g. for GPU-memory reasons), set \
             torch_force_batch_inter = true instead of repeating the device id."
        );
    }

    // lazy one-time initialization: embedded interpreter + torch import + CUDA context
    // warmup. Fail fast (before spending minutes on ao2mo for large systems) if the
    // torch runtime is unavailable.
    timerecords.new_item("torch_setup", "lazy init of embedded python + torch (first torch-engine call)");
    timerecords.count_start("torch_setup");
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        ensure_py_modules(py).map(|_| ())?;
        check_cuda_availability(py)
    })
    .unwrap_or_else(|e| panic!("ri_pt2 engine = \"torch\" initialization (pyo3) failed: {e}"));
    timerecords.count("torch_setup");

    // perform ao2mo (CPU; rimatr if present, else batched int3c2e + j2c solve);
    // times "ao2mo"/"j2c"/"j3c"/"decomp" internally, same as the CPU new-driver
    let cderi_xvo = ri_jk::obtain_cderi_xvo_restricted::<T>(
        scf_data, timerecords, None,
        Some(vir_list.as_slice()),
        Some(occ_list.as_slice()),
    );

    timerecords.count_start("c_r5dft");
    let (eng_os, eng_ss) = Python::with_gil(|py| -> PyResult<(f64, f64)> {
        let (bridge, addons) = ensure_py_modules(py)?;
        check_cuda_availability(py)?;
        set_kernel_env(py, fold_str, batch)?;

        // dtype string for the foreign buffer (f32 or f64 drives the torch matmul precision)
        let dtype = if TypeId::of::<T>() == TypeId::of::<f64>() { "f64" } else { "f32" };

        // rust col-major (naux, nvir, nocc) == python row-major (nocc, nvir, naux): same memory
        let naux = cderi_xvo.shape()[0];
        let nocc = occ_energy.size();
        let nvir = vir_energy.size();
        assert_eq!(cderi_xvo.shape(), &[naux, nvir, nocc], "cderi_xvo shape must be (naux, nvir, nocc)");

        // zero-copy: hand the rust col-major buffer to python as a read-only torch CPU view
        // (the reversed row-major view of the same memory). The view must be f-contiguous;
        // `obtain_cderi_xvo_restricted` always produces one, so a non-contig view is a caller
        // bug -> panic rather than silently materializing a (large) contiguous copy.
        assert!(cderi_xvo.f_contig(), "cderi_xvo must be column-major (f-contiguous)");
        let torch_from_ptr = bridge.getattr("torch_from_ptr")?;
        let cderi_addr: usize = cderi_xvo.raw().as_ptr().wrapping_add(cderi_xvo.offset()) as usize;
        let occ_owned: Tsr<f64> = occ_energy.into_contig(ColMajor);
        let vir_owned: Tsr<f64> = vir_energy.into_contig(ColMajor);
        let cderi_torch = torch_from_ptr.call1((cderi_addr, (nocc, nvir, naux), dtype))?;
        let occ_torch = torch_from_ptr.call1((occ_owned.raw().as_ptr() as usize, (nocc,), "f64"))?;
        let vir_torch = torch_from_ptr.call1((vir_owned.raw().as_ptr() as usize, (nvir,), "f64"))?;
        // occ_owned / vir_owned stay alive until the end of this scope, so their pointers
        // remain valid for the python calls below.

        // intra+inter (batched) evaluation runs when several devices are listed, or
        // when forced on a single device (GPU-memory remedy); otherwise the plain
        // whole-tensor single-device intra kernel
        if devices.len() > 1 || force_batch_inter {
            // batched intra + inter driver; `devices` are distinct logical CUDA ids
            // (after CUDA_VISIBLE_DEVICES filtering). With one device, the occ space
            // is split into `nbatch` clusters on that GPU (peak memory ~ 1/nbatch).
            let driver = addons.getattr("dfmp2_kernel_multi_gpu_cderi_cpu")?;
            let result = driver.call1((
                cderi_torch, occ_torch, vir_torch,
                Some(devices), nbatch, Option::<f64>::None,
                use_tf32, verbose,
            ))?;
            let os: f64 = result.get_item("e_corr_os")?.extract()?;
            let ss: f64 = result.get_item("e_corr_ss")?.extract()?;
            Ok((os, ss))
        } else {
            // single-device intra-pair contraction (device = "cuda", verified available above)
            let kernel = addons.getattr("get_dfmp2_energy_pair_intra")?;
            let result = kernel.call1((cderi_torch, occ_torch, vir_torch, false, "cuda", use_tf32))?;
            let bi1_sum = result.get_item(0)?.call_method0("sum")?.getattr("item")?.call0()?.extract::<f64>()?;
            let bi2_sum = result.get_item(1)?.call_method0("sum")?.getattr("item")?.call0()?.extract::<f64>()?;
            Ok((bi1_sum, bi1_sum - bi2_sum))
        }
    })
    .unwrap_or_else(|e| panic!("ri_pt2 engine = \"torch\" pair-energy contraction (pyo3) failed: {e}"));
    timerecords.count("c_r5dft");

    let eng_tot = eng_os + eng_ss;
    [eng_tot, eng_os, eng_ss]
}

/// Same contract as [`super::pt2_pair_eng::evaluate_riupt2_eng`], but the three
/// pair-energy contractions (αα / ββ same-spin through the ss-only intra kernel,
/// αβ opposite-spin through the unrestricted kernel) run on a CUDA device via
/// the embedded-CPython torch driver `dfump2_kernel_one_gpu`.
///
/// `occidx`/`viridx` are per-spin and follow the caller's frozen-core filtering
/// (applied in [`xdh_calculations`](super::xdh_calculations), same as the
/// new-driver path). The occupation weights are required to be canonical UHF
/// (occ = 1, vir = 0 per spin channel, NO restricted `/2` rescale): the python
/// kernel does not implement the fractional-occupation weights of the CPU
/// kernel, so a violation is a hard error rather than a silently different
/// energy.
///
/// Of the `[ctrl.ri_pt2]` torch options, the single-device path honors `fp_mode`
/// (`FP64`/`FP32`/`TF32`), `torch_fold` and `torch_batch`. The batched intra+inter
/// evaluation (`torch_devices` with 2+ entries or `torch_force_batch_inter`) is NOT
/// available for UHF yet — its python driver implements the closed-shell
/// bi-orthogonal fold only — so requesting it is a hard error rather than a
/// silently different energy.
///
/// Returns `[eng_tot, eng_os, eng_ss]` with `eng_tot = eng_os + eng_ss`, where
/// `eng_os = sum(pair_ab)` and `eng_ss = 0.25 * (sum(pair_aa) + sum(pair_bb))`
/// (the full-matrix ss convention), matching the pure-rust driver.
pub fn evaluate_riupt2_eng_torch<T>(
    scf_data: &SCF,
    timerecords: &mut TimeRecords,
    occidx: [Option<&[usize]>; 2],
    viridx: [Option<&[usize]>; 2],
) -> [f64; 3]
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    const A: usize = 0;
    const B: usize = 1;

    let device = DeviceBLAS::default();

    let mo_energy = scf_data.eigenvalues.as_slice().to_rstsr(&device);
    let mo_occupation = scf_data.occupation.as_slice().to_rstsr(&device);

    // apply occ and vir orbital indices per spin (same convention as the CPU
    // new-driver); when None, default to the conventional contiguous range
    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo;
    let num_mo = mo_energy.shape()[0];

    let occ_lists: [Vec<usize>; 2] = [A, B].map(|spin| {
        occidx[spin]
            .map(|x| x.to_vec())
            .unwrap_or_else(|| (idx_core..idx_lumo[spin]).collect())
    });
    let vir_lists: [Vec<usize>; 2] = [A, B].map(|spin| {
        viridx[spin]
            .map(|x| x.to_vec())
            .unwrap_or_else(|| (idx_lumo[spin]..num_mo).collect())
    });

    // slice each spin channel to 1D then index_select
    let occ_energy = [
        mo_energy.i((.., A)).index_select(-1, &occ_lists[A]),
        mo_energy.i((.., B)).index_select(-1, &occ_lists[B]),
    ];
    let vir_energy = [
        mo_energy.i((.., A)).index_select(-1, &vir_lists[A]),
        mo_energy.i((.., B)).index_select(-1, &vir_lists[B]),
    ];

    // The python kernel assumes canonical UHF occupations (its pair-energy fold
    // has no n_ij * n_ab weight factor). Anything else (smearing, fractional
    // occupations) would give a silently different energy than the CPU kernel.
    for spin in [A, B] {
        let occ_spin = mo_occupation.i((.., spin)).index_select(-1, &occ_lists[spin]);
        let vir_spin = mo_occupation.i((.., spin)).index_select(-1, &vir_lists[spin]);
        let occ_canonical = occ_spin.to_vec().iter().all(|&x| (x - 1.0).abs() < 1e-8);
        let vir_canonical = vir_spin.to_vec().iter().all(|&x| x.abs() < 1e-8);
        assert!(
            occ_canonical && vir_canonical,
            "ri_pt2 engine = \"torch\": the python DF-UMP2 kernel supports canonical UHF \
             occupations only (occ = 1, vir = 0 per spin channel); fractional occupations \
             (smearing/thermal) are not implemented yet. Use engine = \"cpu\" for this system."
        );
    }

    // copy the engine options out of the card (scf_data stays free for the ao2mo call)
    let ri_pt2_opt = &scf_data.mol.ctrl.ri_pt2;
    let devices = ri_pt2_opt.torch_devices.clone();
    let force_batch_inter = ri_pt2_opt.torch_force_batch_inter;
    let use_tf32 = matches!(ri_pt2_opt.fp_mode, PT2FPMode::TF32);
    let fold_str = match ri_pt2_opt.torch_fold {
        PT2TorchFoldMode::F32Acc => "f32acc",
        PT2TorchFoldMode::F64 => "f64",
        PT2TorchFoldMode::F32 => "f32",
    };
    let batch = ri_pt2_opt.torch_batch;
    let verbose = std::env::var("MP2_VERBOSE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);

    // batched intra+inter (multi-device or forced) implements the closed-shell
    // bi-orthogonal fold only; it cannot serve the unrestricted ss/os blocks
    assert!(
        devices.len() == 1 && !force_batch_inter,
        "ri_pt2 engine = \"torch\" with a UHF reference currently supports the single-device \
         kernel only (dfump2_kernel_one_gpu); torch_devices with 2+ entries and \
         torch_force_batch_inter are not yet available for UHF. Run with the default \
         torch_devices = [0] and torch_force_batch_inter = false."
    );

    // lazy one-time initialization: embedded interpreter + torch import + CUDA context
    // warmup. Fail fast (before spending minutes on ao2mo for large systems) if the
    // torch runtime is unavailable.
    timerecords.new_item("torch_setup", "lazy init of embedded python + torch (first torch-engine call)");
    timerecords.count_start("torch_setup");
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        ensure_py_modules(py).map(|_| ())?;
        check_cuda_availability(py)
    })
    .unwrap_or_else(|e| panic!("ri_pt2 engine = \"torch\" initialization (pyo3) failed: {e}"));
    timerecords.count("torch_setup");

    // perform ao2mo (CPU; rimatr if present, else batched int3c2e + j2c solve);
    // times "ao2mo"/"j2c"/"j3c"/"decomp" internally, same as the CPU new-driver
    let cderi_xvo = ri_jk::obtain_cderi_xvo_unrestricted::<T>(
        scf_data, timerecords, None,
        [Some(vir_lists[A].as_slice()), Some(vir_lists[B].as_slice())],
        [Some(occ_lists[A].as_slice()), Some(occ_lists[B].as_slice())],
    );

    timerecords.count_start("c_r5dft");
    let (eng_os, eng_ss) = Python::with_gil(|py| -> PyResult<(f64, f64)> {
        let (bridge, addons) = ensure_py_modules(py)?;
        check_cuda_availability(py)?;
        set_kernel_env(py, fold_str, batch)?;

        // dtype string for the foreign buffer (f32 or f64 drives the torch matmul precision)
        let dtype = if TypeId::of::<T>() == TypeId::of::<f64>() { "f64" } else { "f32" };

        // rust col-major (naux, nvir_s, nocc_s) == python row-major (nocc_s, nvir_s, naux)
        // per spin: same memory, reversed row-major view (see evaluate_ript2_eng_torch)
        let naux = cderi_xvo[A].shape()[0];
        let nocc = [occ_energy[A].size(), occ_energy[B].size()];
        let nvir = [vir_energy[A].size(), vir_energy[B].size()];
        assert_eq!(cderi_xvo[A].shape(), &[naux, nvir[A], nocc[A]], "cderi_xvo[A] shape must be (naux, nvir, nocc)");
        assert_eq!(cderi_xvo[B].shape(), &[naux, nvir[B], nocc[B]], "cderi_xvo[B] shape must be (naux, nvir, nocc)");
        assert!(cderi_xvo[A].f_contig() && cderi_xvo[B].f_contig(), "cderi_xvo must be column-major (f-contiguous)");

        let torch_from_ptr = bridge.getattr("torch_from_ptr")?;
        // index_select outputs are consumed here (moved into contiguous copies
        // whose pointers are handed to python); cderi_xvo stays owned above
        let [occ_energy_a, occ_energy_b] = occ_energy;
        let [vir_energy_a, vir_energy_b] = vir_energy;
        let occ_owned: [Tsr<f64>; 2] = [
            occ_energy_a.into_contig(ColMajor),
            occ_energy_b.into_contig(ColMajor),
        ];
        let vir_owned: [Tsr<f64>; 2] = [
            vir_energy_a.into_contig(ColMajor),
            vir_energy_b.into_contig(ColMajor),
        ];
        let cderi_addr = |spin: usize| {
            cderi_xvo[spin].raw().as_ptr().wrapping_add(cderi_xvo[spin].offset()) as usize
        };
        let cderi_torch = [
            torch_from_ptr.call1((cderi_addr(A), (nocc[A], nvir[A], naux), dtype))?,
            torch_from_ptr.call1((cderi_addr(B), (nocc[B], nvir[B], naux), dtype))?,
        ];
        let occ_torch = [
            torch_from_ptr.call1((occ_owned[A].raw().as_ptr() as usize, (nocc[A],), "f64"))?,
            torch_from_ptr.call1((occ_owned[B].raw().as_ptr() as usize, (nocc[B],), "f64"))?,
        ];
        let vir_torch = [
            torch_from_ptr.call1((vir_owned[A].raw().as_ptr() as usize, (nvir[A],), "f64"))?,
            torch_from_ptr.call1((vir_owned[B].raw().as_ptr() as usize, (nvir[B],), "f64"))?,
        ];
        // occ_owned / vir_owned stay alive until the end of this scope, so their pointers
        // remain valid for the python calls below.

        // single-device UMP2 driver: uploads both spins' cderi once, runs the
        // aa/bb ss-only intra + ab opposite-spin contractions
        let driver = addons.getattr("dfump2_kernel_one_gpu")?;
        let result = driver.call1((
            cderi_torch[A].clone(), cderi_torch[B].clone(),
            occ_torch[A].clone(), occ_torch[B].clone(),
            vir_torch[A].clone(), vir_torch[B].clone(),
            "cuda", use_tf32, verbose,
        ))?;
        let os: f64 = result.get_item("e_corr_os")?.extract()?;
        let ss: f64 = result.get_item("e_corr_ss")?.extract()?;
        Ok((os, ss))
    })
    .unwrap_or_else(|e| panic!("ri_pt2 engine = \"torch\" UMP2 pair-energy contraction (pyo3) failed: {e}"));
    timerecords.count("c_r5dft");

    let eng_tot = eng_os + eng_ss;
    [eng_tot, eng_os, eng_ss]
}

/// Fetch (or lazily materialize) the vendored python modules `_rstsr_bridge` and
/// `dfmp2_addons` from the embedded sources.
///
/// Both modules are created once per process via `PyModule::from_code` and registered
/// in `sys.modules`; subsequent calls fetch them from there (so the module-level
/// torch/CUDA warmup in `_rstsr_bridge` runs exactly once). `import torch` is
/// attempted first and mapped to an error message pointing at the required
/// environment.
fn ensure_py_modules(py: Python<'_>) -> PyResult<(Bound<'_, PyModule>, Bound<'_, PyModule>)> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?.downcast_into::<PyDict>()?;
    if let Some(addons) = modules.get_item("dfmp2_addons")? {
        let bridge = modules.get_item("_rstsr_bridge")?.ok_or_else(|| {
            PyRuntimeError::new_err("dfmp2_addons found in sys.modules but _rstsr_bridge is missing")
        })?;
        return Ok((bridge.downcast_into::<PyModule>()?, addons.downcast_into::<PyModule>()?));
    }
    if let Err(e) = py.import("torch") {
        return Err(PyRuntimeError::new_err(format!(
            "failed to import the `torch` python package: {e}\n\
             engine = \"torch\" requires a python environment with torch installed \
             (conda env `rest-torch`: torch with CUDA support); make sure the embedded \
             interpreter resolves to it (build-time PYO3_PYTHON, runtime PYTHONHOME)."
        )));
    }
    // `from_code` takes the source as `&CStr` too (pyo3 0.24); the embedded sources
    // never contain NUL bytes, so the CString conversion cannot fail.
    let bridge_src = std::ffi::CString::new(RSTSR_BRIDGE_PY).expect("embedded _rstsr_bridge.py contains NUL");
    let addons_src = std::ffi::CString::new(DFMP2_ADDONS_PY).expect("embedded dfmp2_addons.py contains NUL");
    let bridge = PyModule::from_code(py, &bridge_src, c"_rstsr_bridge.py", c"_rstsr_bridge")?;
    modules.set_item("_rstsr_bridge", &bridge)?;
    let addons = PyModule::from_code(py, &addons_src, c"dfmp2_addons.py", c"dfmp2_addons")?;
    modules.set_item("dfmp2_addons", &addons)?;
    Ok((bridge, addons))
}

/// Hard-require a CUDA device: `engine = "torch"` does not silently fall back to
/// torch-CPU (that would look like a success while being slower than the rust CPU
/// engine); use `engine = "cpu"` instead.
fn check_cuda_availability(py: Python<'_>) -> PyResult<()> {
    let torch_mod = py.import("torch")?;
    let available: bool = torch_mod.getattr("cuda")?.call_method0("is_available")?.extract()?;
    if !available {
        return Err(PyRuntimeError::new_err(
            "torch reports no available CUDA device (torch.cuda.is_available() == false); \
             engine = \"torch\" hard-requires a CUDA device and does not fall back to \
             torch-CPU. Use engine = \"cpu\" (default) for CPU contraction, or check the \
             CUDA_VISIBLE_DEVICES environment / torch installation.",
        ));
    }
    Ok(())
}

/// Publish the card-controlled kernel knobs (`[ctrl.ri_pt2].fold` / `.batch`) to the
/// python kernel's env interface (`MP2_FOLD` / `MP2_BATCH`, read at call time). The
/// card is authoritative: a value set here overrides the same variable in the shell
/// environment. (`MP2_VERBOSE` stays a pure environment knob and is read by the
/// kernel itself.)
fn set_kernel_env(py: Python<'_>, fold: &str, batch: usize) -> PyResult<()> {
    let environ = py.import("os")?.getattr("environ")?;
    environ.set_item("MP2_FOLD", fold)?;
    // `os.environ` only accepts str values
    environ.set_item("MP2_BATCH", batch.to_string())?;
    Ok(())
}
