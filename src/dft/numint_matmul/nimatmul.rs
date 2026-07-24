// Note on linkage:
// 
// This algorithm implementation strictly requires linkage of openblas.
// If mkl is also linked and precedence is given to mkl, the algorithm will be very slow due to
// threading conflict. Use `patchelf` to remove mkl linkage from the binary if necessary.

use super::prelude::*;

/// Numerical integration driver using matrix-multiplication.
///
/// Holds molecular coordinates, grid weights, and caches AO integral evaluations
/// to avoid recomputation across multiple density/XCPot evaluations.
#[derive(Clone, Debug)]
pub struct NIMatmul<'a> {
    pub cint: CInt,
    pub coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,

    /// Cache for computed AO values, keyed by derivative order (e.g., "deriv0", "deriv1", etc.).
    ///
    /// This cache is not designed to be modified by API caller in usual cases.
    pub cache_tensor: HashMap<String, TsrCow<'a, f64>>,

    /// Number of grid points to process in one chunk.
    ///
    /// Relations of size: full-grid > batch > chunk > per-grid = 1.
    ///
    /// This value is better set around KC of micro-kernel (256-512 for usual x86 server).
    /// Default to be 1536 (3-6 KCs).
    pub nchunk: usize,

    /// Number of grid points to process in one batch.
    ///
    /// Relations of size: full-grid > batch > chunk > per-grid = 1.
    ///
    /// Full grid requires `[ngrids, nao, ncomp]` AO tensor, which can be too large to fit in memory
    /// for big systems. That's why we need to split the grid into batches.
    ///
    /// This value is better set to a proper size, not exceeding available memory, and be multiple
    /// of `nchunk` for better performance.
    /// Default to be 1536 * 1 * nthreads. nthreads is determined at runtime by rayon.
    pub nbatch: usize,
}

impl<'a> NIMatmul<'a> {
    /// Creates a new instance with the given integral engine, grid coordinates, and weights.
    pub fn new(cint: &CInt, coords: &[[f64; 3]], weights: &[f64]) -> Self {
        assert!(coords.len() == weights.len(), "Number of coordinates must match number of weights");
        let nchunk = 1536;
        let nbatch = nchunk * 1 * rayon::current_num_threads();
        Self {
            cint: cint.clone(),
            coords: coords.to_vec(),
            weights: weights.to_vec(),
            cache_tensor: HashMap::new(),
            nchunk,
            nbatch,
        }
    }

    /// Clone everything, except the cached tensors.
    pub fn duplicate(&self) -> Self {
        Self {
            cint: self.cint.clone(),
            coords: self.coords.clone(),
            weights: self.weights.clone(),
            cache_tensor: HashMap::new(),
            nchunk: self.nchunk,
            nbatch: self.nbatch,
        }
    }

    /// Creates a new instance for a subset of the grid points, slicing from `start` to `end`
    /// (exclusive).
    ///
    /// Cached AO tensors for the full grid are sliced accordingly and cached for the batch.
    ///
    /// This is useful for processing the grid in batches to reduce memory usage.
    pub fn split_batch(&self, start: usize, end: usize) -> NIMatmul<'_> {
        assert!(start < end && end <= self.coords.len(), "Invalid batch range");
        let mut new = self.duplicate();
        new.coords = self.coords[start..end].to_vec();
        new.weights = self.weights[start..end].to_vec();
        // if AO cached for the full grid exists, slice and cache the AO for the batch
        let mut cached_tensors = HashMap::new();
        for keys in self.cache_tensor.keys() {
            if keys.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok()).is_some() {
                let ao_full = self.cache_tensor.get(keys).unwrap();
                let ao_batch = ao_full.i(start..end);
                cached_tensors.insert(keys.clone(), ao_batch.into_cow());
            }
        }
        new.cache_tensor = cached_tensors;
        new
    }

    /// Evaluates AO integrals for the given derivative order and returns as a tensor.
    ///
    /// The returned tensor has shape `[ngrids, nao, ncomp]` where `ncomp = AO_DERIV_DIM[deriv]`.
    pub fn prepare_ao(&self, deriv: usize) -> Tsr {
        let eval_name = format!("deriv{}", deriv);
        let (out, shape) = self.cint.eval_gto(&eval_name, &self.coords).into();
        let device = DeviceBLAS::default();
        rt::asarray((out, shape, &device))
    }

    /// Returns cached AO values for the given derivative order, computing and caching if needed.
    ///
    /// When a higher derivative order has already been cached, the required subset is returned
    /// directly without recomputation.
    pub fn get_cached_ao(&mut self, deriv: usize) -> TsrView<'_> {
        assert!(
            deriv < AO_DERIV_DIM.len(),
            "Derivative order {deriv} is too high, max supported is {}",
            AO_DERIV_DIM.len() - 1
        );
        // determine the maximum ao deriv that have already been computed and cached
        let filter_closure = |k: &String| k.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok());
        let max_cached_deriv = self.cache_tensor.keys().filter_map(filter_closure).max();
        // if the requested deriv is already cached, return it
        if let Some(max_deriv) = max_cached_deriv {
            if max_deriv >= deriv {
                let cache_key = format!("ao_deriv{}", max_deriv);
                return self.cache_tensor.get(&cache_key).unwrap().i((.., .., ..AO_DERIV_DIM[deriv]));
            }
        }

        // otherwise, compute and cache all missing ao deriv up to the requested one
        let key = format!("ao_deriv{}", deriv);
        self.cache_tensor.insert(key.clone(), self.prepare_ao(deriv).into_cow());
        self.cache_tensor.get(&key).unwrap().view()
    }

    /// Evaluates density from density matrices.
    ///
    /// # Parameters
    ///
    /// - `dm_list` : density matrices, each of shape `[nao, nao]`; one per set
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_dm(&mut self, dm_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for dm in dm_list {
            ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2-dim");
            ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension");
        }
        let nset = dm_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_dm_with_output(ao, dm_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from orbital coefficients where bra and ket are the same.
    ///
    /// # Parameters
    ///
    /// - `bra_list` : orbital coefficient matrices, each of shape `[nao, nocc_i]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_homogeneous_braket(&mut self, bra_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for bra in bra_list {
            ni_check_shape!(bra.ndim(), 2, "Each braket must be 2-dim");
            ni_check_shape!(bra.shape()[0], nao, "Bra's first dimension must match AO dimension");
        }
        let nset = bra_list.len();
        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_homogeneous_braket_with_output(ao, bra_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from one shared bra and multiple kets.
    ///
    /// # Parameters
    ///
    /// - `bra` : shared orbital coefficient matrix, shape `[nao, nocc]`
    /// - `ket_list` : orbital coefficient matrices, each of shape `[nao, nocc]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_one_bra_mult_ket(&mut self, bra: TsrView, ket_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        ni_check_shape!(bra.ndim(), 2, "Bra must be 2-dim");
        ni_check_shape!(bra.shape()[0], nao, "Bra first dimension must match AO dimension");
        let nocc = bra.shape()[1];
        for ket in ket_list {
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
            ni_check_shape!(ket.shape()[0], nao, "Ket first dimension must match AO dimension");
            ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra");
        }
        let nset = ket_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_one_bra_mult_ket_with_output(ao, bra, ket_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from multiple bra-ket pairs.
    ///
    /// # Parameters
    ///
    /// - `bra_list` : orbital coefficient matrices for bra, each of shape `[nao, nocc_i]`
    /// - `ket_list` : orbital coefficient matrices for ket, each of shape `[nao, nocc_i]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_mult_bra_mult_ket(
        &mut self,
        bra_list: &[TsrView],
        ket_list: &[TsrView],
        den_type: XCDenType,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
            ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim");
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
            ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension");
            ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension");
            ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match");
        }
        let nset = bra_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_mult_bra_mult_ket_with_output(ao, bra_list, ket_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates XC potential (1st order) with vxc_eff.
    ///
    /// # Parameters
    ///
    /// - `vxc_eff` : effective XC potential, shape `[ngrids, nvar]` for RKS, `[ngrids, nvar, 2]`
    ///   for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// XC potential, shape `[nao, nao]` for RKS, `[nao, nao, 2]` for UKS.
    pub fn make_vxc_pot_with_eff(&mut self, vxc_eff: TsrView, den_type: XCDenType, spin: XCSpin) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let mut out = rt::zeros(([nao, nao], &device));
                rks_vxc_pot_with_eff_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), nchunk);
                out
            },
            XCSpin::Polarized => {
                let mut out = rt::zeros(([nao, nao, 2], &device));
                uks_vxc_pot_with_eff_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), nchunk);
                out
            },
        }
    }

    /// Evaluates XC potential (2nd order) with fxc_eff.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]` for RKS, `[ngrids, nvar, 2,
    ///   nvar, 2]` for UKS
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]` for RKS, `[ngrids,
    ///   nvar, 2, nset]` for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// Second-order XC potential, shape `[nao, nao, nset]` for RKS, `[nao, nao, 2, nset]` for UKS.
    pub fn make_fxc_pot_with_eff(&mut self, fxc_eff: TsrView, rho1: TsrView, den_type: XCDenType, spin: XCSpin) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let nset = rho1.shape()[2];
                let mut out = rt::zeros(([nao, nao, nset], &device));
                rks_fxc_pot_with_eff_with_output(
                    den_type,
                    fxc_eff,
                    rho1,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
            XCSpin::Polarized => {
                let nset = rho1.shape()[3];
                let mut out = rt::zeros(([nao, nao, 2, nset], &device));
                uks_fxc_pot_with_eff_with_output(
                    den_type,
                    fxc_eff,
                    rho1,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
        }
    }

    /// Evaluates XC potential (2nd order) with fxc_eff, applying bra transformation.
    ///
    /// This function only works for RKS (spin-unpolarized case).
    ///
    /// Bra is usually the occupied orbital coefficient, which can lower the computational cost.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
    /// - `bra` : bra orbital coefficients, shape `[nao, nocc]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Bra-transformed XC potential, shape `[nao, nocc, nset]`.
    pub fn make_rks_fxc_pot_with_eff_bra_trans(
        &mut self,
        fxc_eff: TsrView,
        rho1: TsrView,
        bra: TsrView,
        den_type: XCDenType,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        let nset = rho1.shape()[2];
        let nocc = bra.shape()[1];
        let mut out = rt::zeros(([nao, nocc, nset], &device));
        rks_fxc_pot_with_eff_bra_trans_with_output(
            den_type,
            fxc_eff,
            rho1,
            ao,
            weights_tsr.view(),
            bra,
            out.view_mut(),
            nchunk,
        );
        out
    }

    /// Evaluates XC potential (2nd order) with fxc_eff, applying bra transformation.
    ///
    /// This function only works for UKS (spin-polarized case).
    ///
    /// Bra is usually the occupied orbital coefficient, which can lower the computational cost.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2]`
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset]`
    /// - `bra` : bra orbital coefficients, first shape `[nao, nocc_alpha]`, second shape `[nao,
    ///   nocc_beta]`
    /// - `den_type` : which density components to compute
    pub fn make_uks_fxc_pot_with_eff_bra_trans(
        &mut self,
        fxc_eff: TsrView,
        rho1: TsrView,
        bra: &[TsrView; 2],
        den_type: XCDenType,
    ) -> [Tsr; 2] {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        let nset = rho1.shape()[3];
        let nocc_alpha = bra[0].shape()[1];
        let nocc_beta = bra[1].shape()[1];
        let mut out_alpha = rt::zeros(([nao, nocc_alpha, nset], &device));
        let mut out_beta = rt::zeros(([nao, nocc_beta, nset], &device));
        uks_fxc_pot_with_eff_bra_trans_with_output(
            den_type,
            fxc_eff,
            rho1,
            ao,
            weights_tsr.view(),
            bra,
            &mut [out_alpha.view_mut(), out_beta.view_mut()],
            nchunk,
        );
        [out_alpha, out_beta]
    }

    /// Evaluates XC potential (3rd order) with kxc_eff.
    ///
    /// # Parameters
    ///
    /// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar, nvar]` for RKS, `[ngrids,
    ///   nvar, 2, nvar, 2, nvar, 2]` for UKS
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset1]` for RKS, `[ngrids,
    ///   nvar, 2, nset1]` for UKS
    /// - `rho2` : second-order density response, shape `[ngrids, nvar, nset2]` for RKS, `[ngrids,
    ///   nvar, 2, nset2]` for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// Third-order XC potential, shape `[nao, nao, nset1, nset2]` for RKS,
    /// `[nao, nao, 2, nset1, nset2]` for UKS.
    pub fn make_kxc_pot_with_eff(
        &mut self,
        kxc_eff: TsrView,
        rho1: TsrView,
        rho2: TsrView,
        den_type: XCDenType,
        spin: XCSpin,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let nset1 = rho1.shape()[2];
                let nset2 = rho2.shape()[2];
                let mut out = rt::zeros(([nao, nao, nset1, nset2], &device));
                rks_kxc_pot_with_eff_with_output(
                    den_type,
                    kxc_eff,
                    rho1,
                    rho2,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
            XCSpin::Polarized => {
                let nset1 = rho1.shape()[3];
                let nset2 = rho2.shape()[3];
                let mut out = rt::zeros(([nao, nao, 2, nset1, nset2], &device));
                uks_kxc_pot_with_eff_with_output(
                    den_type,
                    kxc_eff,
                    rho1,
                    rho2,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
        }
    }
}
