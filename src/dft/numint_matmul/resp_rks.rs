//! Response (Fock and response matrix) implementation for the RKS numint-matmul XC component.
//!
//! The response object [`RRespKSNIMatmul`] holds the minimal state for the fock/response
//! functionality of the RKS numerical-integration XC contribution; the hessian object
//! [`RHessKSNIMatmul`] composes it through `Rc<RefCell<...>>` and delegates its [`RRespAPI`]
//! methods to it, so the same response object can be shared by the hessian driver and (future)
//! other molecular-property drivers.
//!
//! The two grids are separated by purpose: the fock path (`get_fock_rdm`/`get_fock_coeff`) always
//! evaluates on the common (large) grid `ni`, while the response path
//! (`make_response_preparation`/`get_response_bra`) uses the small response grid `ni_resp` when
//! one is attached. The returned fock is the DFT numerical-integration XC contribution only,
//! never the full SCF fock.
//!
//! Main-thread only (not `Send`/`Sync`).

use super::prelude::*;
use crate::analdrv::prelude::*;

use super::hess_rks::{eval_vxc_fxc_from_rho, get_rks_response_bra_batched, make_cpks_vxc_fxc, RHessKSNIMatmul};

/// Response (fock/response matrix) object for the RKS numint-matmul XC contribution.
pub struct RRespKSNIMatmul<'a> {
    /// List of `(scale, functional)` pairs of the XC functional.
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Common (large) numerical-integration grid, used by the fock path.
    pub ni: NIMatmul<'a>,
    /// Optional separate small grid for the response path; `None` reuses `ni`.
    pub ni_resp: Option<NIMatmul<'a>>,
    /// Print progress of the response evaluation.
    pub verbose: bool,
    /// Response intermediates: `mo_coeff [nao, nmo]` and `mo_occ [nmo]` from
    /// [`RRespAPI::make_response_preparation`], plus `cpks_vxc [ngrids, nvar]` and
    /// `cpks_fxc [ngrids, nvar, nvar]` on the response grid (injected by the hessian setup when
    /// the grids are shared, else recomputed during preparation).
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> RRespKSNIMatmul<'a> {
    /// Create a new RKS response object.
    ///
    /// # Parameters
    ///
    /// - `xc_func_list` : list of `(scale, functional)` pairs.
    /// - `ni` : numerical-integration driver over the common (large) grid.
    /// - `verbose` : print progress.
    pub fn new(xc_func_list: Vec<(f64, LibXCFunctional)>, ni: NIMatmul<'a>, verbose: bool) -> Self {
        Self { xc_func_list, ni, ni_resp: None, verbose, intmd: HashMap::new() }
    }

    /// Attach a dedicated small numerical-integration grid (`ni_resp`) for the response path.
    ///
    /// When set, the response (`get_response_bra`) is evaluated on this grid instead of the common
    /// (large) grid, and `cpks_vxc` / `cpks_fxc` are recomputed on it during
    /// [`make_response_preparation`](RRespAPI::make_response_preparation).
    pub fn set_ni_resp(mut self, ni_resp: NIMatmul<'a>) -> Self {
        self.ni_resp = Some(ni_resp);
        self
    }
}

impl<'a> AnalDrvBaseAPI for RRespKSNIMatmul<'a> {}

impl<'a> RRespAPI for RRespKSNIMatmul<'a> {
    /// Fock (the XC numint contribution only) from density matrix, on the common grid `ni`.
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr {
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());
        let rho = self.ni.make_rho_from_dm(&[rdm], den_type);
        let (vxc, _fxc) = eval_vxc_fxc_from_rho(&self.xc_func_list, rho.i((.., .., 0)));
        self.ni.make_vxc_pot_with_eff(vxc.view(), den_type, XCSpin::Unpolarized)
    }

    // note: `get_fock_coeff` keeps the trait default (dm route via `get_dm0_restricted`).

    fn get_response_rdm(&mut self, _rdm: TsrView) -> Tsr {
        unimplemented!("Response matrix (rdm form) is not implemented for RKS numint-matmul yet.")
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(ColMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.into_contig(ColMajor));

        // When a dedicated response grid is set, `cpks_vxc` / `cpks_fxc` were NOT stored during
        // `make_hessian_setup` (the fock grid's vxc/fxc live on a different grid and must not
        // be reused). Recompute them here on the response grid from the ground-state density, using
        // the lean [`make_cpks_vxc_fxc`] (no skeleton intermediates, minimal AO derivative order,
        // density formed from occupied MOs via a bra-ket contraction rather than a full dm0).
        if let Some(ni_resp) = self.ni_resp.as_mut() {
            let mo_coeff = self.intmd["mo_coeff"].view();
            let mo_occ = self.intmd["mo_occ"].view();
            let (vxc, fxc) = make_cpks_vxc_fxc(&self.xc_func_list, ni_resp, mo_coeff, mo_occ);
            self.intmd.insert("cpks_vxc".to_string(), vxc);
            self.intmd.insert("cpks_fxc".to_string(), fxc);
        }
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let ni_resp = self.ni_resp.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff = self.intmd.get("mo_coeff").unwrap();
        let mo_occ = self.intmd.get("mo_occ").unwrap();
        let fxc_eff = self.intmd.get("cpks_fxc").unwrap();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);

        let (resp, _timing) = get_rks_response_bra_batched(
            ni_resp,
            determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec()),
            fxc_eff.view(),
            bra,
            mocc.view(),
            self.verbose,
        );
        resp
    }
}

/// Fock/response delegation from the hessian object to its shared inner response object.
impl<'a> RRespAPI for RHessKSNIMatmul<'a> {
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr {
        self.resp.borrow_mut().get_fock_rdm(rdm)
    }

    fn get_fock_coeff(&mut self, mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
        self.resp.borrow_mut().get_fock_coeff(mo_coeff, mo_occ)
    }

    fn get_response_rdm(&mut self, rdm: TsrView) -> Tsr {
        self.resp.borrow_mut().get_response_rdm(rdm)
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.resp.borrow_mut().make_response_preparation(mo_coeff, mo_occ)
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        self.resp.borrow_mut().get_response_bra(bra)
    }
}
