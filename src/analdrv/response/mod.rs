//! Energy response to orbitals and density matrix.

// fock and response
pub mod trait_rresp;
pub mod trait_uresp;

// interface to REST SCF data (fock and response)
pub mod rresp_interface;
pub mod uresp_interface;

// generalized fock (restricted)
pub mod trait_rgfock;

// generalized fock interface and driver for double-hybrid type methods (restricted)
pub mod rgfock_hcore;
pub mod rgfock_interface;

/// Carrier of the SCF response object of either spin treatment.
///
/// Built once by the task loop ([`crate::analdrv::interface::analdrv_interface`], per the SCF
/// type) and passed by `&mut` to the property drivers (hessian, multipole), so that the response
/// preparation and cached intermediates are shared across tasks.
///
/// The restricted and unrestricted response types ([`RRespSCF`]/[`URespSCF`]) have different
/// signatures (spin arrays) and no unified driver exists, so every consumer extracts the variant
/// it supports through [`Self::expect_r_mut`]/[`Self::expect_u_mut`]; apart from these
/// variant-accessors this is a plain carrier without façade methods.
///
/// [`RRespSCF`]: rresp_interface::RRespSCF
/// [`URespSCF`]: uresp_interface::URespSCF
pub enum RespSCF<'a> {
    /// Restricted response object ([`rscf_resp_interface`](rresp_interface::rscf_resp_interface)).
    R(rresp_interface::RRespSCF<'a>),
    /// Unrestricted response object
    /// ([`uscf_resp_interface`](uresp_interface::uscf_resp_interface)).
    U(uresp_interface::URespSCF<'a>),
}

impl<'a> RespSCF<'a> {
    /// The restricted response object ([`RespSCF::R`]), for consumers supporting restricted
    /// treatments only; panics with `ctx` on the unrestricted variant.
    pub fn expect_r_mut(&mut self, ctx: &str) -> &mut rresp_interface::RRespSCF<'a> {
        match self {
            RespSCF::R(resp) => resp,
            RespSCF::U(_) => panic!("{ctx}: expected the restricted (R) response object variant"),
        }
    }

    /// The unrestricted response object ([`RespSCF::U`]), for consumers supporting unrestricted
    /// treatments only; panics with `ctx` on the restricted variant.
    pub fn expect_u_mut(&mut self, ctx: &str) -> &mut uresp_interface::URespSCF<'a> {
        match self {
            RespSCF::U(resp) => resp,
            RespSCF::R(_) => panic!("{ctx}: expected the unrestricted (U) response object variant"),
        }
    }
}
