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
/// This is a plain carrier with no façade methods: the restricted and unrestricted response
/// types ([`RRespSCF`]/[`URespSCF`]) have different signatures (spin arrays) and no unified driver
/// exists, so every consumer matches on the variant it supports.
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
