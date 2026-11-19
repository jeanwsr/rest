//! Analytic nuclear gradients of GW quasiparticle energies and GW-BSE
//! excited states.
//!
//! The differentiated self-energy is selected by [`GwGradConfig::gw_variant`]:
//!
//! * [`GwVariant::Cd`] / [`GwVariant::Fscd`] — the full-CD G0W0 functional,
//!   implemented by [`GwCdGradEngine`]: static-subtracted continuous
//!   self-energy, direct `naux x naux` screening factorisation (no low-rank
//!   truncation) and exact moving residue frequencies.  The former low-rank
//!   CD derivative driver has been removed; CD always uses this engine.
//! * [`GwVariant::Ac`] — analytic continuation of `Sigma_c` on the imaginary
//!   axis (Thiele Padé), differentiated through the low-rank QP cache
//!   ([`GwGradEngine`]).
//! * [`GwVariant::Wac`] — CD-WAC: the CD imaginary-axis integral with a
//!   Padé-continued screened interaction `W_c,nm(z)`, also through the
//!   low-rank QP cache.
//!
//! The low-rank QP-cache machinery is retained because AC/WAC and the BSE
//! gradient need it; only the LR-CD *gradient* assembly was removed.
//!
//! Shared infrastructure used by both engines:
//!
//! * raw RI tensors `q[P,pq] = (P|pq)` and the metric `J[P,Q] = (P|Q)`;
//! * the low-rank response factors used by AC/WAC and by BSE screening;
//! * canonical CP-HF/CP-KS responses (`analdrv` + RI-JK response) and the
//!   adjoint (Z-vector) contraction of the CP-HF channel;
//! * pair packing and the QP pullback shared with the BSE gradient.
//!
//! The full-CD engine differentiates the static-subtracted functional
//!
//!     Sigma_sub = sum_m [ -1/pi sum_k w_k K(t_m, u_k) (W_m(iu_k) - W_m(0))
//!                         + (1/2 - f_m) W_m(0)
//!                         + s_m pf_m (W_m(zeta_m) - W_m(0)) ],
//!
//! whose subtraction removes the artificial pole-selection jump of the raw
//! decomposition at `omega = e_m`, making the QP energy and its gradient well
//! defined even when a QP energy coincides with a mean-field pole.
use crate::ctrl_io::quasiparticle_methods::GwVariant;
use crate::utilities::linalg::{gemm_nn, gemm_nt, to_mat};
use crate::ri_gw::ac_grad::{pade_derivatives, pade_derivatives_greedy, pade_derivatives_nodes};
use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::ri_rpa::{gauss_legendre_grids, logarithmic_grid, trans_gauss_legendre_grids};
use crate::scf_io::SCF;
use rest_libcint::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dpotrf, _dsolve, _dsyev_inplace, _dsyevd, _dtrtrs,
};
use tensors::{MathMatrix, MatrixFull, MatrixFullSlice};
use num_complex::Complex;

use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};
use std::os::raw::c_char;
use std::time::Instant;

use crate::ri_bse::bse_grad::grad_timing;

/// Fixed work-block size (auxiliary functions) of the parallel perturbation
/// contractions.
///
/// The batches are laid out from the **auxiliary-shell index** alone — never
/// from `rayon::current_num_threads()` — so the per-batch partial sums and
/// their merge order are the same for every thread count and a gradient stays
/// bit-for-bit reproducible under `num_threads`, the same discipline as
/// [`crate::utilities::par_sum`].  Each batch holds at least this many
/// auxiliary functions (one whole shell more at most), which keeps the
/// per-batch integral calls few.
const PERT_BLK: usize = 64;

/// Fixed work-block size (auxiliary functions) of the parallel `bq` fold.
/// Clear a lazily-filled cache buffer, skipping (instead of panicking) when a
/// reader still holds it.
///
/// Every `evict_*` on [`GwGradEngine`] is a **pure memory optimisation**: the
/// matching `ensure_*` rebuilds the buffer on demand, so a skipped release
/// cannot change any result — it only leaves the buffer resident a little
/// longer.  Using `try_borrow_mut` here removes a `RefCell already borrowed`
/// panic that a concurrent reader (a rayon worker still finishing a sweep)
/// could otherwise trigger, while keeping the peak-memory win in the common
/// case.

include!("prelude.rs");
include!("config.rs");
include!("raw.rs");
include!("math.rs");
include!("lr_factor.rs");
include!("qp_cache.rs");
include!("ac_wac_grad.rs");
include!("gradient.rs");
include!("utils.rs");
include!("cd_grad.rs");
include!("adjoint_cpks.rs");
include!("tests.rs");
