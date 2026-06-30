//! RKS-unique analytic Hessian contributions.
//!
//! Free functions that scatter directly into existing buffers passed by &mut
//! slice — zero new allocation. The XC computational kernels themselves live
//! in xc_hessian.rs and are NOT duplicated here.

use crate::scf_io::SCF;
use crate::hessian::memory_monitor;

/// Add the RKS XC contribution (vxc_diag + vxc_deriv2) to the electronic
/// Hessian partial `h_partial` (column-major, n3*n3, modified in place).
///
/// `h_partial` is the existing buffer owned by the caller (the live
/// `result["h_partial"].data` slice); this function writes into it directly
/// using the same column-major indexing as the inline block it replaced:
///   `h_partial[(ia*3+a)*n3 + (ja*3+b)]`.
///
/// `timings` receives the per-stage `(label, elapsed)` pairs the caller
/// records for its timing profile.
pub fn add_vxc_h_partial(
    scf: &SCF,
    h_partial: &mut [f64],
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) {
    let _t_rks_xc = std::time::Instant::now();
    let mol_rks = &scf.mol;
    let nao_xc = mol_rks.num_basis;
    let natm_xc = mol_rks.geom.nfree;
    let n3_xc = natm_xc * 3;
    let xc_type = if mol_rks.xc_data.use_density_gradient() {
        crate::dft::xc_deriv::XCType::GGA
    } else {
        crate::dft::xc_deriv::XCType::LDA
    };
    let aoslices_xc = crate::hessian::xc_hessian::build_aoslices(mol_rks);
    let dm0_xc = &scf.density_matrix[0];

    // Return freed Phase 1-4 memory to OS before the DFT grid sweep.
    memory_monitor::trim_to_os();

    // vxc_diag (computed once, scattered to diagonal atom blocks)
    let _t_diag = std::time::Instant::now();
    // Phase 2: streaming mode — process grid blocks in small concurrent
    // batches instead of collecting all AO data in a cache. Peak AO
    // memory ≈ grid_concurrency() × per_block_size instead of
    // num_blocks × per_block_size.
    let vxc_diag_mat = crate::hessian::xc_hessian::vxc_diag_streaming(scf, xc_type);
    for ia in 0..natm_xc {
        let (p0, p1) = aoslices_xc[ia];
        for a in 0..3 { for b in 0..3 {
            let row0 = (a * 3 + b) * nao_xc;
            let mut s = 0.0;
            for mu in p0..p1 { for nu in 0..nao_xc {
                s += vxc_diag_mat[[row0 + mu, nu]] * dm0_xc[[mu, nu]];
            }}
            h_partial[(ia * 3 + a) * n3_xc + (ia * 3 + b)] += s * 2.0;
        }}
    }
    timings.push(("  rks: vxc_diag", _t_diag.elapsed()));
    // Free vxc_diag intermediates before the heavier vxc_deriv2 sweep.
    drop(vxc_diag_mat);
    memory_monitor::trim_to_os();

    // vxc_deriv2 (per-atom, symmetrized) — streaming with deriv=2.
    let _t_d2 = std::time::Instant::now();
    let vxc_d2 = crate::hessian::xc_hessian::vxc_deriv2_streaming(scf, xc_type);
    for ia in 0..natm_xc {
        for ja in 0..=ia {
            let (q0, q1) = aoslices_xc[ja];
            for a in 0..3 { for b in 0..3 {
                let row0 = (a * 3 + b) * nao_xc;
                let mut s = 0.0;
                for mu in q0..q1 { for nu in 0..nao_xc {
                    s += vxc_d2[ia][[row0 + mu, nu]] * dm0_xc[[mu, nu]];
                }}
                h_partial[(ia * 3 + a) * n3_xc + (ja * 3 + b)] += s * 2.0;
            }}
        }
    }
    for ia in 0..natm_xc {
        for ja in 0..ia {
            for a in 0..3 { for b in 0..3 {
                h_partial[(ja * 3 + b) * n3_xc + (ia * 3 + a)] =
                    h_partial[(ia * 3 + a) * n3_xc + (ja * 3 + b)];
            }}
        }
    }

    timings.push(("  rks: vxc_deriv2", _t_d2.elapsed()));
    timings.push(("  rks: xc_add", _t_rks_xc.elapsed()));
}

/// Compute the RKS vxc_deriv1 contribution to h1ao, per atom.
///
/// Returns `Vec<Vec<f64>>` indexed `[atom][nao*nao]` — the same shape
/// `calc_h1ao` previously built inline for `vxc_d1`. Memory-neutral: this is
/// the small per-atom vec the caller already allocates; no n3*n3 buffer is
/// involved.
pub fn compute_vxc_h1ao(scf: &SCF) -> Vec<Vec<f64>> {
    let mol = &scf.mol;
    let xct = if mol.xc_data.use_density_gradient() {
        crate::dft::xc_deriv::XCType::GGA
    } else {
        crate::dft::xc_deriv::XCType::LDA
    };
    let v = crate::hessian::xc_hessian::vxc_deriv1_streaming(scf, xct);
    v.iter().map(|m| m.iter().copied().collect()).collect()
}
