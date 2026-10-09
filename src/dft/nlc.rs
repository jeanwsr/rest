//! Non-local correlation (NLC) functionals: VV10.
//!
//! Unlike the (semi-)local LDA/GGA terms, the VV10 correlation energy is a
//! double integral over pairs of quadrature points (Vydrov & Van Voorhis,
//! JCP 133, 244103 (2010)). The pair kernel and the functional derivatives
//! with respect to the density (`vrho`) and `sigma = |grad rho|^2` (`vsigma`)
//! follow the conventions of PySCF's `_vv10nlc`, so the grid quantities are
//! assembled with the standard restricted GGA potential code path.
//!
//! VV10 is activated automatically when the parsed functional carries an
//! explicit VV10 component (e.g. `wb97x-v`); there is no control keyword.

use rayon::prelude::*;
use tensors::matrix_blas_lapack::_dgemm;
use tensors::{MathMatrix, MatrixFull, MatrixUpper};

use super::contract_vxc_0;
use super::{prepare_tabulated_sigma, Grids};
use crate::molecule_io::Molecule;

/// Grid points below this density are discarded in both quadratures, matching
/// the reference implementation.
const RHO_THRESHOLD: f64 = 1.0e-8;

/// Per-grid local intermediates shared by the VV10 energy and its derivatives.
struct VV10Grid {
    w0: f64,
    kappa: f64,
    rho_weight: f64,
    dw0_drho: f64,
    dw0_dsigma: f64,
    dkappa_drho: f64,
}

fn build_local(rho: f64, sigma: f64, weight: f64, b: f64, c: f64) -> VV10Grid {
    let pi43 = 4.0 * std::f64::consts::PI / 3.0;
    let kvv = b * 1.5 * std::f64::consts::PI * (9.0 * std::f64::consts::PI).powf(-1.0 / 6.0);

    // t = C * (sigma / rho^2)^2
    let t = c * (sigma / (rho * rho)).powi(2);
    let w0 = (t + pi43 * rho).sqrt();
    let kappa = kvv * rho.powf(1.0 / 6.0);

    let dw0_drho = (0.5 * pi43 * rho - 2.0 * t) / w0;
    let dw0_dsigma = if sigma > 0.0 { t * rho / (sigma * w0) } else { 0.0 };
    let dkappa_drho = kappa / 6.0;

    VV10Grid {
        w0,
        kappa,
        rho_weight: rho * weight,
        dw0_drho,
        dw0_dsigma,
        dkappa_drho,
    }
}

/// VV10 per-grid energy density and GGA-like potential components.
fn vv10_grid_quantities(
    rho: &[f64],
    sigma: &[f64],
    weights: &[f64],
    coords: &[[f64; 3]],
    b: f64,
    c: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = rho.len();
    let beta = (3.0 / (b * b)).powf(0.75) / 32.0;

    let active: Vec<usize> = (0..n).filter(|&i| rho[i] >= RHO_THRESHOLD).collect();
    let local: Vec<VV10Grid> = active
        .iter()
        .map(|&i| build_local(rho[i], sigma[i], weights[i], b, c))
        .collect();
    let active_coords: Vec<[f64; 3]> = active.iter().map(|&i| coords[i]).collect();

    let mut f = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut w = vec![0.0; n];

    // Pair sums are independent for each outer grid point.
    let sums: Vec<(f64, f64, f64)> = active
        .par_iter()
        .zip(local.par_iter())
        .with_min_len(64)
        .map(|(&i_out, loc_out)| {
            let [ox, oy, oz] = coords[i_out];
            let mut sum_f = 0.0;
            let mut sum_u = 0.0;
            let mut sum_w = 0.0;

            for (j, loc_in) in local.iter().enumerate() {
                let dx = active_coords[j][0] - ox;
                let dy = active_coords[j][1] - oy;
                let dz = active_coords[j][2] - oz;
                let r2 = dx * dx + dy * dy + dz * dz;

                let g_in = r2 * loc_in.w0 + loc_in.kappa;
                let g_out = r2 * loc_out.w0 + loc_out.kappa;
                let g_sum = g_out + g_in;

                let t = loc_in.rho_weight / (g_out * g_in * g_sum);
                sum_f += t;
                let tp = t * (1.0 / g_out + 1.0 / g_sum);
                sum_u += tp;
                sum_w += tp * r2;
            }
            (-1.5 * sum_f, sum_u, sum_w)
        })
        .collect();
    for (&i_out, (sf, su, sw)) in active.iter().zip(sums.iter()) {
        f[i_out] = *sf;
        u[i_out] = *su;
        w[i_out] = *sw;
    }

    let mut exc = vec![0.0; n];
    let mut vrho = vec![0.0; n];
    let mut vsigma = vec![0.0; n];
    for (k, &i) in active.iter().enumerate() {
        let loc = &local[k];
        exc[i] = beta + 0.5 * f[i];
        vrho[i] = beta + f[i] + 1.5 * (u[i] * loc.dkappa_drho + w[i] * loc.dw0_drho);
        vsigma[i] = 1.5 * w[i] * loc.dw0_dsigma;
    }
    (exc, vrho, vsigma)
}

/// Evaluate the VV10 correlation energy and AO-basis potential on `grids`.
///
/// VV10 is a spin-independent functional: it depends only on the total
/// density `rho_a + rho_b`. For unrestricted calculations the pair sum is
/// evaluated once on the total density and the same potential is returned
/// for both spin channels (matching the UKS convention). The passed grids must
/// hold dense AO values and AO gradients.
pub fn vv10_exc_vxc(
    mol: &Molecule,
    grids: &Grids,
    dm: &[MatrixFull<f64>],
    b: f64,
    c: f64,
) -> (f64, Vec<MatrixUpper<f64>>) {
    let spin_channel = mol.spin_channel;
    assert!(
        spin_channel == 1 || spin_channel == 2,
        "VV10 supports 1 (restricted) or 2 (unrestricted) spin channels, got {}",
        spin_channel
    );
    let num_grids = grids.coordinates.len();
    let num_basis = mol.num_basis;

    let ao = grids
        .ao
        .as_ref()
        .expect("dense AO values are required for VV10 (ao_cutoff must be 0)");
    let aop = grids
        .aop
        .as_ref()
        .expect("AO gradients are required for VV10");

    // Collapse spin channels to the total density matrix for the spin-free
    // VV10 kernel. Restricted calculations already carry a single total dm.
    let dm_total: MatrixFull<f64> = if spin_channel == 2 {
        dm[0].clone() + dm[1].clone()
    } else {
        dm[0].clone()
    };
    let dm_vec = vec![dm_total];
    let (rho_mat, rhop) =
        grids.prepare_tabulated_density_slots_dm_only(&dm_vec, 1, 0..num_grids);
    let sigma_mat = prepare_tabulated_sigma(&rhop, 1);
    let rho = rho_mat.slice_column(0);
    let sigma = sigma_mat.slice_column(0);

    let (exc, vrho, vsigma) =
        vv10_grid_quantities(rho, sigma, &grids.weights, &grids.coordinates, b, c);

    let exc_total: f64 = rho
        .iter()
        .zip(grids.weights.iter())
        .zip(exc.iter())
        .map(|((&r, &wgt), &e)| r * wgt * e)
        .sum();

    // Standard restricted GGA assembly:
    // vxc_ao = chi * vrho + 4 (grad chi) . (vsigma grad rho), then fold in the
    // quadrature weights before the back transformation to the AO basis.
    let mut vxc_ao = MatrixFull::new([num_basis, num_grids], 0.0);
    contract_vxc_0(&mut vxc_ao, &ao.to_matrixfullslice(), &vrho, None);

    let rhop_total = rhop.get_reducing_matrix(0).unwrap();
    let mut grad_ao = MatrixFull::new([num_basis, num_grids], 0.0);
    for x in 0..3 {
        let aop_x = aop.get_reducing_matrix(x).unwrap();
        let rhop_x = rhop_total.get_slice_x(x);
        contract_vxc_0(&mut grad_ao, &aop_x, rhop_x, None);
    }
    contract_vxc_0(&mut vxc_ao, &grad_ao.to_matrixfullslice(), &vsigma, Some(4.0));

    vxc_ao
        .iter_columns_full_mut()
        .zip(grids.weights.iter())
        .for_each(|(col, &weight)| col.iter_mut().for_each(|v| *v *= weight));

    let mut vxc_mat = MatrixFull::new([num_basis, num_basis], 0.0);
    _dgemm(
        ao,
        (0..num_basis, 0..num_grids),
        'N',
        &vxc_ao,
        (0..num_basis, 0..num_grids),
        'T',
        &mut vxc_mat,
        (0..num_basis, 0..num_basis),
        1.0,
        0.0,
    );
    vxc_mat.self_add(&vxc_mat.transpose());
    vxc_mat.self_multiple(0.5);

    let vxc_upper = vxc_mat.to_matrixupper();
    (exc_total, vec![vxc_upper; spin_channel])
}
