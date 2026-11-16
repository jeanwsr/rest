use super::prelude_dev::*;
use super::pure_schwartz_rij::*;

/// Generate Coulomb (J) matrices using the RI method with per-shell-pair Schwartz screening.
///
/// This function is high-level interface of [`get_vj_ri_schwartz`], using rest_tensors as input
/// and output. Unlike [`generate_vj_ri_direct`](super::direct::generate_vj_ri_direct), all
/// per-geometry data — the shell-pair Schwarz bounds, the aux-shell bounds, and the decomposed
/// 2c-2e Coulomb metric — live in `engine` and are built once per geometry by
/// [`RIJSchwartzEngine::build`].
///
/// # Parameters
///
/// - `engine`: `&`[`RIJSchwartzEngine`]
///
/// - `dms`: `&[MatrixFull<f64>]`
///
///   - Density matrices, each of shape (nao, nao), symmetric; one J matrix is generated per density
///     matrix.
pub fn generate_vj_ri_schwartz(engine: &RIJSchwartzEngine, dms: &[MatrixFull<f64>]) -> Vec<MatrixUpper<f64>> {
    // dm shape: (nao, nao, nset) in f-contig
    let device = DeviceBLAS::default();
    let dms_rstsr = dms.to_rstsr(&device);

    let nao = dms_rstsr.shape()[0];
    let nset = dms_rstsr.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let js_rstsr = get_vj_ri_schwartz(engine, dms_rstsr.view());

    // Tsr -> Vec<MatrixUpper>
    let mut js = vec![];
    for iset in 0..nset {
        let j = js_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        js.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, j) });
    }
    js
}
