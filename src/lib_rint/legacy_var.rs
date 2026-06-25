use crate::basis_io::Basis4Elem;
use crate::geom_io::GeomCell;
use rest_tensors::{MatrixFull, RIFull, TensorOpt};
use std::ops::Range;

use super::basis::{
    load_molecule_shell_shared_from_raw, transform_mo_coeff_to_cartesian_shell_shared,
    BasisFunction,
};
use super::eri_mo_4c_r;

// `eri_mo_4c_r` contracts AO integrals a(s chemist notation `(pq|rs)`.
// <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
#[inline(always)]
fn eri_mo_asym_r(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    let v1 = eri_mo_4c_r(bfs, c, spin, p, r, q, s);
    let v2 = eri_mo_4c_r(bfs, c, spin, p, s, q, r);
    v1 - v2
}

fn f_v_ai(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    a: usize,
    i: usize,
    occ_list: &[usize],
) -> f64 {
    let mut s = 0.0;
    for &k in occ_list {
        s += eri_mo_asym_r(bfs, c, spin, a, k, i, k); // <a k || i k>
    }
    s
}

// Singles contribution:
// var1 = \sum_{i,a} |f_ai|^2  with  f_ai = \sum_k <a k || i k>.
fn var_singles(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let mut s = 0.0;
    for &i in occ_list {
        for &a in vir_list {
            let v_ai = f_v_ai(bfs, c, spin, a, i, occ_list);
            let term = v_ai * v_ai;
            if term.is_finite() {
                s += term;
            }
        }
    }
    s
}

// Doubles contribution:
// var2 = \sum_{i,j,a,b} (ia|jb) [2 (ia|jb) - (ib|ja)] = \sum_{i,j,a,b} (ia|jb) <ij||ab>.
fn var_doubles(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let mut s = 0.0;
    for &i in occ_list {
        for &j in occ_list {
            for &a in vir_list {
                for &b in vir_list {
                    // Strict closed-shell MP2 numerator term:
                    // (ia|jb) * [2(ia|jb) - (ib|ja)]
                    let v_dir = eri_mo_4c_r(bfs, c, spin, i, a, j, b); // (ia|jb)
                    let v_ex = eri_mo_4c_r(bfs, c, spin, i, b, j, a); // (ib|ja)
                    let term = v_dir * (2.0 * v_dir - v_ex);
                    if term.is_finite() {
                        s += term;
                    }
                }
            }
        }
    }
    s
}

fn ri_vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0, |acc, (x, y)| acc + x * y)
}

fn f_v_ai_ri(
    ri3mo: &RIFull<f64>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
    a: usize,
    i: usize,
    occ_list: &[usize],
) -> f64 {
    let row_start = row_range.start;
    let col_start = col_range.start;

    assert!(row_range.contains(&a));
    assert!(col_range.contains(&i));

    let a_loc = a - row_start;
    let i_loc = i - col_start;
    let ri_i = ri3mo.get_reducing_matrix(i_loc).expect("ri3mo col i");

    let mut s = 0.0;
    for &k in occ_list {
        assert!(row_range.contains(&k));
        assert!(col_range.contains(&k));

        let k_row = k - row_start;
        let k_col = k - col_start;
        let ri_k = ri3mo.get_reducing_matrix(k_col).expect("ri3mo col k");

        let v1 = ri_vec_dot(ri_i.get_slice_x(a_loc), ri_k.get_slice_x(k_row)); // (a i|k k)
        let v2 = ri_vec_dot(ri_k.get_slice_x(a_loc), ri_i.get_slice_x(k_row)); // (a k|k i)
        s += v1 - v2;
    }

    s
}

fn var_singles_ri(
    ri3mo: &RIFull<f64>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let mut s = 0.0;
    for &i in occ_list {
        for &a in vir_list {
            let v_ai = f_v_ai_ri(ri3mo, row_range, col_range, a, i, occ_list);
            let term = v_ai * v_ai;
            if term.is_finite() {
                s += term;
            }
        }
    }
    s
}

fn var_doubles_ri(
    ri3mo: &RIFull<f64>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let row_start = row_range.start;
    let col_start = col_range.start;

    for &a in vir_list {
        assert!(row_range.contains(&a));
    }
    for &i in occ_list {
        assert!(col_range.contains(&i));
    }

    let mut s = 0.0;
    for &i in occ_list {
        let i_loc = i - col_start;
        let ri_i = ri3mo.get_reducing_matrix(i_loc).expect("ri3mo col i");
        for &j in occ_list {
            let j_loc = j - col_start;
            let ri_j = ri3mo.get_reducing_matrix(j_loc).expect("ri3mo col j");
            for &a in vir_list {
                let a_loc = a - row_start;
                let ai = ri_i.get_slice_x(a_loc);
                let aj = ri_j.get_slice_x(a_loc);
                for &b in vir_list {
                    let b_loc = b - row_start;
                    let bj = ri_j.get_slice_x(b_loc);
                    let bi = ri_i.get_slice_x(b_loc);
                    let v_dir = ri_vec_dot(ai, bj); // (ia|jb)
                    let v_ex = ri_vec_dot(bi, aj); // (ib|ja)
                    let term = v_dir * (2.0 * v_dir - v_ex);
                    if term.is_finite() {
                        s += term;
                    }
                }
            }
        }
    }
    s
}

pub fn var_vee_hf_r_ri3mo(
    ri3mo: &RIFull<f64>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let var1 = var_singles_ri(ri3mo, row_range, col_range, occ_list, vir_list);
    let var2 = var_doubles_ri(ri3mo, row_range, col_range, occ_list, vir_list);

    var1 + var2
}

pub fn var_vee_hf_r_ri3mo_geom(
    _geom: &GeomCell,
    _basis4elem: &[Basis4Elem],
    _eigenvectors: &[MatrixFull<f64>; 2],
    _spin: usize,
    occ_list: &[usize],
    vir_list: &[usize],
    ri3mo: &RIFull<f64>,
    row_range: &Range<usize>,
    col_range: &Range<usize>,
) -> f64 {
    var_vee_hf_r_ri3mo(ri3mo, row_range, col_range, occ_list, vir_list)
}

pub fn var_vee_doubles_hf_r(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let var1 = var_singles(bfs, c, spin, occ_list, vir_list);
    let var2 = var_doubles(bfs, c, spin, occ_list, vir_list);

    var1 + var2
}

pub fn var_vee_doubles_hf_r_geom(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    eigenvectors: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
    vir_list: &[usize],
) -> f64 {
    let bfs: Vec<BasisFunction> = load_molecule_shell_shared_from_raw(geom, basis4elem)
        .expect("failed to build bfs from GeomCell/BasCell");
    let nao_cart = bfs.len();
    let eigenvectors_cart = [
        transform_mo_coeff_to_cartesian_shell_shared(
            geom,
            basis4elem,
            &eigenvectors[0],
            nao_cart,
            "var_vee_doubles_hf_r_geom[alpha]",
            false,
        ),
        transform_mo_coeff_to_cartesian_shell_shared(
            geom,
            basis4elem,
            &eigenvectors[1],
            nao_cart,
            "var_vee_doubles_hf_r_geom[beta]",
            false,
        ),
    ];

    var_vee_doubles_hf_r(&bfs, &eigenvectors_cart, spin, occ_list, vir_list)
}
