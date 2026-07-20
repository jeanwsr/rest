use super::prelude::*;

use LibXCSpin::*;
use XCDenType::*;

// https://stackoverflow.com/a/65563202/7740992
pub fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

/* #region libxc-to-xcfun index convention change */

pub const fn libxc_to_xcfun_mapping_parts(den_type: XCDenType, deriv: usize) -> Option<&'static [usize]> {
    match (den_type, deriv) {
        (RHO, _) => None,
        (SIGMA, 0) => Some(&[0]),
        (SIGMA, 1) => Some(&[1, 2, 3, 4, 5]),
        (SIGMA, 2) => Some(&[6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (SIGMA, 3) => Some(&[
            21, 22, 25, 26, 27, 23, 28, 29, 30, 34, 35, 36, 37, 38, 39, 24, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
        ]),
        (SIGMA, 4) => Some(&[
            56, 57, 61, 62, 63, 58, 64, 65, 66, 73, 74, 75, 76, 77, 78, 59, 67, 68, 69, 79, 80, 81, 82, 83, 84, 91, 92,
            93, 94, 95, 96, 97, 98, 99, 100, 60, 70, 71, 72, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        ]),
        (TAU, 0) => Some(&[0]),
        (TAU, 1) => Some(&[1, 2, 3, 4, 5, 6, 7]),
        (TAU, 2) => Some(&[
            8, 9, 11, 12, 13, 17, 18, 10, 14, 15, 16, 19, 20, 21, 22, 23, 27, 28, 24, 25, 29, 30, 26, 31, 32, 33, 34,
            35,
        ]),
        (TAU, 3) => Some(&[
            36, 37, 40, 41, 42, 49, 50, 38, 43, 44, 45, 51, 52, 55, 56, 57, 67, 68, 58, 59, 69, 70, 60, 71, 72, 79, 80,
            81, 39, 46, 47, 48, 53, 54, 61, 62, 63, 73, 74, 64, 65, 75, 76, 66, 77, 78, 82, 83, 84, 85, 86, 87, 95, 96,
            88, 89, 97, 98, 90, 99, 100, 107, 108, 109, 91, 92, 101, 102, 93, 103, 104, 110, 111, 112, 94, 105, 106,
            113, 114, 115, 116, 117, 118, 119,
        ]),
        (TAU, 4) => Some(&[
            120, 121, 125, 126, 127, 137, 138, 122, 128, 129, 130, 139, 140, 145, 146, 147, 163, 164, 148, 149, 165,
            166, 150, 167, 168, 181, 182, 183, 123, 131, 132, 133, 141, 142, 151, 152, 153, 169, 170, 154, 155, 171,
            172, 156, 173, 174, 184, 185, 186, 190, 191, 192, 210, 211, 193, 194, 212, 213, 195, 214, 215, 234, 235,
            236, 196, 197, 216, 217, 198, 218, 219, 237, 238, 239, 199, 220, 221, 240, 241, 242, 252, 253, 254, 255,
            124, 134, 135, 136, 143, 144, 157, 158, 159, 175, 176, 160, 161, 177, 178, 162, 179, 180, 187, 188, 189,
            200, 201, 202, 222, 223, 203, 204, 224, 225, 205, 226, 227, 243, 244, 245, 206, 207, 228, 229, 208, 230,
            231, 246, 247, 248, 209, 232, 233, 249, 250, 251, 256, 257, 258, 259, 260, 261, 262, 275, 276, 263, 264,
            277, 278, 265, 279, 280, 295, 296, 297, 266, 267, 281, 282, 268, 283, 284, 298, 299, 300, 269, 285, 286,
            301, 302, 303, 313, 314, 315, 316, 270, 271, 287, 288, 272, 289, 290, 304, 305, 306, 273, 291, 292, 307,
            308, 309, 317, 318, 319, 320, 274, 293, 294, 310, 311, 312, 321, 322, 323, 324, 325, 326, 327, 328, 329,
        ]),
        // Laplacian-dependent functionals are not supported yet
        (LAPL, _) => unimplemented!(),
        // We currently only support up to 4th order derivatives (exc, vxc, kxc, fxc, lxc)
        (_, 5..) => unimplemented!(),
    }
}

#[doc = include_str!("libxc-xcfun-trans.md")]
pub fn libxc_to_xcfun_mapping(den_type: XCDenType, spin: LibXCSpin, deriv: usize) -> Option<Vec<usize>> {
    // some cases that do not require reordering
    if deriv <= 1 || spin == Unpolarized || den_type == RHO {
        return None;
    }
    // now we assume spin polarized and order >= 2
    Some((0..=deriv).flat_map(|x| libxc_to_xcfun_mapping_parts(den_type, x).unwrap().to_vec()).collect())
}

pub fn libxc_transform_xcfun_indices(
    xc0: TsrView<'_>,
    den_type: XCDenType,
    spin: LibXCSpin,
    deriv: usize,
) -> TsrCow<'_> {
    // sanity check
    assert!(xc0.ndim() == 2, "xc0 must be a 2-dim tensor");
    let indices = libxc_to_xcfun_mapping(den_type, spin, deriv);
    if let Some(indices) = indices {
        xc0.index_select(-1, &indices).into_cow()
    } else {
        xc0.into_cow()
    }
}

/* #endregion libxc-to-xcfun index convention change */

/* #region xlen utility */

/// Get the number of components (xlen) for a given density type and spin polarization.
///
/// This value should match the first derivative length.
pub const fn get_xc_xlen(den_type: XCDenType, spin: LibXCSpin) -> usize {
    match (den_type, spin) {
        (RHO, Unpolarized) => 1,
        (RHO, Polarized) => 2,
        (SIGMA, Unpolarized) => 2,
        (SIGMA, Polarized) => 5,
        (TAU, Unpolarized) => 3,
        (TAU, Polarized) => 7,
        (LAPL, _) => unimplemented!(),
    }
}

/// Generates raveled unique indices for the Cartesian product of a given number
/// of variables and order.
pub fn product_uniq_indices(xlen: usize, order: usize) -> Vec<usize> {
    // Generate all unique combinations with replacement
    let uniq_idx: Vec<Vec<usize>> =
        (0..xlen).combinations_with_replacement(order).map(|v| v.into_iter().collect()).collect();

    // Create a mapping from sorted indices to their position in uniq_idx
    let mut index_map = std::collections::HashMap::new();
    for (pos, indices) in uniq_idx.iter().enumerate() {
        index_map.insert(indices.clone(), pos);
    }

    // Generate all possible Cartesian product indices
    let cartesian_product = (0..order).map(|_| 0..xlen).multi_cartesian_product();

    // For each index in the Cartesian product, find its sorted version and lookup
    // the unique position
    cartesian_product
        .map(|indices| {
            let mut sorted = indices.clone();
            sorted.sort();
            *index_map.get(&sorted).unwrap()
        })
        .collect()
}

/* #endregion xlen utility */

/* #region unfold sigma */

pub fn vxc_unfold_sigma_spin0(
    frho: &mut [f64],
    fsigma: &[f64],
    rho: &[f64],
    ncounts: usize,
    nvar: usize,
    ngrids: usize,
) {
    let ncg = ncounts * ngrids;
    let nvg = nvar * ngrids;

    // Define accessor macros matching the C version's pattern
    macro_rules! fr_at {
        // f_rho index
        ($g:expr, $x:expr, $n:expr) => {
            frho[$g + $x * ngrids + $n * nvg]
        };
    }
    macro_rules! fs_at {
        // f_rho index
        ($g:expr, $n:expr, $x:expr) => {
            fsigma[$g + $n * ngrids + $x * ncg]
        };
    }
    macro_rules! rho_at {
        ($g:expr, $x:expr) => {
            rho[$g + $x * ngrids]
        };
    }

    for n in 0..ncounts {
        for g in 0..ngrids {
            // Main computation block
            fr_at!(g, 0, n) = fs_at!(g, n, 0);
            fr_at!(g, 1, n) = fs_at!(g, n, 1) * rho_at!(g, 1) * 2.0;
            fr_at!(g, 2, n) = fs_at!(g, n, 1) * rho_at!(g, 2) * 2.0;
            fr_at!(g, 3, n) = fs_at!(g, n, 1) * rho_at!(g, 3) * 2.0;
        }
    }

    if nvar > 4 {
        assert_eq!(nvar, 5, "MGGA case requires exactly 5 variables");
        for n in 0..ncounts {
            for g in 0..ngrids {
                fr_at!(g, 4, n) = fs_at!(g, n, 2);
            }
        }
    }
}

pub fn vxc_unfold_sigma_spin1(
    frho: &mut [f64],
    fsigma: &[f64],
    rho: &[f64],
    ncounts: usize,
    nvar: usize,
    ngrids: usize,
) {
    let ncg = ncounts * ngrids;
    let nvg = nvar * ngrids;

    // Helper macros to access the arrays by indices
    macro_rules! fr_at {
        // f_rho index
        ($g:expr, $x:expr, $a:expr, $n:expr) => {
            frho[$g + $x * ngrids + ($a + $n * 2) * nvg]
        };
    }
    macro_rules! fs_at {
        // f_sigma index
        ($g:expr, $n:expr, $x:expr) => {
            fsigma[$g + $n * ngrids + $x * ncg]
        };
    }
    macro_rules! rho_at {
        ($g:expr, $x:expr, $a:expr) => {
            rho[$g + $x * ngrids + $a * nvg]
        };
    }

    for n in 0..ncounts {
        for g in 0..ngrids {
            // Main computation block
            fr_at!(g, 0, 0, n) = fs_at!(g, n, 0);
            fr_at!(g, 0, 1, n) = fs_at!(g, n, 1);
            fr_at!(g, 1, 0, n) = fs_at!(g, n, 2) * rho_at!(g, 1, 0) * 2.0 + fs_at!(g, n, 3) * rho_at!(g, 1, 1);
            fr_at!(g, 1, 1, n) = fs_at!(g, n, 3) * rho_at!(g, 1, 0) + 2.0 * fs_at!(g, n, 4) * rho_at!(g, 1, 1);
            fr_at!(g, 2, 0, n) = fs_at!(g, n, 2) * rho_at!(g, 2, 0) * 2.0 + fs_at!(g, n, 3) * rho_at!(g, 2, 1);
            fr_at!(g, 2, 1, n) = fs_at!(g, n, 3) * rho_at!(g, 2, 0) + 2.0 * fs_at!(g, n, 4) * rho_at!(g, 2, 1);
            fr_at!(g, 3, 0, n) = fs_at!(g, n, 2) * rho_at!(g, 3, 0) * 2.0 + fs_at!(g, n, 3) * rho_at!(g, 3, 1);
            fr_at!(g, 3, 1, n) = fs_at!(g, n, 3) * rho_at!(g, 3, 0) + 2.0 * fs_at!(g, n, 4) * rho_at!(g, 3, 1);
        }
    }

    if nvar > 4 {
        assert_eq!(nvar, 5, "MGGA case requires exactly 5 variables");
        for n in 0..ncounts {
            for g in 0..ngrids {
                fr_at!(g, 4, 0, n) = fs_at!(g, n, 5);
                fr_at!(g, 4, 1, n) = fs_at!(g, n, 6);
            }
        }
    }
}

pub fn unfold_sigma(
    rho: TsrView,
    xc_val: TsrView,
    spin: LibXCSpin,
    order: usize,
    nvar: usize,
    xlen: usize,
    reserve: usize,
) -> Tsr {
    assert!(nvar >= 4);
    let nvar_spin = if spin == Unpolarized { nvar } else { 2 * nvar };
    let ngrids = rho.shape()[0];
    // check dimensions
    assert!(xc_val.shape()[0] == ngrids, "xc_val length mismatch");
    assert!(xc_val.ndim() == 2, "xc_val must be a 2-dim tensor");
    match spin {
        Unpolarized => assert!(rho.shape() == &[ngrids, nvar]),
        Polarized => assert!(rho.shape() == &[ngrids, nvar, 2]),
    };

    let n_transform = order - reserve;
    let mut xc_tensor_shape = vec![ngrids];
    match spin {
        Unpolarized => xc_tensor_shape.extend(vec![nvar; n_transform]),
        Polarized => xc_tensor_shape.extend(vec![[nvar, 2]; n_transform].iter().flatten()),
    }
    xc_tensor_shape.extend(vec![xlen; reserve]);
    let mut xc_tensor: Tsr = unsafe { rt::empty((xc_tensor_shape, xc_val.device())) };

    let idx = product_uniq_indices(xlen, order);
    for (it, &io) in idx.iter().enumerate() {
        // please note that currently, RSTSR's `tensor.raw()` returns the pointer
        // (slice) of original data, instead of offsetted pointer points to the first
        // element of tensor.
        // So we need additionally define an offsetted slice.
        let xc_val_offsetted = &xc_val.raw()[xc_val.offset()..];
        xc_tensor.raw_mut()[it * ngrids..(it + 1) * ngrids]
            .copy_from_slice(&xc_val_offsetted[io * ngrids..(io + 1) * ngrids]);
    }

    // also note the raw usage, rho is not assured to be offset-zero.
    let rho_raw = &rho.raw()[rho.offset()..];

    let mut buf = unsafe { xc_tensor.empty_like() };
    for i in 0..n_transform {
        std::mem::swap(&mut xc_tensor, &mut buf);
        let ncounts = xlen.pow((order - 1 - i) as u32) * nvar_spin.pow(i as u32);

        match spin {
            Unpolarized => vxc_unfold_sigma_spin0(xc_tensor.raw_mut(), buf.raw(), rho_raw, ncounts, nvar, ngrids),
            Polarized => vxc_unfold_sigma_spin1(xc_tensor.raw_mut(), buf.raw(), rho_raw, ncounts, nvar, ngrids),
        }
    }

    xc_tensor
}

/* #endregion unfold sigma */

#[allow(clippy::deref_addrof)]
pub fn transform_xc_inner(rho: TsrView, xc_val: TsrView, den_type: XCDenType, spin: LibXCSpin, order: usize) -> Tsr {
    if order >= 4 {
        panic!("currently only support order < 4 (exc, vxc, kxc, fxc). You specified order {order}");
    }

    // sanity check for dimensions
    let ngrids = rho.shape()[0];
    let nvar = den_type.num_nvar();
    let xlen = get_xc_xlen(den_type, spin);
    // check dimensions
    ni_check_shape!(xc_val.shape()[0], ngrids, "xc_val length (grids) mismatch");
    ni_check_shape!(xc_val.ndim(), 2, "xc_val must be a 2-dim tensor");
    // check shape [ngrids, nvar, nspin if exist], otherwise panic
    match spin {
        Unpolarized => ni_check_shape!(rho.ndim(), 2, "rho must be a 2-dim tensor"),
        Polarized => {
            ni_check_shape!(rho.ndim(), 3, "rho must be a 3-dim tensor");
            ni_check_shape!(rho.shape()[2], 2, "rho last dimension should be 2 for polarized case");
        },
    };
    ni_check_shape!(rho.shape()[0], ngrids, "rho first dimension must be grids");
    ni_check_shape!(rho.shape()[1] >= nvar, "rho second dimension (variables) should be larger than {nvar}");
    let rho = rho.change_contig(ColMajor);
    let xc_val = xc_val.to_contig(ColMajor);
    // double check input tensor
    // since we are using some raw functionality to get the raw slice, the offset must be zero.

    // offsets of xc_val
    let mut offsets = vec![0];
    offsets.extend((0..=order).map(|o| count_combinations(xlen + o, o)));
    let offset_max = offsets.last().unwrap();
    ni_check_shape!(xc_val.shape()[1] >= *offset_max, "xc_val length should be larger than {offset_max}");

    // offsets match current order
    let (p0, p1) = (offsets[order], offsets[order + 1]);

    // quick return for LDA
    if den_type == RHO {
        let xc_out = xc_val.i((.., p0..p1));
        if spin == Unpolarized {
            // shape: [ngrids, 1, 1, ..., 1]
            //                 | [1]*order |
            let mut shape = vec![ngrids];
            shape.extend(vec![1; order]);
            return xc_out.into_shape(shape);
        } else {
            let indices = product_uniq_indices(xlen, order);
            let xc_out = xc_out.index_select(-1, &indices);
            // shape: [ngrids, 1, 2, 1, 2, ..., 1, 2]
            //                 | [1, 2] * order    |
            let mut shape = vec![ngrids];
            shape.extend(vec![[1, 2]; order].into_iter().flatten());
            return xc_out.into_shape(shape);
        }
    }

    let mut xc_tensor = unfold_sigma(rho.view(), xc_val.i((.., p0..p1)), spin, order, nvar, xlen, 0);

    if order <= 1 {
        // quick return for 0/1-order derivatives, which does not involve pair derivatives of sigma
        return xc_tensor;
    }

    if spin == Unpolarized {
        // currently we can only handle order = 2, 3 cases
        // for order > 3, following code is not correct
        let n_pairs = 1; // only correct for order = 2, 3
        let (p0, p1) = (offsets[order - n_pairs], offsets[order - n_pairs + 1]);
        let xc_sub = unfold_sigma(rho.view(), xc_val.i((.., p0..p1)), spin, order - n_pairs, nvar, xlen, n_pairs);
        let xc_sub: Tsr = 2.0 * xc_sub.i((Ellipsis, 1));
        match order {
            2 => *&mut xc_tensor.i_mut((.., 1..4, 1..4)).diagonal_mut((0, -1, -2)) += xc_sub,
            3 => {
                let permute_order_list = [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]];
                for permute_order in permute_order_list {
                    let mut xc_tensor_perm = xc_tensor.view_mut().into_transpose(&permute_order);
                    *&mut xc_tensor_perm.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0, -1, -2)) += &xc_sub;
                }
            },
            _ => unreachable!(),
        }
    } else {
        // currently we can only handle order = 2, 3 cases
        // for order > 3, following code is not correct
        let n_pairs = 1; // only correct for order = 2, 3
        let (p0, p1) = (offsets[order - n_pairs], offsets[order - n_pairs + 1]);
        let xc_sub = unfold_sigma(rho.view(), xc_val.i((.., p0..p1)), spin, order - n_pairs, nvar, xlen, n_pairs);
        // just the sigma components, spin expanded
        let xc_sub = xc_sub.i((Ellipsis, 2..5));
        let xc_sub = xc_sub.index_select(-1, &[0, 1, 1, 2]);
        let xc_sub_shape = {
            let mut xc_sub_shape = xc_sub.shape().clone();
            xc_sub_shape.pop();
            xc_sub_shape.extend(vec![2, 2]);
            xc_sub_shape
        };
        let mut xc_sub = xc_sub.into_shape(xc_sub_shape);
        *&mut xc_sub.i_mut((Ellipsis, 0, 0)) *= 2.0;
        *&mut xc_sub.i_mut((Ellipsis, 1, 1)) *= 2.0;
        match order {
            2 => {
                let permute_spin = [0, 2, 4, 1, 3];
                let mut xc_tensor_spin = xc_tensor.view_mut().into_transpose(&permute_spin);
                // the case of order=2 does not require xc_sub to permute by spin indices
                *&mut xc_tensor_spin.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0, -1, -2)) += &xc_sub;
            },
            3 => {
                let xc_tensor_permute_spin = [0, 2, 4, 6, 1, 3, 5];
                let mut xc_tensor_spin = xc_tensor.view_mut().into_transpose(&xc_tensor_permute_spin);
                let xc_sub_permute_spin = [0, 2, 3, 4, 1];
                let xc_sub_spin = xc_sub.transpose(&xc_sub_permute_spin);

                let permute_order_list = [[0, 1, 2, 3, 4, 5, 6], [0, 2, 3, 1, 5, 6, 4], [0, 3, 1, 2, 6, 4, 5]];
                for permute_order in permute_order_list {
                    let mut xc_tensor_perm = xc_tensor_spin.view_mut().into_transpose(&permute_order);
                    *&mut xc_tensor_perm.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0, -1, -2)) += &xc_sub_spin;
                }
            },
            _ => unreachable!(),
        }
    }

    xc_tensor
}
