/*
Toolkits for handling transformations of XC derivative tensor from functional library.
See also:
https://github.com/pyscf/pyscf/blob/master/pyscf/dft/xc_deriv.py
https://gitee.com/ajz34/showcase_transform_xc
author: ajz34 (Zhenyu Zhu)
*/

use itertools::Itertools;
use rstsr::prelude::*;
use rstsr_core::prelude_dev::*;
use std::iter::repeat_n;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XCType {
    HF,
    LDA,
    GGA,
    MGGA,
}

// https://stackoverflow.com/a/65563202/7740992
pub fn count_combinations(n: usize, r: usize) -> usize {
    if r > n { 0 } else { (1..=r).fold(1, |acc, val| acc * (n - val + 1) / val) }
}

// manually advanced-index the last dimension
pub fn indexed_map_last_dim<T, B>(data: TensorView<T, B>, indices: &[usize]) -> Tensor<T, B>
where
    T: Copy,
    B: DeviceAPI<T, Raw: Clone> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let shape = data.shape().clone();
    let mut shape_new = shape.clone();
    shape_new.pop();
    shape_new.extend(vec![indices.len()]);
    let device = data.device();
    let mut result: Tensor<T, B> = unsafe { rt::empty((shape_new, device)) };

    for (i, &idx) in indices.iter().enumerate() {
        result.i_mut((Ellipsis, i)).assign(&data.i((Ellipsis, idx)));
    }
    result
}

pub const fn get_gga_sort(key: (usize, usize)) -> Option<&'static [usize]> {
    match key {
        (1, 1) => Some(&[0, 1, 2, 3, 4, 5]),
        (1, 2) => Some(&[6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (1, 3) => Some(&[
            21, 22, 25, 26, 27, 23, 28, 29, 30, 34, 35, 36, 37, 38, 39, 24, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55,
        ]),
        (1, 4) => Some(&[
            56, 57, 61, 62, 63, 58, 64, 65, 66, 73, 74, 75, 76, 77, 78, 59, 67, 68, 69, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 60, 70, 71, 72, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
            116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        ]),
        _ => None,
    }
}

pub const fn get_mgga_sort(key: (usize, usize)) -> Option<&'static [usize]> {
    match key {
        (0, 1) => Some(&[0, 1, 2, 3]),
        (0, 2) => Some(&[4, 5, 7, 6, 8, 9]),
        (0, 3) => Some(&[10, 11, 14, 12, 15, 16, 13, 17, 18, 19]),
        (0, 4) => Some(&[20, 21, 25, 22, 26, 27, 23, 28, 29, 30, 24, 31, 32, 33, 34]),
        (1, 1) => Some(&[0, 1, 2, 3, 4, 5, 6, 7]),
        (1, 2) => Some(&[8, 9, 11, 12, 13, 23, 24, 10, 14, 15, 16, 25, 26, 17, 18, 19, 27, 28, 20, 21, 29, 30, 22, 31, 32, 33, 34, 35]),
        (1, 3) => Some(&[
            36, 37, 40, 41, 42, 71, 72, 38, 43, 44, 45, 73, 74, 49, 50, 51, 77, 78, 52, 53, 79, 80, 54, 81, 82, 89, 90, 91, 39, 46, 47, 48,
            75, 76, 55, 56, 57, 83, 84, 58, 59, 85, 86, 60, 87, 88, 92, 93, 94, 61, 62, 63, 95, 96, 64, 65, 97, 98, 66, 99, 100, 107, 108,
            109, 67, 68, 101, 102, 69, 103, 104, 110, 111, 112, 70, 105, 106, 113, 114, 115, 116, 117, 118, 119,
        ]),
        (1, 4) => Some(&[
            120, 121, 125, 126, 127, 190, 191, 122, 128, 129, 130, 192, 193, 137, 138, 139, 198, 199, 140, 141, 200, 201, 142, 202, 203,
            216, 217, 218, 123, 131, 132, 133, 194, 195, 143, 144, 145, 204, 205, 146, 147, 206, 207, 148, 208, 209, 219, 220, 221, 155,
            156, 157, 225, 226, 158, 159, 227, 228, 160, 229, 230, 249, 250, 251, 161, 162, 231, 232, 163, 233, 234, 252, 253, 254, 164,
            235, 236, 255, 256, 257, 267, 268, 269, 270, 124, 134, 135, 136, 196, 197, 149, 150, 151, 210, 211, 152, 153, 212, 213, 154,
            214, 215, 222, 223, 224, 165, 166, 167, 237, 238, 168, 169, 239, 240, 170, 241, 242, 258, 259, 260, 171, 172, 243, 244, 173,
            245, 246, 261, 262, 263, 174, 247, 248, 264, 265, 266, 271, 272, 273, 274, 175, 176, 177, 275, 276, 178, 179, 277, 278, 180,
            279, 280, 295, 296, 297, 181, 182, 281, 282, 183, 283, 284, 298, 299, 300, 184, 285, 286, 301, 302, 303, 313, 314, 315, 316,
            185, 186, 287, 288, 187, 289, 290, 304, 305, 306, 188, 291, 292, 307, 308, 309, 317, 318, 319, 320, 189, 293, 294, 310, 311,
            312, 321, 322, 323, 324, 325, 326, 327, 328, 329,
        ]),
        _ => None,
    }
}

pub const fn get_xc_nvar_xlen(xctype: XCType, spin: usize) -> (usize, usize) {
    match (xctype, spin) {
        (XCType::HF, 0) => (1, 1),
        (XCType::HF, _) => (1, 2),
        (XCType::LDA, 0) => (1, 1),
        (XCType::LDA, _) => (1, 2),
        (XCType::GGA, 0) => (4, 2),
        (XCType::GGA, _) => (4, 5),
        (XCType::MGGA, 0) => (5, 3),
        (XCType::MGGA, _) => (5, 7),
    }
}

pub fn libxc_to_xcfun_indices(xctype: XCType, spin: usize, deriv: usize) -> Option<Vec<usize>> {
    if deriv <= 1 {
        return None;
    }

    match xctype {
        XCType::LDA | XCType::HF => None,
        XCType::GGA => match spin {
            0 => None,
            _ => Some((1..=deriv).flat_map(|i| get_gga_sort((spin, i)).unwrap().to_vec()).collect()),
        },
        XCType::MGGA => Some((1..=deriv).flat_map(|i| get_mgga_sort((spin, i)).unwrap().to_vec()).collect()),
    }
}

/// Generates raveled unique indices for the Cartesian product of a given number
/// of variables and order.
pub fn product_uniq_indices(nvars: usize, order: usize) -> Vec<usize> {
    // Generate all unique combinations with replacement
    let uniq_idx: Vec<Vec<usize>> = (0..nvars).combinations_with_replacement(order).map(|v| v.into_iter().collect()).collect();

    // Create a mapping from sorted indices to their position in uniq_idx
    let mut index_map = std::collections::HashMap::new();
    for (pos, indices) in uniq_idx.iter().enumerate() {
        index_map.insert(indices.clone(), pos);
    }

    // Generate all possible Cartesian product indices
    let cartesian_product = (0..order).map(|_| 0..nvars).multi_cartesian_product();

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

/// Generates unique pair combinations from a list of indices.
pub fn pair_combinations(lst: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let n = lst.len();

    // quick return
    if n == 0 {
        return vec![];
    } else if n == 2 {
        return vec![vec![(lst[0], lst[1])]];
    }

    // pair combinations requires length is even
    if n % 2 != 0 {
        panic!("pair_combinations requires an even number of elements");
    }

    let a = lst[0];
    let mut results = Vec::new();

    for i in 1..n {
        let pair = (a, lst[i]);
        let mut remaining = Vec::with_capacity(n - 2);
        remaining.extend_from_slice(&lst[1..i]);
        remaining.extend_from_slice(&lst[i + 1..]);

        let mut rests = pair_combinations(&remaining);
        for rest in &mut rests {
            rest.push(pair);
        }
        results.extend(rests);
    }

    results
}

pub fn diagonal_indices(idx: &[usize], order: usize) -> Vec<Vec<usize>> {
    assert!(order > 0);
    let n = idx.len();
    let mut diag_idx: Vec<Vec<usize>> = vec![idx.to_vec()];

    for _ in 1..order {
        let last_dim = diag_idx.last().unwrap().clone();
        diag_idx = diag_idx.into_iter().map(|x| x.into_iter().flat_map(|val| repeat_n(val, n)).collect()).collect();
        diag_idx.push(repeat_n(last_dim, n).flatten().collect());
    }

    diag_idx.into_iter().flat_map(|x| repeat_n(x, 2).collect::<Vec<Vec<usize>>>()).collect()
}

pub fn xc_indices_transform<T, B>(xc0: TensorView<T, B>, xctype: XCType, spin: usize, deriv: usize) -> Tensor<T, B>
where
    T: Copy,
    B: DeviceAPI<T, Raw: Clone> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    // sanity check
    assert!(xc0.ndim() == 2, "xc0 must be a 2D tensor");
    let indices = libxc_to_xcfun_indices(xctype, spin, deriv);
    if let Some(indices) = indices { indexed_map_last_dim(xc0.view(), &indices) } else { xc0.to_owned() }
}

pub fn vxc_unfold_sigma_spin0(frho: &mut [f64], fsigma: &[f64], rho: &[f64], ncounts: usize, nvar: usize, ngrids: usize) {
    let ncg = ncounts * ngrids;
    let nvg = nvar * ngrids;

    // Define accessor macros matching the C version's pattern
    macro_rules! frho_at {
        ($g:expr, $x:expr, $n:expr) => {
            frho[$g + $x * ngrids + $n * nvg]
        };
    }
    macro_rules! fsigma_at {
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
            frho_at!(g, 0, n) = fsigma_at!(g, n, 0);
            frho_at!(g, 1, n) = fsigma_at!(g, n, 1) * rho_at!(g, 1) * 2.0;
            frho_at!(g, 2, n) = fsigma_at!(g, n, 1) * rho_at!(g, 2) * 2.0;
            frho_at!(g, 3, n) = fsigma_at!(g, n, 1) * rho_at!(g, 3) * 2.0;
        }
    }

    if nvar > 4 {
        assert_eq!(nvar, 5, "MGGA case requires exactly 5 variables");
        for n in 0..ncounts {
            for g in 0..ngrids {
                frho_at!(g, 4, n) = fsigma_at!(g, n, 2);
            }
        }
    }
}

pub fn vxc_unfold_sigma_spin1(frho: &mut [f64], fsigma: &[f64], rho: &[f64], ncounts: usize, nvar: usize, ngrids: usize) {
    let ncg = ncounts * ngrids;
    let nvg = nvar * ngrids;

    // Helper macros to access the arrays by indices
    macro_rules! frho_at {
        ($g:expr, $x:expr, $a:expr, $n:expr) => {
            frho[$g + $x * ngrids + ($a + $n * 2) * nvg]
        };
    }
    macro_rules! fsigma_at {
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
            frho_at!(g, 0, 0, n) = fsigma_at!(g, n, 0);
            frho_at!(g, 0, 1, n) = fsigma_at!(g, n, 1);
            frho_at!(g, 1, 0, n) = fsigma_at!(g, n, 2) * rho_at!(g, 1, 0) * 2.0 + fsigma_at!(g, n, 3) * rho_at!(g, 1, 1);
            frho_at!(g, 1, 1, n) = fsigma_at!(g, n, 3) * rho_at!(g, 1, 0) + 2.0 * fsigma_at!(g, n, 4) * rho_at!(g, 1, 1);
            frho_at!(g, 2, 0, n) = fsigma_at!(g, n, 2) * rho_at!(g, 2, 0) * 2.0 + fsigma_at!(g, n, 3) * rho_at!(g, 2, 1);
            frho_at!(g, 2, 1, n) = fsigma_at!(g, n, 3) * rho_at!(g, 2, 0) + 2.0 * fsigma_at!(g, n, 4) * rho_at!(g, 2, 1);
            frho_at!(g, 3, 0, n) = fsigma_at!(g, n, 2) * rho_at!(g, 3, 0) * 2.0 + fsigma_at!(g, n, 3) * rho_at!(g, 3, 1);
            frho_at!(g, 3, 1, n) = fsigma_at!(g, n, 3) * rho_at!(g, 3, 0) + 2.0 * fsigma_at!(g, n, 4) * rho_at!(g, 3, 1);
        }
    }

    if nvar > 4 {
        assert_eq!(nvar, 5, "MGGA case requires exactly 5 variables");
        for n in 0..ncounts {
            for g in 0..ngrids {
                frho_at!(g, 4, 0, n) = fsigma_at!(g, n, 5);
                frho_at!(g, 4, 1, n) = fsigma_at!(g, n, 6);
            }
        }
    }
}

pub fn unfold_sigma<B>(
    rho: TensorView<f64, B>,
    xc_val: TensorView<f64, B>,
    spin: usize,
    order: usize,
    nvar: usize,
    xlen: usize,
    reserve: usize,
) -> Tensor<f64, B>
where
    B: DeviceAPI<f64, Raw = Vec<f64>> + DeviceCreationAnyAPI<f64> + OpAssignAPI<f64, IxD>,
{
    assert!(nvar >= 4);
    let nvar_spin = if spin == 0 { nvar } else { 2 * nvar };
    let ngrids = rho.shape()[0];
    // check dimensions
    assert!(xc_val.shape()[0] == ngrids, "xc_val length mismatch");
    assert!(xc_val.ndim() == 2, "xc_val must be a 2D tensor");
    match spin {
        0 => assert!(rho.shape() == &[ngrids, nvar]),
        _ => assert!(rho.shape() == &[ngrids, nvar, 2]),
    };

    let n_transform = order - reserve;
    let mut xc_tensor_shape = vec![ngrids];
    match spin {
        0 => xc_tensor_shape.extend(vec![nvar; n_transform]),
        _ => xc_tensor_shape.extend(vec![[nvar, 2]; n_transform].iter().flatten()),
    }
    xc_tensor_shape.extend(vec![xlen; reserve]);
    let mut xc_tensor: Tensor<f64, B> = unsafe { rt::empty((xc_tensor_shape, xc_val.device())) };

    let idx = product_uniq_indices(xlen, order);
    for (it, &io) in idx.iter().enumerate() {
        // please note that currently, RSTSR's `tensor.raw()` returns the pointer
        // (slice) of original data, instead of offsetted pointer points to the first
        // element of tensor.
        // So we need additionally define an offsetted slice.
        let xc_val_offsetted = &xc_val.raw()[xc_val.offset()..];
        xc_tensor.raw_mut()[it * ngrids..(it + 1) * ngrids].copy_from_slice(&xc_val_offsetted[io * ngrids..(io + 1) * ngrids]);
    }

    let mut buf = unsafe { xc_tensor.empty_like() };
    for i in 0..n_transform {
        std::mem::swap(&mut xc_tensor, &mut buf);
        let ncounts = xlen.pow((order - 1 - i) as u32) * nvar_spin.pow(i as u32);

        match spin {
            0 => vxc_unfold_sigma_spin0(xc_tensor.raw_mut(), buf.raw(), rho.raw(), ncounts, nvar, ngrids),
            _ => vxc_unfold_sigma_spin1(xc_tensor.raw_mut(), buf.raw(), rho.raw(), ncounts, nvar, ngrids),
        }
    }

    xc_tensor
}

pub fn rho6to5<B>(rho: TensorView<f64, B>) -> Tensor<f64, B>
where
    B: DeviceAPI<f64, Raw = Vec<f64>> + DeviceCreationAnyAPI<f64> + OpAssignAPI<f64, IxD> + OpAssignArbitaryAPI<f64, IxD, IxD>,
{
    assert!(rho.ndim() == 2 || rho.ndim() == 3, "rho must be a 2D or 3D tensor");
    match rho.shape()[1] {
        5 => rho.to_owned(),
        6 => {
            let mut shape = rho.shape().to_vec();
            shape[1] = 5;
            let mut rho5 = unsafe { rt::empty((shape, rho.device())) };
            rho5.i_mut((.., 0..4)).assign(rho.i((.., 0..4)));
            rho5.i_mut((.., 4)).assign(rho.i((.., 5)));
            rho5
        },
        _ => panic!("rho must have 5 or 6 components"),
    }
}

#[allow(clippy::deref_addrof)]
pub fn transform_xc_inner<B>(
    rho: TensorView<f64, B>,
    xc_val: TensorView<f64, B>,
    xctype: XCType,
    spin: usize,
    order: usize,
) -> Tensor<f64, B>
where
    B: DeviceAPI<f64, Raw = Vec<f64>>
        + DeviceCreationAnyAPI<f64>
        + OpAssignAPI<f64, IxD>
        + OpAssignArbitaryAPI<f64, IxD, IxD>
        + OpMulAPI<f64, f64, f64, IxD>
        + OpAddAssignAPI<f64, f64, IxD>
        + OpMulAssignAPI<f64, f64, IxD>,
{
    assert!(order < 4, "currently only support order < 4 (exc, vxc, kxc, fxc)");

    // sanity check for dimensions
    let ngrids = rho.shape()[0];
    let (nvar, xlen) = get_xc_nvar_xlen(xctype, spin);
    // check dimensions
    assert!(xc_val.shape()[0] == ngrids, "xc_val length (grids) mismatch");
    assert!(xc_val.ndim() == 2, "xc_val must be a 2D tensor");
    // special case for MGGA that rho may have 5/6 components
    let rho = match xctype {
        XCType::MGGA => rho6to5(rho),
        _ => rho.to_owned(),
    };
    // change shape into [ngrids, nvar, nspin if exist], otherwise panic
    let rho = match spin {
        0 => rho.into_shape([ngrids, nvar]),
        _ => rho.into_shape([ngrids, nvar, 2]),
    };
    assert!(rho.f_contig(), "rho must be f-contiguous");
    assert!(xc_val.f_contig(), "xc_val must be f-contiguous");

    // offsets of xc_val
    let mut offsets = vec![0];
    offsets.extend((0..=order).map(|o| count_combinations(xlen + o, o)));
    let offset_max = offsets.last().unwrap();
    assert!(xc_val.shape()[1] >= *offset_max, "xc_val length (offset) mismatch");

    // offsets match current order
    let (p0, p1) = (offsets[order], offsets[order + 1]);

    // quick return for LDA and HF
    if xctype == XCType::LDA || xctype == XCType::HF {
        let xc_out = xc_val.i((.., p0..p1));
        if spin == 0 {
            // shape: [ngrids, 1, 1, ..., 1]
            //                 | [1]*order |
            let mut shape = vec![ngrids];
            shape.extend(vec![1; order]);
            return xc_out.into_shape(shape);
        } else {
            let indices = product_uniq_indices(xlen, order);
            let xc_out = indexed_map_last_dim(xc_out.view(), &indices);
            // shape: [ngrids, 1, 2, 1, 2, ..., 1, 2]
            //                 | [1, 2] * order    |
            let mut shape = vec![ngrids];
            shape.extend(vec![[1, 2]; order].into_iter().flatten());
            return xc_out.into_shape(shape);
        }
    }

    let mut xc_tensor = unfold_sigma(rho.view(), xc_val.i((.., p0..p1)), spin, order, nvar, xlen, 0);

    if order <= 1 {
        // quick return for 0/1-order derivatives, which does not involve pair
        // derivatives of sigma
        return xc_tensor;
    }

    if spin == 0 {
        // currently we can only handle order = 2, 3 cases
        // for order > 3, following code is not correct
        let n_pairs = 1; // only correct for order = 2, 3
        let (p0, p1) = (offsets[order - n_pairs], offsets[order - n_pairs + 1]);
        let xc_sub = unfold_sigma(rho.view(), xc_val.i((.., p0..p1)), spin, order - n_pairs, nvar, xlen, n_pairs);
        let xc_sub: Tensor<f64, B> = 2.0 * xc_sub.i((Ellipsis, 1));
        match order {
            2 => *&mut xc_tensor.i_mut((.., 1..4, 1..4)).diagonal_mut((0_isize, -1, -2)) += xc_sub,
            3 => {
                let permute_order_list = [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]];
                for permute_order in permute_order_list {
                    let mut xc_tensor_perm = xc_tensor.view_mut().into_transpose(&permute_order);
                    *&mut xc_tensor_perm.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0_isize, -1, -2)) += &xc_sub;
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
        let xc_sub = indexed_map_last_dim(xc_sub, &[0, 1, 1, 2]);
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
                *&mut xc_tensor_spin.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0_isize, -1, -2)) += &xc_sub;
            },
            3 => {
                let xc_tensor_permute_spin = [0, 2, 4, 6, 1, 3, 5];
                let mut xc_tensor_spin = xc_tensor.view_mut().into_transpose(&xc_tensor_permute_spin);
                let xc_sub_permute_spin = [0, 2, 3, 4, 1];
                let xc_sub_spin = xc_sub.transpose(&xc_sub_permute_spin);

                let permute_order_list = [[0, 1, 2, 3, 4, 5, 6], [0, 2, 3, 1, 5, 6, 4], [0, 3, 1, 2, 6, 4, 5]];
                for permute_order in permute_order_list {
                    let mut xc_tensor_perm = xc_tensor_spin.view_mut().into_transpose(&permute_order);
                    *&mut xc_tensor_perm.i_mut((Ellipsis, 1..4, 1..4)).diagonal_mut((0_isize, -1, -2)) += &xc_sub_spin;
                }
            },
            _ => unreachable!(),
        }
    }

    xc_tensor
}