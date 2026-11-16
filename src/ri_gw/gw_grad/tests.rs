#[cfg(test)]
mod layout_tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    }

    fn naive_j(nao: usize, naux: usize, i3: &[f64], di: &[f64], dj: &[f64], dm: &[f64], j: &MatrixFull<f64>) -> Vec<f64> {
        let n2 = nao * nao;
        let mut rho = vec![0.0f64; naux];
        let mut drho = vec![0.0f64; naux];
        for p in 0..naux {
            let mut a1 = 0.0;
            let mut a2 = 0.0;
            for k in 0..n2 {
                a1 += i3[k + p * n2] * dm[k];
                a2 += di[k + p * n2] * dm[k];
            }
            rho[p] = a1;
            drho[p] = a2;
        }
        let c = _dsolve(j, &rho).unwrap();
        let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
        let rhs: Vec<f64> = drho.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
        let dc = _dsolve(j, &rhs).unwrap();
        let mut out = vec![0.0f64; n2];
        for p in 0..naux {
            for k in 0..n2 {
                out[k] += di[k + p * n2] * c[p] + i3[k + p * n2] * dc[p];
            }
        }
        out
    }

    fn naive_k(nao: usize, naux: usize, nocc: usize, i3: &[f64], di: &[f64], dj: &[f64], cocc: &[f64], j: &MatrixFull<f64>) -> Vec<f64> {
        let n2 = nao * nao;
        let mut out = vec![0.0f64; n2];
        for i in 0..nocc {
            let mut y_i = vec![0.0f64; nao * naux];
            let mut dy_i = vec![0.0f64; nao * naux];
            for p in 0..naux {
                for mu in 0..nao {
                    let mut a1 = 0.0;
                    let mut a2 = 0.0;
                    for nu in 0..nao {
                        let ci = cocc[nu + i * nao];
                        a1 += i3[mu + nu * nao + p * n2] * ci;
                        a2 += di[mu + nu * nao + p * n2] * ci;
                    }
                    y_i[mu + p * nao] = a1;
                    dy_i[mu + p * nao] = a2;
                }
            }
            let mut g = vec![0.0f64; naux * nao];
            for mu in 0..nao {
                let rhs: Vec<f64> = (0..naux).map(|p| y_i[mu + p * nao]).collect();
                let sol = _dsolve(j, &rhs).unwrap();
                for p in 0..naux {
                    g[p + mu * naux] = sol[p];
                }
            }
            let dk1 = gemm_nn(&dy_i, nao, naux, &g, naux, nao);
            let tmp = gemm_nn(dj, naux, naux, &g, naux, nao);
            let mut tmp2 = vec![0.0f64; naux * nao];
            for nu in 0..nao {
                let rhs: Vec<f64> = (0..naux).map(|p| tmp[p + nu * naux]).collect();
                let sol = _dsolve(j, &rhs).unwrap();
                for p in 0..naux {
                    tmp2[p + nu * naux] = sol[p];
                }
            }
            let term = gemm_nn(&y_i, nao, naux, &tmp2, naux, nao);
            for row in 0..nao {
                for col in 0..nao {
                    let k = row + col * nao;
                    let k_t = col + row * nao;
                    out[k] += 2.0 * (dk1[k] + dk1[k_t] - term[k]);
                }
            }
        }
        out
    }

    /// naive reference of the old scalar qx_from_u
    fn naive_qx(nao: usize, naux: usize, nmo: usize, c: &[f64], q: &[f64], di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = nao * nao;
        let mut qx = vec![0.0f64; naux * nmo * nmo];
        for p_aux in 0..naux {
            let di_block = &di[p_aux * n2..(p_aux + 1) * n2];
            let mut tmp = vec![0.0f64; nao * nmo];
            for mu in 0..nao {
                for q_mo in 0..nmo {
                    let mut acc = 0.0;
                    for nu in 0..nao {
                        acc += di_block[mu + nu * nao] * c[nu + q_mo * nao];
                    }
                    tmp[mu + q_mo * nao] = acc;
                }
            }
            for q_mo in 0..nmo {
                for p_mo in 0..nmo {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += c[mu + p_mo * nao] * tmp[mu + q_mo * nao];
                    }
                    qx[(p_mo + q_mo * nmo) * naux + p_aux] = acc;
                }
            }
        }
        let um = to_mat(u, nmo, nmo);
        for p_aux in 0..naux {
            let mut block = vec![0.0f64; nmo * nmo];
            for col in 0..nmo * nmo {
                block[col] = q[col * naux + p_aux];
            }
            let utq = gemm_nt(u, nmo, nmo, &block, nmo, nmo);
            let bmat = to_mat(&block, nmo, nmo);
            let mut qum = MatrixFull::new([nmo, nmo], 0.0);
            _dgemm_full(&bmat, 'N', &um, 'N', &mut qum, 1.0, 0.0);
            for col in 0..nmo * nmo {
                block[col] = qx[col * naux + p_aux] + utq[col] + qum.data[col];
            }
            for pi in 0..nmo {
                for pj in (pi + 1)..nmo {
                    let a = pi + pj * nmo;
                    let b = pj + pi * nmo;
                    let s = 0.5 * (block[a] + block[b]);
                    block[a] = s;
                    block[b] = s;
                }
            }
            for col in 0..nmo * nmo {
                qx[col * naux + p_aux] = block[col];
            }
        }
        qx
    }

    /// the new qx algorithm, copied verbatim from qx_from_u (synthetic inputs)
    fn new_qx(nao: usize, naux: usize, nmo: usize, c: &[f64], q: &[f64], di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = nao * nao;
        let npair = nmo * nmo;
        let mut qx = vec![0.0f64; naux * npair];
        {
            let cm = to_mat(c, nao, nmo);
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p in 0..naux {
                ip.data.copy_from_slice(&di[p * n2..(p + 1) * n2]);
                _dgemm_full(&ip, 'N', &cm, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&cm, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..npair {
                    qx[p + col * naux] = sl.data[col];
                }
            }
        }
        // pair-slowest repack of q, as in GwGradEngine::new_with_raw
        let q_pm = {
            let mut v = vec![0.0f64; naux * npair];
            for pair in 0..npair {
                for p in 0..naux {
                    v[pair + p * npair] = q[pair * naux + p];
                }
            }
            v
        };
        // chunked pair-major gemm, as in qx_from_u
        let um = to_mat(u, nmo, nmo);
        let chunk_aux = (4_194_304 / (nmo * nmo)).clamp(1, naux);
        let mut p0 = 0;
        while p0 < naux {
            let n_chunk = (naux - p0).min(chunk_aux);
            let nc = n_chunk * nmo;
            let q2c = to_mat(&q_pm[p0 * npair..(p0 + n_chunk) * npair], nmo, nc);
            let mut t1v = MatrixFull::new([nmo, nc], 0.0);
            _dgemm_full(&um, 'T', &q2c, 'N', &mut t1v, 1.0, 0.0);
            for p in p0..p0 + n_chunk {
                let base = (p - p0) * npair;
                for pj in 0..npair {
                    let pj_swap = (pj % nmo) * nmo + pj / nmo;
                    qx[p + pj * naux] += t1v.data[base + pj] + t1v.data[base + pj_swap];
                }
                for pi in 0..nmo {
                    for pj in (pi + 1)..nmo {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (qx[p + a * naux] + qx[p + b * naux]);
                        qx[p + a * naux] = s;
                        qx[p + b * naux] = s;
                    }
                }
            }
            p0 += n_chunk;
        }
        qx
    }

    #[test]
    fn j_k_qx_match_naive_reference() {
        let mut seed = 42u64;
        let nao = 12;
        let naux = 17;
        let nocc = 4;
        let nmo = nao;
        let n2 = nao * nao;
        // SPD metric
        let mut a = vec![0.0f64; naux * naux];
        for x in a.iter_mut() {
            *x = rnd(&mut seed);
        }
        let am = to_mat(&a, naux, naux);
        let mut spd = MatrixFull::new([naux, naux], 0.0);
        _dgemm_full(&am, 'T', &am, 'N', &mut spd, 1.0, 0.0);
        for i in 0..naux {
            spd.data[i + i * naux] += naux as f64;
        }
        // Cholesky factor + multi-RHS solve closure, as in GwGradEngine
        let mut chol = spd.clone();
        _dpotrf(&mut chol, 'L');
        // in-place multi-column solve, same signature as `j_deriv_from_impl` expects
        let solve = |b: &mut [f64], ncol: usize| {
            let mut bm = to_mat(b, naux, ncol);
            assert!(_dtrtrs(&chol, &mut bm, 'L', 'N', 'N'));
            assert!(_dtrtrs(&chol, &mut bm, 'L', 'T', 'N'));
            b.copy_from_slice(&bm.data);
        };
        // sanity: cholesky solve == dgesv solve
        let rhs: Vec<f64> = (0..naux).map(|_| rnd(&mut seed)).collect();
        let mut x1 = rhs.clone();
        solve(&mut x1, 1);
        let x2 = _dsolve(&spd, &rhs).unwrap();
        let err = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1.0e-10, "chol solve err {err}");

        let i3: Vec<f64> = (0..n2 * naux).map(|_| 0.2 * rnd(&mut seed)).collect();
        let di: Vec<f64> = (0..n2 * naux).map(|_| 0.2 * rnd(&mut seed)).collect();
        let dj: Vec<f64> = {
            let mut v: Vec<f64> = (0..naux * naux).map(|_| rnd(&mut seed)).collect();
            for i in 0..naux {
                for j in 0..i {
                    v[i + j * naux] = v[j + i * naux];
                }
            }
            v
        };
        let dm: Vec<f64> = {
            let mut v: Vec<f64> = (0..n2).map(|_| rnd(&mut seed)).collect();
            for i in 0..nao {
                for j in 0..i {
                    v[i + j * nao] = v[j + i * nao];
                }
            }
            v
        };
        let cocc: Vec<f64> = (0..nao * nocc).map(|_| rnd(&mut seed)).collect();
        let cmo: Vec<f64> = (0..nao * nmo).map(|_| rnd(&mut seed)).collect();
        let u: Vec<f64> = (0..nmo * nmo).map(|_| rnd(&mut seed)).collect();
        // pair-symmetric q
        let q: Vec<f64> = {
            let mut v: Vec<f64> = (0..naux * nmo * nmo).map(|_| rnd(&mut seed)).collect();
            for p in 0..naux {
                for pi in 0..nmo {
                    for pj in 0..pi {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (v[a * naux + p] + v[b * naux + p]);
                        v[a * naux + p] = s;
                        v[b * naux + p] = s;
                    }
                }
            }
            v
        };

        let jn = naive_j(nao, naux, &i3, &di, &dj, &dm, &spd);
        let jnew = j_deriv_from_impl(nao, naux, &i3, &di, &dj, &dm, &solve);
        let ej = jn.iter().zip(jnew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(ej < 1.0e-12, "j_deriv mismatch {ej}");

        let kn = naive_k(nao, naux, nocc, &i3, &di, &dj, &cocc, &spd);
        let knew = k_deriv_from_impl(nao, naux, nocc, &i3, &di, &dj, &cocc, &solve);
        let ek = kn.iter().zip(knew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(ek < 1.0e-12, "k_deriv mismatch {ek}");

        let qn = naive_qx(nao, naux, nmo, &cmo, &q, &di, &u);
        let qnew = new_qx(nao, naux, nmo, &cmo, &q, &di, &u);
        let eq = qn.iter().zip(qnew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(eq < 1.0e-12, "qx mismatch {eq}");
    }

    /// link + correctness smoke test of the complex symmetric LU solve used
    /// by the full-CD screening (zgetrf/zgetrs)
    #[test]
    fn cd_complex_lu_solve_matches_reference() {
        use num_complex::ComplexFloat;
        let n = 9usize;
        let mut seed = 7u64;
        // complex symmetric A = B B^T with complex B (symmetric, indefinite ok)
        let bmat: Vec<Complex<f64>> = (0..n * n).map(|_| Complex::new(rnd(&mut seed), rnd(&mut seed))).collect();
        let mut a_re = vec![0.0f64; n * n];
        let mut a_im = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..n {
                    acc += bmat[i + k * n] * bmat[j + k * n];
                }
                a_re[i + j * n] = acc.re;
                a_im[i + j * n] = acc.im;
            }
        }
        let factor = CdLuFactor::factor(&a_re, &a_im, n);
        let ncol = 3usize;
        let rhs: Vec<Complex<f64>> = (0..n * ncol).map(|_| Complex::new(rnd(&mut seed), rnd(&mut seed))).collect();
        let mut x = rhs.clone();
        factor.solve_in_place(&mut x, ncol);
        // verify A X = B column by column
        for c in 0..ncol {
            for i in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for j in 0..n {
                    acc += Complex::new(a_re[i + j * n], a_im[i + j * n]) * x[j + c * n];
                }
                let err = (acc - rhs[i + c * n]).abs();
                assert!(err < 1.0e-10, "complex LU solve mismatch {err}");
            }
        }
        // residue_dz/de consistency: d/dz of residue_d == residue_dz
        let (delta, z, eta) = (-0.7, 0.31, 1.0e-3);
        let h = 1.0e-6;
        let fd = (residue_d(delta, z + h, eta) - residue_d(delta, z - h, eta)) / (2.0 * h);
        let an = residue_dz(delta, z, eta);
        assert!((fd - an).abs() < 1.0e-8, "residue_dz mismatch {fd} vs {an}");
        let fdd = (residue_d(delta + h, z, eta) - residue_d(delta - h, z, eta)) / (2.0 * h);
        let and = residue_de(delta, z, eta);
        assert!((fdd - and).abs() < 1.0e-8, "residue_de mismatch {fdd} vs {and}");
        let fd0 = (static_d0(delta + h, eta) - static_d0(delta - h, eta)) / (2.0 * h);
        let an0 = static_d0e(delta, eta);
        assert!((fd0 - an0).abs() < 1.0e-8, "static_d0e mismatch {fd0} vs {an0}");
    }
}
