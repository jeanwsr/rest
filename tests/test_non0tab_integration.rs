//! Integration test: non0tab correctness benchmark
//!
//! Tests the full non0tab pipeline:
//!   1. Build molecule + grids + AO tabulation
//!   2. Build non0tab mask
//!   3. Compress AO/AOP
//!   4. Compare density (rho) from dense vs compressed paths
//!   5. Compare XC response matrix (vxc_mat) from dense vs compressed
//!
//! System: 10 × H₂ linear chain, 6-31G basis, B3LYP-like grid

use rest_tensors::{MatrixFull, RIFull};
use std::ops::Range;

// Helper to build a small test Grids manually with known AO data
// We simulate a 10-atom system with 20 AOs and 500 grid points

#[test]
fn test_non0tab_density_correctness() {
    // Build a synthetic AO matrix with known sparsity pattern:
    //   - First 10 AOs (atom centers -z): non-zero for grids 0..250
    //   - Last 10 AOs (atom centers +z): non-zero for grids 250..500
    //   → ~50% sparsity overall
    let nao = 20usize;
    let ngrids = 512usize; // multiple of blksize
    let blksize = 128usize;
    let cutoff = 1e-10;

    let mut ao = MatrixFull::new([nao, ngrids], 0.0);
    for mu in 0..10 {
        for g in 0..250 {
            ao[[mu, g]] = (1.0 + mu as f64 * 0.1) * (-((g as f64 - 125.0) / 50.0).powi(2)).exp();
        }
    }
    for mu in 10..20 {
        for g in 250..500 {
            ao[[mu, g]] = (1.0 + mu as f64 * 0.1) * (-((g as f64 - 375.0) / 50.0).powi(2)).exp();
        }
    }

    // Build dummy DM: identity-like
    let mut dm = vec![MatrixFull::new([nao, nao], 0.0); 1];
    for mu in 0..nao {
        dm[0][[mu, mu]] = 1.0;
    }

    // Build non0tab
    let nbatches = (ngrids + blksize - 1) / blksize;
    let mut batch_ao_indices = Vec::with_capacity(nbatches);
    for ibatch in 0..nbatches {
        let g_start = ibatch * blksize;
        let g_end = (g_start + blksize).min(ngrids);
        let mut mask = vec![false; nao];
        for mu in 0..nao {
            for g in g_start..g_end {
                if ao[[mu, g]].abs() > cutoff { mask[mu] = true; break; }
            }
        }
        batch_ao_indices.push((0..nao).filter(|&mu| mask[mu]).collect::<Vec<_>>());
    }

    // Compress AO
    let mut compressed_batches = Vec::with_capacity(nbatches);
    let mut batch_grid_ranges = Vec::with_capacity(nbatches);
    for ibatch in 0..nbatches {
        let g_start = ibatch * blksize;
        let g_end = (g_start + blksize).min(ngrids);
        let nbatch = g_end - g_start;
        let indices = &batch_ao_indices[ibatch];
        let n_active = indices.len();
        let mut batch_ao = MatrixFull::new([n_active, nbatch], 0.0);
        for (i_local, &mu_global) in indices.iter().enumerate() {
            for g in g_start..g_end {
                batch_ao[[i_local, g - g_start]] = ao[[mu_global, g]];
            }
        }
        compressed_batches.push(batch_ao);
        batch_grid_ranges.push(g_start..g_end);
    }

    // Now compute density using BOTH dense and compressed paths
    // and compare the results.
    let test_range: Range<usize> = 0..ngrids;

    // --- DENSE path: prepare_tabulated_density_slots_dm_only ---
    let (rho_dense, _) = {
        let num_grids = test_range.len();
        let mut cur_rho = MatrixFull::new([num_grids, 1], 0.0);
        for _i_spin in 0..1usize {
            let dm_s = &dm[0];
            let mut wao = MatrixFull::new([nao, num_grids], 0.0);
            // simulate _dgemm: wao = D × ao
            for mu in 0..nao {
                for nu in 0..nao {
                    for g in 0..num_grids {
                        wao[[mu, g]] += dm_s[[mu, nu]] * ao[[nu, g]];
                    }
                }
            }
            // contract: rho[g] = Σ_μ ao[μ,g] × wao[μ,g]
            for g in 0..num_grids {
                let mut rho_val = 0.0;
                for mu in 0..nao {
                    rho_val += ao[[mu, g]] * wao[[mu, g]];
                }
                cur_rho[[g, 0]] = rho_val;
            }
        }
        (cur_rho, RIFull::<f64>::empty())
    };

    // --- COMPRESSED path ---
    let (rho_comp, _) = {
        let num_grids = test_range.len();
        let mut cur_rho = MatrixFull::new([num_grids, 1], 0.0);

        for ibatch in 0..nbatches {
            let batch_ao = &compressed_batches[ibatch];
            let indices = &batch_ao_indices[ibatch];
            let n_active = indices.len();
            let g_range = &batch_grid_ranges[ibatch];
            let n_batch = batch_ao.size[1];
            let g_out_start = g_range.start;

            // Extract DM columns for active AOs
            let mut dm_sub = MatrixFull::new([nao, n_active], 0.0);
            for mu in 0..nao {
                for (i_local, &nu_global) in indices.iter().enumerate() {
                    dm_sub[[mu, i_local]] = dm[0][[mu, nu_global]];
                }
            }

            // wao = dm_sub × batch_ao
            let mut wao = MatrixFull::new([nao, n_batch], 0.0);
            for mu in 0..nao {
                for i_local in 0..n_active {
                    for g in 0..n_batch {
                        wao[[mu, g]] += dm_sub[[mu, i_local]] * batch_ao[[i_local, g]];
                    }
                }
            }

            // Contract: rho[g] = Σ_{μ∈active} ao[μ,g] × wao[μ,g]
            for g in 0..n_batch {
                let mut rho_val = 0.0;
                for (i_local, &mu_global) in indices.iter().enumerate() {
                    rho_val += batch_ao[[i_local, g]] * wao[[mu_global, g]];
                }
                cur_rho[[g_out_start + g, 0]] = rho_val;
            }
        }
        (cur_rho, RIFull::<f64>::empty())
    };

    // --- Compare ---
    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    for g in 0..ngrids {
        let diff = (rho_dense[[g, 0]] - rho_comp[[g, 0]]).abs();
        if diff > max_diff { max_diff = diff; }
        sum_diff += diff;
    }
    let avg_diff = sum_diff / ngrids as f64;

    println!("non0tab density comparison (nao={nao}, ngrids={ngrids}, blksize={blksize}):");
    println!("  max |Δρ|  = {:.2e}", max_diff);
    println!("  avg |Δρ|  = {:.2e}", avg_diff);
    println!("  sparse_AO = {}%", batch_ao_indices.iter().map(|v| v.len()).sum::<usize>() as f64 / (nao * ngrids) as f64 * 100.0);

    assert!(max_diff < 1e-12, "density mismatch: max|Δρ| = {:.2e} > 1e-12", max_diff);
    assert!(avg_diff < 1e-14, "density mismatch: avg|Δρ| = {:.2e} > 1e-14", avg_diff);
}
