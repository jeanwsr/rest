//! Scaling / memory benchmark: AO-basis ("memory-efficient") vs MO-basis BSE
//! matvec.  Run with `--nocapture` to see the table.
//!
//! ```text
//! cargo test --release --test bench_bse_matvec_aovsmo -- --nocapture --ignored
//! ```

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_bse;
use pyrest::scf_io::{self, SCF};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::MatrixFull;
use std::time::Instant;

fn input_cut(geom: &str, basis: &str, aux: &str, cutoff: f64) -> String {
    format!(
        r##"
[ctrl]
print_level = 0
num_threads = 4
xc = "pbe0"
basis_path = "{basis}"
basis_type = "spheric"
auxbas_path = "{aux}"
auxbas_type = "spheric"
eri_type = "ri-v"
use_ri_symm = true
charge = 0.0
spin = 1.0
spin_polarization = false
initial_guess = "sad"
mixer = "diis"
max_scf_cycle = 200
scf_acc_rho = 1.0e-7
scf_acc_eev = 1.0e-6
scf_acc_etot = 1.0e-9
[geom]
{geom}
[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
bse_auxbas_path = "{aux}"
bse_cutoff_energy = {cutoff}
"##
    )
}

fn input(geom: &str, basis: &str, aux: &str) -> String {
    input_cut(geom, basis, aux, 1.0e6)
}

/// Same as [`input_cut`] but with **no** BSE-specific auxiliary basis
/// (`bse_auxbas_path` removed): REST then never builds `rimatr_bse` and every
/// consumer falls back to the regular `rimatr` of `auxbas_path`.
fn input_no_bse_aux(geom: &str, basis: &str, aux: &str, cutoff: f64) -> String {
    input_cut(geom, basis, aux, cutoff)
        .lines()
        .filter(|l| !l.trim_start().starts_with("bse_auxbas_path"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_from_input(input: &str) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(input).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf.prepare_bse_integrals(&None);
    scf
}

fn build_cut(geom: &str, basis: &str, aux: &str, cutoff: f64) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_cut(geom, basis, aux, cutoff)).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf.prepare_bse_integrals(&None);
    scf
}

fn build(geom: &str, basis: &str, aux: &str) -> SCF {
    build_cut(geom, basis, aux, 1.0e6)
}

/// Peak resident set size in MB (Linux `VmHWM`, monotonic).
fn rss_peak_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.parse::<f64>().ok()))
                .flatten()
        })
        .unwrap_or(0.0)
        / 1024.0
}

/// Resident set size in MB (Linux).
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.parse::<f64>().ok()))
                .flatten()
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn geom_block(name: &str, xyz: &str) -> String {
    format!("name = \"{name}\"\nunit = \"angstrom\"\nposition = \"\"\"\n{xyz}\"\"\"\n")
}

fn lcg(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        })
        .collect()
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Molecule geometries (angstrom).  `N` fused benzene rings along x.
fn polyacene(n_rings: usize) -> String {
    // Build a linear polyacene: fused hexagons of side 1.39 A, C-C bond ~1.39.
    let a = 1.39f64; // C-C
    let dx = a * 3f64.sqrt() / 2.0;
    let mut xyz = String::new();
    let mut carbons: Vec<(f64, f64)> = Vec::new();
    for k in 0..n_rings {
        let x0 = k as f64 * 2.0 * dx;
        // hexagon centre offset for the "up" ring
        let cx = x0;
        let cy = 0.0;
        for (i, ang) in [90.0f64, 30.0, -30.0, -90.0, -150.0, 150.0].iter().enumerate() {
            let th = ang.to_radians();
            let (px, py) = (cx + a * th.cos(), cy + a * th.sin());
            // dedupe shared vertices
            if !carbons.iter().any(|(qx, qy)| (qx - px).abs() < 1e-6 && (qy - py).abs() < 1e-6) {
                carbons.push((px, py));
            }
            let _ = i;
        }
    }
    // dedupe again across rings (shared C-C bond atoms)
    let mut uniq: Vec<(f64, f64)> = Vec::new();
    for (px, py) in carbons {
        if !uniq.iter().any(|(qx, qy)| (qx - px).abs() < 1e-6 && (qy - py).abs() < 1e-6) {
            uniq.push((px, py));
        }
    }
    for (px, py) in &uniq {
        xyz.push_str(&format!("C  {:.6}  {:.6}  0.000000\n", px, py));
    }
    // hydrogens on the outer perimeter
    for (px, py) in &uniq {
        let r = (px * px + py * py).sqrt();
        let _ = r;
        let mut deg = 0;
        for (qx, qy) in &uniq {
            let d = ((px - qx).powi(2) + (py - qy).powi(2)).sqrt();
            if d > 1e-6 && d < 1.6 { deg += 1; }
        }
        if deg == 2 {
            // outward direction = away from the ring centroid on the same side
            let vy = if *py > 0.0 { -1.0 } else { 1.0 };
            xyz.push_str(&format!("H  {:.6}  {:.6}  0.000000\n", px, py + vy * 1.09));
        }
    }
    xyz
}

fn report(tag: &str, geom: &str, basis: &str, aux: &str) {
    println!("\n=== {tag} / {basis} (+ {aux}) ===");
    let t_build = Instant::now();
    let mut scf = build(geom, basis, aux);
    println!("scf+bse-integrals: {:.2} s", t_build.elapsed().as_secs_f64());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let n_bas = scf.mol.num_basis;
    let n_aux = scf.num_auxbas_bse.unwrap_or(0);
    println!(
        "n_bas = {n_bas}, n_aux = {n_aux}, rimatr_bse = {:.2} GB, unpacked J = {:.2} GB",
        (n_bas * (n_bas + 1) / 2 * n_aux) as f64 * 8.0 / 1e9,
        (n_bas * n_bas * n_aux) as f64 * 8.0 / 1e9
    );
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let (_, _, o, v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    let m = inv.size[0];
    println!("n_o = {o}, n_v = {v}, nov = {}", o * v);

    // ---- MO-basis tensors (what the historical path materialises) ----
    let ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'N');
    let ri_vv = ri_bse::get_submatrix(&scf, 'V', 'V', 'N');
    let ri_oo = ri_bse::get_submatrix(&scf, 'O', 'O', 'N');
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
    let mo_bytes = (ri_ov.data.len() + ri_vv.data.len() + ri_oo.data.len()
        + ri_oo_tilde.data.len()
        + ri_ov_tilde.data.len()) as f64
        * 8.0;
    println!(
        "MO-basis RI tensors: {:.3} GB  (+ [OV,OV] reorganised matrices = {:.3} GB)",
        mo_bytes / 1e9,
        (4.0 * (o * v) as f64 * (o * v) as f64 * 8.0) / 1e9
    );

    // ---- AO-basis context ----
    let t_ao = Instant::now();
    let ctx = match ri_bse::matvec_fast::FastBseContext::build(&scf, &inv) {
        Some(c) => c,
        None => {
            println!("AO build refused (missing AO tensor)");
            return;
        }
    };
    let ao_build = t_ao.elapsed().as_secs_f64();
    println!(
        "AO context: build {ao_build:.2} s, persistent {:.3} GB (folds {:.3} GB), \
         RSS {:.0} MB, PEAK RSS {:.0} MB",
        ctx.bytes as f64 / 1e9,
        ctx.fold_bytes() as f64 / 1e9,
        rss_mb(),
        rss_peak_mb()
    );

    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let z = lcg(o * v, 7);

    // ---- AO matvec ----
    let reps = if n_bas > 200 { 5 } else { 30 };
    let t1 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(ctx.a_block_matvec(&scf, &qp_ctrl, &z));
    }
    let ao_a = t1.elapsed().as_secs_f64() / reps as f64;
    let t2 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(ctx.b_block_matvec(&scf, &qp_ctrl, &z));
    }
    let ao_b = t2.elapsed().as_secs_f64() / reps as f64;

    // ---- MO matvec: the **production** historical entry points ----
    //
    // `ri_bse::matvec::a_block_matvec` / `b_block_matvec`, given exactly the
    // reshaped tensors `ri_bse::feast_solver` hands them.  (An earlier version of
    // this benchmark fused the two historical contraction steps into one
    // five-deep loop, which inflates the MO cost by a factor `O*V/(O+V)`; the
    // production code builds the `[M*O, V]` intermediate once and then contracts
    // BLAS-3, so it is `O(M*O*V*(O+V))` -- the *same* asymptotic cost as the AO
    // path.)
    let mut ri_oo_tilde_hist = ri_oo_tilde.clone();
    ri_oo_tilde_hist.reshape([m * o, o]);
    ri_oo_tilde_hist = ri_oo_tilde_hist.transpose_and_drop();
    ri_oo_tilde_hist.reshape([o * m, o]);
    let mut ri_vv_hist = ri_vv.clone();
    ri_vv_hist.reshape([m * v, v]);
    let mut ri_ov_tilde_hist = ri_ov_tilde.clone();
    ri_ov_tilde_hist.reshape([m * o, v]);
    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([m * o, v]);

    let reps_mo = if n_bas > 200 { 3 } else { 10 };
    let t3 = Instant::now();
    let mut a_ref = vec![0.0f64; o * v];
    for _ in 0..reps_mo {
        a_ref = std::hint::black_box(ri_bse::matvec::a_block_matvec(
            &scf,
            &qp_ctrl,
            &ri_vv_hist,
            &ri_ov,
            &ri_oo_tilde_hist,
            &z,
        ));
    }
    let mo_time = t3.elapsed().as_secs_f64() / reps_mo as f64;
    let t4 = Instant::now();
    let mut b_ref = vec![0.0f64; o * v];
    for _ in 0..reps_mo {
        b_ref = std::hint::black_box(ri_bse::matvec::b_block_matvec(
            &scf,
            &qp_ctrl,
            &ri_ov,
            &ri_ov_b,
            &ri_ov_tilde_hist,
            &z,
        ));
    }
    let mo_b_time = t4.elapsed().as_secs_f64() / reps_mo as f64;

    // ---- correctness against the production entry points ----
    let a_new = ctx.a_block_matvec(&scf, &qp_ctrl, &z);
    let b_new = ctx.b_block_matvec(&scf, &qp_ctrl, &z);
    let rel = max_abs(&a_ref, &a_new) / norm(&a_ref);
    let rel_b = max_abs(&b_ref, &b_new) / norm(&b_ref);

    println!(
        "TIMING  AO A {:.3} ms | MO A {:.3} ms | AO/MO A = {:.2}x   (A rel err {rel:.2e})",
        ao_a * 1e3,
        mo_time * 1e3,
        ao_a / mo_time
    );
    println!(
        "TIMING  AO B {:.3} ms | MO B {:.3} ms | AO/MO B = {:.2}x   (B rel err {rel_b:.2e})",
        ao_b * 1e3,
        mo_b_time * 1e3,
        ao_b / mo_b_time
    );
    let _ = (ri_ov_tilde, ao_b);
}

#[test]
#[ignore]
fn bench_scaling() {
    let small = "N  -2.1988391019   1.8973746268   0.0000000000\nH  -1.1788391019   1.8973746268   0.0000000000\nH  -2.5388353987   1.0925460144  -0.5263586446\nH  -2.5388400276   2.7556271745  -0.4338224694\n";
    report("NH3", &geom_block("NH3", small), "cc-pvdz", "def2-universal-jkfit");

    let benzene = "C  1.3960  0.0000  0.0000\nC  0.6980  1.2092  0.0000\nC -0.6980  1.2092  0.0000\nC -1.3960  0.0000  0.0000\nC -0.6980 -1.2092  0.0000\nC  0.6980 -1.2092  0.0000\nH  2.4820  0.0000  0.0000\nH  1.2410  2.1495  0.0000\nH -1.2410  2.1495  0.0000\nH -2.4820  0.0000  0.0000\nH -1.2410 -2.1495  0.0000\nH  1.2410 -2.1495  0.0000\n";
    report("benzene", &geom_block("C6H6", benzene), "cc-pvdz", "cc-pvdz-rifit");

    for rings in [2usize, 3] {
        let name = match rings { 2 => "naphthalene", _ => "anthracene" };
        let xyz = polyacene(rings);
        println!("\n--- {name}: {} atoms ---", xyz.lines().count());
        report(name, &geom_block(name, &xyz), "cc-pvdz", "cc-pvdz-rifit");
    }
}

/// cc-pVQZ benchmark: AO matvec vs the MO matvec's memory footprint.
///
/// The MO-basis path allocates `[O*V, O*V]` dense matrices in `reorganize_w` and
/// a `[M, N, N]` scratch inside `ao2mo`; the AO path keeps `[M, N, N]` plus
/// `O(M O (O+V))`.  This test measures the AO resident set and the MO path's
/// *required* allocation (without actually making it, once it is too large).
#[test]
#[ignore]
fn bench_ccpvqz() {
    for (name, xyz) in [
        ("benzene", "C  1.3960  0.0000  0.0000\nC  0.6980  1.2092  0.0000\nC -0.6980  1.2092  0.0000\nC -1.3960  0.0000  0.0000\nC -0.6980 -1.2092  0.0000\nC  0.6980 -1.2092  0.0000\nH  2.4820  0.0000  0.0000\nH  1.2410  2.1495  0.0000\nH -1.2410  2.1495  0.0000\nH -2.4820  0.0000  0.0000\nH -1.2410 -2.1495  0.0000\nH  1.2410 -2.1495  0.0000\n"),
    ] {
        let geom = geom_block(name, xyz);
        println!("\n===== {name} / cc-pVQZ =====");
        let r0 = rss_mb();
        let t0 = Instant::now();
        let mut scf = build_cut(&geom, "cc-pvqz", "cc-pvqz-rifit", 1.0e6);
        println!("scf + BSE AO integrals: {:.1} s, RSS {:.0} MB", t0.elapsed().as_secs_f64(), rss_mb());
        scf.gwqp.0 = scf.eigenvalues[0].clone();
        let n_bas = scf.mol.num_basis;
        let n_aux = scf.num_auxbas_bse.unwrap_or(0);
        let packed_gb = (n_bas * (n_bas + 1) / 2 * n_aux) as f64 * 8.0 / 1e9;
        println!(
            "n_bas={n_bas} n_aux={n_aux}: rimatr_bse = {:.2} GB (held by REST)",
            packed_gb
        );
        let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
        let (_, _, o, v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
        let nov = o * v;
        println!(
            "n_o={o} n_v={v} nov={nov}: MO path would need ri_ov/ri_vv/ri_oo(+tilde) = {:.2} GB \
             and [OV,OV] matrices = {:.2} GB",
            (2.0 * 2.0 * n_aux as f64 * nov as f64 + 2.0 * n_aux as f64 * (o * o) as f64) * 8.0 / 1e9,
            4.0 * (nov as f64) * (nov as f64) * 8.0 / 1e9
        );

        let t1 = Instant::now();
        let ctx = match ri_bse::matvec_fast::FastBseContext::build(&scf, &inv) {
            Some(c) => c,
            None => {
                println!("AO build refused");
                continue;
            }
        };
        println!(
            "AO context: build {:.1} s, persistent {:.2} GB, RSS {:.0} MB (was {:.0}), \
             PEAK RSS {:.0} MB",
            t1.elapsed().as_secs_f64(),
            ctx.bytes as f64 / 1e9,
            rss_mb(),
            r0,
            rss_peak_mb()
        );
        let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
        let z = lcg(nov, 7);
        let t2 = Instant::now();
        let a_ao = std::hint::black_box(ctx.a_block_matvec(&scf, &qp_ctrl, &z));
        let a1 = t2.elapsed().as_secs_f64();
        let t3 = Instant::now();
        let b_ao = std::hint::black_box(ctx.b_block_matvec(&scf, &qp_ctrl, &z));
        let b1 = t3.elapsed().as_secs_f64();
        println!("AO matvec: A {:.1} ms, B {:.1} ms", a1 * 1e3, b1 * 1e3);

        // ---- the production MO entry points, on the same tensors ----
        let m = inv.size[0];
        let ri_ov = ri_bse::get_submatrix(&scf, 'O', 'V', 'N');
        let mut ri_vv = ri_bse::get_submatrix(&scf, 'V', 'V', 'N');
        ri_vv.reshape([m * v, v]);
        let mut ri_oo = ri_bse::get_submatrix(&scf, 'O', 'O', 'N');
        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inv, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        drop(ri_oo);
        ri_oo_tilde.reshape([m * o, o]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([o * m, o]);
        let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
        _dgemm_full(&inv, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
        ri_ov_tilde.reshape([m * o, v]);
        let mut ri_ov_b = ri_ov.clone();
        ri_ov_b.reshape([m * o, v]);
        println!("MO tensors built, RSS {:.0} MB, PEAK RSS {:.0} MB", rss_mb(), rss_peak_mb());

        let t4 = Instant::now();
        let a_mo = std::hint::black_box(ri_bse::matvec::a_block_matvec(
            &scf, &qp_ctrl, &ri_vv, &ri_ov, &ri_oo_tilde, &z,
        ));
        let mo_a = t4.elapsed().as_secs_f64();
        let t5 = Instant::now();
        let b_mo = std::hint::black_box(ri_bse::matvec::b_block_matvec(
            &scf, &qp_ctrl, &ri_ov, &ri_ov_b, &ri_ov_tilde, &z,
        ));
        let mo_b = t5.elapsed().as_secs_f64();
        println!(
            "MO matvec: A {:.1} ms, B {:.1} ms | AO/MO A = {:.2}x | AO/MO B = {:.2}x",
            mo_a * 1e3,
            mo_b * 1e3,
            a1 / mo_a,
            b1 / mo_b
        );
        println!(
            "correctness vs production entry points: A rel {:.2e}, B rel {:.2e}",
            max_abs(&a_mo, &a_ao) / norm(&a_mo),
            max_abs(&b_mo, &b_ao) / norm(&b_mo)
        );

        // MO path cost estimate: O(M * O^2 * V^2) FLOPs
        let mo_flops = 2.0 * (n_aux as f64) * (o as f64).powi(2) * (v as f64).powi(2);
        println!("MO path work per matvec = {:.2e} FLOPs (dense [OV,OV] contraction)", mo_flops);
        let ao_flops = 2.0 * (n_aux as f64) * (o as f64) * (v as f64) * ((o + v) as f64);
        println!("AO path work per matvec = {:.2e} FLOPs  -> speedup {:.1}x", ao_flops, mo_flops / ao_flops);
    }
}

/// Does the AO matvec still work when **no** BSE-specific auxiliary basis is
/// given?  REST then never builds `rimatr_bse` and falls back to the regular
/// `rimatr`; the AO matvec must do the same, and the second `[N(N+1)/2, M]`
/// tensor must disappear from the peak.
#[test]
#[ignore]
fn bench_no_bse_aux() {
    let benzene = "C  1.3960  0.0000  0.0000\nC  0.6980  1.2092  0.0000\nC -0.6980  1.2092  0.0000\nC -1.3960  0.0000  0.0000\nC -0.6980 -1.2092  0.0000\nC  0.6980 -1.2092  0.0000\nH  2.4820  0.0000  0.0000\nH  1.2410  2.1495  0.0000\nH -1.2410  2.1495  0.0000\nH -2.4820  0.0000  0.0000\nH -1.2410 -2.1495  0.0000\nH  1.2410 -2.1495  0.0000\n";
    let geom = geom_block("benzene", benzene);
    for (tag, with_bse_aux) in [
        ("WITH bse_auxbas_path", true),
        ("WITHOUT bse_auxbas_path", false),
    ] {
        println!("\n===== benzene/cc-pVQZ, {tag} =====");
        let input = if with_bse_aux {
            input_cut(&geom, "cc-pvqz", "cc-pvqz-rifit", 1.0e6)
        } else {
            input_no_bse_aux(&geom, "cc-pvqz", "cc-pvqz-rifit", 1.0e6)
        };
        let mut scf = build_from_input(&input);
        scf.gwqp.0 = scf.eigenvalues[0].clone();
        let n_bas = scf.mol.num_basis;
        let n_aux = scf.num_auxbas_bse.unwrap_or(scf.mol.num_auxbas);
        let packed_gb = |m: usize| (n_bas * (n_bas + 1) / 2 * m) as f64 * 8.0 / 1e9;
        println!(
            "n_bas={n_bas} n_aux={n_aux}  rimatr={:.2} GB  rimatr_bse={}  RSS after scf+ri: {:.0} MB",
            packed_gb(scf.mol.num_auxbas),
            if scf.rimatr_bse.is_some() {
                format!("{:.2} GB", packed_gb(n_aux))
            } else {
                "absent".to_string()
            },
            rss_mb()
        );
        let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
        let (_, _, o, v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
        let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
        let z = lcg(o * v, 7);
        let ctx = match ri_bse::matvec_fast::FastBseContext::build(&scf, &inv) {
            Some(c) => c,
            None => {
                println!("  AO build refused");
                continue;
            }
        };
        println!(
            "  AO context: persistent {:.2} GB, RSS {:.0} MB, PEAK RSS {:.0} MB",
            ctx.bytes as f64 / 1e9,
            rss_mb(),
            rss_peak_mb()
        );
        let t = Instant::now();
        std::hint::black_box(ctx.a_block_matvec(&scf, &qp_ctrl, &z));
        let a = t.elapsed().as_secs_f64();
        let t = Instant::now();
        std::hint::black_box(ctx.b_block_matvec(&scf, &qp_ctrl, &z));
        let b = t.elapsed().as_secs_f64();
        println!("  AO matvec: A {:.1} ms, B {:.1} ms", a * 1e3, b * 1e3);
    }
}

// ---------------------------------------------------------------------------
// Where the resident set actually goes.
// ---------------------------------------------------------------------------

extern "C" {
    /// glibc: return free arena pages to the OS.  Lets us separate *live* data
    /// from memory the allocator is merely holding on to after large temporaries.
    fn malloc_trim(pad: usize) -> i32;
}

fn trim() {
    unsafe {
        malloc_trim(0);
    }
}

#[test]
#[ignore]
fn bench_rss_breakdown() {
    let stage = |tag: &str| {
        println!(
            "  {tag:<48} RSS {:>8.0} MB   peak {:>8.0} MB",
            rss_mb(),
            rss_peak_mb()
        )
    };
    let benzene = "C  1.3960  0.0000  0.0000\nC  0.6980  1.2092  0.0000\nC -0.6980  1.2092  0.0000\nC -1.3960  0.0000  0.0000\nC -0.6980 -1.2092  0.0000\nC  0.6980 -1.2092  0.0000\nH  2.4820  0.0000  0.0000\nH  1.2410  2.1495  0.0000\nH -1.2410  2.1495  0.0000\nH -2.4820  0.0000  0.0000\nH -1.2410 -2.1495  0.0000\nH  1.2410 -2.1495  0.0000\n";
    let geom = geom_block("benzene", benzene);
    let input = input_cut(&geom, "cc-pvqz", "cc-pvqz-rifit", 1.0e6);
    let keys = toml::from_str::<serde_json::Value>(&input).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();

    println!("=== RSS breakdown, benzene / cc-pVQZ ===");
    stage("0. test binary");
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let n_bas = mol.num_basis;
    let m_reg = mol.num_auxbas;
    stage("1. + Molecule::build_native (basis, libcint)");

    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    stage("2. + SCF");
    trim();
    stage("3. + malloc_trim (what the SCF keeps live)");

    scf.prepare_bse_integrals(&None);
    let m_bse = scf.num_auxbas_bse.unwrap_or(0);
    stage("4. + prepare_bse_integrals (rimatr_bse)");
    let gb = |n: usize, m: usize| (n * (n + 1) / 2 * m) as f64 * 8.0 / 1e9;
    println!(
        "     known: rimatr = {:.3} GB [{}x{}], rimatr_bse = {:.3} GB [{}x{}]",
        gb(n_bas, m_reg), n_bas * (n_bas + 1) / 2, m_reg,
        gb(n_bas, m_bse), n_bas * (n_bas + 1) / 2, m_bse
    );

    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    stage("5. + inverse dielectric");

    let ctx = ri_bse::matvec_fast::FastBseContext::build(&scf, &inv).expect("AO context");
    let (_, _, o, v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    stage("6. + AO context (folds)");
    println!(
        "     known: folds = {:.3} GB (M x (V^2 + O^2 + 2OV)), D = {:.3} GB, \
         total reported = {:.3} GB   [n_o={o} n_v={v} n_aux={}]",
        ctx.fold_bytes() as f64 / 1e9,
        (inv.size[0] * inv.size[1]) as f64 * 8.0 / 1e9,
        ctx.bytes as f64 / 1e9,
        inv.size[0]
    );

    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let z = lcg(o * v, 7);
    std::hint::black_box(ctx.a_block_matvec(&scf, &qp_ctrl, &z));
    std::hint::black_box(ctx.b_block_matvec(&scf, &qp_ctrl, &z));
    stage("7. + one A and one B matvec");

    drop(ctx);
    trim();
    stage("8. after dropping the AO context + trim");
}
