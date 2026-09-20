//! Benchmark: MO-basis vs AO-basis GW tensors (`gw_tensor_style = "mo" | "ao"`).
//!
//! Two measurements per system:
//!
//! * **tensor construction** — every RI three-centre tensor a full GW run needs:
//!   all `n_mo` rows `(Q|n,m)` (what `compute_ri3mo_row` produces, once per
//!   orbital), the `V[n,m]` matrix and the whole imaginary-axis `W_c` grid;
//! * **end-to-end** — `gw_near_fermi_surface` (the `gw_scheme = "extrapolated"`,
//!   `gw_variant = "cd"` driver) with both routes, reporting wall time and the
//!   process peak RSS.
//!
//! Run with
//! ```text
//! CARGO_TARGET_DIR=target_u cargo test --release --test bench_gw_tensor_ao \
//!     -- --nocapture --ignored --test-threads=1
//! ```

use pyrest::ctrl_io::quasiparticle_methods::QuasiParticle;
use pyrest::molecule_io::Molecule;
use pyrest::ri_gw::tensor_ao::{self, GwAoPlan};
use pyrest::scf_io::{self, SCF};
use std::time::Instant;

// ---------------------------------------------------------------- scaffolding

fn input_for(geom: &str, basis: &str, aux: &str) -> String {
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
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-7
scf_acc_etot = 1.0e-9
[geom]
{geom}
[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
bse_cutoff_energy = 1.0e6
"##
    )
}

fn qp_default() -> QuasiParticle {
    QuasiParticle {
        gw_scheme: "extrapolated".to_string(),
        scgw: "g0w0".to_string(),
        gw_or_bse: "gw".to_string(),
        use_low_rank_contour: false,
        gw_extrapolate_occ_threshold: 100.0,
        gw_extrapolate_vir_threshold: 100.0,
        gw_tensor_style: "mo".to_string(),
        ..Default::default()
    }
}

fn build(geom: &str, basis: &str, aux: &str, style: &str) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_for(geom, basis, aux)[..]).unwrap();
    let (mut ctrl, geom) = pyrest::ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mut qp = qp_default();
    qp.gw_tensor_style = style.to_string();
    ctrl.quasiparticle_methods = Some(qp);
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    pyrest::ri_gw::initialize_qp_g_w(&mut scf);
    scf
}

fn geom_block(name: &str, xyz: &str, unit: &str) -> String {
    format!("name = \"{}\"\nunit = \"{}\"\nposition = \"\"\"\n{}\n\"\"\"\n", name, unit, xyz)
}

/// Fused polyacene (linear acene) skeleton in the xy plane, ring count `n`.
fn polyacene(n: usize) -> String {
    // Anthracene-like: build from fused hexagons, C-C = 1.39 A, C-H = 1.09 A.
    let cc = 1.39_f64;
    let ch = 1.09_f64;
    let mut cs: Vec<(f64, f64)> = Vec::new();
    // hexagon centres along x, separated by sqrt(3)*cc
    let dx = 3.0_f64.sqrt() * cc;
    for ring in 0..n {
        let cx = ring as f64 * dx;
        for k in 0..6 {
            let ang = std::f64::consts::PI / 3.0 * (k as f64) + std::f64::consts::PI / 6.0;
            let x = cx + cc * ang.cos();
            let y = cc * ang.sin();
            // dedupe shared atoms between fused rings
            if !cs.iter().any(|(px, py)| (px - x).abs() < 1e-6 && (py - y).abs() < 1e-6) {
                cs.push((x, y));
            }
        }
    }
    let mut s = String::new();
    for (x, y) in &cs {
        s.push_str(&format!("    C  {:14.8} {:14.8} {:14.8}\n", x, y, 0.0));
    }
    // hydrogens: attach roughly radially, keeping the molecule planar
    let xmin = cs.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let xmax = cs.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    for (x, y) in &cs {
        let mut dirs = 0;
        for (x2, y2) in &cs {
            let d = ((x - x2).powi(2) + (y - y2).powi(2)).sqrt();
            if d > 1e-6 && d < cc * 1.2 {
                dirs += 1;
            }
        }
        if dirs < 3 {
            // outward direction: y sign, or x sign for the terminal carbons
            let (ux, uy) = if *x < xmin + 0.5 * cc {
                (-1.0, 0.0)
            } else if *x > xmax - 0.5 * cc {
                (1.0, 0.0)
            } else if *y > 0.0 {
                (0.0, 1.0)
            } else {
                (0.0, -1.0)
            };
            s.push_str(&format!(
                "    H  {:14.8} {:14.8} {:14.8}\n",
                x + ux * ch,
                y + uy * ch,
                0.0
            ));
        }
    }
    s
}

fn benzene() -> String {
    geom_block("benzene", &polyacene(1), "Angstrom")
}
fn naphthalene() -> String {
    geom_block("naphthalene", &polyacene(2), "Angstrom")
}
fn anthracene() -> String {
    geom_block("anthracene", &polyacene(3), "Angstrom")
}

fn rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

/// Peak RSS is tracked by a sampler thread; `VmHWM` in /proc/self/status is the
/// kernel's own high-water mark and is what we report (it never decreases).
fn vm_hwm_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

// ------------------------------------------------------------------ benches --

/// Cost of the three tensor-construction phases a full GW run performs.
#[test]
#[ignore]
fn bench_tensor_construction() {
    let systems: Vec<(&str, String, &str, &str)> = vec![
        ("benzene/cc-pVDZ", benzene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit"),
        ("naphthalene/cc-pVDZ", naphthalene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit"),
        ("anthracene/cc-pVDZ", anthracene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit"),
    ];

    for (label, geom, basis, aux) in systems {
        // ONE converged SCF, deep-cloned for the AO route.  Two independently
        // converged SCF solutions differ by ~1e-8 in the orbitals, which would
        // swamp the round-off-level agreement the two tensor routes actually have.
        let scf_mo = build(&geom, basis, aux, "mo");
        let scf_ao = scf_mo.clone();
        let (start_mo, num_state, n_o, n_v, _, _) =
            pyrest::ri_gw::get_occupation_parameters(&scf_mo, 'Y');
        let n_mo = num_state - start_mo;
        println!(
            "\n=== {} : n_bas={} n_aux={} n_mo={} ({} occ / {} vir) ===",
            label,
            scf_mo.mol.num_basis,
            scf_mo.rimatr.as_ref().unwrap().0.size[1],
            n_mo,
            n_o,
            n_v
        );

        // ---------------------------------------------------------- MO route --
        let rss0 = rss_mb();
        let t = Instant::now();
        let rows_mo: Vec<_> = (0..n_mo)
            .map(|n| pyrest::ri_gw::compute_ri3mo_row(&scf_mo, n))
            .collect();
        let mo_rows_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v_mo = pyrest::ri_gw::v_matrix_from_scf(&scf_mo);
        let mo_v_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ri_ov_mo = pyrest::ri_bse::get_submatrix(&scf_mo, 'O', 'V', 'Y');
        let wc_mo = pyrest::ri_gw::generate_w_c(
            &scf_mo, &ri_ov_mo, &scf_mo.gwqp.0, &scf_mo.gwqp.1, num_state, n_o, n_v, 20,
        );
        let mo_wc_s = t.elapsed().as_secs_f64();
        let mo_total = mo_rows_s + mo_v_s + mo_wc_s;
        let mo_rss = rss_mb();
        println!(
            "MO: rows {:.2} s | V {:.2} s | W_c(20 freq) {:.2} s | total {:.2} s | RSS {:.0} -> {:.0} MB",
            mo_rows_s, mo_v_s, mo_wc_s, mo_total, rss0, mo_rss
        );

        // ---------------------------------------------------------- AO route --
        let rss0 = rss_mb();
        let t = Instant::now();
        let ctx = GwAoPlan::build(&scf_ao).unwrap();
        let ao_build_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let rows_ao: Vec<_> = (0..n_mo)
            .map(|n| tensor_ao::ri_row(&scf_ao, &ctx, start_mo + n, start_mo, n_mo))
            .collect();
        let ao_rows_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v_ao = tensor_ao::v_matrix(&scf_ao, &ctx);
        let ao_v_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ri_ov_ao = tensor_ao::ri_ov_materialised(&scf_ao, &ctx);
        let wc_ao = pyrest::ri_gw::generate_w_c_ao(&scf_ao, &ctx, 20, true);
        let ao_wc_s = t.elapsed().as_secs_f64();
        let ao_total = ao_build_s + ao_rows_s + ao_v_s + ao_wc_s;
        let ao_rss = rss_mb();
        println!(
            "AO: build {:.2} s | rows {:.2} s | V {:.2} s | W_c(20 freq) {:.2} s | total {:.2} s | RSS {:.0} -> {:.0} MB",
            ao_build_s, ao_rows_s, ao_v_s, ao_wc_s, ao_total, rss0, ao_rss
        );

        // ------------------------------------------------- agreement check ---
        let mut e_rows = 0.0_f64;
        let mut scale = 0.0_f64;
        for (a, b) in rows_mo.iter().zip(rows_ao.iter()) {
            for k in 0..a.data.len() {
                e_rows = e_rows.max((a.data[k] - b.data[k]).abs());
            }
            scale = scale.max(a.data.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
        }
        let mut e_v = 0.0_f64;
        for a in 0..n_mo {
            for b in 0..n_mo {
                e_v = e_v.max((v_mo[[a, b]] - v_ao[[a, b]]).abs());
            }
        }
        let mut e_wc = 0.0_f64;
        for (o1, _, m1) in wc_mo.iter() {
            let found = wc_ao.iter().find(|(o2, _, _)| (o1 - o2).abs() < 1e-12);
            let (_, _, m2) = found.expect("frequency missing");
            for a in 0..num_state {
                for b in 0..num_state {
                    e_wc = e_wc.max((m1[[a, b]] - m2[[a, b]]).abs());
                }
            }
        }
        println!(
            "agreement: max|d ri3mo row| {:.2e} (global scale {:.2e}, relative {:.2e}); max|d V| {:.2e}; max|d W_c| {:.2e}",
            e_rows,
            scale,
            e_rows / scale.max(1e-300),
            e_v,
            e_wc
        );
        println!(
            "SPEED-UP: rows {:.1}x | V {:.1}x | W_c {:.1}x | TOTAL {:.1}x   (AO build included)",
            mo_rows_s / ao_rows_s.max(1e-9),
            mo_v_s / ao_v_s.max(1e-9),
            mo_wc_s / ao_wc_s.max(1e-9),
            mo_total / ao_total
        );
    }
}

/// End-to-end wall time and peak RSS of the extrapolated-CD GW driver.
#[test]
#[ignore]
fn bench_end_to_end() {
    let mut systems = vec![("benzene/cc-pVDZ", benzene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit")];
    if std::env::var("REST_BENCH_E2E_BIG").is_ok() {
        systems.push(("naphthalene/cc-pVDZ", naphthalene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit"));
    }
    for (label, geom, basis, aux) in systems {
        println!("\n=== end-to-end GW: {} (occ/vir window 0.45 Ha, 12 frequencies) ===", label);
        let mut qps: Vec<(String, Vec<f64>, f64, f64, f64)> = Vec::new();
        let base = build(&geom, basis, aux, "mo");
        for style in ["mo", "ao"] {
            // deep clone of the same converged SCF: the only difference between
            // the two runs is `gw_tensor_style`
            let mut scf = base.clone();
            scf.mol.ctrl.quasiparticle_methods.as_mut().unwrap().gw_tensor_style = style.to_string();
            let vxc = pyrest::ri_gw::vxc_ao2mo(&scf);
            let hwm0 = vm_hwm_mb();
            let t = Instant::now();
            // Restrict the explicitly computed window to the frontier region;
            // the rest is returned by the driver's own extrapolation.  The
            // measured wall time therefore excludes the (identical) cost of
            // solving orbitals the user never asked for.
            let qp = pyrest::ri_gw::scgw::gw_near_fermi_surface(&mut scf, 12, &vxc, 0.45, 0.45);
            let dt = t.elapsed().as_secs_f64();
            let hwm = vm_hwm_mb();
            println!(
                "  [{}] wall {:.1} s | peak RSS {:.0} MB (delta {:.0} MB) | HOMO QP {:.8}",
                style,
                dt,
                hwm,
                hwm - hwm0,
                qp[scf.homo[0]]
            );
            qps.push((style.to_string(), qp, dt, hwm, hwm0));
        }
        if qps.len() == 2 {
            let d = qps[0]
                .1
                .iter()
                .zip(qps[1].1.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            println!(
                "  max|dQP| = {:.2e} Ha | time ratio AO/MO = {:.3} | peak RSS MO {:.0} MB vs AO {:.0} MB",
                d,
                qps[1].2 / qps[0].2,
                qps[0].3,
                qps[1].3
            );
            let mut worst: Vec<(usize, f64, f64, f64)> = qps[0]
                .1
                .iter()
                .zip(qps[1].1.iter())
                .enumerate()
                .map(|(i, (a, b))| (i, (a - b).abs(), *a, *b))
                .collect();
            worst.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
            println!("  largest QP deviations:");
            for (i, dd, a, b) in worst.iter().take(6) {
                println!("    orbital {:3}: d = {:.3e}  MO = {:+.10}  AO = {:+.10}", i, dd, a, b);
            }
        }
    }
}

/// Pre-screening behaviour of the AO engine on a real molecule.
#[test]
#[ignore]
fn bench_screening() {
    let scf = build(&naphthalene(), "cc-pvdz", "basis-set-pool/cc-pvdz-rifit", "mo");
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let n_mo = num_state - start_mo;
    let mo_ri_ov = pyrest::ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let scale = mo_ri_ov.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    println!(
        "naphthalene/cc-pVDZ: n_bas={} n_aux={} n_mo={}, max|ri_ov| = {:.4}",
        scf.mol.num_basis, mo_ri_ov.size[0], n_mo, scale
    );
    for tol in [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3] {
        let t = Instant::now();
        let ctx = GwAoPlan::build_with_screening(&scf, tol).unwrap();
        let build_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ri_ov = tensor_ao::ri_ov_materialised(&scf, &ctx);
        let fold_s = t.elapsed().as_secs_f64();
        let mut e = 0.0_f64;
        for k in 0..mo_ri_ov.data.len() {
            e = e.max((mo_ri_ov.data[k] - ri_ov.data[k]).abs());
        }
        let t = Instant::now();
        let row = tensor_ao::ri_row(&scf, &ctx, start_mo + 1, start_mo, n_mo);
        let row_s = t.elapsed().as_secs_f64();
        let _ = (n_o, n_v);
        println!(
            "tol = {:.1e}: pairs {:.2}% | build {:.2} s | ri_ov {:.3} s (err {:.2e}) | one row {:.4} s",
            tol,
            100.0 * (ctx.kept_pairs as f64 / ctx.dims.n_packed as f64),
            build_s,
            fold_s,
            e,
            row_s
        );
    }
}

/// Tuning sweep for the AO `W_c` build: row-block size and worker count.
#[test]
#[ignore]
fn bench_wc_tuning() {
    let geom = benzene();
    let scf = build(&geom, "cc-pvdz", "basis-set-pool/cc-pvdz-rifit", "mo");
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let n_mo = num_state - start_mo;
    let mo_ri_ov = pyrest::ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
    let t = Instant::now();
    let _ = pyrest::ri_gw::generate_w_c(&scf, &mo_ri_ov, &scf.gwqp.0, &scf.gwqp.1, num_state, n_o, n_v, 20);
    println!("MO W_c (20 freq): {:.3} s", t.elapsed().as_secs_f64());

    let ctx = GwAoPlan::build(&scf).unwrap();
    let ri_ov = tensor_ao::ri_ov_materialised(&scf, &ctx);
    for block in [8usize, 16, 25, 50, 114] {
        for workers in [1usize, 4] {
            std::env::set_var("REST_GW_AO_BLOCK", block.to_string());
            std::env::set_var("REST_GW_MAX_WORKERS", workers.to_string());
            let t = Instant::now();
            let _ = pyrest::ri_gw::generate_w_c_ao(&scf, &ctx, 20, workers != 1);
            println!(
                "AO W_c (20 freq): block {:>3} workers {} -> {:.3} s",
                block,
                workers,
                t.elapsed().as_secs_f64()
            );
        }
    }
    std::env::remove_var("REST_GW_AO_BLOCK");
    std::env::remove_var("REST_GW_MAX_WORKERS");
}
