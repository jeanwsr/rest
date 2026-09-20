//! Peak-memory attribution for the two GW tensor routes.
//!
//! Each configuration runs in its **own process** (the harness is driven by env
//! vars and a shell loop) so that `VmHWM` — the kernel's monotonic high-water
//! mark — really is the peak of that configuration, and not carried over from a
//! previous one.
//!
//! ```text
//! REST_PEAK_ROUTE=mo|ao REST_PEAK_WORKERS=<n> REST_PEAK_FREQ=<n> \
//! REST_PEAK_SYS=benzene|naphthalene|anthracene \
//!   cargo test --release --test bench_gw_memory_peaks -- --nocapture --ignored
//! ```

use pyrest::ctrl_io::quasiparticle_methods::QuasiParticle;
use pyrest::molecule_io::Molecule;
use pyrest::ri_gw::tensor_ao::{self, GwAoPlan};
use pyrest::scf_io::{self, SCF};

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
gw_extrapolate_occ_threshold = 0.45
gw_extrapolate_vir_threshold = 0.45
bse_cutoff_energy = 1.0e6
"##
    )
}

fn build(geom: &str, basis: &str, aux: &str, style: &str) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_for(geom, basis, aux)[..]).unwrap();
    let (mut ctrl, geom) = pyrest::ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mut qp = QuasiParticle {
        gw_scheme: "extrapolated".to_string(),
        scgw: "g0w0".to_string(),
        gw_or_bse: "gw".to_string(),
        use_low_rank_contour: false,
        gw_extrapolate_occ_threshold: 0.45,
        gw_extrapolate_vir_threshold: 0.45,
        gw_tensor_style: style.to_string(),
        ..Default::default()
    };
    qp.gw_tensor_style = style.to_string();
    ctrl.quasiparticle_methods = Some(qp);
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    pyrest::ri_gw::initialize_qp_g_w(&mut scf);
    scf
}

fn geom_block(name: &str, xyz: &str) -> String {
    format!("name = \"{}\"\nunit = \"Angstrom\"\nposition = \"\"\"\n{}\n\"\"\"\n", name, xyz)
}

fn polyacene(n: usize) -> String {
    let cc = 1.39_f64;
    let ch = 1.09_f64;
    let mut cs: Vec<(f64, f64)> = Vec::new();
    let dx = 3.0_f64.sqrt() * cc;
    for ring in 0..n {
        let cx = ring as f64 * dx;
        for k in 0..6 {
            let ang = std::f64::consts::PI / 3.0 * (k as f64) + std::f64::consts::PI / 6.0;
            let (x, y) = (cx + cc * ang.cos(), cc * ang.sin());
            if !cs.iter().any(|(px, py)| (px - x).abs() < 1e-6 && (py - y).abs() < 1e-6) {
                cs.push((x, y));
            }
        }
    }
    let mut s = String::new();
    for (x, y) in &cs {
        s.push_str(&format!("    C  {:14.8} {:14.8} {:14.8}\n", x, y, 0.0));
    }
    let xmin = cs.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let xmax = cs.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    for (x, y) in &cs {
        let mut nb = 0;
        for (x2, y2) in &cs {
            let d = ((x - x2).powi(2) + (y - y2).powi(2)).sqrt();
            if d > 1e-6 && d < cc * 1.2 {
                nb += 1;
            }
        }
        if nb < 3 {
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

fn vm(field: &str) -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn mb(bytes: f64) -> f64 {
    bytes / 1048576.0
}

// -------------------------------------------------------------------- main --

/// Polls `VmRSS` in a background thread and keeps the maximum, so that a stage's
/// transient peak is measured even when an earlier stage (the SCF) already left a
/// higher `VmHWM` behind.
struct Peak {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    max_kb: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Peak {
    fn start() -> Self {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let max_kb = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (s2, m2) = (stop.clone(), max_kb.clone());
        let handle = std::thread::spawn(move || {
            while !s2.load(std::sync::atomic::Ordering::Relaxed) {
                let v = vm("VmRSS:") * 1024.0;
                let cur = m2.load(std::sync::atomic::Ordering::Relaxed);
                if (v as u64) > cur {
                    m2.store(v as u64, std::sync::atomic::Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
        });
        Peak { stop, max_kb, handle: Some(handle) }
    }
    fn stop(mut self) -> f64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.max_kb.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.0
    }
}

#[test]
#[ignore]
fn peak_attribution() {
    let route = std::env::var("REST_PEAK_ROUTE").unwrap_or_else(|_| "mo".into());
    let workers: Option<usize> = std::env::var("REST_PEAK_WORKERS").ok().and_then(|s| s.parse().ok());
    let freq: usize = std::env::var("REST_PEAK_FREQ").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let sys = std::env::var("REST_PEAK_SYS").unwrap_or_else(|_| "benzene".into());
    let geom = match sys.as_str() {
        "naphthalene" => geom_block("naphthalene", &polyacene(2)),
        "anthracene" => geom_block("anthracene", &polyacene(3)),
        _ => geom_block("benzene", &polyacene(1)),
    };
    match workers {
        Some(w) => std::env::set_var("REST_GW_MAX_WORKERS", w.to_string()),
        None => std::env::remove_var("REST_GW_MAX_WORKERS"),
    }

    let mut scf = build(&geom, "cc-pvdz", "basis-set-pool/cc-pvdz-rifit", &route);
    let (start_mo, num_state, n_o, n_v, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'Y');
    let n_mo = num_state - start_mo;
    let n = scf.mol.num_basis;
    let m = scf.rimatr.as_ref().unwrap().0.size[1];
    let n_packed = n * (n + 1) / 2;
    let nov = (n_o * n_v) as f64;

    let f = |x: f64| format!("{:8.1}", mb(x));

    println!("@@ @@@ CONFIG route={} workers={:?} nfreq={} sys={}", route, workers, freq, sys);
    println!("@@ @@@ dims n_bas={} n_aux={} n_mo={} O={} V={} n_packed={}", n, m, n_mo, n_o, n_v, n_packed);
    println!("@@ T rimatr             [n_packed,n_aux]   {} MB", f((n_packed * m) as f64 * 8.0));
    println!("@@ T ri_ov               [n_aux,O*V]        {} MB", f(m as f64 * nov * 8.0));
    println!("@@ T v_matrix            [n_mo,n_mo]        {} MB", f((n_mo * n_mo) as f64 * 8.0));
    println!("@@ T w_c_at_freqs        nfreq*[n_mo,n_mo]  {} MB", f((freq * n_mo * n_mo) as f64 * 8.0));
    println!("@@ T MO resp+inv (x1)   2*[n_aux,n_aux]     {} MB", f(2.0 * (m * m) as f64 * 8.0));
    println!("@@ T MO ri3mo blk (x1)  [n_aux,25,n_mo]     {} MB  (x2: rimo + channel)", f((m * 25 * n_mo) as f64 * 8.0));
    let nw_mo = workers.unwrap_or(freq);
    println!("@@ T MO per-worker      resp+inv+2xblk      {} MB", f((2.0 * (m * m) as f64 + 2.0 * (m * 25 * n_mo) as f64) * 8.0));
    println!("@@ T MO aggregate       {} workers          {} MB", nw_mo, f((2.0 * (m * m) as f64 + 2.0 * (m * 25 * n_mo) as f64) * 8.0 * nw_mo as f64));
    println!("@@ T AO fold g          [n_aux,b*n_cols]    {} MB (b=8)", f((m * 8 * n_mo) as f64 * 8.0));
    let nw_ao = workers.unwrap_or_else(rayon::current_num_threads);
    println!("@@ T AO chi scratch     per worker: chi[M,M]+u,us[M,V]+tb[N,M]+xv[N,V]  {} MB",
        f(((m * m) as f64 + 2.0 * (m * n_v) as f64 + (n * m) as f64 + (n * n_v) as f64) * 8.0));
    let chi_scratch = (m * m) as f64 + 2.0 * (m * n_v) as f64 + (n * m) as f64 + (n * n_v) as f64;
    let ao_agg = (m * 8 * n_mo) as f64 * (1.0 + nw_ao as f64)
        + (chi_scratch * nw_ao as f64).max((m * m) as f64 * freq as f64);
    println!(
        "@@ T AO aggregate       g + {} tmp + max(chi*{},{}) inv  {} MB",
        nw_ao, nw_ao, freq, f(ao_agg * 8.0)
    );

    println!("@@ R rss_before_tensor_phase           {:8.1} MB", vm("VmRSS:"));

    if std::env::var("REST_PEAK_DRIVER").is_ok() {
        // Full GW driver in a fresh SCF (the SCF's own peak is already behind us;
        // the sampler reports the RSS high-water of the driver phase).
        let mut d = build(&geom, "cc-pvdz", "basis-set-pool/cc-pvdz-rifit", &route);
        let vxc = pyrest::ri_gw::vxc_ao2mo(&d);
        let base = vm("VmRSS:");
        let p = Peak::start();
        let qp = pyrest::ri_gw::scgw::gw_near_fermi_surface(&mut d, freq, &vxc, 0.45, 0.45);
        let pk = p.stop();
        println!("@@ D driver_rss_base                   {:8.1} MB", base);
        println!("@@ D driver_peak                       {:8.1} MB  (delta {:8.1})", pk, pk - base);
        println!("@@ V {} HOMO_QP {:.10}", route, qp[d.homo[0]]);
        return;
    }

    if route == "ao" {
        let p = Peak::start();
        let ctx = GwAoPlan::build(&scf).unwrap();
        println!("@@ P ao_ctx_build                      {:8.1} MB", p.stop());

        let p = Peak::start();
        let _row = tensor_ao::ri_row(&scf, &ctx, start_mo, start_mo, n_mo);
        println!("@@ P ri_row (O(N^2))                   {:8.1} MB", p.stop());

        let p = Peak::start();
        let _v = tensor_ao::v_matrix(&scf, &ctx);
        println!("@@ P v_matrix                         {:8.1} MB", p.stop());

        let p = Peak::start();
        let _chi = tensor_ao::response_matrix(&scf, &ctx, &scf.gwqp.1, 0.7, 'I', 0.0);
        println!("@@ P response_matrix (no ri_ov)       {:8.1} MB", p.stop());

        let p = Peak::start();
        let _wc = pyrest::ri_gw::generate_w_c_ao(&scf, &ctx, freq, true);
        println!("@@ P generate_w_c_ao                  {:8.1} MB", p.stop());
        drop(_wc);
    } else {
        let p = Peak::start();
        let _v = pyrest::ri_gw::v_matrix_from_scf(&scf);
        println!("@@ P v_matrix_from_scf                {:8.1} MB", p.stop());

        let p = Peak::start();
        let ri_ov = pyrest::ri_bse::get_submatrix(&scf, 'O', 'V', 'Y');
        println!("@@ P ri_ov                            {:8.1} MB", p.stop());

        let p = Peak::start();
        let _wc = pyrest::ri_gw::generate_w_c(&scf, &ri_ov, &scf.gwqp.0, &scf.gwqp.1, num_state, n_o, n_v, freq);
        println!("@@ P generate_w_c                     {:8.1} MB", p.stop());
        drop(_wc);
    }
}
