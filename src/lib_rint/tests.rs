use super::*;
use crate::basis_io::{BasCell, Basis4Elem};
use crate::lib_rint::basis::Shell;
use crate::scf_io::{
    scf_without_build, vj_upper_with_rimatr_sync, vk_upper_with_rimatr_use_dm_only_sync_v02, SCF,
};
use crate::Molecule;
use rest_libcint::prelude::rest_libcint_wrapper::int1e_r;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_temp_id() -> String {
    let count = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}", std::process::id(), count)
}

fn write_temp_ctrl_h2(basis_dir: &str) -> String {
    let temp_id = next_temp_id();
    let path = format!("/tmp/lib_rint_kernel_check_{basis_dir}_{temp_id}.toml")
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
    let num_threads = std::env::var("TMP_EXACT4C_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1);
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = {num_threads}\nrun_lib_rint = false\n\n[geom]\nname = \"H2\"\nunit = \"angstrom\"\nposition = [\n    \"H   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.0000000000   0.0000000000   1.4000000000\",\n]\n"
    );
    fs::write(&path, text).unwrap();
    path
}

fn write_temp_ctrl_h2o(basis_dir: &str) -> String {
    let temp_id = next_temp_id();
    let path = format!("/tmp/lib_rint_kernel_h2o_check_{basis_dir}_{temp_id}.toml")
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
    let num_threads = std::env::var("TMP_EXACT4C_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1);
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = {num_threads}\nrun_lib_rint = false\n\n[geom]\nname = \"H2O\"\nunit = \"angstrom\"\nposition = [\n    \"O   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.7586020000   0.0000000000   0.5042840000\",\n    \"H  -0.7586020000   0.0000000000   0.5042840000\",\n]\n"
    );
    fs::write(&path, text).unwrap();
    path
}

fn write_temp_high_l_he_basis(l: u32) -> (String, String) {
    let temp_id = next_temp_id();
    let basis_dir = format!("/tmp/lib_rint_high_l_basis_l{l}_{temp_id}");
    fs::create_dir_all(&basis_dir).unwrap();
    let basis_path = format!("{basis_dir}/He.json");
    let basis_json = format!(
        r#"{{
"electron_shells": [
    {{
        "function_type": "gto",
        "region": "",
        "angular_momentum": [{l}],
        "exponents": ["0.75"],
        "coefficients": [["1.0"]]
    }}
],
"references": null,
"ecp_potentials": null,
"ecp_electrons": null
}}"#
    );
    fs::write(&basis_path, basis_json).unwrap();

    let ctrl_path = format!("/tmp/lib_rint_high_l_ctrl_l{l}_{temp_id}.toml");
    let ctrl_text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"He_high_l\"\nunit = \"angstrom\"\nposition = [\n    \"He   0.0000000000   0.0000000000   0.0000000000\",\n]\n"
    );
    fs::write(&ctrl_path, ctrl_text).unwrap();
    (ctrl_path, basis_dir)
}

fn assert_high_l_4c_shell_block_matches_scalar_and_libcint(l: u32) {
    let (ctrl_path, basis_dir) = write_temp_high_l_he_basis(l);
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");
    assert_eq!(ao_shells.len(), 1);
    let shell = &ao_shells[0];
    assert_eq!(shell.shell.ang_type, l);

    let t0 = Instant::now();
    let block = int4c_r_shell_block(shell, shell, shell, shell);
    let librint_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let libcint_full = mol.int_ijkl_erifull();
    let libcint_s = t0.elapsed().as_secs_f64();
    let mut max_abs_scalar = 0.0_f64;
    let mut max_abs_libcint = 0.0_f64;
    let mut max_value = 0.0_f64;
    let sample_locs = [
        0_usize,
        shell.ao_len / 3,
        shell.ao_len / 2,
        shell.ao_len.saturating_sub(1),
    ];
    for l_loc in 0..shell.ao_len {
        for k_loc in 0..shell.ao_len {
            let col = l_loc * shell.ao_len + k_loc;
            for j_loc in 0..shell.ao_len {
                for i_loc in 0..shell.ao_len {
                    let row = j_loc * shell.ao_len + i_loc;
                    let got = block[(row, col)];
                    let i_ao = shell.ao_start + i_loc;
                    let j_ao = shell.ao_start + j_loc;
                    let k_ao = shell.ao_start + k_loc;
                    let l_ao = shell.ao_start + l_loc;
                    let libcint = *libcint_full.get(&[i_ao, j_ao, k_ao, l_ao]).unwrap();
                    max_abs_libcint = max_abs_libcint.max((got - libcint).abs());
                    max_value = max_value.max(got.abs()).max(libcint.abs());
                    if sample_locs.contains(&i_loc)
                        && sample_locs.contains(&j_loc)
                        && sample_locs.contains(&k_loc)
                        && sample_locs.contains(&l_loc)
                    {
                        let scalar = eri_ao_4c_r(&ao_bfs, i_ao, j_ao, k_ao, l_ao);
                        max_abs_scalar = max_abs_scalar.max((got - scalar).abs());
                    }
                }
            }
        }
    }
    let tol = 5.0e-8 * max_value.max(1.0);
    let nroots = ((4 * l) / 2 + 1) as usize;
    println!(
        "TMP_EXACT4C_HIGH_L_ERROR l={l} nroots={nroots} ao_len={} libcint_s={libcint_s:.6} lib_rint_s={librint_s:.6} lib_rint_over_libcint={:.3} max_abs_libcint={max_abs_libcint:.3e} max_abs_scalar={max_abs_scalar:.3e} max_value={max_value:.3e} tol={tol:.3e}",
        shell.ao_len,
        librint_s / libcint_s.max(1.0e-12)
    );
    assert!(
        max_abs_scalar <= tol,
        "high-l shell block vs scalar mismatch l={l}: max_abs={max_abs_scalar:.3e}, tol={tol:.3e}"
    );
    assert!(
        max_abs_libcint <= tol,
        "high-l shell block vs libcint mismatch l={l}: max_abs={max_abs_libcint:.3e}, tol={tol:.3e}"
    );

    let _ = fs::remove_file(ctrl_path);
    let _ = fs::remove_dir_all(basis_dir);
}

fn single_component_shell(l: u32, component: [u32; 3]) -> RintShell {
    RintShell {
        atom_idx: 0,
        center: [0.0, 0.0, 0.0],
        shell: Shell {
            ang_type: l,
            exponents: vec![0.75],
            coefficients: vec![vec![1.0]],
        },
        column_idx: 0,
        cart_components: vec![component],
        ao_start: 0,
        ao_len: 1,
        is_aux: false,
    }
}

fn single_component_basis_function(component: [u32; 3]) -> BasisFunction {
    BasisFunction {
        atom_idx: 0,
        lx: component[0],
        ly: component[1],
        lz: component[2],
        exponents: vec![0.75],
        coefficients: vec![1.0],
        center: [0.0, 0.0, 0.0],
    }
}

fn write_temp_ctrl_he2(basis_dir: &str, distance_ang: f64, label: &str) -> String {
    let temp_id = next_temp_id();
    let path =
        format!("/tmp/lib_rint_kernel_he2_{label}_{distance_ang:.3}_{basis_dir}_{temp_id}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nauxbas_path = \"def2-sv(p)-jkfit\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"He2_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n    \"He   0.0000000000   0.0000000000   0.0000000000\",\n    \"He   0.0000000000   0.0000000000   {distance_ang:.10}\",\n]\n"
    );
    fs::write(&path, text).unwrap();
    path
}

fn write_temp_ctrl_diatomic_dimer(
    elem: &str,
    basis_dir: &str,
    distance_ang: f64,
    label: &str,
) -> String {
    let temp_id = next_temp_id();
    let path = format!(
        "/tmp/lib_rint_kernel_{elem}2_{label}_{distance_ang:.3}_{basis_dir}_{temp_id}.toml"
    )
    .replace('(', "")
    .replace(')', "")
    .replace('/', "_")
    .replace(' ', "_");
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nauxbas_path = \"def2-sv(p)-jkfit\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"{elem}2_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n    \"{elem}   0.0000000000   0.0000000000   0.0000000000\",\n    \"{elem}   0.0000000000   0.0000000000   {distance_ang:.10}\",\n]\n"
    );
    fs::write(&path, text).unwrap();
    path
}

fn write_temp_ctrl_methane_dimer(basis_dir: &str, distance_ang: f64, label: &str) -> String {
    let temp_id = next_temp_id();
    let path = format!(
        "/tmp/lib_rint_kernel_ch4_dimer_{label}_{distance_ang:.3}_{basis_dir}_{temp_id}.toml"
    )
    .replace('(', "")
    .replace(')', "")
    .replace('/', "_")
    .replace(' ', "_");
    let h = 1.09_f64 / 3.0_f64.sqrt();
    let mut positions = Vec::new();
    for z0 in [0.0_f64, distance_ang] {
        positions.push(format!(
            "    \"C   0.0000000000   0.0000000000   {z0:.10}\""
        ));
        for (x, y, z) in [(h, h, h), (h, -h, -h), (-h, h, -h), (-h, -h, h)] {
            positions.push(format!("    \"H   {x:.10}   {y:.10}   {:.10}\"", z0 + z));
        }
    }
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nauxbas_path = \"def2-sv(p)-jkfit\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"CH4_dimer_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n{}\n]\n",
        positions.join(",\n")
    );
    fs::write(&path, text).unwrap();
    path
}

fn restrict_operator_to_fragment(
    op: &MatrixFull<f64>,
    aoslice: &[[usize; 4]],
    atoms: &[usize],
) -> MatrixFull<f64> {
    let nao = op.size[0];
    let mut mask = vec![false; nao];
    for &atom in atoms {
        let [_shl0, _shl1, p0, p1] = aoslice[atom];
        for ao in p0..p1 {
            mask[ao] = true;
        }
    }
    let mut frag_op = MatrixFull::new([nao, nao], 0.0_f64);
    for nu in 0..nao {
        if !mask[nu] {
            continue;
        }
        for mu in 0..nao {
            if mask[mu] {
                frag_op[(mu, nu)] = op[(mu, nu)];
            }
        }
    }
    frag_op
}

fn transform_one_body_operator_to_mo(
    op_ao: &MatrixFull<f64>,
    coeff: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    let nao = coeff.size[0];
    let nmo = coeff.size[1];
    assert_eq!(op_ao.size, [nao, nao]);
    let mut tmp = MatrixFull::new([nao, nmo], 0.0_f64);
    tmp.to_matrixfullslicemut().lapack_dgemm(
        &op_ao.to_matrixfullslice(),
        &coeff.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    let mut op_mo = MatrixFull::new([nmo, nmo], 0.0_f64);
    op_mo.to_matrixfullslicemut().lapack_dgemm(
        &coeff.to_matrixfullslice(),
        &tmp.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    op_mo
}

fn occupied_coeff_matrix(coeff: &MatrixFull<f64>, occupation: &[f64]) -> MatrixFull<f64> {
    let nao = coeff.size[0];
    let occ_idx = occupation
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if *value > 1.0e-8 { Some(idx) } else { None })
        .collect::<Vec<_>>();
    let mut c_occ = MatrixFull::new([nao, occ_idx.len()], 0.0_f64);
    for (i_occ, &i_mo) in occ_idx.iter().enumerate() {
        for mu in 0..nao {
            c_occ[(mu, i_occ)] = coeff[(mu, i_mo)];
        }
    }
    c_occ
}

fn matmul_nn(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> MatrixFull<f64> {
    assert_eq!(a.size[1], b.size[0]);
    let mut out = MatrixFull::new([a.size[0], b.size[1]], 0.0_f64);
    out.to_matrixfullslicemut().lapack_dgemm(
        &a.to_matrixfullslice(),
        &b.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    out
}

fn matmul_tn(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> MatrixFull<f64> {
    assert_eq!(a.size[0], b.size[0]);
    let mut out = MatrixFull::new([a.size[1], b.size[1]], 0.0_f64);
    out.to_matrixfullslicemut().lapack_dgemm(
        &a.to_matrixfullslice(),
        &b.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    out
}

fn trace_ct_x(c: &MatrixFull<f64>, x: &MatrixFull<f64>) -> f64 {
    assert_eq!(c.size, x.size);
    let mut trace = 0.0_f64;
    for i in 0..c.size[1] {
        for mu in 0..c.size[0] {
            trace += c[(mu, i)] * x[(mu, i)];
        }
    }
    trace
}

fn trace_product(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
    assert_eq!(a.size[0], a.size[1]);
    assert_eq!(a.size, b.size);
    let mut trace = 0.0_f64;
    for i in 0..a.size[0] {
        for j in 0..a.size[1] {
            trace += a[(i, j)] * b[(j, i)];
        }
    }
    trace
}

fn fragment_dipole_fluctuation_tensor(scf_data: &SCF, atoms: &[usize]) -> [[f64; 3]; 3] {
    let mut cint_data = scf_data.mol.initialize_cint(false);
    let (dipole_raw, dipole_shape) = cint_data.integral_s1::<int1e_r>(None);
    let dipoles = RIFull::from_vec(dipole_shape.try_into().unwrap(), dipole_raw)
        .expect("dipole integral tensor should match libcint shape");
    let dipole_full = (0..3)
        .map(|axis| {
            let mat = dipoles
                .get_reducing_matrix(axis)
                .expect("dipole tensor should have three Cartesian components");
            MatrixFull::from_vec(mat.size.try_into().unwrap(), mat.data.to_vec())
                .expect("dipole slice -> MatrixFull")
        })
        .collect::<Vec<_>>();
    let aoslice = scf_data.mol.aoslice_by_atom();
    let coeff = &scf_data.eigenvectors[0];
    let occ = &scf_data.occupation[0];
    let c_occ = occupied_coeff_matrix(coeff, occ);
    let mut ovlp = scf_data
        .ovlp
        .to_matrixfull()
        .expect("overlap MatrixUpper -> full");
    let s_inv = ovlp
        .lapack_inverse()
        .expect("overlap inverse should exist for closure dipole test");
    let dipole_frag = dipole_full
        .iter()
        .map(|op| restrict_operator_to_fragment(op, &aoslice, atoms))
        .collect::<Vec<_>>();
    let dipole_occ = dipole_frag
        .iter()
        .map(|op| transform_one_body_operator_to_mo(op, &c_occ))
        .collect::<Vec<_>>();
    let mut cov = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mu_b_c = matmul_nn(&dipole_frag[b], &c_occ);
            let s_inv_mu_b_c = matmul_nn(&s_inv, &mu_b_c);
            let mu_a_s_inv_mu_b_c = matmul_nn(&dipole_frag[a], &s_inv_mu_b_c);
            let closure_all = trace_ct_x(&c_occ, &mu_a_s_inv_mu_b_c);
            let occupied_projector = trace_product(&dipole_occ[a], &dipole_occ[b]);
            cov[a][b] = 2.0 * (closure_all - occupied_projector);
        }
    }
    cov
}

fn dipole_coupling_tensor_z(distance_bohr: f64) -> [[f64; 3]; 3] {
    let r2 = distance_bohr * distance_bohr;
    let r5 = distance_bohr.powi(5);
    let r = [0.0_f64, 0.0_f64, distance_bohr];
    let mut tensor = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            tensor[a][b] = (3.0 * r[a] * r[b] - delta * r2) / r5;
        }
    }
    tensor
}

fn fragment_connected_dipole_x(
    cov_a: [[f64; 3]; 3],
    cov_b: [[f64; 3]; 3],
    distance_bohr: f64,
) -> f64 {
    let t = dipole_coupling_tensor_z(distance_bohr);
    let mut x = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            for g in 0..3 {
                for d in 0..3 {
                    x += t[a][b] * t[g][d] * cov_a[a][g] * cov_b[b][d];
                }
            }
        }
    }
    x
}

fn log_log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let sum_x = points.iter().map(|(x, _)| x.ln()).sum::<f64>();
    let sum_y = points.iter().map(|(_, y)| y.abs().ln()).sum::<f64>();
    let sum_xx = points.iter().map(|(x, _)| x.ln() * x.ln()).sum::<f64>();
    let sum_xy = points
        .iter()
        .map(|(x, y)| x.ln() * y.abs().ln())
        .sum::<f64>();
    (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
}

fn write_temp_ctrl_h2_with_aux(basis_dir: &str, aux_basis_dir: &str) -> String {
    let temp_id = next_temp_id();
    let path = format!("/tmp/lib_rint_kernel_ri_check_{basis_dir}_{aux_basis_dir}_{temp_id}.toml")
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
    let text = format!(
        r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "{basis_dir}"
auxbas_path = "{aux_basis_dir}"
basis_type = "Cartesian"
auxbas_type = "Cartesian"
use_auxbas = true
even_tempered_basis = false
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false
[geom]
name = "H2"
unit = "angstrom"
position = [
"H   0.0000000000   0.0000000000   0.0000000000",
"H   0.0000000000   0.0000000000   1.4000000000",
]
"#
    );
    fs::write(&path, text).unwrap();
    path
}

fn sanitize_temp_label(label: &str) -> String {
    label
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_")
        .replace(':', "_")
}

fn write_temp_ctrl_from_positions(
    label: &str,
    basis_dir: &str,
    aux_basis_dir: Option<&str>,
    positions: &[String],
) -> String {
    let temp_id = next_temp_id();
    let aux_label = aux_basis_dir.unwrap_or("no_aux");
    let path = format!(
        "/tmp/lib_rint_kernel_aux_diag_{}_{}_{}_{}.toml",
        sanitize_temp_label(label),
        sanitize_temp_label(basis_dir),
        sanitize_temp_label(aux_label),
        temp_id
    );
    let aux_block = if let Some(aux_basis_dir) = aux_basis_dir {
        format!(
            "auxbas_path = \"{aux_basis_dir}\"\nauxbas_type = \"Cartesian\"\nuse_auxbas = true\n"
        )
    } else {
        "use_auxbas = false\n".to_string()
    };
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\n{aux_block}even_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"{}\"\nunit = \"angstrom\"\nposition = [\n{}\n]\n",
        sanitize_temp_label(label),
        positions.join(",\n")
    );
    fs::write(&path, text).unwrap();
    path
}

fn h2_positions() -> Vec<String> {
    vec![
        "    \"H   0.0000000000   0.0000000000   0.0000000000\"".to_string(),
        "    \"H   0.0000000000   0.0000000000   1.4000000000\"".to_string(),
    ]
}

fn h2o_positions() -> Vec<String> {
    vec![
        "    \"O   0.0000000000   0.0000000000   0.0000000000\"".to_string(),
        "    \"H   0.7586020000   0.0000000000   0.5042840000\"".to_string(),
        "    \"H  -0.7586020000   0.0000000000   0.5042840000\"".to_string(),
    ]
}

fn methane_positions(z0: f64) -> Vec<String> {
    let h = 1.09_f64 / 3.0_f64.sqrt();
    let mut positions = vec![format!(
        "    \"C   0.0000000000   0.0000000000   {z0:.10}\""
    )];
    for (x, y, z) in [(h, h, h), (h, -h, -h), (-h, h, -h), (-h, -h, h)] {
        positions.push(format!("    \"H   {x:.10}   {y:.10}   {:.10}\"", z0 + z));
    }
    positions
}

fn methane_dimer_positions(distance_ang: f64) -> Vec<String> {
    let mut positions = methane_positions(0.0);
    positions.extend(methane_positions(distance_ang));
    positions
}

fn benzene_positions() -> Vec<String> {
    let r_c = 1.397_f64;
    let r_h = 2.479_f64;
    let mut positions = Vec::with_capacity(12);
    for i in 0..6 {
        let theta = std::f64::consts::TAU * i as f64 / 6.0;
        positions.push(format!(
            "    \"C   {:.10}   {:.10}   0.0000000000\"",
            r_c * theta.cos(),
            r_c * theta.sin()
        ));
    }
    for i in 0..6 {
        let theta = std::f64::consts::TAU * i as f64 / 6.0;
        positions.push(format!(
            "    \"H   {:.10}   {:.10}   0.0000000000\"",
            r_h * theta.cos(),
            r_h * theta.sin()
        ));
    }
    positions
}

fn c60_fragment_c10_positions() -> Vec<String> {
    [
        "C       -1.23626350       0.00000000       3.32997662",
        "C       -0.38202643      -1.17575646       3.32997662",
        "C        1.00015818      -0.72665745       3.32997662",
        "C        1.00015818       0.72665745       3.32997662",
        "C       -0.38202643       1.17575646       3.32997662",
        "C        2.34433581      -2.60145768       0.59464214",
        "C        3.19857288      -1.42570122       0.59464214",
        "C        2.96246756      -0.69904376       1.83090564",
        "C        1.96230938      -1.42570122       2.59495851",
        "C        1.58028295      -2.60145768       1.83090564",
    ]
    .iter()
    .map(|line| format!("    \"{line}\""))
    .collect()
}

fn r2_auxbasis_count(geom: &GeomCell, auxbasis4elem: &[Basis4Elem]) -> usize {
    crate::lib_rint::basis::load_aux_rint_shells_from_raw(geom, auxbasis4elem)
        .map(|shells| rint_shell_basis_count(&shells))
        .unwrap_or(0)
}

fn r2_aux_diag_delta(reference: Option<RhfVeeObservables>, value: RhfVeeObservables) -> String {
    if let Some(reference) = reference {
        format!(
            " dEJ_ref={:.12e} dEK_ref={:.12e} dE_ref={:.12e}",
            (value.ej - reference.ej).abs(),
            (value.ek - reference.ek).abs(),
            (value.total - reference.total).abs()
        )
    } else {
        " dEJ_ref=NA dEK_ref=NA dE_ref=NA".to_string()
    }
}

fn compare_r2_direct_with_incore_for_ctrl(ctrl_path: &str, label: &str) -> [f64; 2] {
    let mol = Molecule::build(ctrl_path.to_string(), None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let (ao_shells, _bfs, p_cart) =
        crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.density_matrix[0],
            label,
        );
    let auxbasis4elem = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let aux_shells =
        crate::lib_rint::basis::load_aux_rint_shells_from_raw(&scf_data.mol.geom, &auxbasis4elem)
            .expect("failed to build aux rint shells");
    let dm = vec![p_cart];

    let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
        .expect("failed to build incore RI-r2 matrix");
    let j_incore_u = vj_upper_with_rimatr_r2_sync(&Some(ri.clone()), &dm, 1, 1.0);
    let k_incore_u = vk_upper_with_rimatr_r2_sync(&Some(ri), &dm, 1, 1.0);
    let j_incore = j_incore_u[0].to_matrixfull().unwrap();
    let k_incore = k_incore_u[0].to_matrixfull().unwrap();

    let coeff_cart = [transform_mo_coeff_to_cartesian_shell_shared(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.eigenvectors[0],
        rint_shell_basis_count(&ao_shells),
        label,
        false,
    )];
    let occupation = [scf_data.occupation[0].clone()];
    let (j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
        &ao_shells,
        &aux_shells,
        &dm,
        Some(&coeff_cart),
        Some(&occupation),
        AUXBAS_THRESHOLD,
    )
    .expect("failed to build semi-direct RI-r2 J/K");
    let j_semi = j_semi_u[0].to_matrixfull().unwrap();
    let k_semi = k_semi_u[0].to_matrixfull().unwrap();
    [
        max_abs_diff_matrix(&j_incore, &j_semi),
        max_abs_diff_matrix(&k_incore, &k_semi),
    ]
}

#[test]
fn r2_direct_matches_incore_for_h2_sto3g() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let (ao_shells, _bfs, p_cart) =
        crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.density_matrix[0],
            "r2_direct_matches_incore_for_h2_sto3g",
        );
    let auxbasis4elem = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let aux_shells =
        crate::lib_rint::basis::load_aux_rint_shells_from_raw(&scf_data.mol.geom, &auxbasis4elem)
            .expect("failed to build aux rint shells");
    let dm = vec![p_cart];

    let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
        .expect("failed to build incore RI-r2 matrix");
    let j_incore_u = vj_upper_with_rimatr_r2_sync(&Some(ri.clone()), &dm, 1, 1.0);
    let k_incore_u = vk_upper_with_rimatr_r2_sync(&Some(ri), &dm, 1, 1.0);

    let j_incore = j_incore_u[0].to_matrixfull().unwrap();
    let k_incore = k_incore_u[0].to_matrixfull().unwrap();

    let coeff_cart = [transform_mo_coeff_to_cartesian_shell_shared(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.eigenvectors[0],
        rint_shell_basis_count(&ao_shells),
        "r2_direct_matches_incore_for_h2_sto3g[semi-direct]",
        false,
    )];
    let occupation = [scf_data.occupation[0].clone()];
    let (j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
        &ao_shells,
        &aux_shells,
        &dm,
        Some(&coeff_cart),
        Some(&occupation),
        AUXBAS_THRESHOLD,
    )
    .expect("failed to build semi-direct RI-r2 J/K");
    let j_semi = j_semi_u[0].to_matrixfull().unwrap();
    let k_semi = k_semi_u[0].to_matrixfull().unwrap();
    let d_j_semi = max_abs_diff_matrix(&j_incore, &j_semi);
    let d_k_semi = max_abs_diff_matrix(&k_incore, &k_semi);
    println!("H2/sto-3g RI-r2 semi-direct vs incore: dJ={d_j_semi:.3e} dK={d_k_semi:.3e}");
    assert!(
        d_j_semi < R2_SEMIDIRECT_REGRESSION_TOL,
        "RI-r2 semi-direct J drifted from incore: {d_j_semi}"
    );
    assert!(
        d_k_semi < R2_SEMIDIRECT_REGRESSION_TOL,
        "RI-r2 semi-direct K drifted from incore: {d_k_semi}"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let (ao_shells, _bfs, p_cart) =
        crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.density_matrix[0],
            "r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g",
        );
    let auxbasis4elem = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let aux_shells =
        crate::lib_rint::basis::load_aux_rint_shells_from_raw(&scf_data.mol.geom, &auxbasis4elem)
            .expect("failed to build aux rint shells");
    let dm = vec![p_cart];

    let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
        .expect("failed to build incore RI-r2 matrix");
    let coeff_alpha = transform_mo_coeff_to_cartesian_shell_shared(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.eigenvectors[0],
        rint_shell_basis_count(&ao_shells),
        "r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g[alpha]",
        false,
    );
    let coeff_cart = [coeff_alpha.clone(), MatrixFull::empty()];
    let occupation = [scf_data.occupation[0].clone(), Vec::new()];
    let num_elec_alpha = occupation[0].iter().sum::<f64>();
    let num_elec = [num_elec_alpha, num_elec_alpha, 0.0_f64];
    let k_occ_u =
        vk_upper_with_rimatr_sync_v03(&Some(ri), &coeff_cart, &num_elec, &occupation, 1, 1.0);
    let k_occ = k_occ_u[0].to_matrixfull().unwrap();

    let coeff_semi = [coeff_alpha];
    let occupation_semi = [occupation[0].clone()];
    let (_j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
        &ao_shells,
        &aux_shells,
        &dm,
        Some(&coeff_semi),
        Some(&occupation_semi),
        AUXBAS_THRESHOLD,
    )
    .expect("failed to build semi-direct RI-r2 J/K");
    let k_semi = k_semi_u[0].to_matrixfull().unwrap();
    let d_k = max_abs_diff_matrix(&k_occ, &k_semi);
    println!("H2/sto-3g RI-r2 incore occ-coeff K vs semi-direct: dK={d_k:.3e}");
    assert!(
        d_k < R2_SEMIDIRECT_REGRESSION_TOL,
        "RI-r2 incore occ-coeff K drifted from semi-direct: {d_k}"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
#[ignore = "medium molecule RI-r2 K drift is still under investigation; keep as targeted diagnostic"]
fn r2_direct_matches_incore_for_medium_sto3g_molecules() {
    let cases = [
        ("H2O/sto-3g", write_temp_ctrl_h2o("sto-3g")),
        ("NH3/sto-3g", write_temp_ctrl_nh3("sto-3g")),
    ];

    for (label, ctrl_path) in cases {
        let [d_j_semi, d_k_semi] = compare_r2_direct_with_incore_for_ctrl(&ctrl_path, label);
        println!("{label} RI-r2 semi-direct vs incore: dJ={d_j_semi:.3e} dK={d_k_semi:.3e}");
        assert!(
            d_j_semi < R2_SEMIDIRECT_MEDIUM_REGRESSION_TOL,
            "{label} RI-r2 semi-direct J drifted from incore: {d_j_semi}"
        );
        assert!(
            d_k_semi < R2_SEMIDIRECT_MEDIUM_REGRESSION_TOL,
            "{label} RI-r2 semi-direct K drifted from incore: {d_k_semi}"
        );
        let _ = fs::remove_file(ctrl_path);
    }
}

#[test]
#[ignore]
fn occ_closure_dipole_connected_he2_loglog_slope() {
    let distances_ang = [4.0_f64, 5.0, 6.0, 7.0, 8.0, 10.0];
    let mut points = Vec::new();
    for distance_ang in distances_ang {
        let ctrl_path = write_temp_ctrl_he2("cc-pvdz", distance_ang, "dipole_connected");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);
        let cov_a = fragment_dipole_fluctuation_tensor(&scf_data, &[0]);
        let cov_b = fragment_dipole_fluctuation_tensor(&scf_data, &[1]);
        let distance_bohr = distance_ang / crate::constants::BOHR;
        let x = fragment_connected_dipole_x(cov_a, cov_b, distance_bohr);
        println!(
            "occ-closure dipole X_disp(He2): R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X={x:.16e} logR={:.8} log|X|={:.8}",
            distance_bohr.ln(),
            x.abs().ln(),
        );
        points.push((distance_bohr, x));
        let _ = fs::remove_file(ctrl_path);
    }
    let slope = log_log_slope(&points);
    println!("occ-closure dipole connected He2 log|X| vs log R slope = {slope:.6}");
    assert!(
        (slope + 6.0).abs() < 0.5,
        "expected long-range dipole connected descriptor slope near -6, got {slope}"
    );
}

#[test]
fn r2_shell_block_windows_match_ao_scalar_windows() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");
    let auxbasis4elem = build_default_r2_etb_auxbasis(&mol.geom, &mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let aux_shells =
        crate::lib_rint::basis::load_aux_rint_shells_from_raw(&mol.geom, &auxbasis4elem)
            .expect("failed to build auxiliary rint shells");
    let aux_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&aux_shells)
        .expect("failed to expand auxiliary rint shells");

    let ao_a = &ao_shells[0];
    let ao_b = ao_shells.get(1).unwrap_or(ao_a);
    let aux_a = &aux_shells[0];
    let aux_b = aux_shells.get(1).unwrap_or(aux_a);

    let block_2c = int2c_r2_shell_block(aux_a, aux_b);
    for j in 0..aux_b.ao_len {
        for i in 0..aux_a.ao_len {
            let got = block_2c[(i, j)];
            let expect = eri_ao_2c_r2(&aux_bfs, aux_a.ao_start + i, aux_b.ao_start + j);
            assert!(
                (got - expect).abs() < 1.0e-12,
                "2c shell block mismatch at ({i},{j}): {got} vs {expect}"
            );
        }
    }

    let block_3c = int3c_r2_shell_block(ao_a, ao_b, aux_a);
    for p in 0..aux_a.ao_len {
        for j in 0..ao_b.ao_len {
            for i in 0..ao_a.ao_len {
                let row = j * ao_a.ao_len + i;
                let got = block_3c[(row, p)];
                let expect = eri_ao_3c_r2(
                    &ao_bfs,
                    &aux_bfs,
                    ao_a.ao_start + i,
                    ao_b.ao_start + j,
                    aux_a.ao_start + p,
                );
                assert!(
                    (got - expect).abs() < 1.0e-12,
                    "3c shell block mismatch at row={row}, p={p}: {got} vs {expect}"
                );
            }
        }
    }

    let block_4c = int4c_r2_shell_block(ao_a, ao_b, ao_a, ao_b);
    for l in 0..ao_b.ao_len {
        for k in 0..ao_a.ao_len {
            let col = l * ao_a.ao_len + k;
            for j in 0..ao_b.ao_len {
                for i in 0..ao_a.ao_len {
                    let row = j * ao_a.ao_len + i;
                    let got = block_4c[(row, col)];
                    let expect = eri_ao_4c_r2(
                        &ao_bfs,
                        ao_a.ao_start + i,
                        ao_b.ao_start + j,
                        ao_a.ao_start + k,
                        ao_b.ao_start + l,
                    );
                    assert!(
                        (got - expect).abs() < 1.0e-12,
                        "4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                    );
                }
            }
        }
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r_shell_block_4c_matches_ao_scalar_window() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    let shell_a = &ao_shells[0];
    let shell_b = ao_shells.get(1).unwrap_or(shell_a);
    let shell_c = ao_shells.get(2).unwrap_or(shell_a);
    let shell_d = ao_shells.get(3).unwrap_or(shell_b);

    let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
    for l in 0..shell_d.ao_len {
        for k in 0..shell_c.ao_len {
            let col = l * shell_c.ao_len + k;
            for j in 0..shell_b.ao_len {
                for i in 0..shell_a.ao_len {
                    let row = j * shell_a.ao_len + i;
                    let got = block_4c[(row, col)];
                    let expect = eri_ao_4c_r(
                        &ao_bfs,
                        shell_a.ao_start + i,
                        shell_b.ao_start + j,
                        shell_c.ao_start + k,
                        shell_d.ao_start + l,
                    );
                    assert!(
                        (got - expect).abs() < 1.0e-12,
                        "1/r 4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                    );
                }
            }
        }
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r_shell_block_into_data_matches_allocating_shell_block() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");

    let shell_a = &ao_shells[0];
    let shell_b = ao_shells.get(1).unwrap_or(shell_a);
    let shell_c = ao_shells.get(2).unwrap_or(shell_a);
    let shell_d = ao_shells.get(3).unwrap_or(shell_b);

    let expected = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
    let mut entries = Vec::new();
    let mut data = Vec::new();
    let shape = int4c_r_shell_block_batched_into_data(
        shell_a,
        shell_b,
        shell_c,
        shell_d,
        &mut entries,
        &mut data,
    );

    assert_eq!(shape, expected.size);
    assert_eq!(data.len(), expected.data.len());
    for (idx, (got, expect)) in data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (*got - *expect).abs() < 1.0e-12,
            "reused 1/r 4c shell block mismatch at data[{idx}]: {got} vs {expect}"
        );
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r_shell_block_with_cached_primitive_pairs_matches_default() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");

    let shell_a = &ao_shells[0];
    let shell_b = ao_shells.get(1).unwrap_or(shell_a);
    let shell_c = ao_shells.get(2).unwrap_or(shell_a);
    let shell_d = ao_shells.get(3).unwrap_or(shell_b);
    let expected = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);

    let shell_coeffs = build_rint_shell_coefficient_cache(&ao_shells);
    let ab_pairs = build_primitive_pairs_4c(
        shell_a,
        shell_b,
        &shell_coeffs[0],
        &shell_coeffs[ao_shells
            .iter()
            .position(|shell| std::ptr::eq(shell, shell_b))
            .unwrap_or(0)],
        distance_squared(&shell_a.center, &shell_b.center),
    );
    let c_idx = ao_shells
        .iter()
        .position(|shell| std::ptr::eq(shell, shell_c))
        .unwrap_or(0);
    let d_idx = ao_shells
        .iter()
        .position(|shell| std::ptr::eq(shell, shell_d))
        .unwrap_or(0);
    let cd_pairs = build_primitive_pairs_4c(
        shell_c,
        shell_d,
        &shell_coeffs[c_idx],
        &shell_coeffs[d_idx],
        distance_squared(&shell_c.center, &shell_d.center),
    );
    let mut entries = Vec::new();
    let mut data = Vec::new();
    let shape = int4c_r_shell_block_batched_into_data_with_pairs(
        shell_a,
        shell_b,
        shell_c,
        shell_d,
        &ab_pairs,
        &cd_pairs,
        &mut entries,
        &mut data,
    );

    assert_eq!(shape, expected.size);
    for (idx, (got, expect)) in data.iter().zip(expected.data.iter()).enumerate() {
        assert!(
            (*got - *expect).abs() < 1.0e-12,
            "cached-pair 1/r 4c shell block mismatch at data[{idx}]: {got} vs {expect}"
        );
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r_shell_block_4c_matches_ao_scalar_high_ang_ccpvdz() {
    let ctrl_path = write_temp_ctrl_h2o("cc-pvdz");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    let shell_a = ao_shells
        .iter()
        .max_by_key(|shell| shell.shell.ang_type)
        .expect("missing AO shell");
    let shell_b = shell_a;
    let shell_c = ao_shells
        .iter()
        .find(|shell| shell.shell.ang_type == 1)
        .unwrap_or(shell_a);
    let shell_d = ao_shells
        .iter()
        .find(|shell| shell.shell.ang_type == 0)
        .unwrap_or(shell_a);

    let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
    for l in 0..shell_d.ao_len {
        for k in 0..shell_c.ao_len {
            let col = l * shell_c.ao_len + k;
            for j in 0..shell_b.ao_len {
                for i in 0..shell_a.ao_len {
                    let row = j * shell_a.ao_len + i;
                    let got = block_4c[(row, col)];
                    let expect = eri_ao_4c_r(
                        &ao_bfs,
                        shell_a.ao_start + i,
                        shell_b.ao_start + j,
                        shell_c.ao_start + k,
                        shell_d.ao_start + l,
                    );
                    assert!(
                        (got - expect).abs() < 1.0e-10,
                        "high-ang 1/r 4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                    );
                }
            }
        }
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
#[ignore = "expensive diagnostic; use targeted full-tensor mismatch diagnostics first"]
fn r_shell_block_4c_matches_ao_scalar_def2_tzvpp_all_shells() {
    let ctrl_path = write_temp_ctrl_h2o("def2-tzvpp");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    for (a_idx, shell_a) in ao_shells.iter().enumerate() {
        for (b_idx, shell_b) in ao_shells.iter().enumerate() {
            for (c_idx, shell_c) in ao_shells.iter().enumerate() {
                for (d_idx, shell_d) in ao_shells.iter().enumerate() {
                    let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
                    for l in 0..shell_d.ao_len {
                        for k in 0..shell_c.ao_len {
                            let col = l * shell_c.ao_len + k;
                            for j in 0..shell_b.ao_len {
                                for i in 0..shell_a.ao_len {
                                    let row = j * shell_a.ao_len + i;
                                    let got = block_4c[(row, col)];
                                    let expect = eri_ao_4c_r(
                                        &ao_bfs,
                                        shell_a.ao_start + i,
                                        shell_b.ao_start + j,
                                        shell_c.ao_start + k,
                                        shell_d.ao_start + l,
                                    );
                                    assert!(
                                        (got - expect).abs()
                                            <= 1.0e-10 * expect.abs().max(1.0),
                                        "def2-tzvpp shell block mismatch shells=({a_idx},{b_idx},{c_idx},{d_idx}) ang=({},{},{},{}) local=({i},{j},{k},{l}) row={row} col={col} got={got:.16e} expect={expect:.16e}",
                                        shell_a.shell.ang_type,
                                        shell_b.shell.ang_type,
                                        shell_c.shell.ang_type,
                                        shell_d.shell.ang_type,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn rys_roots_weights_r_reproduce_boys_moments_through_nine_roots() {
    for nroots in 1..=9 {
        for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
            let (roots, weights) = rys_roots_weights_r(nroots, t);
            let moments = boys_vec(2 * nroots - 1, t);
            assert_eq!(roots.len(), nroots);
            assert_eq!(weights.len(), nroots);
            for moment_idx in 0..=(2 * nroots - 1) {
                let got = roots
                    .iter()
                    .zip(weights.iter())
                    .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                    .sum::<f64>();
                let expect = moments[moment_idx];
                let tol = if nroots <= 6 { 1.0e-9 } else { 5.0e-8 } * expect.abs().max(1.0);
                assert!(
                    (got - expect).abs() <= tol,
                    "Rys moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: {got} vs {expect}"
                );
            }
        }
    }
}

#[test]
fn rys_roots_weights_r_reproduce_boys_moments_through_twenty_five_roots() {
    for nroots in 4..=25 {
        for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
            let moments = boys_vec(2 * nroots, t);
            let (roots, weights) = rys_roots_weights_r(nroots, t);
            for idx in 0..nroots {
                assert!(
                    roots[idx].is_finite() && roots[idx] >= -1.0e-12 && roots[idx] <= 1.0 + 1.0e-12,
                    "invalid root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    roots[idx]
                );
                assert!(
                    weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                    "invalid weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    weights[idx]
                );
            }
            for moment_idx in 0..=(2 * nroots - 1) {
                let got = roots
                    .iter()
                    .zip(weights.iter())
                    .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                    .sum::<f64>();
                let expect = moments[moment_idx];
                assert!(
                    (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                    "generic moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                );
            }
        }
    }
}

#[test]
fn rys_roots_weights_r2_reproduce_moments_through_six_roots() {
    for nroots in 1..=6 {
        for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
            let moments = boys_vec_r2(2 * nroots, t);
            let (roots, weights) = rys_roots_weights_r2(nroots, t);
            assert_eq!(roots.len(), nroots);
            assert_eq!(weights.len(), nroots);
            for idx in 0..nroots {
                assert!(
                    roots[idx].is_finite() && roots[idx] >= -1.0e-12 && roots[idx] <= 1.0 + 1.0e-12,
                    "invalid R2 root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    roots[idx]
                );
                assert!(
                    weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                    "invalid R2 weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    weights[idx]
                );
            }
            for moment_idx in 0..=(2 * nroots - 1) {
                let got = roots
                    .iter()
                    .zip(weights.iter())
                    .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                    .sum::<f64>();
                let expect = moments[moment_idx];
                assert!(
                    (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                    "R2 moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                );
            }
        }
    }
}

#[test]
fn rys_roots_weights_r2_keeps_legacy_low_root_path() {
    for nroots in 1..=6 {
        for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
            let moments = boys_vec_r2(2 * nroots, t);
            let (roots_ref, weights_ref) =
                rys_roots_weights_r2_from_moments(nroots, &moments, false)
                    .expect("legacy R2 roots/weights should be available");
            let (roots, weights) = rys_roots_weights_r2(nroots, t);
            assert_eq!(roots.len(), roots_ref.len());
            assert_eq!(weights.len(), weights_ref.len());
            for idx in 0..nroots {
                assert!(
                    (roots[idx] - roots_ref[idx]).abs() <= 1.0e-13,
                    "R2 low-root baseline root changed nroots={nroots}, t={t}, idx={idx}: got={:.16e}, legacy={:.16e}",
                    roots[idx],
                    roots_ref[idx]
                );
                assert!(
                    (weights[idx] - weights_ref[idx]).abs() <= 1.0e-13,
                    "R2 low-root baseline weight changed nroots={nroots}, t={t}, idx={idx}: got={:.16e}, legacy={:.16e}",
                    weights[idx],
                    weights_ref[idx]
                );
            }
        }
    }
}

#[test]
fn rys_roots_weights_r2_handles_high_roots_with_stable_moments() {
    for nroots in [7_usize, 9, 15, 25, 33, 49, 65] {
        for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 20.0, 80.0] {
            let moments = r2_stable_moments_from_measure(2 * nroots, t)
                .expect("failed to build stable R2 moments");
            let (roots, weights) = rys_roots_weights_r2(nroots, t);
            assert_eq!(roots.len(), nroots);
            assert_eq!(weights.len(), nroots);
            for idx in 0..nroots {
                assert!(
                    roots[idx].is_finite() && roots[idx] >= -1.0e-12 && roots[idx] <= 1.0 + 1.0e-12,
                    "invalid stable R2 root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    roots[idx]
                );
                assert!(
                    weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                    "invalid stable R2 weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                    weights[idx]
                );
            }
            for moment_idx in 0..=(2 * nroots - 1) {
                let got = roots
                    .iter()
                    .zip(weights.iter())
                    .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                    .sum::<f64>();
                let expect = moments[moment_idx];
                assert!(
                    (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                    "stable R2 moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                );
            }
        }
    }
}

#[test]
fn r2_stable_moments_match_legacy_gm_convention() {
    for mmax in [12_usize, 24, 50, 98, 130] {
        for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 8.0] {
            let legacy = boys_vec_r2(mmax, t);
            let stable =
                r2_stable_moments_from_measure(mmax, t).expect("failed to build stable R2 moments");
            for m in 0..=mmax {
                let expect = legacy[m];
                let got = stable[m];
                let tol = 5.0e-10 * expect.abs().max(1.0);
                assert!(
                    (got - expect).abs() <= tol,
                    "stable R2 moment changed legacy convention mmax={mmax}, m={m}, t={t}: got={got:.16e}, legacy={expect:.16e}, tol={tol:.3e}"
                );
            }
        }
    }
}

#[test]
fn boys_f0_f1_matches_general_boys_slice() {
    for t in [
        0.0_f64, 1.0e-12, 1.0e-8, 1.0e-6, 1.0e-5, 0.2, 1.0, 2.0, 20.0, 80.0,
    ] {
        let mut reference = [0.0_f64; 2];
        boys_slice(1, t, &mut reference);
        let (f0, f1) = boys_f0_f1(t);
        assert!(
            (f0 - reference[0]).abs() <= 1.0e-12 * reference[0].abs().max(1.0),
            "F0 mismatch at t={t}: {f0} vs {}",
            reference[0]
        );
        assert!(
            (f1 - reference[1]).abs() <= 1.0e-10 * reference[1].abs().max(1.0),
            "F1 mismatch at t={t}: {f1} vs {}",
            reference[1]
        );
    }
}

#[test]
fn boys_f0_matches_general_boys_slice() {
    for t in [
        0.0_f64, 1.0e-12, 1.0e-8, 1.0e-6, 1.0e-5, 0.2, 0.8, 1.0, 2.0, 20.0, 80.0,
    ] {
        let mut reference = [0.0_f64; 1];
        boys_slice(0, t, &mut reference);
        let f0 = boys_f0(t);
        assert!(
            (f0 - reference[0]).abs() <= 1.0e-13 * reference[0].abs().max(1.0),
            "F0 mismatch at t={t}: {f0} vs {}",
            reference[0]
        );
    }
}

#[test]
fn exact4c_scalar_def2_tzvpp_pfff_matches_libcint() {
    let ctrl_path = write_temp_ctrl_h2o("def2-tzvpp");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    let indices = [41_usize, 26_usize, 26_usize, 26_usize];
    let got = eri_ao_4c_r(&ao_bfs, indices[0], indices[1], indices[2], indices[3]);
    let eri4_libcint = mol.int_ijkl_erifull();
    let expect = *eri4_libcint
        .get(&[indices[0], indices[1], indices[2], indices[3]])
        .unwrap();
    assert!(
        (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
        "def2-tzvpp p-f-f-f exact 4c mismatch at {:?}: got={got:.16e}, libcint={expect:.16e}",
        indices
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn exact4c_shell_block_def2_qzvp_gggg_matches_scalar() {
    let ctrl_path = write_temp_ctrl_h2o("def2-qzvp");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    let shell = ao_shells
        .iter()
        .find(|shell| shell.shell.ang_type == 4)
        .expect("def2-qzvp H2O should contain g shells");
    let block = int4c_r_shell_block(shell, shell, shell, shell);
    let got = block[(0, 0)];
    let ao = shell.ao_start;
    let expect = eri_ao_4c_r(&ao_bfs, ao, ao, ao, ao);
    assert!(
        got.abs() > 1.0e-8,
        "def2-qzvp g-g-g-g shell block unexpectedly vanished for ao={ao}: got={got:.16e}"
    );
    assert!(
        (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
        "def2-qzvp g-g-g-g shell block mismatch ao={ao}: got={got:.16e}, scalar={expect:.16e}"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn exact4c_shell_block_single_component_l5_to_l8_matches_scalar() {
    for l in 5_u32..=8 {
        let component = [l, 0, 0];
        let shell = single_component_shell(l, component);
        let ao_bfs = vec![single_component_basis_function(component)];
        let block = int4c_r_shell_block(&shell, &shell, &shell, &shell);
        let got = block[(0, 0)];
        let expect = eri_ao_4c_r(&ao_bfs, 0, 0, 0, 0);
        assert!(
            (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
            "single-component exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
        );
    }
}

#[test]
fn r2_4c_shell_block_single_component_l5_to_l12_matches_scalar() {
    for l in 5_u32..=12 {
        let component = [l, 0, 0];
        let shell = single_component_shell(l, component);
        let ao_bfs = vec![single_component_basis_function(component)];
        let block = int4c_r2_shell_block(&shell, &shell, &shell, &shell);
        let got = block[(0, 0)];
        let expect = eri_ao_4c_r2(&ao_bfs, 0, 0, 0, 0);
        assert!(
            (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
            "single-component R2 exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
        );
    }
}

#[test]
#[ignore = "artificial R2 l=13..20 shell-block comparison is too slow for default debug tests"]
fn r2_4c_shell_block_single_component_l13_to_l20_matches_scalar() {
    for l in 13_u32..=20 {
        let component = [l, 0, 0];
        let shell = single_component_shell(l, component);
        let ao_bfs = vec![single_component_basis_function(component)];
        let block = int4c_r2_shell_block(&shell, &shell, &shell, &shell);
        let got = block[(0, 0)];
        let expect = eri_ao_4c_r2(&ao_bfs, 0, 0, 0, 0);
        assert!(
            (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
            "single-component R2 exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
        );
    }
}

#[test]
#[ignore = "full h/i Cartesian shell-block comparison is too slow for debug default tests"]
fn exact4c_shell_block_artificial_h_and_i_match_scalar_and_libcint() {
    for l in [5_u32, 6_u32] {
        assert_high_l_4c_shell_block_matches_scalar_and_libcint(l);
    }
}

#[test]
fn rys_roots_weights_r_into_matches_allocating_api() {
    for nroots in 1..=9 {
        for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 20.0, 80.0] {
            let (roots_ref, weights_ref) = rys_roots_weights_r(nroots, t);
            let mut roots = [0.0_f64; 16];
            let mut weights = [0.0_f64; 16];
            let got_nroots = rys_roots_weights_r_into(nroots, t, &mut roots, &mut weights);
            assert_eq!(got_nroots, nroots);
            for i in 0..nroots {
                assert!(
                    (roots[i] - roots_ref[i]).abs() < 1.0e-11,
                    "root mismatch nroots={nroots}, t={t}, i={i}: {} vs {}",
                    roots[i],
                    roots_ref[i]
                );
                assert!(
                    (weights[i] - weights_ref[i]).abs() < 1.0e-11,
                    "weight mismatch nroots={nroots}, t={t}, i={i}: {} vs {}",
                    weights[i],
                    weights_ref[i]
                );
            }
        }
    }
}

#[test]
fn r_full_shell_blocks_match_ao_scalar_h2_sto3g() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");

    let full = int4c_r_full_from_shell_blocks(&ao_shells);
    let nao = ao_bfs.len();
    for sig in 0..nao {
        for lam in 0..nao {
            for nu in 0..nao {
                for mu in 0..nao {
                    let got = full[int4c_r_full_index(mu, nu, lam, sig, nao)];
                    let expect = eri_ao_4c_r(&ao_bfs, mu, nu, lam, sig);
                    assert!(
                        (got - expect).abs() < 1.0e-12,
                        "full 1/r 4c mismatch at ({mu},{nu},{lam},{sig}): {got} vs {expect}"
                    );
                }
            }
        }
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r_full_shell_blocks_parallel_matches_serial_h2_sto3g() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");

    let serial = int4c_r_full_from_shell_blocks(&ao_shells);
    let parallel = int4c_r_full_from_shell_blocks_parallel(&ao_shells);

    assert_eq!(parallel.len(), serial.len());
    for (idx, (got, expect)) in parallel.iter().zip(serial.iter()).enumerate() {
        assert!(
            (*got - *expect).abs() < 1.0e-12,
            "parallel full 1/r 4c mismatch at data[{idx}]: {got} vs {expect}"
        );
    }

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn contracted_rint_shells_keep_multicontraction_shells() {
    use crate::lib_rint::basis::{
        expand_shell_shared_to_contracted_rint_shells, expand_shell_shared_to_rint_shells,
        normalize_raw_shells_to_shell_shared, RawShellShared,
    };

    let raw = RawShellShared {
        atom_idx: 0,
        center: [0.0, 0.0, 0.0],
        l: 1,
        exponents: vec![3.0, 0.8],
        coeff_columns: vec![vec![0.6, 0.4], vec![0.2, 0.9]],
        is_aux: false,
    };
    let normalized = normalize_raw_shells_to_shell_shared(&[raw]).unwrap();
    let split_shells = expand_shell_shared_to_rint_shells(&normalized).unwrap();
    let contracted_shells = expand_shell_shared_to_contracted_rint_shells(&normalized).unwrap();

    assert_eq!(split_shells.len(), 2);
    assert_eq!(contracted_shells.len(), 1);
    assert_eq!(contracted_shells[0].cart_len, 3);
    assert_eq!(contracted_shells[0].nctr(), 2);
    assert_eq!(contracted_shells[0].ao_start, 0);
    assert_eq!(contracted_shells[0].ao_len, 6);
}

#[test]
fn contracted_r_full_matches_split_r_full_for_multicontraction_shells() {
    use crate::lib_rint::basis::{
        expand_shell_shared_to_contracted_rint_shells, expand_shell_shared_to_rint_shells,
        normalize_raw_shells_to_shell_shared, RawShellShared,
    };

    let raw_shells = vec![
        RawShellShared {
            atom_idx: 0,
            center: [0.0, 0.0, 0.0],
            l: 0,
            exponents: vec![2.1, 0.7],
            coeff_columns: vec![vec![0.8, 0.3], vec![0.1, 0.9]],
            is_aux: false,
        },
        RawShellShared {
            atom_idx: 1,
            center: [0.0, 0.0, 1.4],
            l: 1,
            exponents: vec![1.8, 0.5],
            coeff_columns: vec![vec![0.7, 0.4], vec![0.3, 0.8]],
            is_aux: false,
        },
    ];
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells).unwrap();
    let split_shells = expand_shell_shared_to_rint_shells(&normalized).unwrap();
    let contracted_shells = expand_shell_shared_to_contracted_rint_shells(&normalized).unwrap();

    let split_full = int4c_r_full_from_shell_blocks(&split_shells);
    let contracted_full = int4c_r_full_from_contracted_shell_blocks(&contracted_shells);

    assert_eq!(contracted_full.len(), split_full.len());
    for (idx, (got, expect)) in contracted_full.iter().zip(split_full.iter()).enumerate() {
        assert!(
            (*got - *expect).abs() < 1.0e-12,
            "contracted full 1/r mismatch at data[{idx}]: {got} vs {expect}"
        );
    }
}

#[test]
fn contracted_r2_full_matches_split_r2_full_for_multicontraction_shells() {
    use crate::lib_rint::basis::{
        expand_shell_shared_to_contracted_rint_shells, expand_shell_shared_to_rint_shells,
        normalize_raw_shells_to_shell_shared, RawShellShared,
    };

    let raw_shells = vec![
        RawShellShared {
            atom_idx: 0,
            center: [0.0, 0.0, 0.0],
            l: 0,
            exponents: vec![2.1, 0.7],
            coeff_columns: vec![vec![0.8, 0.3], vec![0.1, 0.9]],
            is_aux: false,
        },
        RawShellShared {
            atom_idx: 1,
            center: [0.0, 0.0, 1.4],
            l: 1,
            exponents: vec![1.8, 0.5],
            coeff_columns: vec![vec![0.7, 0.4], vec![0.3, 0.8]],
            is_aux: false,
        },
    ];
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells).unwrap();
    let split_shells = expand_shell_shared_to_rint_shells(&normalized).unwrap();
    let contracted_shells = expand_shell_shared_to_contracted_rint_shells(&normalized).unwrap();

    let split_full = int4c_r2_full_from_shell_blocks(&split_shells);
    let contracted_full = int4c_r2_full_from_contracted_shell_blocks(&contracted_shells);

    assert_eq!(contracted_full.len(), split_full.len());
    for (idx, (got, expect)) in contracted_full.iter().zip(split_full.iter()).enumerate() {
        assert!(
            (*got - *expect).abs() < 1.0e-12,
            "contracted full R2 mismatch at data[{idx}]: {got} vs {expect}"
        );
    }
}

fn tmp_exact4c_deterministic_density(nao: usize) -> MatrixFull<f64> {
    let mut p = MatrixFull::new([nao, nao], 0.0);
    for mu in 0..nao {
        for nu in 0..=mu {
            let value = if mu == nu {
                1.0 / (1.0 + mu as f64)
            } else {
                ((mu + 3 * nu + 1) % 11 + 1) as f64 * 1.0e-2
            };
            p.set2d([mu, nu], value);
            p.set2d([nu, mu], value);
        }
    }
    p
}

fn tmp_exact4c_contract_jk_from_libcint_full(
    eri4_libcint: &rest_tensors::ERIFull<f64>,
    p: &MatrixFull<f64>,
    nao: usize,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let mut j = MatrixFull::new([nao, nao], 0.0);
    let mut k = MatrixFull::new([nao, nao], 0.0);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut j_munu = 0.0_f64;
            let mut k_munu = 0.0_f64;
            for lam in 0..nao {
                for sig in 0..nao {
                    let p_lamsig = *p.get(&[lam, sig]).unwrap();
                    j_munu += p_lamsig * eri4_libcint.get(&[mu, nu, lam, sig]).unwrap();
                    k_munu += p_lamsig * eri4_libcint.get(&[mu, lam, nu, sig]).unwrap();
                }
            }
            j.set2d([mu, nu], j_munu);
            k.set2d([mu, nu], k_munu);
        }
    }
    (j, k)
}

fn tmp_exact4c_contract_jk_from_librint_full(
    eri4_librint: &[f64],
    p: &MatrixFull<f64>,
    nao: usize,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let mut j = MatrixFull::new([nao, nao], 0.0);
    let mut k = MatrixFull::new([nao, nao], 0.0);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut j_munu = 0.0_f64;
            let mut k_munu = 0.0_f64;
            for lam in 0..nao {
                for sig in 0..nao {
                    let p_lamsig = *p.get(&[lam, sig]).unwrap();
                    j_munu += p_lamsig * eri4_librint[int4c_r_full_index(mu, nu, lam, sig, nao)];
                    k_munu += p_lamsig * eri4_librint[int4c_r_full_index(mu, lam, nu, sig, nao)];
                }
            }
            j.set2d([mu, nu], j_munu);
            k.set2d([mu, nu], k_munu);
        }
    }
    (j, k)
}

#[test]
fn r2_exact_jk_generation_uses_shell_block_full_tensor() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");
    let p = tmp_exact4c_deterministic_density(ao_bfs.len());

    let (j_scalar, k_scalar) = build_jk_from_p_kernel(&ao_bfs, &p, eri_ao_4c_r2);
    let (j_block, k_block) =
        build_jk_from_r2_shell_block_full_tensor(&mol.geom, &mol.basis4elem, &p);

    assert!(
        max_abs_diff_matrix(&j_scalar, &j_block) < 1.0e-10,
        "R2 J from shell-block full tensor drifted from AO scalar"
    );
    assert!(
        max_abs_diff_matrix(&k_scalar, &k_block) < 1.0e-10,
        "R2 K from shell-block full tensor drifted from AO scalar"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
fn r2_rhf_exact_observables_use_shell_block_generation() {
    let ctrl_path = write_temp_ctrl_h2("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand AO rint shells");
    let p = tmp_exact4c_deterministic_density(ao_bfs.len());

    let (j_scalar, k_scalar) = build_jk_from_p_kernel(&ao_bfs, &p, eri_ao_4c_r2);
    let expected = {
        let (ej, ek, total) = ej_ek_from_p_jk_rhf(&p, &j_scalar, &k_scalar);
        RhfVeeObservables { ej, ek, total }
    };
    let got = lib_vee_rhf_r2_observables_exact(&mol.geom, &mol.basis4elem, &p);

    assert!(
        (got.ej - expected.ej).abs() < 1.0e-10,
        "R2 exact RHF EJ drifted after shell-block generation switch"
    );
    assert!(
        (got.ek - expected.ek).abs() < 1.0e-10,
        "R2 exact RHF EK drifted after shell-block generation switch"
    );
    assert!(
        (got.total - expected.total).abs() < 1.0e-10,
        "R2 exact RHF total drifted after shell-block generation switch"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
#[ignore = "temporary exact 4c shell splitting diagnostics"]
fn tmp_diagnose_exact4c_shell_splitting_libcint_vs_librint() {
    let cases_env = std::env::var("TMP_EXACT4C_CASES").unwrap_or_else(|_| String::from("sto-3g"));
    let cases = cases_env
        .split(',')
        .map(str::trim)
        .filter(|case| !case.is_empty())
        .map(|case| (case.to_string(), write_temp_ctrl_h2o(case)))
        .collect::<Vec<_>>();
    let summary_path = std::env::var("TMP_EXACT4C_SUMMARY_FILE").ok();
    if let Some(path) = &summary_path {
        let _ = fs::remove_file(path);
    }
    let emit_summary_line = |line: String| {
        println!("{line}");
        if let Some(path) = &summary_path {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("failed to open exact 4c shell splitting summary file");
            writeln!(file, "{line}")
                .expect("failed to write exact 4c shell splitting summary file");
        }
    };

    let pair_count = |n: usize| n * (n + 1) / 2;
    let quartet_count = |n: usize| {
        let npair = pair_count(n);
        npair * (npair + 1) / 2
    };

    for (basis_name, ctrl_path) in cases {
        let mol = match std::panic::catch_unwind(|| Molecule::build(ctrl_path.clone(), None)) {
            Ok(Ok(mol)) => mol,
            Ok(Err(err)) => {
                emit_summary_line(format!(
                    "TMP_EXACT4C_SHELL_SPLIT_SKIP basis={basis_name} reason=build_failed detail={err:?}"
                ));
                let _ = fs::remove_file(ctrl_path);
                continue;
            }
            Err(_) => {
                emit_summary_line(format!(
                    "TMP_EXACT4C_SHELL_SPLIT_SKIP basis={basis_name} reason=build_panicked"
                ));
                let _ = fs::remove_file(ctrl_path);
                continue;
            }
        };
        let ao_shells = match crate::lib_rint::basis::load_molecule_rint_shells_from_raw(
            &mol.geom,
            &mol.basis4elem,
        ) {
            Ok(shells) => shells,
            Err(err) => {
                emit_summary_line(format!(
                    "TMP_EXACT4C_SHELL_SPLIT_SKIP basis={basis_name} reason=rint_shell_load_failed detail={err:?}"
                ));
                let _ = fs::remove_file(ctrl_path);
                continue;
            }
        };

        let libcint_nshell = mol.cint_fdqc.len();
        let librint_nshell = ao_shells.len();
        let nao = rint_shell_basis_count(&ao_shells);
        let libcint_pairs = pair_count(libcint_nshell);
        let librint_pairs = pair_count(librint_nshell);
        let libcint_quartets = quartet_count(libcint_nshell);
        let librint_quartets = build_unique_4c_shell_quartet_tasks(&ao_shells).len();

        let shell_coeffs = build_rint_shell_coefficient_cache(&ao_shells);
        let primitive_pair_cache = build_shell_pair_primitive_pair_cache(&ao_shells, &shell_coeffs);
        let mut librint_primitive_pair_sum = 0_usize;
        for left_idx in 0..librint_nshell {
            for right_idx in 0..=left_idx {
                librint_primitive_pair_sum +=
                    primitive_pair_cache[shell_pair_rank(left_idx, right_idx)].len();
            }
        }
        let mut librint_primitive_quartet_sum = 0_usize;
        let mut librint_block_value_sum = 0_usize;
        for (a_idx, b_idx, c_idx, d_idx) in build_unique_4c_shell_quartet_tasks(&ao_shells) {
            let ab = primitive_pair_cache[shell_pair_rank(a_idx, b_idx)].len();
            let cd = primitive_pair_cache[shell_pair_rank(c_idx, d_idx)].len();
            librint_primitive_quartet_sum += ab * cd;
            librint_block_value_sum += ao_shells[a_idx].ao_len
                * ao_shells[b_idx].ao_len
                * ao_shells[c_idx].ao_len
                * ao_shells[d_idx].ao_len;
        }

        let mut cint_primitive_pair_sum = 0_usize;
        let mut cint_block_value_sum = 0_usize;
        let mut cint_primitive_quartet_est = 0_usize;
        for a_idx in 0..libcint_nshell {
            for b_idx in 0..=a_idx {
                let ab_nprim = mol.cint_bas[a_idx][2] as usize * mol.cint_bas[b_idx][2] as usize;
                cint_primitive_pair_sum += ab_nprim;
            }
        }
        for a_idx in 0..libcint_nshell {
            for b_idx in 0..=a_idx {
                let ab_rank = shell_pair_rank(a_idx, b_idx);
                let ab_nprim = mol.cint_bas[a_idx][2] as usize * mol.cint_bas[b_idx][2] as usize;
                for c_idx in 0..=a_idx {
                    for d_idx in 0..=c_idx {
                        if shell_pair_rank(c_idx, d_idx) <= ab_rank {
                            let cd_nprim =
                                mol.cint_bas[c_idx][2] as usize * mol.cint_bas[d_idx][2] as usize;
                            cint_primitive_quartet_est += ab_nprim * cd_nprim;
                            cint_block_value_sum += mol.cint_fdqc[a_idx][1]
                                * mol.cint_fdqc[b_idx][1]
                                * mol.cint_fdqc[c_idx][1]
                                * mol.cint_fdqc[d_idx][1];
                        }
                    }
                }
            }
        }

        emit_summary_line(format!(
            "TMP_EXACT4C_SHELL_SPLIT basis={basis_name} nao={nao} libcint_nshell={libcint_nshell} librint_nshell={librint_nshell} shell_ratio={:.3} libcint_pairs={libcint_pairs} librint_pairs={librint_pairs} pair_ratio={:.3} libcint_quartets={libcint_quartets} librint_quartets={librint_quartets} quartet_ratio={:.3} libcint_primitive_pair_sum={cint_primitive_pair_sum} librint_primitive_pair_sum={librint_primitive_pair_sum} primitive_pair_ratio={:.3} libcint_primitive_quartet_est={cint_primitive_quartet_est} librint_primitive_quartet_sum={librint_primitive_quartet_sum} primitive_quartet_ratio={:.3} libcint_block_value_sum={cint_block_value_sum} librint_block_value_sum={librint_block_value_sum} block_value_ratio={:.3}",
            librint_nshell as f64 / libcint_nshell.max(1) as f64,
            librint_pairs as f64 / libcint_pairs.max(1) as f64,
            librint_quartets as f64 / libcint_quartets.max(1) as f64,
            librint_primitive_pair_sum as f64 / cint_primitive_pair_sum.max(1) as f64,
            librint_primitive_quartet_sum as f64 / cint_primitive_quartet_est.max(1) as f64,
            librint_block_value_sum as f64 / cint_block_value_sum.max(1) as f64,
        ));

        let _ = fs::remove_file(ctrl_path);
    }
}

#[test]
fn r2_3c_libcint_pair_screening_matches_unscreened_h2o_sto3g() {
    let ctrl_path = write_temp_ctrl_h2o("sto-3g");
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

    let ao_shells =
        crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
            .expect("failed to build AO rint shells");
    let auxbasis4elem = build_default_r2_etb_auxbasis(&mol.geom, &mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let aux_shells =
        crate::lib_rint::basis::load_aux_rint_shells_from_raw(&mol.geom, &auxbasis4elem)
            .expect("failed to build auxiliary rint shells");

    let mut max_diff = 0.0_f64;
    for left_shell in ao_shells.iter() {
        for right_shell in ao_shells.iter() {
            let left_min = left_shell.ao_start;
            let right_max = right_shell.ao_start + right_shell.ao_len - 1;
            if left_min > right_max {
                continue;
            }
            for aux_shell in aux_shells.iter() {
                let screened = int3c_r2_shell_block(left_shell, right_shell, aux_shell);
                let unscreened =
                    int3c_r2_shell_block_batched_unscreened(left_shell, right_shell, aux_shell);
                max_diff = max_diff.max(max_abs_diff_matrix(&screened, &unscreened));
            }
        }
    }

    println!("H2O/sto-3g r2 3c libcint-pair-screening max_diff={max_diff:.3e}");
    assert!(
        max_diff < 1.0e-12,
        "libcint-like primitive pair screening drifted from unscreened 3c blocks: {max_diff}"
    );

    let _ = fs::remove_file(ctrl_path);
}

fn write_temp_ctrl_h2_with_etb(basis_dir: &str, etb_beta: f64) -> (String, String) {
    let temp_id = next_temp_id();
    let safe_basis = basis_dir
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
    let aux_dir = format!("/tmp/lib_rint_etb_aux_{safe_basis}_{temp_id}");
    let path = format!("/tmp/lib_rint_kernel_etb_check_{safe_basis}_{temp_id}.toml");
    let text = format!(
        r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "{basis_dir}"
auxbas_path = "{aux_dir}"
basis_type = "Cartesian"
auxbas_type = "Cartesian"
use_auxbas = true
even_tempered_basis = true
etb_start_atom_number = 1
etb_beta = {etb_beta}
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false
[geom]
name = "H2"
unit = "angstrom"
position = [
"H   0.0000000000   0.0000000000   0.0000000000",
"H   0.0000000000   0.0000000000   1.4000000000",
]
"#
    );
    fs::write(&path, text).unwrap();
    (path, aux_dir)
}
fn write_temp_ctrl_nh3(basis_dir: &str) -> String {
    let temp_id = next_temp_id();
    let path = format!("/tmp/lib_rint_kernel_nh3_{basis_dir}_{temp_id}.toml")
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
    let text = format!(
        "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"NH3\"\nunit = \"angstrom\"\nposition = [\n    \"N   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.0000000000   1.5000000000   1.0000000000\",\n    \"H   1.4000000000   1.1000000000   0.0000000000\",\n    \"H   1.2000000000   0.0000000000   1.3000000000\",\n]\n"
    );
    fs::write(&path, text).unwrap();
    path
}
fn load_pyscf_autoaux_basis4elem(mol: &Molecule, basis_name: &str) -> Vec<Basis4Elem> {
    let atoms = crate::lib_rint::basis::parse_geomcell(&mol.geom)
        .expect("failed to parse geometry for PySCF autoaux bridge");
    let atom_spec = atoms
        .iter()
        .map(|(elem, center)| {
            format!(
                "{} {:.16} {:.16} {:.16}",
                elem, center[0], center[1], center[2]
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let basis_spec = basis_name.to_lowercase();
    let script = r#"
import json
import sys
from pyscf import gto
from pyscf.df.autoaux import autoaux

atom_spec = sys.argv[1]
basis_spec = sys.argv[2]
mol = gto.M(atom=atom_spec, basis=basis_spec, cart=True)
aux = autoaux(mol)

payload = {}
for elem, shells in aux.items():
    out_shells = []
    for shell in shells:
        l = int(shell[0])
        exps = []
        coeffs = []
        for pair in shell[1:]:
            exps.append(float(pair[0]))
            coeffs.append(float(pair[1]))
        out_shells.append({
            "l": l,
            "exponents": exps,
            "coefficients": coeffs,
        })
    payload[elem] = out_shells
print(json.dumps(payload))
"#;
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(&atom_spec)
        .arg(&basis_spec)
        .output()
        .expect("failed to run python for PySCF autoaux");
    assert!(
        output.status.success(),
        "PySCF autoaux bridge failed: stdout={}; stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("failed to parse JSON from PySCF autoaux");

    atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, (elem, _))| {
            let shells_json = payload
                .get(elem)
                .unwrap_or_else(|| panic!("PySCF autoaux output missing element {elem}"));
            let electron_shells = shells_json
                .as_array()
                .unwrap()
                .iter()
                .map(|shell| {
                    let l = shell["l"].as_i64().unwrap() as i32;
                    let exponents = shell["exponents"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<_>>();
                    let coeffs = shell["coefficients"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<_>>();
                    BasCell {
                        function_type: Some(String::from("gto")),
                        region: None,
                        angular_momentum: vec![l],
                        exponents,
                        coefficients: vec![coeffs.clone()],
                        native_coefficients: vec![coeffs],
                    }
                })
                .collect::<Vec<_>>();
            Basis4Elem {
                electron_shells,
                references: None,
                ecp_potentials: None,
                ecp_electrons: None,
                global_index: (atom_idx, atom_idx),
            }
        })
        .collect()
}
fn compare_ri_r_with_fixed_density(
    ao_bfs: &[BasisFunction],
    p_cart: &MatrixFull<f64>,
    aux_bfs: &[BasisFunction],
) -> (f64, f64, f64, f64, f64, usize) {
    let dm_vec = vec![p_cart.clone()];
    let ri_r = prepare_rimatr_for_r_sync(ao_bfs, aux_bfs)
        .expect("failed to build RI-r matrix for fixed-density comparison");
    let j_ri_u = vj_upper_with_rimatr_r_sync(&Some(ri_r.clone()), &dm_vec, 1, 1.0);
    let k_ri_u = vk_upper_with_rimatr_r_sync(&Some(ri_r), &dm_vec, 1, 1.0);
    let j_ri = j_ri_u[0].to_matrixfull().unwrap();
    let k_ri = k_ri_u[0].to_matrixfull().unwrap();
    let (j_ex, k_ex) = build_jk_from_p_kernel(ao_bfs, p_cart, eri_ao_4c_r);
    let (ej_ri, ek_ri, e_ri) = ej_ek_from_p_jk_rhf(p_cart, &j_ri, &k_ri);
    let (ej_ex, ek_ex, e_ex) = ej_ek_from_p_jk_rhf(p_cart, &j_ex, &k_ex);

    (
        max_abs_diff_matrix(&j_ri, &j_ex),
        max_abs_diff_matrix(&k_ri, &k_ex),
        (ej_ri - ej_ex).abs(),
        (ek_ri - ek_ex).abs(),
        (e_ri - e_ex).abs(),
        aux_bfs.len(),
    )
}
fn reconstruct_eri_ao_4c_from_rimatr(
    rimatr: &MatrixFull<f64>,
    basbas2baspar: &MatrixFull<usize>,
    mu: usize,
    nu: usize,
    lam: usize,
    sig: usize,
) -> f64 {
    let munu = *basbas2baspar.get(&[mu, nu]).unwrap();
    let lamsig = *basbas2baspar.get(&[lam, sig]).unwrap();
    let naux = rimatr.size[1];
    let mut value = 0.0_f64;
    for aux in 0..naux {
        value += rimatr.get(&[munu, aux]).unwrap() * rimatr.get(&[lamsig, aux]).unwrap();
    }
    value
}
fn build_jk_from_reconstructed_rimatr(
    rimatr: &MatrixFull<f64>,
    basbas2baspar: &MatrixFull<usize>,
    p: &MatrixFull<f64>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let nao = p.size[0];
    let mut j = MatrixFull::new([nao, nao], 0.0_f64);
    let mut k = MatrixFull::new([nao, nao], 0.0_f64);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut j_munu = 0.0_f64;
            let mut k_munu = 0.0_f64;
            for lam in 0..nao {
                for sig in 0..nao {
                    let p_lamsig = *p.get(&[lam, sig]).unwrap();
                    let eri_j =
                        reconstruct_eri_ao_4c_from_rimatr(rimatr, basbas2baspar, mu, nu, lam, sig);
                    let eri_k =
                        reconstruct_eri_ao_4c_from_rimatr(rimatr, basbas2baspar, mu, lam, nu, sig);
                    j_munu += p_lamsig * eri_j;
                    k_munu += p_lamsig * eri_k;
                }
            }
            j.set2d([mu, nu], j_munu);
            k.set2d([mu, nu], k_munu);
        }
    }
    (j, k)
}
fn write_temp_ctrl_h2o_sto3g_cartesian() -> String {
    let temp_id = next_temp_id();
    let path = format!("/tmp/lib_rint_grid_h2o_sto3g_cart_{temp_id}.toml");
    let text = r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "sto-3g"
basis_type = "Cartesian"
use_auxbas = false
even_tempered_basis = false
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false

[geom]
name = "H2O"
unit = "angstrom"
position = [
"O   0.0000000000   0.0000000000   0.0000000000",
"H   0.0000000000  -0.7571600000   0.5862600000",
"H   0.0000000000   0.7571600000   0.5862600000",
]
"#;
    fs::write(&path, text).unwrap();
    path
}
fn write_temp_ctrl_li_open_shell_with_aux(basis_dir: &str, aux_basis_dir: &str) -> String {
    let temp_id = next_temp_id();
    let path =
        format!("/tmp/lib_rint_open_shell_ri_check_{basis_dir}_{aux_basis_dir}_{temp_id}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
    let text = format!(
        r#"[ctrl]
print_level = 0
xc = "hf"
eri_type = "ri_v"
basis_path = "{basis_dir}"
auxbas_path = "{aux_basis_dir}"
basis_type = "spheric"
auxbas_type = "spheric"
use_auxbas = true
even_tempered_basis = false
charge = 0.0
spin = 2.0
spin_polarization = true
initial_guess = "sad"
mixer = "diis"
num_max_diis = 8
start_diis_cycle = 1
mix_param = 0.6
max_scf_cycle = 80
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-8
scf_acc_etot = 1.0e-10
num_threads = 1
run_lib_rint = false
[geom]
name = "Li"
unit = "angstrom"
position = [
"Li   0.0000000000   0.0000000000   0.0000000000",
]
"#
    );
    fs::write(&path, text).unwrap();
    path
}
fn generate_pyscf_ref_4c(basis_name: &str, out4: &str) {
    let script = r#"
import sys
from pyscf import gto
basis, out4 = sys.argv[1:3]
mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis=basis, unit='Angstrom', cart=True)
eri4 = mol.intor('int2e_cart', aosym='s1')
with open(out4, 'w', encoding='utf-8') as fh:
for i in range(eri4.shape[0]):
    for j in range(eri4.shape[1]):
        for k in range(eri4.shape[2]):
            for l in range(eri4.shape[3]):
                fh.write(f'{i} {j} {k} {l} {eri4[i,j,k,l]:.15e}\n')
"#;
    let status = Command::new("python3")
        .args(["-c", script, basis_name, out4])
        .status()
        .expect("failed to launch python3");
    assert!(
        status.success(),
        "PySCF 4c reference generation failed for {basis_name}"
    );
}
fn generate_pyscf_ref_2c_3c_4c(basis_name: &str, out2: &str, out3: &str, out4: &str) {
    let script = r#"
import sys
from pyscf import gto, df
basis, out2, out3, out4 = sys.argv[1:5]
atom = 'H 0 0 0; H 0 0 1.4'
mol = gto.M(atom=atom, basis=basis, unit='Angstrom', cart=True)
auxmol = gto.M(atom=atom, basis=basis, unit='Angstrom', cart=True)
eri2 = auxmol.intor('int2c2e_cart')
eri3 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_cart', aosym='s1')
eri4 = mol.intor('int2e_cart', aosym='s1')
with open(out2, 'w', encoding='utf-8') as fh:
for i in range(eri2.shape[0]):
    for j in range(eri2.shape[1]):
        fh.write(f'{i} {j} {eri2[i,j]:.15e}\n')
with open(out3, 'w', encoding='utf-8') as fh:
for i in range(eri3.shape[0]):
    for j in range(eri3.shape[1]):
        for k in range(eri3.shape[2]):
            fh.write(f'{i} {j} {k} {eri3[i,j,k]:.15e}\n')
with open(out4, 'w', encoding='utf-8') as fh:
for i in range(eri4.shape[0]):
    for j in range(eri4.shape[1]):
        for k in range(eri4.shape[2]):
            for l in range(eri4.shape[3]):
                fh.write(f'{i} {j} {k} {l} {eri4[i,j,k,l]:.15e}\n')
"#;
    let status = Command::new("python3")
        .args(["-c", script, basis_name, out2, out3, out4])
        .status()
        .expect("failed to launch python3");
    assert!(
        status.success(),
        "PySCF 2c/3c/4c reference generation failed for {basis_name}"
    );
}
fn read_ref_values(path: &str, expected: usize) -> Vec<f64> {
    let text = fs::read_to_string(path).unwrap();
    let mut out = Vec::with_capacity(expected);
    for line in text.lines() {
        let value = line
            .split_whitespace()
            .last()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        out.push(value);
    }
    assert_eq!(
        out.len(),
        expected,
        "unexpected reference length for {path}"
    );
    out
}
#[derive(Clone, Copy, Debug)]
struct DiffStats {
    count: usize,
    bad: usize,
    max_abs: f64,
    max_rel: f64,
    sum_abs: f64,
    sum_rel: f64,
}
impl DiffStats {
    fn new() -> Self {
        Self {
            count: 0,
            bad: 0,
            max_abs: 0.0,
            max_rel: 0.0,
            sum_abs: 0.0,
            sum_rel: 0.0,
        }
    }
    fn push(&mut self, got: f64, reference: f64, tol_abs: f64, tol_rel: f64) {
        let abs = (got - reference).abs();
        let rel = abs / reference.abs().max(1.0e-300);
        self.count += 1;
        self.sum_abs += abs;
        self.sum_rel += rel;
        self.max_abs = self.max_abs.max(abs);
        self.max_rel = self.max_rel.max(rel);
        if abs > tol_abs && rel > tol_rel {
            self.bad += 1;
        }
    }
    fn summary(&self) -> String {
        format!(
            "count={} bad_count={} max_abs={:.12e} avg_abs={:.12e} max_rel={:.12e} avg_rel={:.12e}",
            self.count,
            self.bad,
            self.max_abs,
            if self.count > 0 {
                self.sum_abs / self.count as f64
            } else {
                0.0
            },
            self.max_rel,
            if self.count > 0 {
                self.sum_rel / self.count as f64
            } else {
                0.0
            },
        )
    }
}
fn max_abs_diff_matrix(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
    assert_eq!(a.size, b.size);
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridKernel {
    R,
    R2,
}
#[derive(Clone, Debug)]
struct GridQuartetRef {
    label: String,
    kernel: GridKernel,
    ijkl: [usize; 4],
    value: f64,
}
fn read_grid_quartet_refs(path: &str) -> Vec<GridQuartetRef> {
    let text = fs::read_to_string(path).unwrap();
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 {
            continue;
        }
        let kernel = match fields[1] {
            "1/r12" | "K=1/r12" => GridKernel::R,
            "1/r12²" | "1/r12^2" | "K=1/r12^2" => GridKernel::R2,
            _ => continue,
        };
        let n = fields.len();
        rows.push(GridQuartetRef {
            label: fields[0].to_string(),
            kernel,
            ijkl: [
                fields[n - 5].parse().unwrap(),
                fields[n - 4].parse().unwrap(),
                fields[n - 3].parse().unwrap(),
                fields[n - 2].parse().unwrap(),
            ],
            value: fields[n - 1].parse().unwrap(),
        });
    }
    rows
}

#[test]
fn compare_li_open_shell_ri_r_with_rest_standard_path() {
    let basis_name = "def2-tzvp";
    let aux_name = "def2-tzvp-rifit";
    let ctrl_path = write_temp_ctrl_li_open_shell_with_aux(basis_name, aux_name);
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let mut scf_data = crate::scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    assert!(
        scf_data.density_matrix.len() >= 2,
        "open-shell SCF should provide alpha and beta density matrices"
    );

    let dm_spin = scf_data.density_matrix[0..2].to_vec();
    let lib_obs = lib_vee_uhf_r_observables_with_auxbasis(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.mol.auxbas4elem,
        &dm_spin,
    );

    let ri_rest = Some(scf_data.mol.prepare_rimatr_for_ri_v_rayon(None));
    let j_rest_u = vj_upper_with_rimatr_sync(&ri_rest, &dm_spin, 2, 1.0);
    let k_rest_u = vk_upper_with_rimatr_use_dm_only_sync_v02(&ri_rest, &dm_spin, 2, 1.0);

    let dm_a_upper = dm_spin[0].to_matrixupper();
    let dm_b_upper = dm_spin[1].to_matrixupper();
    let mut j_total_upper = j_rest_u[0].clone();
    j_total_upper
        .data
        .iter_mut()
        .zip(j_rest_u[1].data.iter())
        .for_each(|(to, from)| *to += *from);

    let ej_rest = 0.5
        * (SCF::par_energy_contraction(&dm_a_upper, &j_total_upper)
            + SCF::par_energy_contraction(&dm_b_upper, &j_total_upper));
    let ek_rest = -0.5
        * (SCF::par_energy_contraction(&dm_a_upper, &k_rest_u[0])
            + SCF::par_energy_contraction(&dm_b_upper, &k_rest_u[1]));
    let e_rest = ej_rest + ek_rest;

    let d_ej = (lib_obs.ej - ej_rest).abs();
    let d_ek = (lib_obs.ek - ek_rest).abs();
    let d_e = (lib_obs.total - e_rest).abs();

    println!(
        "Li open-shell {basis_name} + {aux_name}: lib_rint(EJ, EK, E) = ({:.16e}, {:.16e}, {:.16e})",
        lib_obs.ej, lib_obs.ek, lib_obs.total
    );
    println!(
        "Li open-shell {basis_name} + {aux_name}: rest_std(EJ, EK, E) = ({:.16e}, {:.16e}, {:.16e})",
        ej_rest, ek_rest, e_rest
    );
    println!(
        "Li open-shell {basis_name} + {aux_name}: |dEJ| = {:.3e}, |dEK| = {:.3e}, |dE| = {:.3e}",
        d_ej, d_ek, d_e
    );

    assert!(
        d_ej < 2.0e-5 && d_ek < 1.0e-5 && d_e < 1.0e-5,
        "open-shell lib_rint r-energy mismatch vs REST standard path: dEJ={d_ej:.3e}, dEK={d_ek:.3e}, dE={d_e:.3e}"
    );

    let _ = fs::remove_file(ctrl_path);
}

#[test]
#[ignore = "diagnostic only; compares REST RI-r against exact-r without a regression assertion"]
fn diagnose_h2_rest_internal_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared() {
    let cases = [
        ("cc-pvdz", "cc-pvdz-rifit"),
        ("def2-tzvp", "def2-tzvp-rifit"),
    ];

    for (basis_name, aux_name) in cases {
        let ctrl_path = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let (ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "diagnose_h2_rest_internal_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared",
            );

        let dm_vec_rest = vec![scf_data.density_matrix[0].clone()];
        let ri_rest = Some(scf_data.mol.prepare_rimatr_for_ri_v_rayon(None));
        let j_rest_u = vj_upper_with_rimatr_sync(&ri_rest, &dm_vec_rest, 1, 1.0);
        let k_rest_u = vk_upper_with_rimatr_use_dm_only_sync_v02(&ri_rest, &dm_vec_rest, 1, 1.0);

        let mut j_total_u = j_rest_u[0].clone();
        j_total_u
            .data
            .iter_mut()
            .zip(j_rest_u[1].data.iter())
            .for_each(|(to, from)| *to += *from);
        let j_rest = j_total_u.to_matrixfull().unwrap();
        let k_rest = k_rest_u[0].to_matrixfull().unwrap();

        let (j_ex, k_ex) = build_jk_from_p_kernel(&ao_bfs, &p_cart, eri_ao_4c_r);
        let (ej_rest, ek_rest, e_rest) = ej_ek_from_p_jk_rhf(&p_cart, &j_rest, &k_rest);
        let (ej_ex, ek_ex, e_ex) = ej_ek_from_p_jk_rhf(&p_cart, &j_ex, &k_ex);

        let d_j = max_abs_diff_matrix(&j_rest, &j_ex);
        let d_k = max_abs_diff_matrix(&k_rest, &k_ex);
        let d_ej = (ej_rest - ej_ex).abs();
        let d_ek = (ek_rest - ek_ex).abs();
        let d_e = (e_rest - e_ex).abs();

        println!(
            "{basis_name} + {aux_name} converged RHF density (SCF energy = {:.16e}) REST-internal-RI-r vs exact-r: dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
            scf_data.scf_energy, d_j, d_k, d_ej, d_ek, d_e
        );

        let _ = fs::remove_file(ctrl_path);
    }
}

#[test]
#[ignore = "diagnostic only; compares RI-r auxiliary choices against exact-r without a regression assertion"]
fn diagnose_h2_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared() {
    let cases = [
        ("cc-pvdz", "cc-pvdz-rifit"),
        ("def2-tzvp", "def2-tzvp-rifit"),
    ];

    for (basis_name, aux_name) in cases {
        let ctrl_path = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let (ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "diagnose_h2_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared",
            );
        let aux_bfs = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
            &scf_data.mol.geom,
            &scf_data.mol.auxbas4elem,
        )
        .expect("failed to load auxiliary basis (shell_shared_from_raw)");

        let (d_j, d_k, d_ej, d_ek, d_e, naux) =
            compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs);

        println!(
            "{basis_name} + {aux_name} converged RHF density (SCF energy = {:.16e}) RI-r vs exact-r: naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
            scf_data.scf_energy, naux, d_j, d_k, d_ej, d_ek, d_e
        );

        let _ = fs::remove_file(ctrl_path);
    }
}
#[test]
#[ignore = "diagnostic only; depends on the external PySCF autoaux bridge"]
fn diagnose_h2_ri_r_with_converged_rhf_density_across_aux_generators() {
    let cases = [
        ("cc-pvdz", "cc-pvdz-rifit"),
        ("def2-tzvp", "def2-tzvp-rifit"),
    ];

    for (basis_name, aux_name) in cases {
        let ctrl_path = write_temp_ctrl_h2(basis_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let dm = scf_data.density_matrix[0].clone();
        let (ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &dm,
                "diagnose_h2_ri_r_with_converged_rhf_density_across_aux_generators",
            );

        let ctrl_rifit = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
        let mol_rifit = Molecule::build(ctrl_rifit.clone(), None).unwrap();
        let aux_bfs_rifit = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
            &scf_data.mol.geom,
            &mol_rifit.auxbas4elem,
        )
        .expect("failed to expand rifit aux basis");
        let (d_j_rifit, d_k_rifit, d_ej_rifit, d_ek_rifit, d_e_rifit, naux_rifit) =
            compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_rifit);

        let (ctrl_etb, aux_dir_etb) = write_temp_ctrl_h2_with_etb(basis_name, 1.3);
        let mol_etb = Molecule::build(ctrl_etb.clone(), None).unwrap();
        let aux_bfs_etb = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
            &scf_data.mol.geom,
            &mol_etb.auxbas4elem,
        )
        .expect("failed to expand ETB aux basis");
        let (d_j_etb, d_k_etb, d_ej_etb, d_ek_etb, d_e_etb, naux_etb) =
            compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_etb);

        let autoaux_basis4elem = load_pyscf_autoaux_basis4elem(&scf_data.mol, basis_name);
        let aux_bfs_autoaux = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
            &scf_data.mol.geom,
            &autoaux_basis4elem,
        )
        .expect("failed to expand PySCF autoaux basis");
        let (d_j_autoaux, d_k_autoaux, d_ej_autoaux, d_ek_autoaux, d_e_autoaux, naux_autoaux) =
            compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_autoaux);

        println!(
            "{basis_name} converged RHF density (exact SCF energy = {:.16e})",
            scf_data.scf_energy
        );
        println!(
            "  rifit        naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
            naux_rifit, d_j_rifit, d_k_rifit, d_ej_rifit, d_ek_rifit, d_e_rifit
        );
        println!(
            "  ETB(beta=1.3) naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
            naux_etb, d_j_etb, d_k_etb, d_ej_etb, d_ek_etb, d_e_etb
        );
        println!(
            "  PySCF autoaux naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
            naux_autoaux, d_j_autoaux, d_k_autoaux, d_ej_autoaux, d_ek_autoaux, d_e_autoaux
        );

        let _ = fs::remove_file(ctrl_path);
        let _ = fs::remove_file(ctrl_rifit);
        let _ = fs::remove_file(ctrl_etb);
        let _ = fs::remove_dir_all(aux_dir_etb);
    }
}
#[test]
#[ignore = "diagnostic only; scans ETB beta values and is too slow for default regression tests"]
fn diagnose_h2_ri_r_with_converged_rhf_density_with_etb_shell_shared() {
    let basis_cases = ["cc-pvdz", "def2-tzvp"];
    let beta_cases = [2.0_f64, 1.7_f64, 1.5_f64, 1.3_f64];

    for basis_name in basis_cases {
        let ctrl_path = write_temp_ctrl_h2(basis_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let dm = scf_data.density_matrix[0].clone();
        let (ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &dm,
                "diagnose_h2_ri_r_with_converged_rhf_density_with_etb_shell_shared",
            );

        for etb_beta in beta_cases {
            let (ctrl_path_etb, aux_dir) = write_temp_ctrl_h2_with_etb(basis_name, etb_beta);
            let mol_etb = Molecule::build(ctrl_path_etb.clone(), None).unwrap();
            let aux_bfs = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                &scf_data.mol.geom,
                &mol_etb.auxbas4elem,
            )
            .expect("failed to load ETB auxiliary basis (shell_shared_from_raw)");

            assert_eq!(ao_bfs.len(), p_cart.size[0]);
            assert_eq!(ao_bfs.len(), p_cart.size[1]);
            assert_eq!(aux_bfs.len(), mol_etb.num_auxbas);
            assert!(
                !aux_bfs.is_empty(),
                "ETB auxiliary basis should not be empty for {basis_name}"
            );

            let (d_j, d_k, d_ej, d_ek, d_e, naux) =
                compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs);

            println!(
                "{basis_name} + ETB(beta={etb_beta:.3}) converged RHF density (exact SCF energy = {:.16e}): naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                scf_data.scf_energy, naux, d_j, d_k, d_ej, d_ek, d_e
            );

            let _ = fs::remove_file(ctrl_path_etb);
            let _ = fs::remove_dir_all(aux_dir);
        }

        let _ = fs::remove_file(ctrl_path);
    }
}
#[test]
fn default_r2_path_uses_etb_beta_1p7_when_auxbasis_is_missing() {
    let basis_name = "cc-pvdz";
    let ctrl_path = write_temp_ctrl_h2(basis_name);
    let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let default_obs = lib_vee_rhf_r2_observables(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.density_matrix[0],
    );

    let auxbasis4elem = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
        .expect("default r2 ETB auxiliary basis should be available");
    let explicit_obs = lib_vee_rhf_r2_observables_with_auxbasis(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &auxbasis4elem,
        &scf_data.density_matrix[0],
    );

    assert!((default_obs.ej - explicit_obs.ej).abs() < 1.0e-12);
    assert!((default_obs.ek - explicit_obs.ek).abs() < 1.0e-12);
    assert!((default_obs.total - explicit_obs.total).abs() < 1.0e-12);

    let _ = fs::remove_file(ctrl_path);
}
#[test]
#[ignore = "diagnostic only; compares RI-r2 auxiliary basis choices"]
fn diagnose_h2_ri_r2_with_rifit_and_etb_auxbasis() {
    let cases = [
        ("cc-pvdz", "cc-pvdz-rifit"),
        ("def2-tzvp", "def2-tzvp-rifit"),
        ("def2-qzvpp", "def2-qzvpp-rifit"),
    ];

    for (basis_name, aux_name) in cases {
        let ctrl_path = write_temp_ctrl_h2(basis_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let ctrl_rifit = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
        let mol_rifit = Molecule::build(ctrl_rifit.clone(), None).unwrap();
        let rifit_obs = lib_vee_rhf_r2_observables_with_auxbasis(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &mol_rifit.auxbas4elem,
            &scf_data.density_matrix[0],
        );

        println!(
            "{basis_name} RI-r2 rifit: naux={} EJ={:.16e} EK={:.16e} E={:.16e}",
            mol_rifit.num_auxbas, rifit_obs.ej, rifit_obs.ek, rifit_obs.total
        );

        for etb_beta in [1.3_f64, 1.7_f64] {
            let (ctrl_etb, aux_dir_etb) = write_temp_ctrl_h2_with_etb(basis_name, etb_beta);
            let mol_etb = Molecule::build(ctrl_etb.clone(), None).unwrap();
            let etb_obs = lib_vee_rhf_r2_observables_with_auxbasis(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &mol_etb.auxbas4elem,
                &scf_data.density_matrix[0],
            );
            println!(
                "  ETB(beta={etb_beta:.1}) naux={} EJ={:.16e} EK={:.16e} E={:.16e} |vs rifit| dEJ={:.12e} dEK={:.12e} dE={:.12e}",
                mol_etb.num_auxbas,
                etb_obs.ej,
                etb_obs.ek,
                etb_obs.total,
                (etb_obs.ej - rifit_obs.ej).abs(),
                (etb_obs.ek - rifit_obs.ek).abs(),
                (etb_obs.total - rifit_obs.total).abs()
            );
            let _ = fs::remove_file(ctrl_etb);
            let _ = fs::remove_dir_all(aux_dir_etb);
        }

        let _ = fs::remove_file(ctrl_path);
        let _ = fs::remove_file(ctrl_rifit);
    }
}

#[test]
#[ignore]
fn diagnose_r2_auxbasis_rifit_etb_exact_representative_set() {
    struct AuxDiagCase {
        label: &'static str,
        basis_dir: &'static str,
        rifit_aux_dir: Option<&'static str>,
        positions: Vec<String>,
        run_exact: bool,
        heavy: bool,
    }

    let cases = vec![
        AuxDiagCase {
            label: "H2/sto-3g",
            basis_dir: "sto-3g",
            rifit_aux_dir: None,
            positions: h2_positions(),
            run_exact: true,
            heavy: false,
        },
        AuxDiagCase {
            label: "H2/cc-pvdz",
            basis_dir: "cc-pvdz",
            rifit_aux_dir: Some("cc-pvdz-rifit"),
            positions: h2_positions(),
            run_exact: true,
            heavy: false,
        },
        AuxDiagCase {
            label: "H2O/sto-3g",
            basis_dir: "sto-3g",
            rifit_aux_dir: None,
            positions: h2o_positions(),
            run_exact: true,
            heavy: false,
        },
        AuxDiagCase {
            label: "H2O/def2-svp",
            basis_dir: "def2-svp",
            rifit_aux_dir: Some("def2-svp-rifit"),
            positions: h2o_positions(),
            run_exact: false,
            heavy: false,
        },
        AuxDiagCase {
            label: "CH4/sto-3g",
            basis_dir: "sto-3g",
            rifit_aux_dir: None,
            positions: methane_positions(0.0),
            run_exact: true,
            heavy: false,
        },
        AuxDiagCase {
            label: "CH4/def2-svp",
            basis_dir: "def2-svp",
            rifit_aux_dir: Some("def2-svp-rifit"),
            positions: methane_positions(0.0),
            run_exact: false,
            heavy: false,
        },
        AuxDiagCase {
            label: "benzene/def2-svp",
            basis_dir: "def2-svp",
            rifit_aux_dir: Some("def2-svp-rifit"),
            positions: benzene_positions(),
            run_exact: false,
            heavy: true,
        },
        AuxDiagCase {
            label: "CH4-CH4/def2-svp",
            basis_dir: "def2-svp",
            rifit_aux_dir: Some("def2-svp-rifit"),
            positions: methane_dimer_positions(4.0),
            run_exact: false,
            heavy: true,
        },
        AuxDiagCase {
            label: "C60-frag-C10/def2-svp",
            basis_dir: "def2-svp",
            rifit_aux_dir: Some("def2-svp-rifit"),
            positions: c60_fragment_c10_positions(),
            run_exact: false,
            heavy: true,
        },
    ];

    let filter = std::env::var("REST_R2_AUX_DIAG_FILTER").ok();
    let include_heavy = std::env::var("REST_R2_AUX_DIAG_INCLUDE_HEAVY")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    println!(
        "RI_R2_AUX_DIAG_BEGIN filter={} include_heavy={} columns: label basis source naux EJ EK E deltas",
        filter.as_deref().unwrap_or("ALL"),
        include_heavy
    );

    let mut ran = 0_usize;
    for case in cases {
        if case.heavy && !include_heavy {
            println!(
                "RI_R2_AUX_DIAG_SKIP label=\"{}\" reason=heavy_set_requires_REST_R2_AUX_DIAG_INCLUDE_HEAVY",
                case.label
            );
            continue;
        }
        if let Some(filter) = &filter {
            if !case.label.contains(filter)
                && !case.basis_dir.contains(filter)
                && !case.rifit_aux_dir.unwrap_or("").contains(filter)
            {
                continue;
            }
        }
        ran += 1;

        let ctrl_path =
            write_temp_ctrl_from_positions(case.label, case.basis_dir, None, &case.positions);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);
        let natom = scf_data.mol.geom.elem.len().max(1) as f64;

        let exact_obs = if case.run_exact {
            let exact = lib_vee_rhf_r2_observables_exact(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
            );
            println!(
                "RI_R2_AUX_DIAG label=\"{}\" basis={} source=exact naux=NA EJ={:.16e} EK={:.16e} E={:.16e}",
                case.label, case.basis_dir, exact.ej, exact.ek, exact.total
            );
            Some(exact)
        } else {
            None
        };

        let etb_aux = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
            .expect("default r2 ETB auxiliary basis should be available");
        let etb_naux = r2_auxbasis_count(&scf_data.mol.geom, &etb_aux);
        let etb_obs = lib_vee_rhf_r2_observables_with_auxbasis(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &etb_aux,
            &scf_data.density_matrix[0],
        );
        println!(
            "RI_R2_AUX_DIAG label=\"{}\" basis={} source=ETB_beta_1.7 naux={} EJ={:.16e} EK={:.16e} E={:.16e}{}",
            case.label,
            case.basis_dir,
            etb_naux,
            etb_obs.ej,
            etb_obs.ek,
            etb_obs.total,
            r2_aux_diag_delta(exact_obs, etb_obs)
        );

        if let Some(rifit_aux_dir) = case.rifit_aux_dir {
            let ctrl_rifit = write_temp_ctrl_from_positions(
                case.label,
                case.basis_dir,
                Some(rifit_aux_dir),
                &case.positions,
            );
            let mol_rifit = Molecule::build(ctrl_rifit.clone(), None).unwrap();
            let rifit_naux = r2_auxbasis_count(&scf_data.mol.geom, &mol_rifit.auxbas4elem);
            let rifit_obs = lib_vee_rhf_r2_observables_with_auxbasis(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &mol_rifit.auxbas4elem,
                &scf_data.density_matrix[0],
            );
            let d_ej_etb = (rifit_obs.ej - etb_obs.ej).abs();
            let d_ek_etb = (rifit_obs.ek - etb_obs.ek).abs();
            let d_e_etb = (rifit_obs.total - etb_obs.total).abs();
            println!(
                "RI_R2_AUX_DIAG label=\"{}\" basis={} source=rifit aux={} naux={} EJ={:.16e} EK={:.16e} E={:.16e}{} dEJ_ETB={:.12e} dEK_ETB={:.12e} dE_ETB={:.12e} dE_ETB_per_atom={:.12e}",
                case.label,
                case.basis_dir,
                rifit_aux_dir,
                rifit_naux,
                rifit_obs.ej,
                rifit_obs.ek,
                rifit_obs.total,
                r2_aux_diag_delta(exact_obs, rifit_obs),
                d_ej_etb,
                d_ek_etb,
                d_e_etb,
                d_e_etb / natom
            );
            assert!(rifit_obs.total.is_finite());
            let _ = fs::remove_file(ctrl_rifit);
        }

        assert!(etb_obs.total.is_finite());
        let _ = fs::remove_file(ctrl_path);
    }
    assert!(ran > 0, "RI-r2 auxiliary basis diagnostic ran no cases");
    println!("RI_R2_AUX_DIAG_END ran_cases={ran}");
}
#[test]
fn r2_optimized_contractions_match_reference_shell_shared() {
    let basis_cases = ["cc-pvdz", "def2-tzvp"];

    for basis_name in basis_cases {
        let ctrl_path = write_temp_ctrl_h2(basis_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let dm = scf_data.density_matrix[0].clone();
        let (ao_shells, _ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &dm,
                "r2_optimized_contractions_match_reference_shell_shared",
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build default ETB auxiliary rint shells");
        let ri_r2 = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
            .expect("failed to build RI-r2 matrix for contraction regression");
        let dm_vec = vec![p_cart.clone()];

        let j_ref_u = vj_upper_with_rimatr_sync_internal(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
        let k_ref_u = vk_upper_with_rimatr_sync_internal(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
        let j_opt_u = vj_upper_with_rimatr_r2_sync(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
        let k_opt_u = vk_upper_with_rimatr_r2_sync(&Some(ri_r2), &dm_vec, 1, 1.0);

        let j_ref = j_ref_u[0].to_matrixfull().unwrap();
        let k_ref = k_ref_u[0].to_matrixfull().unwrap();
        let j_opt = j_opt_u[0].to_matrixfull().unwrap();
        let k_opt = k_opt_u[0].to_matrixfull().unwrap();

        let (ej_ref, ek_ref, e_ref) = ej_ek_from_p_jk_rhf(&p_cart, &j_ref, &k_ref);
        let (ej_opt, ek_opt, e_opt) = ej_ek_from_p_jk_rhf(&p_cart, &j_opt, &k_opt);

        assert!(
            max_abs_diff_matrix(&j_ref, &j_opt) < 1.0e-10,
            "{basis_name} optimized r2 J contraction drifted from reference"
        );
        assert!(
            max_abs_diff_matrix(&k_ref, &k_opt) < 1.0e-10,
            "{basis_name} optimized r2 K contraction drifted from reference"
        );
        assert!(
            (ej_ref - ej_opt).abs() < 1.0e-10,
            "{basis_name} optimized r2 E_J drifted from reference"
        );
        assert!(
            (ek_ref - ek_opt).abs() < 1.0e-10,
            "{basis_name} optimized r2 E_K drifted from reference"
        );
        assert!(
            (e_ref - e_opt).abs() < 1.0e-10,
            "{basis_name} optimized r2 total energy drifted from reference"
        );

        let _ = fs::remove_file(ctrl_path);
    }
}
