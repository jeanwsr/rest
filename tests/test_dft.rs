use std::collections::HashMap;
use std::io::Read;
use rest_tensors::{MatrixFull, ParMathMatrix};
use pyrest::basis_io::{BasCell, Basis4Elem, cint_norm_factor, gto_value, };
use pyrest::dft::{DFA4REST, Grids};
use pyrest::dft::libxc_helper::{xc_func_init, lda_exc_vxc};
use pyrest::utilities;
use regex::Regex;

#[test]
fn debug_num_density_for_atom() {
    let angular = 2;
    let num_basis = 2*angular+1;
    let mut dm = [
        //MatrixFull::from_vec([1,1], vec![2.0]).unwrap(),
        //MatrixFull::from_vec([num_basis,num_basis], vec![
        //    1.0,0.0,0.0,
        //    0.0,1.0,0.0,
        //    0.0,0.0,0.0]).unwrap(), 
        MatrixFull::from_vec([5,5], vec![
            2.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0]).unwrap(), 
        MatrixFull::empty()];
    let mut alpha_min_h: HashMap<usize, f64> = HashMap::new();
    alpha_min_h.insert(angular,0.122);
    let alpha_max_h: f64 = 0.122;
    let basis4elem = vec![Basis4Elem {
        electron_shells: vec![
            BasCell {
                function_type: None,
                region: None,
                angular_momentum: vec![angular as i32],
                exponents: vec![0.122],
                coefficients: vec![vec![1.0/cint_norm_factor(angular as i32, 0.122)]],
                native_coefficients: vec![vec![1.0/cint_norm_factor(angular as i32, 0.122)]]
            }
        ],
        references: None,
        ecp_electrons: None,
        ecp_potentials: None,
        global_index: (0,0)
    }];
    let center_coordinates_bohr = vec![(0.0,0.0,0.0)];
    let proton_charges = vec![1];
    let grids = Grids::build_nonstd(
        center_coordinates_bohr.clone(), 
        proton_charges.clone(), 
        vec![alpha_min_h], 
        vec![alpha_max_h], &mut None);
    let mut total_density = 0.0;
    let mut count:usize =0;
    grids.coordinates.iter().zip(grids.weights.iter()).for_each(|(r,w)| {
        let mut density_r_sum = 0.0;
        let mut density_r:Vec<f64> = vec![];
        basis4elem.iter().zip(center_coordinates_bohr.iter()).for_each(|(elem,geom_nonstd)| {
            let geom = [geom_nonstd.0,geom_nonstd.1,geom_nonstd.2];
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            //density_r.extend(gto_value(r, &tmp_geom, elem, &mol.ctrl.basis_type));
            //let tmp_vec = gto_value(r, &tmp_geom, elem, &"spheric".to_string());
            let tmp_vec = gto_value(r, &tmp_geom, elem, &"spheric".to_string());
            //println!("debug 0: len {}", &tmp_vec.len());
            density_r.extend(tmp_vec);
            //if count<=10 {println!("{:?},{:?},{:?}", density_r,elem.electron_shells[0].exponents, elem.electron_shells[0].coefficients[0])};
        });
        //println!{"debug 1"};
        let mut density_rr = MatrixFull::from_vec([num_basis,1],density_r).unwrap();
        //println!{"debug 2"};
        dm.iter_mut().for_each(|dm_s| {
            let mut tmp_mat = MatrixFull::new([num_basis,1],0.0);
            tmp_mat.lapack_dgemm(&mut density_rr, dm_s, 'T', 'N', 1.0, 0.0);
            if count<=10 {println!("count: {},{:?}",count, &tmp_mat.data)};
            density_r_sum += tmp_mat.data.iter().zip(density_rr.data.iter()).fold(0.0, |acc,(a,b)| {acc + a*b});
        });
        if count<=10 {println!("{:?},{},{}", r,w,density_r_sum)};
        count += 1;
        //println!{"debug 3"};
        total_density += density_r_sum * w;
    });

    println!("Total density: {}", total_density);
    
}

#[test]
fn test_libxc() {
    let rho:Vec<f64> = vec![0.1,0.2,0.3,0.4,0.5,0.6,0.8];
    let _sigma:Vec<f64> = vec![0.2,0.3,0.4,0.5,0.6,0.7];
    //&rho.par_iter().for_each(|c| {println!("{:16.8}",c)});
    //let mut exc:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let mut vrho:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let func_id: usize = ffi_xc::XC_GGA_X_XPBE as usize;
    let spin_channel: usize = 1;

    let my_xc = DFA4REST::parse_scf("lda_x_slater", spin_channel); 
    //let mut my_xc = XcFuncType::xc_func_init_fdqc(&"pw-lda",spin_channel); 



    let mut exc = MatrixFull::new([rho.len()/spin_channel,1],0.0);
    let mut vrho = MatrixFull::new([rho.len()/spin_channel,spin_channel],0.0);
    my_xc.dfa_compnt_scf.iter().zip(my_xc.dfa_paramr_scf.iter()).for_each(|(xc_func, xc_para)| {
        //let mut new_vec = vec![-0.34280861230056237, -0.43191178672272906, -0.494415573788165, -0.5441747517896713, -0.586194481347579, -0.622924588811561, -0.6856172246011247];
        //let tmp_c = (new_vec.as_mut_ptr(), new_vec.len(),new_vec.capacity());
        //let new_vec = unsafe{Vec::from_raw_parts(tmp_c.0, tmp_c.1, tmp_c.2)};
        //new_vec.par_iter().for_each(|c| {println!("{:16.8e}",c)});
        let xc_func = my_xc.init_libxc(xc_func);

        let (tmp_exc, tmp_vrho) = lda_exc_vxc(&xc_func,&rho);
        //let tmp_exc_2 = tmp_exc.clone();
        //println!("WARNNING:: unsolved rayon par_iter problem. It should be relevant to be the fact that tmp_exc is prepared by libxc via ffi");
        //println!("tmp_vec_2 copied from tmp_exc: {:?},{},{}", &tmp_exc_2, tmp_exc_2.len(),tmp_exc_2.capacity());
        //println!("tmp_vec");
        //&tmp_exc.iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        //println!("tmp_vec_2");
        //&tmp_exc_2.par_iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        //println!("tmp_vec_2");
        //&tmp_exc.iter().for_each(|c| {
        //    println!("{:16.8e}",c);
        //});
        let tmp_exc = MatrixFull::from_vec([rho.len()/spin_channel,1],tmp_exc).unwrap();
        let tmp_vrho = MatrixFull::from_vec([rho.len()/spin_channel,spin_channel],tmp_vrho).unwrap();
        //println!("{:?}", &tmp_exc.data);
        //println!("{:?}", &tmp_vrho.data);
        exc.par_self_scaled_add(&tmp_exc,*xc_para);
        vrho.par_self_scaled_add(&tmp_vrho,*xc_para);
        //exc.data.par_iter_mut().zip(tmp_exc.data.par_iter()).for_each(|(c,p)| {
        //    println!("{:16.8e},{:16.8e}",c,p);
        //});
        //println!("{:?}", &exc.data);
        //println!("{:?}", &vrho.data);
    });
    println!("{:?}", exc.data);
    println!("{:?}", vrho.data);
}

#[test]
#[ignore]
#[allow(unused)]
fn read_grid() {
    let mut grids_file = std::fs::File::open("/home/igor/Documents/Package-Pool/Rust/rest/grids").unwrap();
    let mut content = String::new();
    grids_file.read_to_string(&mut content);
    //println!("{}",&content);
    let re1 = Regex::new(r"(?x)\s*
        (?P<x>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'x' position
        \s+
        (?P<y>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'y' position
        \s+
        (?P<z>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*,# the 'z' position
        \s+
        (?P<w>[\+-]?\d+.\d+[eE][\+-]?\d+)\s*# the 'w' weight
        \s*\n").unwrap();
    //if let Some(cap)  = re1.captures(&content) {
    //    println!("{:?}", &cap)
    //}
    for cap in re1.captures_iter(&content) {
        let x:f64 = cap[1].parse().unwrap();
        let y:f64 = cap[2].parse().unwrap();
        let z:f64 = cap[3].parse().unwrap();
        let w:f64 = cap[4].parse().unwrap();
        println!("{:16.8} {:16.8} {:16.8} {:16.8}", x,y,z,w);
    }
}
#[test]
fn debug_transpose() {
    let len_a = 111_usize;
    let len_b = 40000_usize;
    let orig_a:Vec<f64> = (0..len_a*len_b).map(|i| {i as f64}).collect();
    let a_mat = MatrixFull::from_vec([len_a,len_b],orig_a).unwrap();
    let dt0 = utilities::init_timing();
    let b_mat = a_mat.transpose_and_drop();
    //b_mat.formated_output(10, "full");
    let _dt1 = utilities::timing(&dt0, Some("old transpose"));
    let orig_a:Vec<f64> = (0..len_a*len_b).map(|i| {i as f64}).collect();
    let a_mat = MatrixFull::from_vec([len_a,len_b],orig_a).unwrap();
    let dt0 = utilities::init_timing();
    let c_mat = a_mat.transpose_and_drop();
    //b_mat.formated_output(10, "full");
    let _dt1 = utilities::timing(&dt0, Some("new transpose"));
    b_mat.data.iter().zip(c_mat.data.iter()).for_each(|(b,c)| {
        assert!(*b==*c);
    });
}





#[test]
fn test_rsh_cam_coeff_raw() {
    // Test raw libxc CAM coefficients match pyscf expectations
    // Each entry: (libxc_id, expected_omega, expected_alpha, expected_beta, expected_hyb)
    // hyb = alpha + beta (pyscf convention for RSH)
    let ref_data = vec![
        (464, 0.300, 1.0, -0.842294, 0.157706),    // WB97X
        (466, 0.300, 1.0, -0.833,    0.167),        // WB97X-V
        (471, 0.200, 1.0, -0.777964, 0.222036),     // WB97X-D
        (433, 0.330, 0.65, -0.46,   0.19),          // CAM-B3LYP
        (400, 0.330, 1.0, -1.0,     0.0),           // LC-BLYP
        (478, 0.400, 1.0, -1.0,     0.0),           // LC-wPBE
        (428, 0.110, 0.0,  0.25,    0.25),          // HSE06
        (427, 0.1061, 0.0, 0.25,    0.25),          // HSE03
    ];

    for (libxc_id, exp_omega, exp_alpha, exp_beta, exp_hyb) in &ref_data {
        let func = xc_func_init(*libxc_id, 1);
        assert!(func.is_hyb_cam(), "ID {} should be RSH/CAM", libxc_id);
        let (omega, alpha, beta) = func.cam_coef().unwrap_or((0.0, 0.0, 0.0));
        let hyb = alpha + beta;
        assert!((omega - exp_omega).abs() < 1e-3, "ID {}: omega mismatch: got {} expected {}", libxc_id, omega, exp_omega);
        assert!((alpha - exp_alpha).abs() < 1e-3, "ID {}: alpha mismatch: got {} expected {}", libxc_id, alpha, exp_alpha);
        assert!((beta - exp_beta).abs() < 1e-3, "ID {}: beta mismatch: got {} expected {}", libxc_id, beta, exp_beta);
        assert!((hyb - exp_hyb).abs() < 1e-4, "ID {}: hyb mismatch: got {} expected {}", libxc_id, hyb, exp_hyb);
        println!("ID {}: omega={:.4} alpha={:.4} beta={:+.4} hyb={:.6}  OK", libxc_id, omega, alpha, beta, hyb);
    }
    println!("All {} RSH libxc raw CAM coefficient tests passed.", ref_data.len());
}

#[test]
fn test_rsh_parameters_wb97x() {
    let dfa = DFA4REST::new("wb97x", 1, 0);
    assert!(dfa.is_hybrid());
    assert!(dfa.is_rsh());
    assert!((dfa.omega().unwrap() - 0.3).abs() < 1e-9);
    assert!((dfa.rsh_alpha().unwrap() - 1.0).abs() < 1e-9);
    assert!((dfa.dfa_hybrid_scf - 0.157706).abs() < 1e-5);
    assert_eq!(dfa.dfa_compnt_scf, vec![464]);
}

#[test]
fn test_rsh_parameters_cam_b3lyp() {
    let dfa = DFA4REST::new("cam-b3lyp", 1, 0);
    assert!(dfa.is_hybrid());
    assert!(dfa.is_rsh());
    assert!((dfa.omega().unwrap() - 0.33).abs() < 1e-9);
    assert!((dfa.rsh_alpha().unwrap() - 0.65).abs() < 1e-5);
    assert!((dfa.dfa_hybrid_scf - 0.19).abs() < 1e-5);
    assert_eq!(dfa.dfa_compnt_scf, vec![433]);
    // also test case variation
    let dfa2 = DFA4REST::new("CAM-B3LYP", 1, 0);
    assert!((dfa.omega().unwrap() - dfa2.omega().unwrap()).abs() < 1e-9);
    assert!((dfa.dfa_hybrid_scf - dfa2.dfa_hybrid_scf).abs() < 1e-9);
}

#[test]
fn test_rsh_parameters_lc_blyp() {
    let dfa = DFA4REST::new("lc-blyp", 1, 0);
    assert!(dfa.is_hybrid());
    assert!(dfa.is_rsh());
    assert!((dfa.omega().unwrap() - 0.33).abs() < 1e-9);
    assert!((dfa.rsh_alpha().unwrap() - 1.0).abs() < 1e-9);
    // LC-BLYP has hyb=0 (alpha=1, beta=-1 → hyb=0, LR-only HF)
    assert!((dfa.dfa_hybrid_scf - 0.0).abs() < 1e-8);
    assert_eq!(dfa.dfa_compnt_scf, vec![400]);
}

#[test]
fn test_rsh_parameters_hse06() {
    let dfa = DFA4REST::new("hse06", 1, 0);
    assert!(dfa.is_hybrid());
    assert!(dfa.is_rsh());
    assert!((dfa.omega().unwrap() - 0.11).abs() < 1e-9);
    // HSE06 has alpha=0 (no additional LR-HF), beta=0.25 (SR-HF only)
    assert!((dfa.rsh_alpha().unwrap() - 0.0).abs() < 1e-5);
    assert!((dfa.dfa_hybrid_scf - 0.25).abs() < 1e-5);
    assert_eq!(dfa.dfa_compnt_scf, vec![428]);
}

#[test]
fn test_non_rsh_b3lyp() {
    let dfa = DFA4REST::new("b3lyp", 1, 0);
    assert!(dfa.is_hybrid(), "B3LYP should be hybrid");
    assert!(!dfa.is_rsh(), "B3LYP should NOT be range-separated");
    assert!(dfa.omega().is_none());
    assert!(dfa.rsh_alpha().is_none());
}

#[test]
fn test_non_rsh_pbe() {
    let dfa = DFA4REST::new("pbe", 1, 0);
    assert!(!dfa.is_hybrid());
    assert!(!dfa.is_rsh());
    assert!(dfa.omega().is_none());
    assert!(dfa.rsh_alpha().is_none());
    assert_eq!(dfa.dfa_hybrid_scf, 0.0);
}

#[test]
fn test_cam_b3lyp_case_insensitive() {
    let names = ["cam-b3lyp", "CAMB3LYP"];
    let mut dfas = vec![];
    for name in &names {
        dfas.push(DFA4REST::new(name, 1, 0));
    }
    for i in 1..dfas.len() {
        assert!((dfas[0].omega().unwrap() - dfas[i].omega().unwrap()).abs() < 1e-9);
        assert!((dfas[0].rsh_alpha().unwrap() - dfas[i].rsh_alpha().unwrap()).abs() < 1e-9);
        assert!((dfas[0].dfa_hybrid_scf - dfas[i].dfa_hybrid_scf).abs() < 1e-9);
    }
}

#[test]
fn test_rsh_spin_polarized() {
    // Test RSH initialization with spin=2 (open-shell)
    let dfa_closed = DFA4REST::new("cam-b3lyp", 1, 0);
    let dfa_open = DFA4REST::new("cam-b3lyp", 2, 0);
    // RSH parameters should be spin-independent
    assert!((dfa_closed.omega().unwrap() - dfa_open.omega().unwrap()).abs() < 1e-9);
    assert!((dfa_closed.rsh_alpha().unwrap() - dfa_open.rsh_alpha().unwrap()).abs() < 1e-9);
    assert!((dfa_closed.dfa_hybrid_scf - dfa_open.dfa_hybrid_scf).abs() < 1e-9);
}

#[test]
fn test_rsh_hyb_override() {
    // Verify that for RSH functionals, dfa_hybrid_scf is overridden to alpha+beta
    // NOT the raw xc_hyb_exx_coeff (which may return 1.0)
    let dfa = DFA4REST::new("lc-blyp", 1, 0);
    // LC-BLYP: raw xc_hyb_exx_coef = 1.0 (because it's a hybrid),
    // but alpha+beta = 0.0 (pure LR-HF, no SR-HF)
    // The override should set hyb to 0.0
    assert!((dfa.dfa_hybrid_scf - 0.0).abs() < 1e-8,
        "LC-BLYP hyb should be 0 (alpha+beta=0), got {}", dfa.dfa_hybrid_scf);

    let dfa2 = DFA4REST::new("cam-b3lyp", 1, 0);
    // CAM-B3LYP: raw xc_hyb_exx_coef = ?, alpha+beta = 0.19
    assert!((dfa2.dfa_hybrid_scf - 0.19).abs() < 1e-3,
        "CAM-B3LYP hyb should be ~0.19 (alpha+beta), got {}", dfa2.dfa_hybrid_scf);
}

#[test]
fn test_rsh_nonstd_parse() {
    // Test parse_scf_nonstd with RSH components (e.g. custom mixing)
    let dfa = DFA4REST::parse_scf_nonstd(
        &vec!["HYB_GGA_XC_WB97X_V".to_string()],
        &vec![0.5],  // 50% scaling
        &0.167,       // hyb = alpha+beta
        1,
    );
    assert!(dfa.is_rsh(), "Nonstd WB97X-V should be RSH");
    assert!((dfa.omega().unwrap() - 0.3).abs() < 1e-9);
    assert!((dfa.rsh_alpha().unwrap() - 1.0).abs() < 1e-9);
    assert!((dfa.dfa_hybrid_scf - 0.167).abs() < 1e-5);

    // Test with HSE06 nonstd
    let dfa2 = DFA4REST::parse_scf_nonstd(
        &vec!["HYB_GGA_XC_HSE06".to_string()],
        &vec![1.0],
        &0.25,
        1,
    );
    assert!(dfa2.is_rsh());
    assert!((dfa2.omega().unwrap() - 0.11).abs() < 1e-9);
    assert!((dfa2.rsh_alpha().unwrap() - 0.0).abs() < 1e-5);
}

#[test]
fn test_rsh_summary_does_not_panic() {
    // Ensure summary() works without panicking for RSH and non-RSH
    for name in &["cam-b3lyp", "lc-blyp", "hse06", "b3lyp", "pbe"] {
        let dfa = DFA4REST::new(name, 1, 0);
        dfa.summary(0);
    }
}

#[test]
fn test_all_rsh_via_auto_resolver() {
    // Test that the auto-resolver (generic libxc name lookup) works for RSH
    let dfa = DFA4REST::new("HYB_GGA_XC_WB97X_V", 1, 0);
    assert!(dfa.is_rsh());
    assert!((dfa.omega().unwrap() - 0.3).abs() < 1e-9);
    assert_eq!(dfa.dfa_compnt_scf, vec![466]);

    let dfa2 = DFA4REST::new("HYB_GGA_XC_HSE06", 1, 0);
    assert!(dfa2.is_rsh());
    assert!((dfa2.omega().unwrap() - 0.11).abs() < 1e-9);
    assert_eq!(dfa2.dfa_compnt_scf, vec![428]);
}
