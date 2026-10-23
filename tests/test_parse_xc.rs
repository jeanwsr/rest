use pyrest::dft::parse_xc::parse::{parse, parse_1step, parse_and_derive, merge_components, parse_pass3};
use pyrest::dft::parse_xc::ComponentType;


#[test]
fn test_parse_xc_param() {
    let input1 = "0.5*PBE + 0.5*B88, PBE(_beta=0.1)";
    let final_components = parse_1step(input1);
    assert_eq!(final_components.len(), 3);
    assert!((final_components[0].factor - 0.5).abs() <= 1e-9);
    assert_eq!(final_components[0].id, 101);
    assert_eq!(final_components[1].id, 106);
    assert_eq!(final_components[2].factor, 1.0);
    assert_eq!(final_components[2].id, 130);
    assert!((final_components[2].param_keyword.get("_beta").unwrap().as_f64().unwrap() - 0.1).abs() <= 1e-9);
}

#[test]
fn test_parse_xc_hybrid() {
    let input1 = ".2*HF + 0.08*LDA + 0.72*B88, 0.81*LYP + 0.19*VWN3";
    let final_components = parse_1step(input1);
    assert_eq!(final_components.len(), 5);
    assert_eq!(final_components[0].component_type, ComponentType::HF);
    assert_eq!(final_components[0].factor, 0.2);
    assert_eq!(final_components[1].id, 1);
    assert_eq!(final_components[2].id, 106);
    assert_eq!(final_components[3].id, 131);
    assert_eq!(final_components[4].id, 8);
    let input2 = "B3LYP";
    let final_dfa = parse(input2);
    assert!((final_dfa.get_hybrid_scf(1) - 0.2).abs() <= 1e-9);
    let input3 = "0.2*HF + 0.5*B3LYP";
    let final_dfa3 = parse(input3);
    assert!((final_dfa3.get_hybrid_scf(1) - 0.3).abs() <= 1e-9);
}

#[test]
fn test_is_hybrid() {
    let input1 = "B3LYP";
    let final_dfa = parse_and_derive(input1, 1, 0);
    assert!(final_dfa.is_hybrid());
    let input2 = "PBE";
    let final_dfa2 = parse_and_derive(input2, 1, 0);
    assert!(!final_dfa2.is_hybrid());
}

#[test]
fn test_is_fifth_dfa() {
    let input1 = "XYG3";
    let final_dfa = parse_and_derive(input1, 1, 0);
    assert!(final_dfa.is_fifth_dfa());
    // let input2 = "ZRPS";
    // let final_dfa2 = parse_and_init(input2, 1);
    // assert!(final_dfa2.is_fifth_dfa());
    let input3 = "PBE0";
    let final_dfa3 = parse_and_derive(input3, 1, 0);
    assert!(!final_dfa3.is_fifth_dfa());
}

#[test]
fn test_pass3_lincomb() {
    let input = "K1 + 0.5*K2 - 0.5*K3 + K4 - K5 +0.111*K6 -21*K_7";
    let param_captures = vec![];
    let (fac, funcs, _params) = parse_pass3(input, &param_captures);
    assert_eq!(fac, vec![1.0, 0.5, -0.5, 1.0, -1.0, 0.111, -21.0]);
    assert_eq!(funcs, vec!["K1", "K2", "K3", "K4", "K5", "K6", "K_7"]);
}

#[test]
fn test_parse_xc_merge() {
    let input1 = "BLYP + 0.1*X_B88";
    let final_components = parse_1step(input1);
    let merged = merge_components(final_components);
    assert_eq!(merged.len(), 2);
    assert!((merged[0].factor - 1.1).abs() <= 1e-9);
    assert_eq!(merged[0].id, 106);
    let input2 = "BLYP + 0.1*LYP";
    let final_components2 = parse_1step(input2);
    let merged2 = merge_components(final_components2);
    assert_eq!(merged2.len(), 2);
    assert!((merged2[1].factor - 1.1).abs() <= 1e-9);
}

#[test]
fn test_parse_xc_simple() {
    let input1 = "APBE,";
    let final_components = parse_1step(input1);
    assert_eq!(final_components.len(), 1);
    assert_eq!(final_components[0].id, 184);
    let input2 = "LDA0";
    let final_components2 = parse_1step(input2);
    assert_eq!(final_components2[0].id, 177);
    let input3 = "Xpbe,";
    let final_components3 = parse_1step(input3);
    assert_eq!(final_components3[0].id, 123);
    let input4 = "gga_x_pbe_gaussian";
    let final_components4 = parse_1step(input4);
    assert_eq!(final_components4[0].id, 321);
}

#[test]
fn test_parse_xc_dash() {
    let input1 = "M06-L";
    let final_components = parse_1step(input1);
    assert_eq!(final_components.len(), 2);
    assert_eq!(final_components[0].id, 203);
    assert_eq!(final_components[1].id, 233);
    let input2 = "m06-l,m06-2x";
    let final_components2 = parse_1step(input2);
    assert_eq!(final_components2.len(), 2);
}

// mean to fail
#[test]
#[should_panic]
fn test_parse_xc_fail() {
    let input1 = "B3LYP,";
    let _final_components = parse_1step(input1);
}

#[test]
fn test_parse_xc_vv10() {
    // wb97x-v parses into HYB_GGA_XC_WB97X_V (libxc 466) plus an explicit
    // VV10 component; sanity must pass and the (b, C) parameters are taken
    // from libxc and handed to the NLC evaluation.
    let dfa = parse_and_derive("wB97x-v", 1, 0);
    let (san, _) = dfa.check_sanity();
    assert!(san);
    let xc = dfa.to_dfa4rest();
    assert_eq!(xc.dfa_compnt_scf, vec![466]);
    assert_eq!(xc.nlc_vv10, Some((6.0, 0.01)));

    // The VV10 flag on the libxc functional alone must not activate NLC:
    // wb97x-d3bj keeps using D3 and must not carry a VV10 component.
    let dfa_d3 = parse_and_derive("wb97x-d3bj", 1, 0);
    let (san_d3, _) = dfa_d3.check_sanity();
    assert!(san_d3);
    assert_eq!(dfa_d3.to_dfa4rest().nlc_vv10, None);
}

#[test]
fn test_parse_xc_cam_b3lyp() {
    let input1 = "CAM-B3LYP";
    let dfa = parse_and_derive(input1, 1, 0);
    let (san,_) = dfa.check_sanity();
    assert!(san);
    let (omega, alpha, _beta) = dfa.get_rsh_scf(1);
    assert!((omega.unwrap() - 0.33).abs() <= 1e-7);
    assert!((alpha - 0.65).abs() <= 1e-7);
    assert!((dfa.dfa_hybrid_scf - 0.19).abs() <= 1e-7);
}

#[test]
fn test_parse_xc_rsh() {
    let input1 = "0.2*SR_HF(0.2) + 0.3*LR_HF(0.2) + 0.5*B3LYP";
    let dfa = parse_and_derive(input1, 1, 0);
    let (san,_) = dfa.check_sanity();
    assert!(san);
    let (omega, alpha, beta) = dfa.get_rsh_scf(1);
    assert!((omega.unwrap() - 0.2).abs() <= 1e-7);
    assert!((alpha - 0.4).abs() <= 1e-7);
    assert!(((alpha + beta) - 0.3).abs() <= 1e-7);
    assert!((dfa.dfa_hybrid_scf - 0.3).abs() <= 1e-7);
}

#[test]
fn test_parse_xc_equivalent_rsh() {
    let input1 = "RSH(0.33,0.65,-0.46) + 0.46*ITYH + 0.35*B88, 0.19*VWN5 + 0.81*LYP";
    let input2 = "CAM-B3LYP";
    let dfa1 = parse_and_derive(input1, 1, 0);
    let dfa2 = parse_and_derive(input2, 1, 0);
    assert!(dfa1.get_rsh_scf(1) == dfa2.get_rsh_scf(1));
    assert!((dfa1.dfa_hybrid_scf - dfa2.dfa_hybrid_scf).abs() <= 1e-7);
}

#[test]
#[should_panic]
fn test_parse_xc_rsh_fail() {
    let input1 = "0.2*RSH(0.1, 0.5, -0.1) + 0.5*CAM-B3LYP";
    let _dfa = parse_and_derive(input1, 1, 0);
}

use pyrest::dft::DFA4REST;
use pyrest::dft::DFAFamily;
use pyrest::dft::parse_xc::{DFAComponent, DFAdef, func_list_from_components};
use std::collections::HashMap;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1.0e-8
}

#[test]
fn rsh_lrsr_rshf_decomposition() {
    let hf_comp = |factor: f64, func: &str| DFAComponent {
        factor,
        func: func.to_string(),
        func_full_name: String::new(),
        id: 0,
        param_positional: vec![0.33],
        param_keyword: HashMap::new(),
        component_type: ComponentType::RSHF,
        reference: Vec::new(),
    };
    let mut dfa = pyrest::dft::parse_xc::DFAdef::new();
    dfa.spin_channel = 1;
    dfa.xc_scf = Some(vec![hf_comp(0.65, "SR_HF"), hf_comp(0.19, "LR_HF")]);
    let (omega, c_lr, c_sr) = dfa.get_rsh_scf_lrsr(1);
    assert_eq!(omega, Some(0.33));
    assert!(close(c_lr, 0.19), "c_LR = {c_lr}");
    assert!(close(c_sr, 0.65), "c_SR = {c_sr}");
    assert!(dfa.is_rsh());
}

fn assert_round_trip(original: &DFA4REST, name: &str) {
    let rt = pyrest::dft::parse_xc::DFAdef::from_dfa4rest(original).to_dfa4rest();
    assert_eq!(original.spin_channel, rt.spin_channel, "spin_channel for {name}");
    assert_eq!(original.dfa_compnt_scf, rt.dfa_compnt_scf, "compnt_scf for {name}");
    assert_eq!(original.dfa_paramr_scf, rt.dfa_paramr_scf, "paramr_scf for {name}");
    assert!(close(original.dfa_hybrid_scf, rt.dfa_hybrid_scf), "hybrid_scf for {name}");
    assert_eq!(original.dfa_rsh_scf, rt.dfa_rsh_scf, "rsh for {name}");
    assert_eq!(original.dfa_family_pos, rt.dfa_family_pos, "family_pos for {name}");
    assert_eq!(original.dfa_compnt_pos, rt.dfa_compnt_pos, "compnt_pos for {name}");
    assert_eq!(original.dfa_paramr_pos, rt.dfa_paramr_pos, "paramr_pos for {name}");
    match (original.dfa_hybrid_pos, rt.dfa_hybrid_pos) {
        (None, None) => {},
        (Some(a), Some(b)) => assert!(close(a, b), "hybrid_pos for {name}"),
        _ => panic!("hybrid_pos presence mismatch for {name}"),
    }
    match (&original.dfa_paramr_adv, &rt.dfa_paramr_adv) {
        (None, None) => {},
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len(), "paramr_adv len for {name}");
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(close(*x, *y), "paramr_adv for {name}");
            }
        },
        _ => panic!("paramr_adv presence mismatch for {name}"),
    }
}

#[test]
fn legacy_postscf_names_covered_by_parse_xc() {
    // (name, expected fifth-rung family) for every entry of the legacy
    // DFA4REST parse_postscf table (dft/mod.rs). parse_xc is the default
    // xc_parser and must accept all of them.
    let expected: Vec<(&str, DFAFamily)> = vec![
        ("b2gpplyp", DFAFamily::PT2),
        ("b2plyp", DFAFamily::PT2),
        ("mp2", DFAFamily::PT2),
        ("pbe0dh", DFAFamily::PT2),
        ("pbe-qidh", DFAFamily::PT2),
        ("scs-mp2", DFAFamily::PT2),
        ("xdh-pbe0", DFAFamily::PT2),
        ("xyg2", DFAFamily::PT2),
        ("xyg3", DFAFamily::PT2),
        ("xyg7", DFAFamily::PT2),
        ("xygjos", DFAFamily::PT2),
        ("zrps", DFAFamily::SBGE2),
        ("r-xyg2", DFAFamily::SBGE2),
        ("r-xyg3", DFAFamily::SBGE2),
        ("r-xyg7", DFAFamily::SBGE2),
        ("r-xygjos", DFAFamily::SBGE2),
        ("r-xdh7", DFAFamily::SCSRPA),
        ("scsrpa", DFAFamily::SCSRPA),
        ("rpa@b3lyp", DFAFamily::RPA),
        ("rpa@pbe", DFAFamily::RPA),
    ];
    for (name, family) in expected {
        let dfa = pyrest::dft::parse_xc::parse_and_derive(name, 1, 0);
        assert_eq!(dfa.family_pos(), Some(family), "family mismatch for {name}");
    }
    // the sBGE2-renormalized variants: (os, ss) factors transcribed from the
    // legacy table, exercising the SBGE2_OS/SBGE2_SS tokens
    let checks: Vec<(&str, [f64; 2], f64)> = vec![
        ("r-xyg2", [0.3387, 0.3387], 0.8016),
        ("r-xyg3", [0.3211, 0.3211], 0.8033),
        ("r-xyg7", [0.4052, 0.2589], 0.8971),
        ("r-xygjos", [0.4364, 0.0], 0.7731),
    ];
    for (name, os_ss, hybrid) in checks {
        let dfa = pyrest::dft::parse_xc::parse_and_derive(name, 1, 0);
        let comp = dfa
            .fifth_rung_component()
            .unwrap_or_else(|| panic!("{name} must carry a fifth-rung component"));
        assert_eq!(comp.param_positional.len(), 2, "os/ss len for {name}");
        assert!(close(comp.param_positional[0], os_ss[0]), "os for {name}");
        assert!(close(comp.param_positional[1], os_ss[1]), "ss for {name}");
        assert!(close(dfa.dfa_hybrid_nscf.unwrap(), hybrid), "hybrid for {name}");
    }
}

#[test]
fn rsh_parity_parse_xc_vs_legacy() {
    for name in ["cam-b3lyp", "lc-wpbe"] {
        let legacy = pyrest::dft::DFA4REST::new(name, 1, 0);
        let pxc = pyrest::dft::parse_xc::parse_and_derive(name, 1, 0).to_dfa4rest();
        assert_eq!(legacy.dfa_rsh_scf, pxc.dfa_rsh_scf, "rsh triple mismatch for {name}");
        assert!(
            close(legacy.dfa_hybrid_scf, pxc.dfa_hybrid_scf),
            "hybrid mismatch for {name}"
        );
    }
}

#[test]
fn from_dfa4rest_round_trip_legacy() {
    for name in ["hf", "b3lyp", "pbe0", "cam-b3lyp", "wb97x-v", "xyg3", "xygjos", "b2plyp", "scsrpa"] {
        let original = pyrest::dft::DFA4REST::new(name, 1, 0);
        assert_round_trip(&original, name);
    }
}

#[test]
fn from_dfa4rest_round_trip_deep_learning() {
    let original = pyrest::dft::DFA4REST::parse_scf_dldft(&"dl_dfa_scf".to_string(), 1);
    assert_round_trip(&original, "dl_dfa_scf");
}

#[test]
fn parse_standard_functionals() {
    let dfa = pyrest::dft::parse_xc::parse_and_derive("hf", 1, 0);
    assert!(close(dfa.dfa_hybrid_scf, 1.0));
    assert!(dfa.family_pos().is_none());
    assert!(!dfa.is_dfa_scf());
    assert!(!dfa.is_fifth_dfa());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("b3lyp", 1, 0);
    assert!(close(dfa.dfa_hybrid_scf, 0.2));
    assert!(dfa.is_dfa_scf());
    assert!(dfa.family_pos().is_none());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("pbe0", 1, 0);
    assert!(close(dfa.dfa_hybrid_scf, 0.25));
    assert!(dfa.is_dfa_scf());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("wb97x_v", 1, 0);
    assert!(dfa.is_dfa_scf());
    assert!(dfa.is_rsh());
}

#[test]
fn parse_fifth_rung() {
    let dfa = pyrest::dft::parse_xc::parse_and_derive("xyg3", 1, 0);
    assert_eq!(dfa.family_pos(), Some(DFAFamily::PT2));
    assert!(dfa.is_fifth_dfa());
    assert!(!dfa.is_rpa());
    assert!(dfa.is_dfa_scf());
    assert!(close(dfa.dfa_hybrid_scf, 0.2));
    assert!(close(dfa.dfa_hybrid_nscf.unwrap(), 0.8033));
    let pt2 = dfa.fifth_rung_component().expect("xyg3 must carry a PT2 component");
    assert_eq!(pt2.param_positional.len(), 2);
    assert!(close(pt2.param_positional[0], 0.3211));
    assert!(close(pt2.param_positional[1], 0.3211));

    let dfa = pyrest::dft::parse_xc::parse_and_derive("dsd-blyp-d3bj", 1, 0);
    assert_eq!(dfa.family_pos(), Some(DFAFamily::PT2));
    let pt2 = dfa.fifth_rung_component().expect("dsd-blyp-d3bj must carry a PT2 component");
    assert!(close(pt2.param_positional[0], 0.47));
    assert!(close(pt2.param_positional[1], 0.40));
    assert!(dfa.has_dispersion());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("rpa@b3lyp", 1, 0);
    assert_eq!(dfa.family_pos(), Some(DFAFamily::RPA));
    assert!(dfa.is_rpa());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("scsrpa", 1, 0);
    assert_eq!(dfa.family_pos(), Some(DFAFamily::SCSRPA));
    assert!(dfa.is_fifth_dfa());
    assert!(!dfa.is_rpa());
    assert!(dfa.fifth_rung_component().is_some());
}

#[test]
fn parse_dispersion() {
    let dfa = pyrest::dft::parse_xc::parse_and_derive("pbe0-d3bj", 1, 0);
    let disp = dfa.get_dispersion().expect("pbe0-d3bj must carry a Disp component");
    assert_eq!(disp.func.to_uppercase(), "DFTD3");
    assert!(!dfa.is_nlc());

    let dfa = pyrest::dft::parse_xc::parse_and_derive("wb97x-v", 1, 0);
    assert!(dfa.is_nlc());
    let (b, c) = dfa.vv10_params().expect("wb97x-v must carry VV10 parameters");
    assert!(b > 0.0 && c > 0.0);
}

#[test]
fn update_pt2_params_overrides() {
    let mut dfa = pyrest::dft::parse_xc::parse_and_derive("xyg3", 1, 0);
    dfa.update_pt2_params(Some(0.5), None);
    let pt2 = dfa.fifth_rung_component().unwrap();
    assert!(close(pt2.param_positional[0], 0.5));
    assert!(close(pt2.param_positional[1], 0.3211));

    let mut dfa = pyrest::dft::parse_xc::parse_and_derive("xyg3", 1, 0);
    dfa.update_pt2_params(None, None);
    let pt2 = dfa.fifth_rung_component().unwrap();
    assert!(close(pt2.param_positional[0], 0.3211));
    assert!(close(pt2.param_positional[1], 0.3211));

    let mut dfa = pyrest::dft::parse_xc::parse_and_derive("b3lyp", 1, 0);
    dfa.update_pt2_params(Some(0.5), Some(0.5));
    assert!(dfa.fifth_rung_component().is_none());
}

#[test]
fn to_xc_data_matches_to_dfa4rest_scf_side() {
    for name in ["hf", "b3lyp", "pbe0", "cam-b3lyp", "wb97x-v", "pbe0-d3bj", "xyg3"] {
        let dfa = pyrest::dft::parse_xc::parse_and_derive(name, 1, 0);
        let full = dfa.to_dfa4rest();
        let narrow = dfa.to_xc_data();
        assert_eq!(full.dfa_compnt_scf, narrow.dfa_compnt_scf, "compnt_scf mismatch for {name}");
        assert_eq!(full.dfa_paramr_scf, narrow.dfa_paramr_scf, "paramr_scf mismatch for {name}");
        assert!(close(full.dfa_hybrid_scf, narrow.dfa_hybrid_scf), "hybrid mismatch for {name}");
        assert_eq!(full.dfa_rsh_scf, narrow.dfa_rsh_scf, "rsh mismatch for {name}");
        assert_eq!(full.nlc_vv10, narrow.nlc_vv10, "nlc mismatch for {name}");
    }
}

#[test]
fn func_list_door_matches_legacy_arrays() {
    for name in ["xyg3", "dsd-blyp-d3bj", "rpa@b3lyp"] {
        let dfa = pyrest::dft::parse_xc::parse_and_derive(name, 1, 0);
        let full = dfa.to_dfa4rest();
        let legacy: Option<Vec<(f64, usize)>> =
            match (&full.dfa_compnt_pos, &full.dfa_paramr_pos) {
                (Some(code), Some(param)) if code.is_empty() => None,
                (Some(code), Some(param)) => Some(
                    code.iter()
                        .zip(param.iter())
                        .map(|(&c, &p)| (p, c))
                        .collect(),
                ),
                (None, None) => None,
                _ => panic!("pos arrays must be present or absent together"),
            };
        let door = pyrest::dft::parse_xc::func_list_from_components(
            dfa.xc_nscf.as_deref(),
            1,
        )
        .map(|l| {
            l.into_iter()
                .map(|(f, func)| (f, func.number() as usize))
                .collect::<Vec<_>>()
        });
        assert_eq!(legacy, door, "func list mismatch for {name}");
    }
}

#[test]
fn test_parse_xc_part_decomposition() {
    // pbe parses into separate exchange and correlation components
    let pbe = parse_and_derive("pbe", 1, 0);
    let x = pbe.x_part(1).expect("pbe has an exchange part");
    let x_funcs = x.scf_func_list(1);
    assert_eq!(x_funcs.len(), 1);
    assert_eq!(x_funcs[0].1.number(), 101); // gga_x_pbe
    let c = pbe.c_part(1).expect("pbe has a correlation part");
    let c_funcs = c.scf_func_list(1);
    assert_eq!(c_funcs.len(), 1);
    assert_eq!(c_funcs[0].1.number(), 130); // gga_c_pbe

    // blyp: b88 (106) is exchange, lyp (131) is correlation
    let blyp = parse_and_derive("blyp", 1, 0);
    assert_eq!(blyp.x_part(1).expect("blyp x").scf_func_list(1)[0].1.number(), 106);
    assert_eq!(blyp.c_part(1).expect("blyp c").scf_func_list(1)[0].1.number(), 131);

    // combined exchange-correlation functionals cannot be split natively
    let b3lyp = parse_and_derive("b3lyp", 1, 0);
    assert!(b3lyp.x_part(1).is_none(), "b3lypg is a combined functional");
    assert!(b3lyp.c_part(1).is_none());
}

#[test]
fn test_parse_xc_part_hf_kept_in_x() {
    // an HF reference: the exact-exchange term belongs to the x part
    let hf = parse_and_derive("hf", 1, 0);
    let x = hf.x_part(1).expect("hf has an exchange part");
    let comps = x.xc_scf.expect("hf x part carries components");
    assert!(comps.iter().all(|comp| !matches!(comp.component_type, ComponentType::Libxc)));
}
#[test]
fn test_parse_xc_part_of_name() {
    use pyrest::dft::parse_xc::xc_part_of_name;
    use libxc::prelude::LibXCFunctionalKind;

    // libxc-style pure names resolve via libxc itself (no REST-level alias needed)
    let x = xc_part_of_name("lda_x", 1, LibXCFunctionalKind::Exchange)
        .expect("lda_x is a pure exchange functional");
    assert_eq!(x.scf_func_list(1)[0].1.number(), 1);
    let c = xc_part_of_name("gga_c_pbe", 1, LibXCFunctionalKind::Correlation)
        .expect("gga_c_pbe is a pure correlation functional");
    assert_eq!(c.scf_func_list(1)[0].1.number(), 130);

    // REST-level names resolve through the parse
    let pbe_x = xc_part_of_name("pbe", 1, LibXCFunctionalKind::Exchange)
        .expect("pbe has an exchange part");
    assert_eq!(pbe_x.scf_func_list(1)[0].1.number(), 101);

    // combined exchange-correlation functionals have no separate parts
    assert!(xc_part_of_name("b3lyp", 1, LibXCFunctionalKind::Exchange).is_none());
    assert!(xc_part_of_name("b3lyp", 1, LibXCFunctionalKind::Correlation).is_none());
    // a wrong-kind request on a pure functional yields None
    assert!(xc_part_of_name("lda_x", 1, LibXCFunctionalKind::Correlation).is_none());
}
