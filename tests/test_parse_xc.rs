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
    let input1 = "wB97x-v";
    let dfa = parse_and_derive(input1, 1, 0);
    let (san,_) = dfa.check_sanity();
    assert!(!san);
    let input2 = "SCAN-VV10";
    let dfa2 = parse_and_derive(input2, 1, 0);
    let (san2,_) = dfa2.check_sanity();
    assert!(!san2);
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
