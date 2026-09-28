#[test]
fn style_keywords_round_trip() {
    use pyrest::ctrl_io::quasiparticle_methods::{parse_quasiparticle_keywords, style_is_ao};
    let src = r#"
[quasiparticle_methods]
bse_matvec_style = "ao"
gw_tensor_style = "ao"
"#;
    let keys = toml::from_str::<serde_json::Value>(src).unwrap();
    let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
    assert_eq!(qp.bse_matvec_style, "ao");
    assert_eq!(qp.gw_tensor_style, "ao");
    assert!(style_is_ao(&qp.bse_matvec_style) && style_is_ao(&qp.gw_tensor_style));

    let src = r#"
[quasiparticle_methods]
bse_matvec_style = "mo"
gw_tensor_style = "mo"
"#;
    let keys = toml::from_str::<serde_json::Value>(src).unwrap();
    let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
    assert_eq!(qp.bse_matvec_style, "mo");
    assert_eq!(qp.gw_tensor_style, "mo");
    assert!(!style_is_ao(&qp.bse_matvec_style));

    // historical spellings still map onto the new canonical values
    for (old, new) in [
        ("\"fast\"", "mo"),
        ("\"memory-efficient\"", "ao"),
        ("\"pair\"", "pair"),
        ("\"pairs\"", "pair"),
        ("\"list\"", "pair"),
        ("\"pair-list\"", "pair"),
        ("\"sparse\"", "pair"),
    ] {
        let src = format!("[quasiparticle_methods]\nbse_matvec_style = {}\n", old);
        let keys = toml::from_str::<serde_json::Value>(&src).unwrap();
        let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
        assert_eq!(qp.bse_matvec_style, new, "{} -> {}", old, new);
    }
    // the pair-list backend is neither "ao" nor "mo"
    {
        use pyrest::ctrl_io::quasiparticle_methods::style_is_pair;
        let keys =
            toml::from_str::<serde_json::Value>("[quasiparticle_methods]\nbse_matvec_style = \"pair\"\n")
                .unwrap();
        let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
        assert!(style_is_pair(&qp.bse_matvec_style));
        assert!(!style_is_ao(&qp.bse_matvec_style));
    }
    // the pair-list screening tolerance round-trips
    {
        let keys = toml::from_str::<serde_json::Value>(
            "[quasiparticle_methods]\nbse_pair_screening_tol = 3.0e-3\n",
        )
        .unwrap();
        let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
        assert_eq!(qp.bse_pair_screening_tol, 3.0e-3);
        assert_eq!(
            qp.to_toml().get("bse_pair_screening_tol").unwrap().as_float().unwrap(),
            3.0e-3
        );
    }
    // default: the BSE matvec defaults to AO, GW stays on MO (commit 4f32667)
    let keys = toml::from_str::<serde_json::Value>("[quasiparticle_methods]\n").unwrap();
    let qp = parse_quasiparticle_keywords(&keys).unwrap().unwrap();
    assert_eq!(qp.bse_matvec_style, "ao");
    assert_eq!(qp.gw_tensor_style, "mo");
    // to_toml exports the new keys
    let t = qp.to_toml();
    assert_eq!(t.get("bse_matvec_style").unwrap().as_str().unwrap(), "ao");
    assert_eq!(t.get("gw_tensor_style").unwrap().as_str().unwrap(), "mo");
    assert!(t.get("gw_ao_screening_tol").is_some());
    println!("style keyword round-trip OK");
}
