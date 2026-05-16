use lazy_static::lazy_static;
use std::collections::HashMap;
use serde_json;
use crate::dft::DFAFamily;

#[derive(Clone,PartialEq,Debug)]
pub enum ComponentType {
    HF,
    PT2,
    RPA,
    SCSRPA,
    SBGE2,
    Disp,
    Libxc,
    Unknown,
}

pub const NSCF_COMPONENTS: [ComponentType; 5] = [
    ComponentType::PT2, 
    ComponentType::RPA, 
    ComponentType::SCSRPA, 
    ComponentType::SBGE2,
    ComponentType::Disp,
];

impl ComponentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComponentType::HF => "HF",
            ComponentType::PT2 => "PT2",
            ComponentType::RPA => "RPA",
            ComponentType::SCSRPA => "SCSRPA",
            ComponentType::SBGE2 => "SBGE2",
            ComponentType::Disp => "Disp",
            ComponentType::Libxc => "Libxc",
            ComponentType::Unknown => "Unknown",
        }
    }

    pub fn to_dfa_family(&self) -> Option<DFAFamily> {
        match self {
            ComponentType::HF => None,
            ComponentType::PT2 => Some(DFAFamily::PT2),
            ComponentType::RPA => Some(DFAFamily::RPA),
            ComponentType::SCSRPA => Some(DFAFamily::SCSRPA),
            ComponentType::SBGE2 => Some(DFAFamily::SBGE2),
            ComponentType::Disp => None,
            ComponentType::Libxc => None,
            ComponentType::Unknown => None,
        }
    }
}

lazy_static! {
    pub static ref AVAIL_FUNC : HashMap<String, usize> = get_available_functionals();

    pub static ref CODES: HashMap<&'static str, usize> = HashMap::from([
        // LDA
        ("LDA", 1),
        ("SLATER", 1),
        ("VWN3", 8),
        ("VWNRPA", 8),
        ("VWN5", 7),
        // HYB_GGA
        ("PBE0", 406),
        ("B3LYPG", 402),
        ("X3LYPG", 411),
    ]);

    pub static ref ALIAS: HashMap<&'static str, &'static str> = HashMap::from([
        // LDA
        ("SVWN", "SLATER,VWN"),
        // GGA
        ("BLYP", "B88,LYP"),
        ("BP86", "B88,P86"),
        ("PW91", "PW91,PW91"),
        ("PBE", "PBE,PBE"),
        ("REVPBE", "PBE_R,PBE"),
        ("PBESOL", "PBE_SOL,PBE_SOL"),
        ("SOGGA", "SOGGA,PBE"),
        ("OLYP", "OPTX,LYP"),
        ("OPBE", "OPTX,PBE"),
        ("RPBE", "RPBE,PBE"),
        ("BPBE", "B88,PBE"),
        ("XPBE", "XPBE,XPBE"),
        ("MPW91", "MPW91,PW91"),
        ("SOGGA11", "SOGGA11,SOGGA11"),
        ("KT1", "KT1,VWN"),
        ("KT2", "GGA_XC_KT2"),
        ("KT3", "GGA_XC_KT3"),
        ("GAM", "GAM,GAM"),
        ("N12", "N12,N12"),
        ("PBEOP", "PBE,OP_PBE"),
        ("BOP", "B88,OP_B88"),
        // MGGA
        ("PKZB", "PKZB,PKZB"),
        ("TPSS", "TPSS,TPSS"),
        ("REVTPSS", "REVTPSS,REVTPSS"),
        ("SCAN", "SCAN,SCAN"),
        ("RSCAN", "RSCAN,RSCAN"),
        ("R2SCAN", "R2SCAN,R2SCAN"),
        ("SCANL", "SCANL,SCANL"),
        ("R2SCANL", "R2SCANL,R2SCANL"),
        ("BLOC", "BLOC,TPSSLOC"),
        ("MS0", "MS0,REGTPSS"),
        ("MS1", "MS1,REGTPSS"),
        ("MS2", "MS2,REGTPSS"),
        ("MS2H", "MS2H,REGTPSS"),
        ("MVS", "MVS,REGTPSS"),
        ("MVSH", "MVSH,REGTPSS"),
        ("M06_L", "M06_L,M06_L"),
        ("M11_L", "M11_L,M11_L"),
        ("MN12_L", "MN12_L,MN12_L"),
        ("MN15_L", "MN15_L,MN15_L"),
        ("MBEEF", "MBEEF,PBE_SOL"),
        ("REVSCAN", "REVSCAN,REVSCAN"),
        // ("REVSCAN_VV10", "REVSCAN,REVSCAN_VV10"),
        // ("SCAN_VV10", "SCAN,SCAN_VV10"),
        // ("SCAN_RVV10", "SCAN,SCAN_RVV10"),
        ("R2SCAN", "R2SCAN,R2SCAN"),
        // HYB_LDA
        ("HFPW92", "HF,PW_MOD"),
        ("SPW92", "SLATER,PW_MOD"),
        // HYB_GGA
        ("B3LYP5", ".2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN"),
        ("HFLYP", "HF,LYP"),
        ("SOGGA11_X", "SOGGA11_X,SOGGA11_X"),
        ("N12_SX", "N12_SX,N12_SX"),
        // HYB_MGGA
        ("DLDF", "DLDF,DLDF"),
        ("M06_SX", "M06_SX,M06_SX"),
        ("MN12_SX", "MN12_SX,MN12_SX"),
        ("MN15", "MN15,MN15"),
        ("SCAN0", "SCAN0,SCAN"),
        ("M05", "M05,M05"),
        ("M06", "M06,M06"),
        ("M05_2X", "M05_2X,M05_2X"),
        ("M06_2X", "M06_2X,M06_2X"),
        ("CF22D", "CF22D,CF22D"),
        // CAM_, LC_, _X, _L, _SX, _2X
        ("CAMB3LYP", "CAM_B3LYP"),
        ("LCBLYP", "LC_BLYP"),
        ("LCWPBE", "LC_WPBE"),
        ("SOGGA11X", "SOGGA11_X"),
        ("M06L", "M06_L"),
        ("M11L", "M11_L"),
        ("MN12L", "MN12_L"),
        ("MN15L", "MN15_L"),
        ("N12SX", "N12_SX"),
        ("M06SX", "M06_SX"),
        ("MN12SX", "MN12_SX"),
        ("M052X", "M05_2X"),
        ("M062X", "M06_2X"),
        ("M06HF", "M06_HF"),
    ]);

    pub static ref NAME_WITH_DASH:HashMap<&'static str, &'static str> = HashMap::from([
        ("SR-HF", "SR_HF"),
        ("LR-HF", "LR_HF"),
        ("B97-1", "B97_1"),
        ("B97-2", "B97_2"),
        ("B97-3", "B97_3"),
        ("B97-K", "B97_K"),
        ("M05-2X", "M05_2X"),
        ("M06-2X", "M06_2X"),
        ("M06-HF", "M06_HF"),
        ("M06-SX", "M06_SX"),
        ("M06-L", "M06_L"),
        ("M08-HX", "M08_HX"),
        ("M08-SO", "M08_SO"),
        ("M11-L", "M11_L"),
        ("MN12-L", "MN12_L"),
        ("MN15-L", "MN15_L"),
        ("N12-SX", "N12_SX"),
        ("MN12-SX", "MN12_SX"),
        ("CAM-B3LYP", "CAM_B3LYP"),
        ("LC-BLYP", "LC_BLYP"),
        ("LC-WPBE", "LC_WPBE"),
        ("SOGGA11-X", "SOGGA11_X"),
    ]);

    pub static ref NAME_WITHOUT_UNDERSCORE:HashMap<&'static str, &'static str> = HashMap::from([
        ("M052X", "M05_2X"),
        ("M062X", "M06_2X"),
        ("M06HF", "M06_HF"),
        ("M06SX", "M06_SX"),
        ("M06L", "M06_L"),
        ("M08HX", "M08_HX"),
        ("M08SO", "M08_SO"),
        ("M11L", "M11_L"),
        ("MN12L", "MN12_L"),
        ("MN15L", "MN15_L"),
        ("N12SX", "N12_SX"),
        ("MN12SX", "MN12_SX"),
        ("SOGGA11X", "SOGGA11_X"),
    ]);

    pub static ref WHITELIST_NONLIBXC:HashMap<&'static str, ComponentType> = HashMap::from([
        ("HF", ComponentType::HF),
        // todo SR_HF
        ("MP2", ComponentType::PT2),
        ("MP2_OS", ComponentType::PT2),
        ("MP2_SS", ComponentType::PT2),
        ("RPA", ComponentType::RPA),
        ("SCSRPA", ComponentType::SCSRPA),
        ("SBGE2", ComponentType::SBGE2),
        ("DFTD3", ComponentType::Disp),
        ("D3", ComponentType::Disp),
        ("DFTD4", ComponentType::Disp),
        ("D4", ComponentType::Disp),
        ("VV10", ComponentType::Disp),
    ]);

    pub static ref MULTISTEP:HashMap<String, XC2step> = load_json_functionals();

    pub static ref MULTISTEP_NAME_WITH_DASH:HashMap<String, &'static str> = get_name_with_dash();
    pub static ref MULTISTEP_ALIAS:HashMap<&'static str, &'static str> = HashMap::from([
        // lowercase
        ("xygjos", "xygj_os"),
        ("revxygjos", "revxygj_os"),
        ("xygjos5", "xygj_os5"),
        ("xdhpbe0", "xdh_pbe0"),
        ("scsmp2", "scs_mp2"),
        ("b2gpplyp", "b2gp_plyp"),
        ("pbe0dh", "pbe0_dh"),
        ("pbeqidh", "pbe_qidh"),
        ("dsdpbep86", "dsd_pbep86"),
        ("dsdpbep86-d3bj", "dsd_pbep86_d3bj"),
        ("dsdpbeb95-d3bj", "dsd_pbeb95_d3bj"),
        ("dsdpbepbe-d3bj", "dsd_pbepbe_d3bj"),
        ("dsdblyp-d3bj", "dsd_blyp_d3bj"),
    ]);
}

pub fn get_name_with_dash() -> HashMap<String, &'static str> {
    // for each key in MULTISTEP, if it contains "_", add an entry with "_" replaced by "-"
    let mut map = HashMap::new();
    for key in MULTISTEP.keys() {
        if key.contains("_") {
            let new_key = key.replace("_", "-");
            map.insert(new_key, key.as_str());
        }
    }
    map
}

pub fn get_name(id: usize) -> String {
    let name = libxc::util::libxc_functional_get_name(id as i32)
        .unwrap_or_default();
    name.to_uppercase()
}

pub fn get_available_functionals() -> HashMap<String, usize> {
    let ids = libxc::util::libxc_available_functional_numbers();
    let mut available_functionals = HashMap::new();
    for id in ids {
        let name = libxc::util::libxc_functional_get_name(id)
            .unwrap_or_default();
        available_functionals.insert(name.to_uppercase(), id as usize);
    }
    available_functionals
}

#[derive(Clone,PartialEq,Debug)]
pub struct XC2step {
    pub name: String,
    pub code_scf: String,
    pub code: String,
    pub reference: Vec<String>,
}

pub fn load_json_functionals() -> HashMap<String, XC2step> {
    let jsonfile = include_str!("./family_xdh.json");
    let mut map1 = load_json(jsonfile);
    let jsonfile2 = include_str!("./family_xdh_add.json");
    map1.extend(load_json(jsonfile2));
    let jsonfile3 = include_str!("./family_bdh.json");
    map1.extend(load_json(jsonfile3));
    map1
}

pub fn load_user_json_functionals() -> Option<HashMap<String, XC2step>> {
    if let Ok(data_path) = std::env::var("REST_DATA_DIR") {
        println!("Detected REST_DATA_DIR environment variable: {}", data_path);
        // read all json files in data_path, then load them and construct HashMap<String, XC2step>
        let mut map = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&data_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") 
                    && path.file_name().and_then(|s| s.to_str()).map(|s| s.starts_with("xc")).unwrap_or(false) {
                        if let Ok(json_str) = std::fs::read_to_string(&path) {
                            let json_map = load_json(&json_str);
                            map.extend(json_map);
                            println!("Loaded JSON file: {:?}", path);
                        } else {
                            println!("Failed to read JSON file: {:?}", path);
                        }
                    }
                }
            }
        } else {
            println!("Failed to read directory: {}", data_path);
        }
        return Some(map);
    }
    None
}

fn load_json(json_str: &str) -> HashMap<String, XC2step> {
    // let json_str = std::fs::read_to_string(jsonfile).expect("Failed to read JSON file");
    // load data from json_str, then construct HashMap<String, XC2step>
    let v: serde_json::Value = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    let mut map = HashMap::new();
    if let serde_json::Value::Object(obj) = v {
        for (key, value) in obj {
            if let serde_json::Value::Object(inner_obj) = value {
                let name = key.to_string();
                let code_scf = inner_obj.get("code_scf").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let code = inner_obj.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let refs = inner_obj.get("ref"); // could be array or string 
                let reference = if let Some(serde_json::Value::Array(ref_array)) = refs {
                    ref_array.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect()
                } else {
                    refs.and_then(|v| v.as_str()).map(|s| vec![s.to_string()]).unwrap_or(Vec::new())
                };
                map.insert(key.to_lowercase(), XC2step { name, code_scf, code, reference });
            }
        }
    }
    map
}
