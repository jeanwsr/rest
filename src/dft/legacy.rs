//! The legacy XC definition (`DFA4REST`): the code-array representation of an XC
//! functional plus the legacy parser bundle. On the parse_xc route this struct only
//! exists to be converted via `DFAdef::from_dfa4rest`; everything computational left
//! for the free functions in `dft` (funcs currency). Dies with the legacy parser.
use libxc::prelude::*;
use crate::dft::DFAFamily;
use crate::dft::libxc_helper::{xc_code_fdqc, xc_func_init};

#[derive(Clone)]
pub struct DFA4REST {
    pub spin_channel: usize,
    pub dfa_compnt_scf: Vec<usize>,
    pub dfa_paramr_scf: Vec<f64>,
    pub dfa_hybrid_scf: f64,
    // (omega, alpha, beta) in libxc convention; note beta is usually not used in computation.
    pub dfa_rsh_scf: Option<(f64, f64, f64)>,
    pub dfa_family_pos: Option<DFAFamily>,
    pub dfa_compnt_pos: Option<Vec<usize>>,
    pub dfa_paramr_pos: Option<Vec<f64>>,
    pub dfa_hybrid_pos: Option<f64>,
    pub dfa_paramr_adv: Option<Vec<f64>>,
    pub nlc_vv10: Option<(f64, f64)>,
}
impl DFA4REST {
    pub fn new_xc(spin_channel: usize, print_level: usize) -> DFA4REST {
        DFA4REST { 
            spin_channel, 
            dfa_compnt_scf: vec![], 
            dfa_paramr_scf: vec![], 
            dfa_hybrid_scf: 0.0,
            dfa_rsh_scf: None,
            dfa_family_pos: None, 
            dfa_compnt_pos: None, 
            dfa_paramr_pos: None, 
            dfa_hybrid_pos: None, 
            dfa_paramr_adv: None,
            nlc_vv10: None,
        }
    }
    
    pub fn summary(&self, print_level: usize) {
        if print_level > 0 {
            println!("==== DFA Summary (legacy) ====");
            println!("Spin channel: {}", self.spin_channel);
            println!("SCF DFA components: {:?}", self.dfa_compnt_scf);
            println!("SCF DFA parameters: {:?}", self.dfa_paramr_scf);
            println!("SCF DFA hybrid coeff: {:16.8}", self.dfa_hybrid_scf);
            if self.is_rsh() {
                println!("Range-Separated Hybrid parameters:");
                println!("  omega: {:16.8}", self.omega().unwrap());
                println!("  alpha (LR-HF coeff): {:16.8}", self.rsh_alpha().unwrap());
            }
            if let Some(dfatype) = &self.dfa_family_pos {
                println!("Post-SCF DFA family: {}", dfatype.to_name());
                if let Some(dfacomp) = &self.dfa_compnt_pos {
                    println!("Post-SCF DFA components: {:?}", dfacomp);
                }
                if let Some(dfaparam) = &self.dfa_paramr_pos {
                    println!("Post-SCF DFA parameters: {:?}", dfaparam);
                }
                if let Some(dfahybrid) = &self.dfa_hybrid_pos {
                    println!("Post-SCF DFA hybrid coeff: {:16.8}", dfahybrid);
                }
            } else {
                println!("Post-SCF DFA: None");
            }
            if let Some(dfaparam_adv) = &self.dfa_paramr_adv {
                println!("Advanced DFA parameters: {:?}", dfaparam_adv);
            }
        }
        if print_level > 1 {
            self.describe();
        }
        println!("==== End of Summary ====");
    }

    pub fn describe(&self) {
        println!("==== detailed info of the scf functional ====");
        &self.dfa_compnt_scf.iter().for_each(|xc_func| {
            println!("{}", self.init_libxc(xc_func).describe())
        });
        if let (Some(dfatype),Some(dfacomp)) = 
            (&self.dfa_family_pos, &self.dfa_compnt_pos) {
            //match dfatype {
            //    DFAFamily::PT2 => println!("XYG3-type functional '{}' is employed", &name),
            //    DFAFamily::RPA => println!("RPA-type functional '{}' is employed", &name),
            //    _ => println!("Standard DFA '{}' is employed", &name),
            //}
            println!("==== detailed info of the post-scf functional ====");
            dfacomp.into_iter().for_each(|xc_func| {
                println!("{}", self.init_libxc(xc_func).describe())
            })
        };
    }

    pub fn new_nonstandard(
        spin_channel: usize, 
        print_level:  usize, 
        xc_namelist:  &Option<Vec<String>>, 
        xc_paramlist: &Option<Vec<f64>>, 
        dfa_hybrid_scf: &Option<f64>,
    ) -> DFA4REST {
        let mut dfa = if let (Some(codelist), Some(paramlist), Some(dfa_hybrid_scf)) = (&xc_namelist, &xc_paramlist, &dfa_hybrid_scf) {
            DFA4REST::parse_scf_nonstd(codelist, paramlist, dfa_hybrid_scf, spin_channel)
        } else {
            panic!("xc_namelist, xc_paramlist and dfa_hybrid_scf should be provided for a non-standard setting of DFA")
        };
        dfa
    }

    pub fn new_deep_learning(spin_channel: usize, print_level: usize, xc_model:&Option<String>) -> DFA4REST {
        let mut dfa = if let Some(xc_model) = xc_model {
            DFA4REST::parse_scf_dldft(xc_model, spin_channel)
        } else {
            panic!("xc_model should be provided for deep-learning DFA model")
        };
        dfa
    }

    pub fn update_pt2_params(&mut self, os_factor:Option<f64>, ss_factor:Option<f64>) {
        match self.dfa_family_pos {
            Some(DFAFamily::PT2) => {
                let mut params_adv = self.dfa_paramr_adv.take().unwrap_or(vec![1.0, 1.0]);
                if let Some(osf) = os_factor {
                    params_adv[0] = osf;
                }
                if let Some(ssf) = ss_factor {
                    params_adv[1] = ssf;
                }
                self.dfa_paramr_adv = Some(params_adv);
            },
            _ => { }
        }
    }

    pub fn parse_scf_dldft(xc_model:&String, spin_channel: usize) -> DFA4REST {

        let mut codelist: Vec<&str> = vec![];
        let mut paralist: Vec<f64> = vec![];
        let mut dfa_hybrid_scf: f64 = 0.0;
        if xc_model.eq("dl_dfa_scf") {
            // 毕升的机器学习杂化泛函，fake成b3lyp，但是初始为BLYP
            codelist = vec!["lda_x_slater", "gga_x_b88", "lda_c_vwn_rpa", "gga_c_lyp"];
            paralist = vec![0.00, 1.00, 0.00, 1.00];
            dfa_hybrid_scf = 0.00001;
        };

        // Parse the xc functionals
        let dfa_compnt_scf = codelist.iter().map(|xc| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
        }).flatten().collect::<Vec<usize>>();
        // Parse the xc parameters
        let dfa_paramr_scf = codelist.iter().zip(paralist.iter()).map(|(xc, param)| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *param).collect::<Vec<f64>>()
        }).flatten().collect::<Vec<f64>>();
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

        DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf,
            dfa_rsh_scf,
            nlc_vv10: None,
        }
    }

    pub fn new(name: &str, spin_channel: usize, print_level: usize) -> DFA4REST {
        
        let tmp_name = name.to_lowercase();
        let post_dfa = DFA4REST::parse_postscf(&tmp_name, spin_channel);
        match post_dfa {
            Some(dfa) => {
                dfa
            },
            None => {
                let dfa = DFA4REST::parse_scf(&tmp_name, spin_channel);
                dfa
            },
        }
    }

    pub fn libxc_code_fdqc(name: &str) -> [usize;3] {
        xc_code_fdqc(name)
    }

    pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> Vec<usize> {
        let xc_code = DFA4REST::libxc_code_fdqc(name);
        xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
    }

    pub fn init_libxc(&self, xc_code: &usize) -> LibXCFunctional {
        xc_func_init(*xc_code, self.spin_channel)
    }

    pub fn init_libxc_and_set_param(&self, xc_code: &usize) -> LibXCFunctional {
        let mut func = self.init_libxc(xc_code);
        if let Some(omega) = self.omega() {
            if func.ext_param_names().contains(&"_omega".to_string()) {
                func.set_ext_param_by_name("_omega", omega);
            }
        }
        func
    }

    pub fn get_hybrid_libxc(dfa_compnt_scf: &Vec<usize>,spin_channel:usize) -> f64 {
        let mut hybrid_coeff = None;
        for xc_func in dfa_compnt_scf {
            let func = xc_func_init(*xc_func, spin_channel);
            let hyb_exx_coeff = if let Some((_omega, alpha, beta)) = func.cam_coef() {
                alpha + beta // for RSH, return alpha + beta (matches pyscf)
            } else {
                // for non-RSH, return hybrid coefficient; if not available, return 0.0
                func.hyb_exx_coef().unwrap_or(0.0)
            };
            if hyb_exx_coeff.abs() > 1e-10 {
                if hybrid_coeff.is_some() {
                    panic!("Multiple hybrid functionals are specified in the DFA components for SCF. Currently this is not supported.");
                }
                hybrid_coeff = Some(hyb_exx_coeff);
            }
        }
        hybrid_coeff.unwrap_or(0.0)
    }

    pub fn get_rsh_libxc(dfa_compnt_scf: &Vec<usize>, spin_channel: usize) -> Option<(f64, f64, f64)> {
        let mut result = None;
        for xc_func in dfa_compnt_scf {
            let func = xc_func_init(*xc_func, spin_channel);
            if func.is_hyb_cam() {
                let (omega, alpha, beta) = func.cam_coef().unwrap_or((0.0, 0.0, 0.0));
                if alpha.abs() < 1e-10 && beta.abs() < 1e-10 {
                    continue;
                }
                if result.is_some() {
                    panic!("Multiple RSH functionals are specified in the DFA components for SCF. Currently this is not supported.");
                }
                result = Some((omega, alpha, alpha + beta));
            }
        }
        result
    }

    pub fn is_rsh(&self) -> bool {
        self.omega().is_some()
    }

    pub fn omega(&self) -> Option<f64> {
        self.rsh_params().map(|(omega, _, _)| omega)
    }

    pub fn rsh_alpha(&self) -> Option<f64> {
        self.rsh_params().map(|(_, alpha, _)| alpha)
    }

    pub fn rsh_params(&self) -> Option<(f64, f64, f64)> {
        self.dfa_rsh_scf
    }

    pub fn parse_scf(name: &str, spin_channel: usize) -> DFA4REST {
        let tmp_name = name.to_lowercase();
        let legacy_nlc_ids: Option<Vec<usize>> = match tmp_name.as_str() {
            "wb97x-v" => Some(vec![466]),
            "wb97m-v" => Some(vec![531]),
            _ => None,
        };
        let dfa_compnt_scf = match &legacy_nlc_ids {
            Some(ids) => ids.clone(),
            None => DFA4REST::xc_func_init_fdqc(&tmp_name, spin_channel),
        };
        let mut dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
        // The HF reference carries the full exact exchange:
        // dfa_hybrid_scf = 1.0 so that every response consumer (TDDFT/
        // stability matvecs, gradients, hessians, energy components) sees the
        // correct exchange scaling. Keyed on the requested name ("hf"), not on
        // the component list. The SCF's own Fock build dispatches on the empty
        // component list into the dedicated `generate_hf_hamiltonian_*` paths,
        // which apply the exchange at unit strength directly and do not read
        // this field.
        if tmp_name == "hf" {
            dfa_hybrid_scf = 1.0;
        }
        let dfa_paramr_scf =  vec![1.0;dfa_compnt_scf.len()];
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

        let mut dfa = DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf,
            dfa_rsh_scf,
            nlc_vv10: None,
        };
        if legacy_nlc_ids.is_some() {
            dfa.nlc_vv10 = dfa
                .dfa_compnt_scf
                .iter()
                .map(|id| dfa.init_libxc(id))
                .find_map(|func| func.vv10_coef());
        }
        dfa
    }

    pub fn parse_scf_nonstd(codelist: &Vec<String>, paramlist: &Vec<f64>, dfa_hybrid_scf: &f64, spin_channel: usize) -> DFA4REST {
        if codelist.len()!=paramlist.len() {panic!("codelist (len: {}) does not match paramlist (len: {})", codelist.len(), paramlist.len())}
        // Parse the xc functionals
        let dfa_compnt_scf = codelist.iter().map(|xc| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *x).collect::<Vec<usize>>()
        }).flatten().collect::<Vec<usize>>();
        // Parse the xc parameters
        let dfa_paramr_scf = codelist.iter().zip(paramlist.iter()).map(|(xc, param)| {
            let xc_code = DFA4REST::libxc_code_fdqc(xc);
            xc_code.iter().filter(|x| **x!=0).map(|x| *param).collect::<Vec<f64>>()
        }).flatten().collect::<Vec<f64>>();
        let dfa_rsh_scf = DFA4REST::get_rsh_libxc(&dfa_compnt_scf, spin_channel);

        println!("==== IGOR debug for nonstd DFT parse ====");
        println!("codelist: {:?}, xc_hybrid: {:16.8}", codelist, dfa_hybrid_scf);
        println!("dfa_compnt_scf: {:?}", &dfa_compnt_scf);
        println!("dfa_paramr_scf: {:?}", &dfa_paramr_scf);
        println!("==== IGOR debug for nonstd DFT parse ====");
        DFA4REST {
            spin_channel,
            dfa_family_pos: None,
            dfa_compnt_pos: None,
            dfa_paramr_pos: None,
            dfa_hybrid_pos: None,
            dfa_paramr_adv: None,
            dfa_compnt_scf,
            dfa_paramr_scf,
            dfa_hybrid_scf: *dfa_hybrid_scf,
            dfa_rsh_scf,
            nlc_vv10: None,
        }
    }

    pub fn parse_postscf(name: &str, spin_channel: usize) -> Option<DFA4REST> {
        let tmp_name = name.to_lowercase();
        if tmp_name.starts_with("dsd") {
            panic!(
                "DSD-type functional '{}' is no longer supported by the legacy xc parser: \
                the DSD* definitions in legacy parser are confusing. Please set xc_parser = \"parse_xc\" \
                in the ctrl input instead, e.g.\n \
                xc_parser = \"parse_xc\"\n \
                xc = \"dsdpbep86-d3bj\"\n \
                or \n \
                xc_parser = \"parse_xc\"\n \
                xc = \"dsdpbep86\"\n \
                (no dispersion version)",
                name
            );
        }
        if tmp_name.eq("xyg3") {
            // XYG3 functional
            // Proc. Natl. Acad. Sci. U.S.A. 106, 13, 4963-4968 (2009); https://pnas.org/doi/full/10.1073/pnas.0901093106
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![-0.0140,0.2107,0.00,0.6789]);
            let dfa_hybrid_pos = Some(0.8033);
            let dfa_paramr_adv = Some(vec![0.3211,0.3211]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xygjos") {
            // XYGJ-OS functional
            // Proc. Natl. Acad. Sci. U.S.A. 108, 50, 19896-19900 (2011); https://pnas.org/doi/full/10.1073/pnas.1115123108
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2269,0.000,0.2309,0.2754]);
            let dfa_hybrid_pos = Some(0.7731);
            let dfa_paramr_adv = Some(vec![0.4364,0.0000]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xyg7") {
            // XYG7 functional 
            // J. Phys. Chem. Lett. 12, 10, 2638-2644 (2021); https://doi.org/10.1021/acs.jpclett.1c00360
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2055,-0.1408,0.4056,0.1159]);
            let dfa_hybrid_pos = Some(0.8971);
            let dfa_paramr_adv = Some(vec![0.4052,0.2589]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xyg2") {
            // XYG2 functional
            // Yan, W., PhD thesis, Fudan University, Shanghai, China (2022).
            // Precision Chemistry (2026); https://doi.org/10.1021/prechem.5c00432
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![0.00,0.1984,0.00,0.6613]);
            let dfa_hybrid_pos = Some(0.8016);
            let dfa_paramr_adv = Some(vec![0.3387,0.3387]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("xdh-pbe0") {
            // xDH-PBE0 functional
            // J. Chem. Phys. 136, 174103 (2012); https://doi.org/10.1063/1.3703893
            let dfa_family_pos = Some(DFAFamily::PT2);
            let pos_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.1665, 0.5292]);
            let dfa_hybrid_pos = Some(0.8335);
            let dfa_paramr_adv = Some(vec![0.5428, 0.0000]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["pbe0"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0; dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("zrps") {
            // ZRPS
            // Phys. Rev. Lett. 117, 133002 (2016); https://doi.org/10.1103/PhysRevLett.117.133002
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["gga_x_pbe","gga_c_pbe"];
            let scf_dfa = ["pbe0"];
            let dfa_paramr_pos = Some(vec![0.5,0.75]);
            let dfa_hybrid_pos = Some(0.5);
            let dfa_paramr_adv = Some(vec![0.25,0.00]);

            // Now initialize ZRPS
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_family_scf = DFAFamily::HybridGGA;
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("rpa@b3lyp") {
            let dfa_family_pos = Some(DFAFamily::RPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("rpa@pbe") {
            let dfa_family_pos = Some(DFAFamily::RPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0]);

            let dfa_family_scf = DFAFamily::GGA;
            let scf_dfa = ["pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 0.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("scsrpa") {
            // scsRPA
            // J. Phys. Chem. Lett. 10, 10, 2617-2623 (2019); https://doi.org/10.1021/acs.jpclett.9b00946
            let dfa_family_pos = Some(DFAFamily::SCSRPA);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.2,0.75]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["pbe0"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xdh7") {
            // R-xDH7
            // JACS Au 4, 8, 3205-3216 (2024); https://doi.org/10.1021/jacsau.4c00488
            let dfa_family_pos = Some(DFAFamily::SCSRPA);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.3600,-0.2917,0.4937,-0.4301]);
            let dfa_hybrid_pos = Some(0.9081);
            let dfa_paramr_adv = Some(vec![0.8624,0.2359]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("mp2") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.0,1.0]);

            let scf_dfa = [];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 1.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("scs-mp2") {
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(vec![]);
            let dfa_paramr_pos = Some(vec![]);
            let dfa_hybrid_pos = Some(1.0);
            let dfa_paramr_adv = Some(vec![1.2,0.333333333]);

            let scf_dfa = [];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = 1.0;
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("b2plyp") {
            // below are some popular B2PLYP-type DH functionals
            // B2PLYP
            // J. Chem. Phys. 2006, 124, 034108.
            // KS part
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_b88", "gga_c_lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.47, 0.73];
            let dfa_hybrid_scf = 0.53;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.27, 0.27]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("b2gpplyp") {
            // B2GP-PLYP
            // J. Phys. Chem. A 2008, 112, 12868–12886.
            // KS part
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_b88", "gga_c_lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.35, 0.64];
            let dfa_hybrid_scf = 0.65;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.36, 0.36]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("pbe-qidh") {
            // PBE-QIDH
            // J. Chem. Phys. 2014, 141, 031101.
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.306639, 0.666667];
            let dfa_hybrid_scf = 0.693361;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.333333, 0.333333]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("pbe0dh") {
            // PBE0-DH
            // J. Chem. Phys. 2011, 135, 024106.
            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["gga_x_pbe", "gga_c_pbe"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![0.50, 0.875];
            let dfa_hybrid_scf = 0.50;
            // PT2 part
            let dfa_family_pos = Some(DFAFamily::PT2);
            let dfa_paramr_adv = Some(vec![0.125, 0.125]);
            let dfa_compnt_pos: Option<Vec<usize>> = Some(dfa_compnt_scf.clone());
            let dfa_paramr_pos = Some(dfa_paramr_scf.clone());
            let dfa_hybrid_pos = Some(dfa_hybrid_scf);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg3") {
            // Renormalized XYG3 functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYG3 framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![-0.0140,0.2107,0.00,0.6789]);
            let dfa_hybrid_pos = Some(0.8033);
            let dfa_paramr_adv = Some(vec![0.3211,0.3211]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xygjos") {
            // Renormalized XYGJOS functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYGJOS framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2269,0.000,0.2309,0.2754]);
            let dfa_hybrid_pos = Some(0.7731);
            let dfa_paramr_adv = Some(vec![0.4364,0.0000]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg7") {
            // Renormalized XYG7 functional (experimental)
            // Replaces PT2 correlation with sBGE2 in the post-SCF part
            // No publication yet - experimental test of sBGE2 in XYG7 framework
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            //let dfa_compnt_pos: Option<Vec<XcFuncType>> = Some(pos_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect());
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect());
            let dfa_paramr_pos = Some(vec![0.2055,-0.1408,0.4056,0.1159]);
            let dfa_hybrid_pos = Some(0.8971);
            let dfa_paramr_adv = Some(vec![0.4052,0.2589]);

            let dfa_family_scf = DFAFamily::HybridGGA;
            let scf_dfa = ["b3lyp"];
            //let dfa_compnt_scf: Vec<XcFuncType> = scf_dfa.iter().map(|xc| {
            //    DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
            //    .flatten().collect();
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf,spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else if tmp_name.eq("r-xyg2") {
            // Renormalized XYG2 functional (experimental)
            let dfa_family_pos = Some(DFAFamily::SBGE2);
            let pos_dfa = ["lda_x_slater", "gga_x_b88","lda_c_vwn_rpa","gga_c_lyp"];
            let dfa_compnt_pos: Option<Vec<usize>> = Some(pos_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten()
                .collect());
            let dfa_paramr_pos = Some(vec![0.00,0.1984,0.00,0.6613]);
            let dfa_hybrid_pos = Some(0.8016);
            let dfa_paramr_adv = Some(vec![0.3387,0.3387]);

            let scf_dfa = ["b3lyp"];
            let dfa_compnt_scf: Vec<usize> = scf_dfa.iter().map(|xc| {
                DFA4REST::xc_func_init_fdqc(*xc, spin_channel).into_iter()})
                .flatten().collect();
            let dfa_paramr_scf = vec![1.0;dfa_compnt_scf.len()];
            let dfa_hybrid_scf = DFA4REST::get_hybrid_libxc(&dfa_compnt_scf, spin_channel);
            Some(DFA4REST{
                spin_channel,
                dfa_compnt_scf,
                dfa_paramr_scf,
                dfa_hybrid_scf,
                dfa_paramr_adv,
                dfa_family_pos,
                dfa_compnt_pos,
                dfa_paramr_pos,
                dfa_hybrid_pos,
                dfa_rsh_scf: None,
            nlc_vv10: None,
        })
        } else {
            None
        }
    }

    pub fn is_dfa_scf(&self) -> bool {
        self.dfa_compnt_scf.len() !=0
    }

    pub fn is_hybrid(&self) -> bool {
        self.dfa_hybrid_scf.abs() >= 1.0e-6 || self.is_rsh()
    }

    pub fn is_fifth_dfa(&self) -> bool {
        match self.dfa_family_pos {
            None => false,
            _ => true
        }
    }

    pub fn is_rpa(&self) -> bool {
        match self.dfa_family_pos {
            Some(DFAFamily::RPA) => true,
            _ => false
        }
    }


    pub fn use_density_gradient(&self) -> bool {
        self
            .dfa_compnt_pos
            .iter()
            .flatten()
            .chain(self.dfa_compnt_scf.iter())
            .any(|xc_func_id| {
                // Only LDA and HybLDA do not need density gradient
                !matches!(self.init_libxc(xc_func_id).family(), libxc::enums::LibXCFamily::LDA | libxc::enums::LibXCFamily::HybLDA)
            })
    }

}
impl DFA4REST {
    pub fn use_kinetic_density(&self) -> bool {
        self
            .dfa_compnt_pos
            .iter()
            .flatten()
            .chain(self.dfa_compnt_scf.iter())
            .any(|xc_func_id| self.init_libxc(xc_func_id).needs_tau())
    }

}
