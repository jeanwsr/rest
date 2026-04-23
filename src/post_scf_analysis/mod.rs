pub mod rand_wf_real_space;
pub mod cube_build;
pub mod molden_build;
pub mod mulliken;
pub mod strong_correlation_correction;
pub mod rrs_pbc;
pub mod spin_correction;

use std::path::Path;
use rest_libcint::prelude::rest_libcint_wrapper::int1e_r;
use tensors::{MathMatrix, MatrixFull, RIFull};

use crate::constants::{ANG, AU2DEBYE, SPECIES_INFO};
use crate::dft::{DFAFamily};
use crate::geom_io::get_mass_charge;
use crate::grad::{formated_force, formated_force_ev, numerical_force};
use crate::mpi_io::MPIOperator;
use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon, open_shell_sbge2_rayon, close_shell_sbge2_detailed_rayon, open_shell_sbge2_detailed_rayon};
use crate::ri_rpa::scsrpa::{evaluate_osrpa_correlation_rayon, evaluate_spin_response_rayon, evaluate_special_radius_only};
use crate::ri_rpa::{evaluate_rpa_correlation, evaluate_rpa_correlation_rayon};
use crate::ri_gw;
use crate::ri_bse;
use crate::scf_io::{SCF, SCFType, print_force_for_ghost_point_charges};
use crate::ri_pt2::{close_shell_pt2_rayon, open_shell_pt2_rayon};
use crate::utilities::TimeRecords;

use self::molden_build::{gen_header, gen_molden};
use self::strong_correlation_correction::scc15_for_rxdh7;

pub fn post_scf_output(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) {
    scf_data.mol.ctrl.outputs.iter().for_each(|output_type| {
        if output_type.eq("fchk") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {scf_data.save_fchk_of_gaussian();}
            } else {
                scf_data.save_fchk_of_gaussian();
            }
        } else if output_type.eq("fciqmc_dump") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {fciqmc_dump(&scf_data)}
            } else {
                fciqmc_dump(&scf_data);
            }
        } else if output_type.eq("wfn_in_real_space") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating the wavefunction in real space");
            } else {
                let np = 100;
                let slater_determinant = rand_wf_real_space::slater_determinant(&scf_data, np);
                let output = serde_json::to_string(&slater_determinant).unwrap();
                let mut file = std::fs::File::create("./wf_in_real_space.txt").unwrap();
                std::io::Write::write(&mut file, output.as_bytes());
            }
        } else if output_type.eq("cube_orb") {
            println!("Now generating the cube files for given orbitals");
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating orbital cube files");
            } else {
                cube_build::get_cube_orb(&scf_data);
            }
        } else if output_type.eq("tabulated_exc") {
            println!("Now tabulating e_[xc] to each grid points");
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for tabulating e_[xc] to each grid points");
            } else {
                if let Some(grids) = &scf_data.grids {
                    let dm = &scf_data.density_matrix;
                    let mo = &scf_data.eigenvectors;
                    let occ = &scf_data.occupation;
                    scf_data.mol.xc_data.post_tabulated_exc(grids, dm, mo, occ);
                } else {
                    panic!("The grids are not yet initialized");
                }
            }
        } else if output_type.eq("molden") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating molden file");
            } else {
                molden_build::gen_molden(&scf_data);
            }
        } else if output_type.eq("hamiltonian") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {save_hamiltonian(&scf_data);}
            } else {
                save_hamiltonian(&scf_data);
            }
        } else if output_type.eq("geometry") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    save_geometry(&scf_data);
                    scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
                }
            } else {
                save_geometry(&scf_data);
                scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
            }
        } else if output_type.eq("overlap") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {save_overlap(&scf_data);}
            } else {
               save_overlap(&scf_data);
            }
        } else if output_type.eq("multiwfn") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating multiwfn file");
            } else {
                gen_molden(&scf_data);
            }
        } else if output_type.eq("deeph") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    save_hamiltonian(&scf_data);
                    save_geometry(&scf_data);
                    scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
                }
            } else {
                save_hamiltonian(&scf_data);
                save_geometry(&scf_data);
                scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
            }
        } else if output_type.eq("dipole") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    let dp = evaluate_dipole_moment(scf_data, None);
                    println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
                }
            } else {
                let dp = evaluate_dipole_moment(scf_data, None);
                println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
            }
        } else if output_type.eq("num_force") {
            let displace = match scf_data.mol.geom.unit {
                crate::geom_io::GeomUnit::Angstrom => scf_data.mol.ctrl.nforce_displacement/ANG,
                crate::geom_io::GeomUnit::Bohr => scf_data.mol.ctrl.nforce_displacement,
            };
            let (energy, num_force) = numerical_force(scf_data, displace, mpi_operator);
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    println!("Total atomic forces [a.u.]: ");
                    println!("{}", formated_force(&num_force, &scf_data.mol.geom.elem));
                    println!("Total atomic forces [ev/ang]: ");
                    println!("{}", formated_force_ev(&num_force, &scf_data.mol.geom.elem));
                }
            } else {
                println!("Total atomic forces [a.u.]: ");
                println!("{}", formated_force(&num_force, &scf_data.mol.geom.elem));
                println!("Total atomic forces [ev/ang]: ");
                println!("{}", formated_force_ev(&num_force, &scf_data.mol.geom.elem));
            }
        } else if output_type.eq("force_for_ghost_point_charges") {
            print_force_for_ghost_point_charges(&scf_data)
        }
    });
}

pub fn write_scf_attribute<T>(group: &hdf5::Group, dataset_name: &str, value: &[T]) 
where 
    T: Clone + hdf5::H5Type
{
    if let Ok(dataset) = group.dataset(dataset_name) {
        match dataset.write_raw(value) {
            Ok(_) => (),
            Err(e) => println!("Error writing dataset {}: {:?}", dataset_name, e),
        }
    } else {
        let builder = group.new_dataset_builder();
        match builder.with_data(value).create(dataset_name) {
            Ok(_) => (),
            Err(e) => println!("Error creating dataset {}: {:?}", dataset_name, e),
        }
    }
}

pub fn save_chkfile(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    //if path.exists() {std::fs::remove_file(chkfile).unwrap()};
    //let file = hdf5::File::create(chkfile).unwrap();
    //let scf = file.create_group("scf").unwrap();
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    println!("write chkfile: {}", chkfile);
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };

    write_scf_attribute(&scf, "e_tot", &[scf_data.scf_energy]);
    write_scf_attribute(&scf, "num_basis", &[scf_data.mol.num_basis]);
    write_scf_attribute(&scf, "spin_channel", &[scf_data.mol.spin_channel]);
    write_scf_attribute(&scf, "num_states", &[scf_data.mol.num_state]);

    // let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("mo_coeff")});
    let mut eigenvectors: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.eigenvectors[i_spin].transpose();
        eigenvectors.extend(tmp_eigenvectors.data.iter());
        if let SCFType::ROHF = scf_data.scftype { // ROHF: only process i_spin = 0, since alpha/beta eigenvectors are the same
            break
        }
    }
    write_scf_attribute(&scf, "mo_coeff", &eigenvectors);

    let mut eigenvalues: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        eigenvalues.extend(scf_data.eigenvalues[i_spin].iter());
        if let SCFType::ROHF = scf_data.scftype { // ROHF: only process i_spin = 0, since alpha/beta eigenvalues are the same
            break 
        }
    }
    write_scf_attribute(&scf, "mo_energy", &eigenvalues);

    let mut occ: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        occ.extend(scf_data.occupation[i_spin].iter());
    }
    // for compatibility with old rest, may be removed in the future
    write_scf_attribute(&scf, "mo_occupation", &occ);
    // for compatibility with pyscf
    write_scf_attribute(&scf, "mo_occ", &occ);

    file.close();
}

pub fn save_hamiltonian(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    let mut hamiltonians: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.hamiltonian[i_spin].to_matrixfull().unwrap();
        hamiltonians.extend(scf_data.eigenvalues[i_spin].iter());
    }
    let is_hamiltonian = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("hamiltonian")});
    if is_hamiltonian {
        let dataset = scf.dataset("hamiltonian").unwrap();
        dataset.write(&hamiltonians);
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&hamiltonians).create("hamiltonian");
    };
    file.close();
}

pub fn save_overlap(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    let mut overlap: Vec<f64> = vec![];
    let overlap = scf_data.ovlp.to_matrixfull().unwrap();
    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("overlap")});
    if is_exist {
        let dataset = scf.dataset("overlap").unwrap();
        dataset.write(&overlap.data);
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&overlap.data).create("overlap");
    };
    file.close();
}

pub fn save_geometry(scf_data: &SCF) {
    let ang = crate::constants::ANG;
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_geom = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("geom")});
    let geom = if is_geom {
        file.group("geom").unwrap()
    } else {
        file.create_group("geom").unwrap()
    };
    let mass_charge = get_mass_charge(&scf_data.mol.geom.elem);
    //let mut geometry: Vec<(f64,f64,f64,f64)> = vec![];
    let mut geometry: Vec<[f64;4]> = vec![];
    mass_charge.iter().zip(scf_data.mol.geom.position.iter_columns_full()).for_each(|(mass_charge, position)| {
        geometry.push([mass_charge.1,position[0]*ang,position[1]*ang,position[2]*ang]);
        //geometry.push((mass_charge.1,position[0]*ang,position[1]*ang,position[2]*ang));
    });

    let is_geom = geom.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("position")});
    if is_geom {
        let dataset = geom.dataset("position").unwrap();
        dataset.write(&geometry);
    } else {
        let builder = geom.new_dataset_builder();
        builder.with_data(&geometry).create("position");
    }
    file.close();
}

pub fn print_out_dfa(scf_data: &SCF) {
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    });
}

pub fn print_out_xc_potentials(scf_data: &SCF) {
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    });
}

pub fn post_ai_correction(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) -> Option<Vec<f64>> {
    let xc_method = &scf_data.mol.ctrl.xc.to_lowercase();
    let post_ai_corr = &scf_data.mol.ctrl.post_ai_correction.to_lowercase();
    let mut scc = 0.0;
    if post_ai_corr.eq("scc15") && xc_method.eq("r-xdh7") {
        scc = scc15_for_rxdh7(scf_data, mpi_operator);
        let total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
        if scf_data.mol.ctrl.print_level>0 {
            println!("E(R-xDH7-SCC15): {:16.8} Ha", total_energy + scc);
        }
        //scf_data.energies.insert("scc23".to_string(), vec![scc]);
        return Some(vec![scc])
    };
    None
}

/// NOTE: only support symmetric RI-V tensors
pub fn post_scf_correlation(scf_data: &mut SCF) {

    let mut timerecords = TimeRecords::new();
    let spin_channel = scf_data.mol.spin_channel;
    let dfa_family_pos = if let Some(tmp_dfa) = &scf_data.mol.xc_data.dfa_family_pos {
        tmp_dfa.clone()
    } else {crate::dft::DFAFamily::Unknown};

    if let None = scf_data.ri3mo {
        let (occ_range, vir_range) = crate::scf_io::determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        if scf_data.mol.ctrl.print_level>1 {
            //println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range);
            let spin_orb_indices = split_indices_by_spin_occ(&scf_data.occupation, 0.5);
            let (alpha_occ, alpha_vir) = &spin_orb_indices[0];
            println!("Occupied orbitals (alpha): {}", format_indices(alpha_occ));
            println!("Virtual orbitals (alpha): {}", format_indices(alpha_vir));
            if matches!(scf_data.scftype, SCFType::UHF | SCFType::ROHF) {
                let (beta_occ,  beta_vir)  = &spin_orb_indices[1];
                println!("Occupied orbitals (beta): {}", format_indices(beta_occ));
                println!("Virtual orbitals (beta): {}", format_indices(beta_vir));
            }
        };
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
    }

    let mut post_corr: Vec<(crate::dft::DFAFamily, [f64;3])> = vec![];
    scf_data.mol.ctrl.post_correlation.iter().filter(|corr| *corr!=&dfa_family_pos).for_each(|corr| {
        match corr {
            crate::dft::DFAFamily::PT2 => {
                timerecords.new_item("PT2", "the PT2 calculation");
                timerecords.count_start("PT2");
                println!("Evaluating the PT2 correlation");
                let mut energy_post = if spin_channel == 1 {
                    close_shell_pt2_rayon(&scf_data).unwrap()
                } else {
                    open_shell_pt2_rayon(&scf_data).unwrap()
                };
                let os_factor = scf_data.mol.ctrl.ri_pt2.os_factor.unwrap_or(1.0);
                let ss_factor = scf_data.mol.ctrl.ri_pt2.ss_factor.unwrap_or(1.0);
                if scf_data.mol.ctrl.print_level > 1 && (os_factor != 1.0 || ss_factor != 1.0) {
                    println!("PT2 scaling factors: OS: {:16.8}, SS: {:16.8}", os_factor, ss_factor);
                }
                let sos_energy = os_factor * energy_post[1];
                let sss_energy = ss_factor * energy_post[2];
                energy_post[0] = sos_energy + sss_energy;
                post_corr.push((crate::dft::DFAFamily::PT2, energy_post));
                timerecords.count("PT2");
            },
            crate::dft::DFAFamily::SBGE2 => {
                timerecords.new_item("sBGE2", "the sBGE2 calculation");
                timerecords.count_start("sBGE2");
                println!("Evaluating the sBGE2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_sbge2_rayon(&scf_data).unwrap()
                } else {
                    //[0.0,0.0,0.0]
                    open_shell_sbge2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::SBGE2, energy_post));
                timerecords.count("sBGE2");

            },
            crate::dft::DFAFamily::RPA => {
                timerecords.new_item("dRPA", "the dRPA calculation");
                timerecords.count_start("dRPA");
                println!("Evaluating the dRPA correlation");
                let energy_post = evaluate_rpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::RPA, [energy_post, 0.0,0.0]));
                timerecords.count("dRPA");
            },
            crate::dft::DFAFamily::SCSRPA => {
                timerecords.new_item("SCSRPA", "the scsRPA calculation");
                timerecords.count_start("SCSRPA");
                println!("Evaluating the scsRPA correlation");
                let energy_post = evaluate_osrpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::SCSRPA, energy_post));
                timerecords.count("SCSRPA");
            }
            _ => {println!("Unknown post-scf correlation methods")}
        }
    });

    if scf_data.mol.ctrl.print_level>0 {
        println!("----------------------------------------------------------------------");
        println!("{:16}: {:>16}, {:>16}, {:>16}","Methods","Total Corr", "OS Corr", "SS Corr");
        println!("----------------------------------------------------------------------");

        post_corr.iter().for_each(|(name,energy)| {
            println!("{:16}: {:16.8}, {:16.8}, {:16.8}", name.to_name(), energy[0], energy[1], energy[2]);
        });
        println!("----------------------------------------------------------------------");
        if scf_data.mol.ctrl.print_level>1 {timerecords.report_all()};
    }
}

pub fn quasiparticle_methods(scf_data:&mut SCF,mpi_operator:&Option<MPIOperator>){
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let output_type=qp_ctrl.gw_or_bse.clone();
    if output_type.eq("gw"){
        let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
        let xc_data=scf_data.mol.xc_data.clone();
        println!("Current XC data:");
        println!("dfa_compnt_scf={:?}",xc_data.dfa_compnt_scf);
        println!("dfa_paramr_scf={:?}",xc_data.dfa_paramr_scf);
        println!("dfa_hybrid_scf={}",xc_data.dfa_hybrid_scf);
        if qp_ctrl.homo_lumo_gw_qp==true{
            ri_gw::get_homo_lumo_qp_only(scf_data,20,&vxc_nn,mpi_operator);
        }else if qp_ctrl.self_energy_spectrum_test==true{
            ri_gw::spectrum_test(scf_data,20);
        }else if qp_ctrl.obtain_vx_vc_terms==true{
            ri_gw::obtain_vx_vc_terms(scf_data);
        }else{
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
            if scf_data.mol.ctrl.print_level>1{
                ri_bse::matvec::test_v_w_contribution(scf_data);
            }
        }
    }else if output_type.eq("bse"){
        // Prepare BSE-specific RI integrals before BSE calculation
        scf_data.prepare_bse_integrals(mpi_operator);

        if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        ri_bse::bse_main(scf_data);
    }else if output_type.eq("damped_bse"){
        if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        let p_induced=ri_bse::damped::damped_bse(scf_data);
        println!("Induced Density Matrix:");
        println!("P Real (Plus Half):\n{:#?}",p_induced.0);
        println!("P Real (Minus Half):\n{:#?}",p_induced.1);
        println!("P Imaginary (Plus Half):\n{:#?}",p_induced.2);
        println!("P Imaginary (Minus Half):\n{:#?}",p_induced.3);
    }else{
        print!("Warning: You entered an invalid quasiparticle method. No quasiparticle methods Were triggered.")
    }
}

fn fciqmc_dump(scf_data: &SCF) {
    if let Some(ri3fn) = &scf_data.ri3fn {
        // prepare RI-V three-center coefficients for HF orbitals
        for i_spin in 0..scf_data.mol.spin_channel {
            let ri3mo = ri3fn.ao2mo_v01(&scf_data.eigenvectors[i_spin]).unwrap();
            for i in 0.. scf_data.mol.num_state {
                let ri3mo_i = ri3mo.get_reducing_matrix(i).unwrap();
                for j in 0.. scf_data.mol.num_state {
                    let ri3mo_ij = ri3mo_i.get_slice_x(j);
                    for k in 0.. scf_data.mol.num_state {
                        let ri3mo_k = ri3mo.get_reducing_matrix(k).unwrap();
                        for l in 0.. scf_data.mol.num_state {
                            let ri3mo_kl = ri3mo_k.get_slice_x(l);
                            let ijkl = ri3mo_ij.iter().zip(ri3mo_kl.iter())
                                .fold(0.0, |acc, (val1, val2)| acc + val1*val2);
                            if ijkl.abs() > 1.0E-8 {
                                println! ("{:16.8} {:5} {:5} {:5} {:5}",ijkl, i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }
}


//pub fn test_hdf5_string() {
//    let file = hdf5::File::create("test_string").unwrap();
//    let geom = file.group("geom").unwrap_or(file.create_group("geom").unwrap());
//    let dd = vec!["string1", "string2"];
//    let builder = geom.new_dataset_builder();
//    builder.with_data(&ndarray::arr1(&dd)).create("elem");
//    file.close();
//}

// evaluate the dipole moment based on the converged density matrix (dm): scf_data.density_matrix
// the dipole moment of the nuclear part (nucl_dip) is given by scf_data.mol.geom.evaluate_dipole_moment()
// the dipole moment of the atomic orbitals (ao_dip) is given by scf_data.mol.int_ij_matrixuppers()
// the dipole moment of the electronic part (el_dip) is given ('ij,ji', ao_dip[x], dm)
pub fn evaluate_dipole_moment(scf_data: &SCF, orig: Option<[f64;3]>) -> [f64;3] {

    let (mut tot_dip, mass_tot) = scf_data.mol.geom.evaluate_dipole_moment(None);

    let mut dm = scf_data.density_matrix[0].clone();
    if scf_data.mol.spin_channel == 2 {
        dm.self_add(&scf_data.density_matrix[1]);
    }

    let mut cint_data = scf_data.mol.initialize_cint(false);

    let p_orig = cint_data.get_common_origin();

    let r_orig: [f64;3] = if let Some(u_orig) = orig {
        u_orig.try_into().unwrap()
    } else {
        p_orig.clone()
    };
    cint_data.set_common_origin(r_orig);

    let (out, out_shape)= cint_data.integral_s1::<int1e_r>(None);
    //let mut out_shape_1 = [0;3];
    //out_shape_1.iter_mut().zip(out_shape.iter()).for_each(|(out_shape_1, &out_shape)| {*out_shape_1 = out_shape});

    let ao_dip = RIFull::from_vec(out_shape.try_into().unwrap(), out).unwrap();

    let mut el_dip = [0.0;3];
    for i in 0..3 {
        let ao_dip_tmp = ao_dip.get_reducing_matrix(i).unwrap();
        el_dip[i] = dm.iter_columns_full().zip(ao_dip_tmp.iter_columns_full()).fold(0.0,|acc_c,(dm_col, ao_dip_col)| {
            let acc_r = dm_col.iter().zip(ao_dip_col.iter()).fold(0.0, |acc_r, (dm_val, ao_dip_val)| {acc_r + dm_val*ao_dip_val});
            acc_c + acc_r
        });
    }

    cint_data.set_common_origin(p_orig);

    //nucl_dip.iter().zip(el_dip.iter()).map(|(nucl, el)| (*nucl - *el)*AU2DEBYE).collect::<Vec<f64>>()

    tot_dip.iter_mut().zip(el_dip.iter()).for_each(|(nucl, el)| *nucl = (*nucl - *el)*AU2DEBYE);

    tot_dip

}

fn split_indices_by_occ(
    spin_occ: &[f64],
    occ_threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut occ_idx = Vec::new();
    let mut vir_idx = Vec::new();

    for (i, &n_occ) in spin_occ.iter().enumerate() {
        if n_occ > occ_threshold {
            occ_idx.push(i);
        } else {
            vir_idx.push(i);
        }
    }

    (occ_idx, vir_idx)
}

pub fn split_indices_by_spin_occ(
    occupations: &[Vec<f64>], // occupations[0]=alpha, occupations[1]=beta
    occ_threshold: f64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    occupations
        .iter()
        .map(|spin_occ| split_indices_by_occ(spin_occ, occ_threshold))
        .collect()
}

fn compress_indices_to_ranges(indices: &[usize]) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();

    let mut start = indices[0];
    let mut prev  = indices[0];

    for &idx in indices.iter().skip(1) {
        if idx == prev + 1 {
            // still contiguous
            prev = idx;
        } else {
            // end current range
            ranges.push((start, prev));
            start = idx;
            prev  = idx;
        }
    }

    // push last range
    ranges.push((start, prev));

    ranges
}

fn format_ranges(ranges: &[(usize, usize)]) -> String {
    ranges
        .iter()
        .map(|(start, end)| {
            if start == end {
                format!("{}", start)
            } else {
                format!("{}-{}", start, end)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub fn format_indices(indices: &[usize]) -> String {
    let ranges = compress_indices_to_ranges(indices);
    format_ranges(&ranges)
}
