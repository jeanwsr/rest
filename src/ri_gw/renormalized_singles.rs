use crate::constants::PI;
use itertools::Itertools;
use std::ops::Range;
use crate::utilities;
use crate::scf_io::SCF;
use rayon::result;
use reqwest::blocking::Response;
use rest_tensors::{RIFull};
use tensors::{matrix_blas_lapack::{_dinverse,_dsyev}, ri, MathMatrix, MatrixFull};
use rest_tensors::matrix::matrix_blas_lapack::{_dgees,_dgemm,_dgemv};
use rest_tensors::MatrixUpper;
use crate::ri_bse;
use crate::ri_rpa;
use crate::molecule_io;
use crate::scf_io;
use crate::dft;
use crate::dft::DFA4REST;
use std::sync::mpsc::channel;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use crate::mpi_io::{MPIOperator,MPIData};
use crate::ri_gw;

#[cfg(target_os = "linux")]
use libc::seccomp_notif;

pub fn generate_rs_hamiltonian(scf_data:&mut SCF,mpi_operator:&Option<MPIOperator>)->(MatrixFull<f64>,MatrixFull<f64>){
    println!("Starts generating rs hamiltonian!");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    //let dfa_oo_hamiltonian=hamiltonian_ao2mo(scf_data,'O');
    //let dfa_vv_hamiltonian=hamiltonian_ao2mo(scf_data,'V');
    //println!("DFA oo hamiltonian");
    //dfa_oo_hamiltonian.formated_output(20,"full");
    //println!("DFA vv hamiltonian");
    //dfa_vv_hamiltonian.formated_output(20,"full");
    scf_data.mol.xc_data.dfa_compnt_scf=vec![];
    scf_io::SCF::generate_hf_hamiltonian_ri_v_dm_only(scf_data,mpi_operator);
    let hf_oo_hamiltonian=hamiltonian_ao2mo(scf_data,'O');
    let hf_vv_hamiltonian=hamiltonian_ao2mo(scf_data,'V');
    (hf_oo_hamiltonian,hf_vv_hamiltonian)
}
pub fn renormalized_singles_diagonalization(scf_data:&mut SCF,w_rs:bool,mpi_operator:&Option<MPIOperator>)->Vec<f64>{
    println!("You are doing renormalized singles GW calculations suggested by Weitao Yang's group:");
    println!("Ye Jin, Neil Qiang Su, and Weitao Yang, J. Phys. Chem. Lett. 10, 3, 447–452 (2019)");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let (hf_oo_hamiltonian,hf_vv_hamiltonian)=generate_rs_hamiltonian(scf_data,mpi_operator);
    if scf_data.mol.ctrl.print_level>1{
        println!("HF oo hamiltonian");
        hf_oo_hamiltonian.formated_output(20,"full");
        println!("HF vv hamiltonian");
        hf_vv_hamiltonian.formated_output(20,"full");
    }
    let (occ_rs_values,oo_renormalized)=diagonalize_and_renormalize(scf_data,&hf_oo_hamiltonian,'O');
    let (vir_rs_values,vv_renormalized)=diagonalize_and_renormalize(scf_data,&hf_vv_hamiltonian,'V');
    println!("Renormalized Singles Results:");
    if scf_data.mol.ctrl.print_level>1{
        println!("eigenenergies:{:?}",scf_data.eigenvalues.clone()[0]);
    }
    if scf_data.mol.ctrl.print_level>2{
        println!("oo:");
    oo_renormalized.formated_output(1000,"full");
    println!("vv:");
    vv_renormalized.formated_output(1000,"full");
    }
    if w_rs==true{
        for i in 0..num_state{
            for j in 0..num_state{
                if i<occ_size{
                    scf_data.eigenvectors[0][[j,i]]=oo_renormalized[[j,i]]
                }else{
                    scf_data.eigenvectors[0][[j,i]]=vv_renormalized[[j,i-occ_size]]
                }            
            }
        }
    }
    occ_rs_values.into_iter().chain(vir_rs_values.into_iter()).collect()
}
pub fn renormalized_singles_diagonalization_fullspace(scf_data:&mut SCF,w_rs:bool,mpi_operator:&Option<MPIOperator>)->Vec<f64>{
    println!("You are doing full-space renormalized singles GW calculations:");
    println!("The full HF Hamiltonian in MO basis is diagonalized and renormalized.");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    
    scf_data.mol.xc_data.dfa_compnt_scf=vec![];
    scf_io::SCF::generate_hf_hamiltonian_ri_v_dm_only(scf_data,mpi_operator);
    let hf_full_hamiltonian=hamiltonian_ao2mo_full(scf_data);
    
    let (mut eigenvectors_opt, eigenvalues_opt, ndim) = _dsyev(&hf_full_hamiltonian, 'V');
    let ndim = ndim as usize;
    let mut eigenvectors = eigenvectors_opt.unwrap();
    let eigenvalues = eigenvalues_opt;
    
    for i in 0..ndim{
        let mut norm=0.0;
        for j in 0..ndim{
            norm+=(eigenvectors[[j,i]].powf(2.0));
        }
        norm=norm.sqrt();
        for j in 0..ndim{
            eigenvectors[[j,i]]*=norm.powf(-1.0);
        }
    }
    
    let previous_coeff=scf_data.eigenvectors[0].clone();
    let mut vectors=MatrixFull::new([num_state,ndim],0.0);
    _dgemm(&previous_coeff,((0..num_state),(0..ndim)),'N',
           &eigenvectors,((0..ndim),(0..ndim)),'N',
           &mut vectors,((0..num_state),(0..ndim)),1.0,0.0);
    
    if w_rs==true{
        for i in 0..num_state{
            for j in 0..num_state{
                scf_data.eigenvectors[0][[j,i]]=vectors[[j,i]]
            }
        }
    }
    
    eigenvalues
}
pub fn hamiltonian_ao2mo(scf_data:&SCF,choice:char)->MatrixFull<f64>{
    println!("now computing:{}",choice);
    let ao_hamiltonian=scf_data.hamiltonian[0].clone().to_matrixfull().unwrap();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let mut range_a:Range<usize>=0..1;
    let mut range_b:Range<usize>=0..1;
    let eigenvecs=scf_data.eigenvectors[0].clone();
    let dimensions = match choice {
        'O' => occ_size,
        'V' => vir_size,
        _ => panic!("invalid choice of RS subspace!"),
    };
    let mut hamiltonian=MatrixFull::new([dimensions,dimensions],0.0);
    let mut element=0.0;
    if scf_data.mol.ctrl.print_level>1{
        println!("Now staring to do ao2mo of {}, have a little patience when doing 'V'",choice);
    }
    let ao_dimensions=ao_hamiltonian.size[0];
    if scf_data.mol.ctrl.print_level>1{
        println!("ao dimensions={}",ao_dimensions);
        println!("mo dimensions={}",dimensions);
    }
    let mut c_t_h_ao=MatrixFull::new([dimensions,ao_dimensions],0.0);
    let mut h_mo=MatrixFull::new([dimensions,dimensions],0.0);
    if choice=='O'{
        _dgemm(&eigenvecs,((0..ao_dimensions),(0..occ_size)),'T',
               &ao_hamiltonian,((0..ao_dimensions),(0..ao_dimensions)),'N',
               &mut c_t_h_ao,((0..occ_size),(0..ao_dimensions)),1.0,0.0);
        _dgemm(&c_t_h_ao,((0..occ_size),(0..ao_dimensions)),'N',
               &eigenvecs,((0..ao_dimensions),(0..occ_size)),'N',
               &mut h_mo,((0..occ_size),(0..occ_size)),1.0,0.0);
    }else{
        _dgemm(&eigenvecs,((0..ao_dimensions),(occ_size..ao_dimensions)),'T',
               &ao_hamiltonian,((0..ao_dimensions),(0..ao_dimensions)),'N',
               &mut c_t_h_ao,((0..dimensions),(0..ao_dimensions)),1.0,0.0);
        _dgemm(&c_t_h_ao,((0..dimensions),(0..ao_dimensions)),'N',
               &eigenvecs,((0..ao_dimensions),(occ_size..ao_dimensions)),'N',
               &mut h_mo,((0..dimensions),(0..dimensions)),1.0,0.0);
    }
    
    h_mo
}
pub fn diagonalize_and_renormalize(scf_data:&SCF,subspace_hamiltonian:&MatrixFull<f64>,choice:char)->(Vec<f64>,MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let mut eigenvalues_for_scf: Vec<f64> = Vec::new();
    let spin_channel=scf_data.mol.ctrl.spin_channel;
    let previous_eigenvecs=scf_data.eigenvectors[0].clone();
    let (mut eigenvectors,mut eigenvalues, ndim) = _dsyev(subspace_hamiltonian, 'V');
    let num_state = scf_data.mol.num_state;
    let previous_coeff=previous_eigenvecs.clone();
    let mut renormalized_singles_mo=eigenvectors.unwrap();
    let mut norm:f64=0.0;
    let ndim=ndim as usize;
    for i in 0..ndim{
        for j in 0..ndim{
            norm+=(&renormalized_singles_mo[[j,i]].powf(2.0));
        }
        norm=norm.sqrt();
        //println!("norm at {} has been collected!",i);
        for j in 0..ndim{
            //println!("processing {}th dimension at {} with norm",j,i);
            renormalized_singles_mo[[j,i]]*=norm.powf(-1.0);
        }
        norm=0.0;
    }
    let mut vectors:MatrixFull<f64>=MatrixFull::new([num_state,ndim],0.0);
    for i in 0..ndim{
        for mu in 0..num_state{
            let mut sum=0.0;
            for j in 0..ndim{
                let ind = match choice {
                'O' => j,
                'V' => num_state - ndim + j,
                _ => panic!("无效的选项: {}", choice), // 或返回默认值如 0
                };
                sum+=renormalized_singles_mo[[i,j]]*previous_coeff[[mu,ind]];
            }
            vectors[[mu,i]]=sum;
            sum=0.0;
        }
    }
    /*eigenvectors_for_scf=vectors;
    if choice=='O'{
        for i in start_mo..homo+1{
            for spin in 0..spin_channel{
                scf_data.eigenvalues[spin][i]=eigenvalues_for_scf[spin][i];
            }
        }
    }else if choice=='V'{
        for i in lumo..num_state{
            for spin in 0..spin_channel{
                scf_data.eigenvalues[spin][i]=eigenvalues_for_scf[spin][i-lumo];
            }
        }
    }
    eigenvectors_for_scf*/
    (eigenvalues,vectors)
}
pub fn hamiltonian_ao2mo_full(scf_data:&SCF)->MatrixFull<f64>{
    let ao_hamiltonian=scf_data.hamiltonian[0].clone().to_matrixfull().unwrap();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let eigenvecs=scf_data.eigenvectors[0].clone();
    let ao_dimensions=ao_hamiltonian.size[0];
    
    let mut c_t_h_ao=MatrixFull::new([num_state,ao_dimensions],0.0);
    let mut h_mo=MatrixFull::new([num_state,num_state],0.0);
    
    _dgemm(&eigenvecs,((0..ao_dimensions),(0..num_state)),'T',
           &ao_hamiltonian,((0..ao_dimensions),(0..ao_dimensions)),'N',
           &mut c_t_h_ao,((0..num_state),(0..ao_dimensions)),1.0,0.0);
    _dgemm(&c_t_h_ao,((0..num_state),(0..ao_dimensions)),'N',
           &eigenvecs,((0..ao_dimensions),(0..num_state)),'N',
           &mut h_mo,((0..num_state),(0..num_state)),1.0,0.0);
    
    h_mo
}

// ===========================================================================
// Renormalized singles: orbital coefficients (not only eigenvalues)
// ===========================================================================
//
// The routines above return the RS eigenvalues, which are used to (re)initialise
// the quasiparticle energies entering the Green's function G (and, with
// `w_rs = true`, the screened interaction W).  The RS diagonalisation also
// produces an orbital rotation U in the Kohn-Sham MO basis,
//
//     H^RS_mo U = U diag(e_RS),      C_rs = C_ks U ,
//
// and the routines below additionally return the AO-basis RS coefficients
// `C_rs` so that the RI three-centre tensor can be recombined from the AO basis
// into the *RS* MO basis (`ao2mo_rayon`, called through
// `SCF::generate_ri3mo_rayon_for_multiple_times` / `v_matrix_from_scf`, always
// reads `scf_data.eigenvectors` at call time).  Installing `C_rs` into
// `scf_data.eigenvectors[0]` is therefore all that is needed for every
// subsequent RI AO2MO step of the GW run to use the RS orbitals.

/// Outcome of a renormalized-singles diagonalisation that returns orbitals.
pub struct RenormalizedSinglesWithOrbitals {
    /// RS eigenvalues; column `n` of `coefficients` is the orbital with energy
    /// `energies[n]` (same ordering convention as `scf_data.eigenvalues[0]`).
    pub energies: Vec<f64>,
    /// Variationally consistent RS orbital coefficients in the AO basis,
    /// `C_rs = C_ks * U`, with the same shape as `scf_data.eigenvectors[0]`.
    pub coefficients: MatrixFull<f64>,
    /// The Kohn-Sham orbital coefficients `C_ks` the rotation was applied to.
    pub ks_coefficients: MatrixFull<f64>,
    /// The orbital rotation `U` in the Kohn-Sham MO basis, `C_rs = C_ks * U`.
    pub rotation: MatrixFull<f64>,
}

/// Column-wise normalisation of a square matrix (mirrors the projection step
/// used by the legacy RS routines).
fn normalize_columns(matr:&mut MatrixFull<f64>){
    let ndim=matr.size[0];
    for i in 0..ndim{
        let mut norm=0.0;
        for j in 0..ndim{
            norm+=matr[[j,i]].powf(2.0);
        }
        norm=norm.sqrt();
        for j in 0..ndim{
            matr[[j,i]]*=norm.powf(-1.0);
        }
    }
}

/// Largest modulus of an off-diagonal element of `Y^T H Y`, where `Y` is `x`
/// (`transposed = false`) or `x^T` (`transposed = true`).  For a variationally
/// consistent rotation this quantity vanishes, which makes it a direct
/// numerical check of the RS orbital rotation.
fn max_offdiagonal_residue(hamiltonian:&MatrixFull<f64>,rotation:&MatrixFull<f64>,transposed:bool)->f64{
    let ndim=rotation.size[0];
    let elem=|row:usize,col:usize| if transposed {rotation[[col,row]]} else {rotation[[row,col]]};
    let mut max_residue=0.0_f64;
    for i in 0..ndim{
        for j in 0..ndim{
            if i==j {continue}
            let mut element=0.0;
            for a in 0..ndim{
                for b in 0..ndim{
                    element+=elem(a,i)*hamiltonian[[a,b]]*elem(b,j);
                }
            }
            if element.abs()>max_residue {max_residue=element.abs()}
        }
    }
    max_residue
}

/// Largest modulus of `U^T U - I`: zero for an orthogonal (unitary) rotation.
fn max_orthogonality_error(rotation:&MatrixFull<f64>)->f64{
    let ndim=rotation.size[0];
    let mut max_error=0.0_f64;
    for i in 0..ndim{
        for j in 0..ndim{
            let mut element=0.0;
            for a in 0..ndim{
                element+=rotation[[a,i]]*rotation[[a,j]];
            }
            if i==j {element-=1.0}
            if element.abs()>max_error {max_error=element.abs()}
        }
    }
    max_error
}

/// Build the RS orbital rotation `U` in the Kohn-Sham MO basis and the
/// associated RS eigenvalues.
///
/// * `full_space = false`: the occupied-occupied and virtual-virtual blocks of
///   the HF Hamiltonian (built from the DFT density matrix) are diagonalised
///   separately; `U` is the corresponding block-diagonal rotation.
/// * `full_space = true`: the complete HF Hamiltonian in the MO basis is
///   diagonalised; `U` is a full orbital rotation.
fn rs_rotation_in_mo_basis(scf_data:&SCF,full_space:bool)->(Vec<f64>,MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    if full_space{
        let hf_full_hamiltonian=hamiltonian_ao2mo_full(scf_data);
        let (rotation,eigenvalues,_)=_dsyev(&hf_full_hamiltonian,'V');
        let mut rotation=rotation.unwrap();
        normalize_columns(&mut rotation);
        if scf_data.mol.ctrl.print_level>1{
            println!("Full-space RS rotation: max|U^T U - I| = {:.3e}",max_orthogonality_error(&rotation));
            println!("Full-space RS rotation: max off-diagonal |U^T H U| = {:.3e}",
                     max_offdiagonal_residue(&hf_full_hamiltonian,&rotation,false));
        }
        (eigenvalues,rotation)
    }else{
        let hf_oo_hamiltonian=hamiltonian_ao2mo(scf_data,'O');
        let hf_vv_hamiltonian=hamiltonian_ao2mo(scf_data,'V');
        let (oo_rotation,oo_eigenvalues,_)=_dsyev(&hf_oo_hamiltonian,'V');
        let (vv_rotation,vv_eigenvalues,_)=_dsyev(&hf_vv_hamiltonian,'V');
        let mut oo_rotation=oo_rotation.unwrap();
        let mut vv_rotation=vv_rotation.unwrap();
        normalize_columns(&mut oo_rotation);
        normalize_columns(&mut vv_rotation);
        if scf_data.mol.ctrl.print_level>1{
            println!("RS subspace rotation quality (max off-diagonal element of H^RS in the rotated basis):");
            println!("  occupied block: U^T H U = {:.3e} (consistent rotation); U H U^T = {:.3e} (transposed rotation)",
                     max_offdiagonal_residue(&hf_oo_hamiltonian,&oo_rotation,false),
                     max_offdiagonal_residue(&hf_oo_hamiltonian,&oo_rotation,true));
            println!("  virtual  block: U^T H U = {:.3e} (consistent rotation); U H U^T = {:.3e} (transposed rotation)",
                     max_offdiagonal_residue(&hf_vv_hamiltonian,&vv_rotation,false),
                     max_offdiagonal_residue(&hf_vv_hamiltonian,&vv_rotation,true));
            println!("  max|U^T U - I|: occupied {:.3e}, virtual {:.3e}",
                     max_orthogonality_error(&oo_rotation),max_orthogonality_error(&vv_rotation));
        }
        // Assemble the block-diagonal rotation.  The occupied block occupies
        // MO columns 0..occ_size and the virtual block the columns
        // occ_size..num_state, exactly as in `hamiltonian_ao2mo`.
        let mut rotation=MatrixFull::new([num_state,num_state],0.0);
        for i in 0..occ_size{
            for j in 0..occ_size{
                rotation[[i,j]]=oo_rotation[[i,j]];
            }
        }
        for i in 0..vir_size{
            for j in 0..vir_size{
                rotation[[occ_size+i,occ_size+j]]=vv_rotation[[i,j]];
            }
        }
        let energies:Vec<f64>=oo_eigenvalues.into_iter().chain(vv_eigenvalues.into_iter()).collect();
        (energies,rotation)
    }
}

/// Renormalized-singles diagonalisation that also returns the RS orbitals in
/// the AO basis.
///
/// The returned coefficients are `C_rs = C_ks U`, i.e. the variationally
/// consistent rotation (the RS Hamiltonian is diagonal in the returned basis).
/// The Kohn-Sham orbitals `C_ks` that define the DFT density matrix — and hence
/// the RS Hamiltonian itself — are the ones present when the routine is called,
/// so the routine must be called *before* the orbitals are overwritten.
pub fn renormalized_singles_with_orbitals(scf_data:&mut SCF,full_space:bool,mpi_operator:&Option<MPIOperator>)->RenormalizedSinglesWithOrbitals{
    if full_space{
        println!("You are doing full-space renormalized singles GW calculations:");
        println!("The full HF Hamiltonian in MO basis is diagonalized, and its orbital coefficients are used.");
    }else{
        println!("You are doing renormalized singles GW calculations suggested by Weitao Yang's group:");
        println!("Ye Jin, Neil Qiang Su, and Weitao Yang, J. Phys. Chem. Lett. 10, 3, 447–452 (2019)");
        println!("The RS orbital coefficients are used for the RI AO-to-MO transformation.");
    }
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    // The RS Hamiltonian is the HF Hamiltonian evaluated with the DFT density
    // matrix, and is expanded in the DFT orbitals: build it before touching the
    // orbital coefficients.
    scf_data.mol.xc_data.dfa_compnt_scf=vec![];
    scf_io::SCF::generate_hf_hamiltonian_ri_v_dm_only(scf_data,mpi_operator);
    let previous_coeff=scf_data.eigenvectors[0].clone();
    let (energies,rotation)=rs_rotation_in_mo_basis(scf_data,full_space);
    // C_rs = C_ks * U
    let ao_dimensions=previous_coeff.size[0];
    let mut coefficients=MatrixFull::new([ao_dimensions,num_state],0.0);
    _dgemm(&previous_coeff,((0..ao_dimensions),(0..num_state)),'N',
           &rotation,((0..num_state),(0..num_state)),'N',
           &mut coefficients,((0..ao_dimensions),(0..num_state)),1.0,0.0);
    if scf_data.mol.ctrl.print_level>0{
        println!("RS orbital coefficients built: C_rs = C_ks * U, size=[{},{}]",ao_dimensions,num_state);
    }
    if scf_data.mol.ctrl.print_level>2{
        verify_rs_representation(scf_data,&coefficients,&rotation);
    }
    RenormalizedSinglesWithOrbitals{energies,coefficients,ks_coefficients:previous_coeff,rotation}
}

/// Debug/verification helper (`print_level > 2`).
///
/// Proves that the RI AO-to-MO transformation really consumes the installed RS
/// coefficients: it builds the full MO three-centre tensor and the full MO dipole
/// matrix in the Kohn-Sham basis, rebuilds them after installing the RS orbitals,
/// and verifies the exact relation between the two,
///
///   (pq|P)_RS = Σ_{rs} U_{rp} U_{sq} (rs|P)_KS ,   μ_RS = Uᵀ μ_KS U ,
///
/// restricted to the occupied/virtual blocks that the GW and BSE stages use.
/// Note that for a full-space RS rotation the sum must run over *all* Kohn-Sham
/// orbitals: an RS "occupied" orbital contains Kohn-Sham virtual components.
///
/// A machine-zero residue shows that every quantity built after
/// `install_rs_orbitals` — v, W and therefore the BSE transition amplitudes — is
/// expanded in the RS MO basis and not in the Kohn-Sham one.  Skipped for large
/// systems, where materialising the full MO tensor would be wasteful.
fn verify_rs_representation(scf_data:&mut SCF,coefficients:&MatrixFull<f64>,rotation:&MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'N');
    if occ_size==0||vir_size==0{return}
    if num_state>64{
        println!("RS representation check skipped (num_state={} > 64).",num_state);
        return;
    }
    // Full MO tensors in the Kohn-Sham basis.
    let full_ks=ri_bse::get_submatrix(scf_data,'F','F','N');
    let ao_dip=ri_bse::dipoles::obtain_ao_dips(scf_data,None);
    let c_ks=scf_data.eigenvectors[0].clone();
    let dipole_ks_full:Vec<[f64;3]>=(0..num_state).flat_map(|j|(0..num_state).map(move |b|(j,b)))
        .map(|(j,b)|ri_bse::dipoles::obtain_mu_ia(&c_ks,&ao_dip,j,b)).collect();
    // Same tensors after installing the renormalized-singles orbitals.
    scf_data.eigenvectors[0]=coefficients.clone();
    let full_rs=ri_bse::get_submatrix(scf_data,'F','F','N');
    let dipole_rs=ri_bse::dipoles::compute_dipole_matrix(scf_data);
    scf_data.eigenvectors[0]=c_ks;
    let num_auxbas=full_ks.size[0];
    let mut max_residue_ri=0.0_f64;
    let mut max_residue_dipole=0.0_f64;
    let mut reference_ri=0.0_f64;
    let mut reference_dipole=0.0_f64;
    for i in 0..occ_size{
        for a in 0..vir_size{
            let mut rotated_dipole=[0.0_f64;3];
            for p in 0..num_auxbas{
                let mut rotated_ri=0.0;
                for j in 0..num_state{
                    for b in 0..num_state{
                        rotated_ri+=rotation[[j,i]]*rotation[[b,occ_size+a]]*full_ks[[p,j+b*num_state]];
                    }
                }
                let reference=full_rs[[p,i+(occ_size+a)*num_state]];
                max_residue_ri=max_residue_ri.max((rotated_ri-reference).abs());
                reference_ri=reference_ri.max(reference.abs());
            }
            for j in 0..num_state{
                for b in 0..num_state{
                    let weight=rotation[[j,i]]*rotation[[b,occ_size+a]];
                    let mu=&dipole_ks_full[j*num_state+b];
                    for x in 0..3{rotated_dipole[x]+=weight*mu[x];}
                }
            }
            for x in 0..3{
                let reference=dipole_rs[[x,i+a*occ_size]];
                max_residue_dipole=max_residue_dipole.max((rotated_dipole[x]-reference).abs());
                reference_dipole=reference_dipole.max(reference.abs());
            }
        }
    }
    println!("RS representation check (RS orbitals vs rotated Kohn-Sham quantities):");
    println!("  max|(ia|P)_RS - Σ U U (rs|P)_KS| = {:.3e}  (reference magnitude {:.3e})",max_residue_ri,reference_ri);
    println!("  max|mu_ia_RS  - U^T mu_KS U|     = {:.3e}  (reference magnitude {:.3e})",max_residue_dipole,reference_dipole);
    println!("  max|U^T U - I|                   = {:.3e}",max_orthogonality_error(rotation));
}

/// Write the renormalized-singles orbitals to `rs_orbitals.dat` (current working
/// directory) so that quantities expressed in the RS MO basis — the BSE
/// transition amplitudes and NTOs above all — can be transformed back to the
/// Kohn-Sham/AO basis.
///
/// Three blocks are written: the Kohn-Sham coefficients `C_ks`, the RS
/// coefficients `C_rs = C_ks U`, and the orthogonal rotation `U`.  With `U` the
/// BSE amplitudes (which live in the RS `(i,a)` product basis) map back to the
/// Kohn-Sham basis through `A_KS = U_o A_RS U_vᵀ`.
///
/// Layout: comment lines start with `#`; every other line is whitespace
/// separated data (readable with `numpy.loadtxt(..., comments="#")` in order).
///
/// ```text
/// # nao nmo
/// nao nmo
/// # rs_eigenvalues (nmo, Hartree)
/// e_rs ...
/// # C_rs (nmo rows, nao columns; one MO per row, AO index fastest)
/// ...
/// # ks_eigenvalues (nmo, Hartree)
/// e_ks ...
/// # C_ks (nmo rows, nao columns)
/// ...
/// # rotation U (nmo rows, nmo columns); C_rs = C_ks * U
/// ...
/// ```
pub fn save_rs_orbitals(coefficients:&MatrixFull<f64>,previous_coeff:&MatrixFull<f64>,
                        rotation:&MatrixFull<f64>,
                        rs_energies:&Vec<f64>,ks_energies:&Vec<f64>,path:&str){
    use std::io::Write;
    let num_ao=coefficients.size[0];
    let num_mo=coefficients.size[1];
    let mut buffer=String::new();
    let join=|values:&[f64]|values.iter().map(|v|format!("{:.16e}",v)).collect::<Vec<_>>().join(" ");
    buffer.push_str("# REST renormalized-singles orbitals: C_rs = C_ks * U\n");
    buffer.push_str("# nao nmo\n");
    buffer.push_str(&format!("{}\t{}\n",num_ao,num_mo));
    buffer.push_str("# rs_eigenvalues (Hartree)\n");
    buffer.push_str(&format!("{}\n",join(&rs_energies[0..num_mo.min(rs_energies.len())])));
    buffer.push_str("# C_rs (nmo rows, nao columns)\n");
    for i in 0..num_mo{
        let column:Vec<f64>=(0..num_ao).map(|mu|coefficients[[mu,i]]).collect();
        buffer.push_str(&format!("{}\n",join(&column)));
    }
    buffer.push_str("# ks_eigenvalues (Hartree)\n");
    buffer.push_str(&format!("{}\n",join(&ks_energies[0..num_mo.min(ks_energies.len())])));
    buffer.push_str("# C_ks (nmo rows, nao columns)\n");
    for i in 0..num_mo{
        let column:Vec<f64>=(0..num_ao).map(|mu|previous_coeff[[mu,i]]).collect();
        buffer.push_str(&format!("{}\n",join(&column)));
    }
    buffer.push_str("# rotation U (nmo rows, nmo columns)\n");
    for j in 0..rotation.size[0]{
        let row:Vec<f64>=(0..rotation.size[1]).map(|i|rotation[[j,i]]).collect();
        buffer.push_str(&format!("{}\n",join(&row)));
    }
    match std::fs::File::create(path){
        Ok(mut file)=>{
            if let Err(err)=file.write_all(buffer.as_bytes()){
                println!("Warning: could not write the RS orbitals to {}: {}",path,err);
            }else{
                println!("The renormalized-singles orbital coefficients have been written to {}",path);
                println!("  (blocks: nao/nmo, e_rs, C_rs, e_ks, C_ks, U = C_ks^T C_rs; see the file header)");
            }
        },
        Err(err)=>println!("Warning: could not create {}: {}",path,err),
    }
}

/// Install RS orbital coefficients into the SCF orbital coefficient matrix.
///
/// Every subsequent RI AO2MO step (`SCF::generate_ri3mo_rayon_for_multiple_times`,
/// `SCF::generate_ri3mo_bse`, `v_matrix_from_scf`, ...) reads
/// `scf_data.eigenvectors` at call time, so this single update is what makes the
/// GW run use the RS orbitals.  The DFT density matrix is intentionally left
/// untouched: it is the density matrix that defines the RS Hamiltonian.
pub fn install_rs_orbitals(scf_data:&mut SCF,coefficients:&MatrixFull<f64>){
    let target=&mut scf_data.eigenvectors[0];
    if target.size!=coefficients.size{
        panic!("RS orbital coefficients have size {:?}, but the SCF orbital coefficient matrix has size {:?}",coefficients.size,target.size);
    }
    let (num_row,num_col)=(coefficients.size[0],coefficients.size[1]);
    for mu in 0..num_row{
        for i in 0..num_col{
            target[[mu,i]]=coefficients[[mu,i]];
        }
    }
    println!("The RI three-centre integrals will be recombined from the AO basis with the renormalized-singles orbital coefficients.");
}