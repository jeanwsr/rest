use super::{SCF, SCFType};
use crate::constants::{SQRT_THRESHOLD};
use crate::mpi_io::MPIOperator;
use rayon::prelude::*;
use tensors::matrix_blas_lapack::{_dgemm_full, _dsymm, _pinv, _get_sqrt_and_inv_sqrt};
use tensors::{BasicMatrix, MathMatrix, MatrixFull, MatrixUpper, TensorOpt, TensorOptMut, TensorSlice};
use log::{debug};


#[derive(Clone)]
pub struct ScfTraceRecord {
    pub num_iter: usize,
    pub mixer: String,
    // the maximum number of stored residual densities
    pub num_max_records: usize,
    pub mix_param: f64,
    pub start_diis_cycle: usize,
    //pub scf_energy : Vec<f64>,
    //pub density_matrix: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvectors: Vec<[MatrixFull<f64>;2]>,
    //pub eigenvalues: Vec<[Vec<f64>;2]>,
    pub scf_energy : f64,
    pub smearing_entropy: f64,
    pub energy_records: Vec<f64>,
    pub prev_hamiltonian: Vec<[MatrixUpper<f64>;2]>,
    pub eigenvectors: [MatrixFull<f64>;2],
    pub eigenvalues: [Vec<f64>;2],
    pub density_matrix: [Vec<MatrixFull<f64>>;2],
    pub target_vector: Vec<[MatrixFull<f64>;2]>,
    pub error_vector: Vec<Vec<f64>>,
    pub sqrt_inv_ovlp: Option<MatrixFull<f64>>,
    pub ediis_density: Vec<[MatrixFull<f64>;2]>,
    pub ediis_energy: Vec<f64>,
    /// Separate Fock history for EDIIS/ADIIS B-matrix, independent from DIIS
    /// target_vector. This allows clearing DIIS history (when transitioning
    /// from EDIIS to DIIS) without losing the EDIIS B-matrix data.
    pub ediis_fock: Vec<[MatrixFull<f64>;2]>,
    /// Flag set when prepare_next_input applied density-matrix extrapolation
    /// (EDIIS/ADIIS) for HF: D_mixed is the next density, skip generate_density_matrix.
    pub extrapolated_density: bool,
    /// Flag set when prepare_next_input applied density-matrix extrapolation
    /// (EDIIS/ADIIS) for DFT: after generate_density_matrix (Roothaan step
    /// from D_mixed → D_idem), rebuild F(D_idem) for correct physical energy.
    pub needs_fock_rebuild: bool,
    /// Persistent flag: true if the density entering prepare_next_input is a
    /// non-idempotent EDIIS/ADIIS extrapolated density. Used to clear DIIS
    /// history before it gets contaminated.
    pub prev_density_extrapolated: bool,
}

impl ScfTraceRecord {
    pub fn new(num_max_records: usize, mix_param: f64, mixer: String,start_diis_cycle: usize) -> ScfTraceRecord {
        if num_max_records==0 {
            println!("Error: num_max_records cannot be 0");
        }
        ScfTraceRecord {
            num_iter: 0,
            mixer,
            mix_param,
            num_max_records,
            start_diis_cycle,
            scf_energy : 0.0,
            smearing_entropy: 0.0,
            energy_records: vec![],
            prev_hamiltonian: vec![[MatrixUpper::empty(),MatrixUpper::empty()]],
            eigenvectors: [MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
            eigenvalues: [Vec::<f64>::new(),Vec::<f64>::new()],
            density_matrix: [vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)],
                             vec![MatrixFull::new([1,1],0.0),
                              MatrixFull::new([1,1],0.0)]],
            target_vector: Vec::<[MatrixFull<f64>;2]>::new(),
            error_vector: Vec::<Vec::<f64>>::new(),
            sqrt_inv_ovlp: None,
            ediis_density: Vec::<[MatrixFull<f64>;2]>::new(),
            ediis_energy: Vec::<f64>::new(),
            ediis_fock: Vec::<[MatrixFull<f64>;2]>::new(),
            extrapolated_density: false,
            needs_fock_rebuild: false,
            prev_density_extrapolated: false,
        }
    }
    pub fn initialize(scf: &SCF) -> ScfTraceRecord {
        /// 
        /// Initialize the scf records which should be involked after the initial guess 
        /// 
        let mut tmp_records = ScfTraceRecord::new(
            scf.mol.ctrl.num_max_diis, 
            scf.mol.ctrl.mix_param.clone(), 
            scf.mol.ctrl.mixer.clone(),
            scf.mol.ctrl.start_diis_cycle.clone()
        );
        tmp_records.scf_energy=scf.scf_energy;
        tmp_records.eigenvectors=scf.eigenvectors.clone();
        tmp_records.eigenvalues=scf.eigenvalues.clone();
        tmp_records.density_matrix=[scf.density_matrix.clone(),scf.density_matrix.clone()];
        let ovlp_full = scf.ovlp.to_matrixfull().unwrap();
        if let Some((_sqrt_a, sqrt_inv_a, _rank)) = _get_sqrt_and_inv_sqrt(&ovlp_full, SQRT_THRESHOLD) {
            tmp_records.sqrt_inv_ovlp = Some(sqrt_inv_a);
        }
        if tmp_records.mixer.eq(&"ediis") || tmp_records.mixer.eq(&"ediis+diis") || tmp_records.mixer.eq(&"adiis+diis") {
            tmp_records.ediis_density.push([scf.density_matrix[0].clone(),
                if scf.mol.spin_channel>1 { scf.density_matrix[1].clone() }
                else { MatrixFull::empty() }]);
            tmp_records.ediis_energy.push(ediis_e0(scf.scf_energy,
                scf.smearing_entropy, Some(scf.current_smear_sigma)));
            tmp_records.ediis_fock.push([scf.hamiltonian[0].to_matrixfull().unwrap(),
                if scf.mol.spin_channel>1 { scf.hamiltonian[1].to_matrixfull().unwrap() }
                else { MatrixFull::empty() }]);
        }
        tmp_records
    }
    /// This subroutine updates:  
    ///     scf_energy:         from previous to the current value  
    ///     scf_eigenvalues:    from previous to the current value  
    ///     scf_eigenvectors:   from previous to the current value  
    ///     scf_density_matrix: [pre, cur]  
    ///     num_iter  
    /// This subroutine should be called after [`scf.check_scf_convergence`] and before [`self.prepare_next_input`]"
    pub fn update(&mut self, scf: &SCF) {

        let spin_channel = scf.mol.spin_channel;

        // now store the scf energy, eigenvectors and eigenvalues of the last two steps
        //let tmp_data =  self.scf_energy[1].clone();
        self.scf_energy=scf.scf_energy;
        self.smearing_entropy=scf.smearing_entropy;
        //let tmp_data =  self.eigenvectors[1].clone();
        self.eigenvectors=scf.eigenvectors.clone();
        //let tmp_data =  self.eigenvalues[1].clone();
        self.eigenvalues=scf.eigenvalues.clone();
        let tmp_data =  self.density_matrix[1].clone();
        self.density_matrix=[tmp_data,scf.density_matrix.clone()];

        self.num_iter +=1;
    }
    /// This subroutine prepares the fock matrix for the next step according different mixing algorithm  
    ///
    /// self.mixer =  
    /// * "direct": the output density matrix in the current step `n0[out]` will be used directly 
    ///             to generate the the input fock matrix of the next step
    /// * "linear": the density matrix used in the next step `n1[in]` is a mix between
    ///             the input density matrix in the current step `n0[in]` and `n0[out]`  
    ///             <span style="text-align:right">`n1[in] = alpha*n0[out] + (1-alpha)*n0[in]`</span>  
    ///             <span style="text-align:right">`       = n0[in] + alpha * Rn0            ` </span>  
    ///             where alpha the mixing parameter obtained from self.mix_param
    ///              and `Rn0 = n0[out]-n0[in]` is the density matrix change in the current step.
    ///             `n1[in]` is then be used to generate the input fock matrix of the next step
    /// * "diis":   the input fock matrix of the next step `f1[in] = sum_{i} c_i*f_i[in]`,
    ///            where `f_i[in]` is the input fock matrix of the ith step and 
    ///            c_i is obtained by the diis altogirhm against the error vector
    ///            of the commutator `(f_i[out]*d_i[out]*s-s*d_i[out]*f_i[out])`, where  
    ///            - `f_i[out]` is the ith output fock matrix,   
    ///            - `d_i[out]` is the ith output density matrix,  
    ///            - `s` is the overlap matrix  
    /// * **Ref**: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
    ///
    pub fn prepare_next_input(&mut self, scf: &mut SCF, mpi_operator: &Option<MPIOperator>) {
        let spin_channel = scf.mol.spin_channel;
        let start_pulay = self.start_diis_cycle;
        let alpha = self.mix_param;
        let beta = 1.0-self.mix_param;
        let mut level_shift_applied = false;
        if self.mixer.eq(&"direct") {
            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();
        }
        else if self.mixer.eq(&"linear") 
            || (self.mixer.eq(&"diis") && self.num_iter<start_pulay) 
        {
            debug!("using linear mixer");
            let mut alpha = self.mix_param;
            let mut beta = 1.0-alpha;
            // n1[in] = a*n0[out] + (1-a)*n0[in] = n0[out]-(1-a)*Rn0 = n0[in] + a*Rn0
            // Rn0 = n0[out]-n0[in]; the residual density in the current iteration
            // n1[in] is the input density for the next iteration
            for i_spin in (0..spin_channel) {
                let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                    .scaled_add(&residual_dm, alpha)
                    .unwrap();
            }
            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();
        } else if self.mixer.eq(&"diis") && self.num_iter>=start_pulay {
            debug!("using diis mixer");
            // 
            // Reference: P. Pulay, Improved SCF Convergence Acceleration, JCC, 1982, 3:556-560.
            // 
            let start_dim = 0usize;
            let mut start_check_oscillation = scf.mol.ctrl.start_check_oscillation;
            //
            // prepare the fock matrix according to the output density matrix of the previous step
            //
            let dt1 = time::Local::now();

            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();

            // Apply level_shift to the output fock matrix BEFORE DIIS target storage.
            // This ensures DIIS operates in the level-shifted subspace, so the DIIS
            // extrapolated Fock matrix is already level-shifted (no post-hoc correction).
            if let Some(level_shift_val) = scf.mol.ctrl.level_shift {
                let ovlp = &scf.ovlp;
                let dm_scaling_factor = match scf.scftype {
                    SCFType::RHF | SCFType::ROHF => 0.5,
                    SCFType::UHF => 1.0,
                };
                match scf.scftype {
                    SCFType::RHF => {
                        let mut fock = scf.hamiltonian.get_mut(0).unwrap();
                        let dm = scf.density_matrix.get(0).unwrap();
                        level_shift_fock(fock, ovlp, level_shift_val, dm, dm_scaling_factor);
                    },
                    SCFType::UHF => {
                        for i_spin in 0..scf.mol.spin_channel {
                            let mut fock = scf.hamiltonian.get_mut(i_spin).unwrap();
                            let dm = scf.density_matrix.get(i_spin).unwrap();
                            level_shift_fock(fock, ovlp, level_shift_val, dm, dm_scaling_factor);
                        }
                    },
                    SCFType::ROHF => {
                        let fock = scf.roothaan_hamiltonian.as_mut().unwrap();
                        let dm = scf.density_matrix[0].clone() + scf.density_matrix[1].clone();
                        level_shift_fock(fock, ovlp, level_shift_val, &dm, dm_scaling_factor);
                    }
                }
                level_shift_applied = true;
            }


            // update the energy records and check the oscillation
            let e_free = if scf.mol.ctrl.smear.is_some() {
                let sigma = scf.current_smear_sigma;
                scf.scf_energy - sigma * scf.smearing_entropy
            } else {
                scf.scf_energy
            };
            self.energy_records.push(e_free);
            let num_step = self.energy_records.len();
            let oscillation_flag = if num_step >=2 {
                let change_1 = self.energy_records[num_step-1] - self.energy_records[num_step-2];
                num_step > start_check_oscillation && change_1 > 0.0
            }else {
                false
            };

            let dt2 = time::Local::now();


            // check if the storage of fock matrix reaches the maximum setting
            if self.target_vector.len() == self.num_max_records {
                self.target_vector.remove(0);
                self.error_vector.remove(0);
            };

            //
            // prepare and store the fock matrix in full formate and the error vector in the current step
            //
            //for i_spin in (0..spin_channel) {
            //    //self.target_vector.push([scf.hamiltonian[i_spin].clone(), scf.hamiltonian[i_spin].clone()]);
            //    scf.hamiltonian[i_spin].formated_output(5, "upper");
            //}
            let (cur_error_vec, cur_target) = generate_diis_error_vector(&scf.hamiltonian, &scf.ovlp, &mut self.density_matrix, spin_channel, &self.sqrt_inv_ovlp);
            self.error_vector.push(cur_error_vec);
            self.target_vector.push(cur_target);


            // solve the DIIS against the error vector
            if let Some(coeff) = diis_solver(&self.error_vector, &self.error_vector.len()) {
                // now extrapolate the fock matrix for the next step
                (0..spin_channel).into_iter().for_each(|i_spin| {
                    let mut next_hamiltonian = MatrixFull::new(self.target_vector[0][i_spin].size.clone(),0.0);
                    coeff.iter().enumerate().for_each(|(i,value)| {
                        next_hamiltonian.self_scaled_add(&self.target_vector[i+start_dim][i_spin], *value);
                    });
                    let next_hamiltonian = next_hamiltonian.to_matrixupper();

                    if oscillation_flag {
                        if scf.mol.ctrl.print_level>0 {
                            println!("Energy increase is detected. Turn on the linear mixing algorithm with (H[DIIS, i-1] + H[DIIS, i+1]).");
                            let length = self.energy_records.len();
                            println!("Prev_Energies: ({:16.8}, {:16.8})", self.energy_records[length-2], self.energy_records[length-1]);
                        }
                        let mut alpha: f64 = self.mix_param;
                        let mut beta = 1.0-alpha;
                        scf.hamiltonian[i_spin].data.par_iter_mut().zip(self.prev_hamiltonian[0][i_spin].data.par_iter()).zip(next_hamiltonian.data.par_iter())
                        .for_each(|((to, prev), new)| {
                            *to = prev*beta + new*alpha;
                        });
                    } else {
                        scf.hamiltonian[i_spin] = next_hamiltonian;
                    }
                });

                // update the previous hamiltonian list to make sure the first item is H[DIIS, i-1]
                // and the second term is H[DIIS, i]
                if self.prev_hamiltonian.len() == 2 {self.prev_hamiltonian.remove(0);};
                self.prev_hamiltonian.push(scf.hamiltonian.clone());



            } else {
                let mut alpha = self.mix_param;
                let mut beta = 1.0-alpha;
                if scf.mol.ctrl.print_level>0 {
                    println!("WARNING: fail to obtain the DIIS coefficients. Turn to use the linear mixing algorithm, and re-invoke DIIS  8 steps later");
                }
                for i_spin in (0..spin_channel) {
                    let residual_dm = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                    scf.density_matrix[i_spin] = self.density_matrix[0][i_spin]
                        .scaled_add(&residual_dm, alpha)
                        .unwrap();
                }
                scf.generate_hf_hamiltonian(mpi_operator);
                level_shift_applied = false;
                self.start_diis_cycle = self.num_iter + 8;
                self.target_vector =  Vec::<[MatrixFull<f64>;2]>::new();
                self.error_vector =  Vec::<Vec::<f64>>::new();
            }
            let dt3 = time::Local::now();
            let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
            if scf.mol.ctrl.print_level>2 {
                println!("Hamiltonian: generation by {:10.2}s and DIIS extrapolation by {:10.2}s", timecost1,timecost2);
            }
            
        } else if self.mixer.eq(&"ediis") && self.num_iter>=start_pulay {
            // EDIIS: energy-DIIS per Kudin, Scuseria, Cancès, JCP 2002.
            // Fock mixing F̃ = ΣcᵢFᵢ is used for BOTH HF and DFT (the paper
            // shows DFT nonlinearity is negligible for convergence acceleration).
            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();
            let cur_density = [scf.density_matrix[0].clone(),
                if spin_channel>1 { scf.density_matrix[1].clone() } else { MatrixFull::empty() }];
            let cur_fock = [scf.hamiltonian[0].to_matrixfull().unwrap(),
                if spin_channel>1 { scf.hamiltonian[1].to_matrixfull().unwrap() } else { MatrixFull::empty() }];
            if self.ediis_density.len() == self.num_max_records {
                self.ediis_density.remove(0); self.ediis_energy.remove(0); self.ediis_fock.remove(0);
            }
            self.ediis_density.push(cur_density); self.ediis_energy.push(ediis_e0(scf.scf_energy,
                scf.smearing_entropy, Some(scf.current_smear_sigma)));
            self.ediis_fock.push(cur_fock);
            let nhist = self.ediis_density.len();
            if nhist >= 2 {
                let bmat = generate_ediis_penalty(&self.ediis_fock, &self.ediis_density, spin_channel);
                let eta = scf.mol.ctrl.ediis_penalty.unwrap_or(0.5);
                let n = bmat.size()[0]; let mut qmat = MatrixFull::new([n, n], 0.0);
                for i in 0..n { for j in 0..n { let bij = bmat.get2d([i, j]).unwrap_or(&0.0); qmat.set2d([i, j], -2.0*eta*bij); } }
                let coeff = ediis_qp_solver(&qmat, &self.ediis_energy, eta);
                // Fock mixing: F̃ = ΣcᵢFᵢ (paper Eq. 10).
                // The SCF loop diagonalizes F̃ → idempotent D_next.
                for i_spin in 0..spin_channel {
                    let mut next_h = MatrixFull::new(self.ediis_fock[0][i_spin].size.clone(), 0.0);
                    for (k, ck) in coeff.iter().enumerate() { 
                        next_h.self_scaled_add(&self.ediis_fock[k][i_spin], *ck); 
                    }
                    scf.hamiltonian[i_spin] = next_h.to_matrixupper();
                }
                if scf.mol.ctrl.print_level > 1 {
                    print!("EDIIS coeff: ["); for ck in &coeff { print!(" {:.4}", ck); } println!(" ]");
                }
            }
            level_shift_applied = false;
        } else if self.mixer.eq(&"ediis+diis") && self.num_iter>=start_pulay {
            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();
            if let Some(ls_val) = scf.mol.ctrl.level_shift {
                let dsf = match scf.scftype { SCFType::RHF|SCFType::ROHF => 0.5, SCFType::UHF => 1.0 };
                match scf.scftype {
                    SCFType::RHF => level_shift_fock(scf.hamiltonian.get_mut(0).unwrap(), &scf.ovlp, ls_val, scf.density_matrix.get(0).unwrap(), dsf),
                    SCFType::UHF => for s in 0..spin_channel { level_shift_fock(scf.hamiltonian.get_mut(s).unwrap(), &scf.ovlp, ls_val, scf.density_matrix.get(s).unwrap(), dsf) },
                    SCFType::ROHF => { let dm = scf.density_matrix[0].clone()+scf.density_matrix[1].clone(); level_shift_fock(scf.roothaan_hamiltonian.as_mut().unwrap(), &scf.ovlp, ls_val, &dm, dsf) }
                }
                level_shift_applied = true;
            }
            let cur_dens = [scf.density_matrix[0].clone(), if spin_channel>1 { scf.density_matrix[1].clone() } else { MatrixFull::empty() }];
            let cur_fock = [scf.hamiltonian[0].to_matrixfull().unwrap(), if spin_channel>1 { scf.hamiltonian[1].to_matrixfull().unwrap() } else { MatrixFull::empty() }];
            let max_rec = self.num_max_records;
            if self.target_vector.len() == max_rec { self.target_vector.remove(0); self.error_vector.remove(0); }
            if self.ediis_density.len() == max_rec { 
                self.ediis_density.remove(0); self.ediis_energy.remove(0); self.ediis_fock.remove(0); 
            }
            let (cur_err, cur_tgt) = generate_diis_error_vector(&scf.hamiltonian, &scf.ovlp, &mut self.density_matrix, spin_channel, &self.sqrt_inv_ovlp);
            self.error_vector.push(cur_err); self.target_vector.push(cur_tgt);
            self.ediis_density.push(cur_dens); self.ediis_energy.push(ediis_e0(scf.scf_energy,
                scf.smearing_entropy, Some(scf.current_smear_sigma)));
            self.ediis_fock.push(cur_fock);
            let num_diis = self.error_vector.len(); let num_ediis = self.ediis_density.len();
            let diis_norm = self.error_vector.last().map(|v| v.iter().map(|x| x*x).sum::<f64>().sqrt()).unwrap_or(1.0);
            let use_ediis = num_ediis >= 2 && (num_diis < 2 || diis_norm > 1e-3);
            let mut ediis_used = false;
            if use_ediis && num_ediis >= 2 {
                let bmat = generate_ediis_penalty(&self.ediis_fock, &self.ediis_density, spin_channel);
                let eta = scf.mol.ctrl.ediis_penalty.unwrap_or(0.5);
                let n = bmat.size()[0]; let mut qmat = MatrixFull::new([n, n], 0.0);
                for i in 0..n { for j in 0..n { let bij = bmat.get2d([i, j]).unwrap_or(&0.0); qmat.set2d([i, j], -2.0*eta*bij); } }
                let coeff = ediis_qp_solver(&qmat, &self.ediis_energy, eta);
                let mut e_pred = 0.0f64;
                for i in 0..n { e_pred += coeff[i] * self.ediis_energy[i]; }
                for i in 0..n { for j in 0..n {
                    let bij = bmat.get2d([i, j]).unwrap_or(&0.0);
                    e_pred -= eta * coeff[i] * coeff[j] * bij;
                }}
                let e0_cur = ediis_e0(scf.scf_energy, scf.smearing_entropy, Some(scf.current_smear_sigma));
                let ediis_ok = e_pred <= e0_cur + 1e-10;
                if !ediis_ok && scf.mol.ctrl.print_level > 1 {
                    println!("[EDIIS] rejected: E_pred={:14.8} > E0_cur={:14.8}", e_pred, e0_cur);
                }
                if ediis_ok {
                    for i_spin in 0..spin_channel {
                        let mut next_h = MatrixFull::new(self.ediis_fock[0][i_spin].size.clone(), 0.0);
                        for (k, ck) in coeff.iter().enumerate() { next_h.self_scaled_add(&self.ediis_fock[k][i_spin], *ck); }
                        scf.hamiltonian[i_spin] = next_h.to_matrixupper();
                    }
                    if scf.mol.ctrl.print_level > 1 { print!("[EDIIS] coeff:"); for ck in &coeff { print!(" {:.4}", ck); } println!(); }
                    ediis_used = true;
                }
            }
            if !ediis_used && num_diis >= 2 {
                if let Some(coeff) = diis_solver(&self.error_vector, &num_diis) {
                    for i_spin in 0..spin_channel {
                        let mut next_h = MatrixFull::new(self.target_vector[0][i_spin].size.clone(), 0.0);
                        for (k, ck) in coeff.iter().enumerate() { next_h.self_scaled_add(&self.target_vector[k][i_spin], *ck); }
                        scf.hamiltonian[i_spin] = next_h.to_matrixupper();
                    }
                    if scf.mol.ctrl.print_level > 1 { print!("[DIIS] coeff:"); for ck in &coeff { print!(" {:.4}", ck); } println!(); }
                } else {
                    for i_spin in 0..spin_channel {
                        let rd = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                        scf.density_matrix[i_spin] = self.density_matrix[0][i_spin].scaled_add(&rd, alpha).unwrap();
                    }
                    scf.generate_hf_hamiltonian(mpi_operator);
                    level_shift_applied = false;
                    self.target_vector.clear(); self.error_vector.clear();
                    self.ediis_density.clear(); self.ediis_energy.clear(); self.ediis_fock.clear();
                }
            }
        } else if self.mixer.eq(&"adiis+diis") && self.num_iter>=start_pulay {
            scf.generate_hf_hamiltonian(mpi_operator);
            scf.grad_dm = scf.get_grad_dm();
            if let Some(ls_val) = scf.mol.ctrl.level_shift {
                let dsf = match scf.scftype { SCFType::RHF|SCFType::ROHF => 0.5, SCFType::UHF => 1.0 };
                match scf.scftype {
                    SCFType::RHF => level_shift_fock(scf.hamiltonian.get_mut(0).unwrap(), &scf.ovlp, ls_val, scf.density_matrix.get(0).unwrap(), dsf),
                    SCFType::UHF => for s in 0..spin_channel { level_shift_fock(scf.hamiltonian.get_mut(s).unwrap(), &scf.ovlp, ls_val, scf.density_matrix.get(s).unwrap(), dsf) },
                    SCFType::ROHF => { let dm = scf.density_matrix[0].clone()+scf.density_matrix[1].clone(); level_shift_fock(scf.roothaan_hamiltonian.as_mut().unwrap(), &scf.ovlp, ls_val, &dm, dsf) }
                }
                level_shift_applied = true;
            }
            let cur_dens = [scf.density_matrix[0].clone(), if spin_channel>1 { scf.density_matrix[1].clone() } else { MatrixFull::empty() }];
            let cur_fock = [scf.hamiltonian[0].to_matrixfull().unwrap(), if spin_channel>1 { scf.hamiltonian[1].to_matrixfull().unwrap() } else { MatrixFull::empty() }];
            let max_rec = self.num_max_records;
            if self.target_vector.len() == max_rec { self.target_vector.remove(0); self.error_vector.remove(0); }
            if self.ediis_density.len() == max_rec { self.ediis_density.remove(0); self.ediis_energy.remove(0); self.ediis_fock.remove(0); }
            let (cur_err, cur_tgt) = generate_diis_error_vector(&scf.hamiltonian, &scf.ovlp, &mut self.density_matrix, spin_channel, &self.sqrt_inv_ovlp);
            self.error_vector.push(cur_err); self.target_vector.push(cur_tgt);
            self.ediis_density.push(cur_dens.clone()); self.ediis_energy.push(ediis_e0(scf.scf_energy,
                scf.smearing_entropy, Some(scf.current_smear_sigma)));
            self.ediis_fock.push(cur_fock);
            let num_diis = self.error_vector.len(); let num_ediis = self.ediis_density.len();
            let diis_norm = self.error_vector.last().map(|v| v.iter().map(|x| x*x).sum::<f64>().sqrt()).unwrap_or(1.0);
            let use_adiis = num_ediis >= 2 && (num_diis < 2 || diis_norm > 1e-3);
            let mut adiis_used = false;
            if use_adiis && num_ediis >= 2 {
                let bmat = generate_adiis_penalty(&self.ediis_density, &self.ediis_fock, spin_channel);
                let mu = scf.mol.ctrl.adiis_penalty.unwrap_or(0.5);
                let bmax = bmat.data.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
                let mu_eff = if bmax > 1e-10 { mu / bmax } else { mu };
                let n = bmat.size()[0]; let mut qmat = MatrixFull::new([n, n], 0.0);
                for i in 0..n { for j in 0..n { let bij = bmat.get2d([i, j]).unwrap_or(&0.0); qmat.set2d([i, j], mu_eff*bij); } }
                let coeff = ediis_qp_solver(&qmat, &self.ediis_energy, mu_eff);
                let mut e_pred = 0.0f64;
                for i in 0..n { e_pred += coeff[i] * self.ediis_energy[i]; }
                for i in 0..n { for j in 0..n {
                    let bij = bmat.get2d([i, j]).unwrap_or(&0.0);
                    e_pred += 0.5 * mu_eff * coeff[i] * coeff[j] * bij;
                }}
                let e0_cur = ediis_e0(scf.scf_energy, scf.smearing_entropy, Some(scf.current_smear_sigma));
                let adiis_ok = e_pred <= e0_cur + 1e-8;
                if !adiis_ok && scf.mol.ctrl.print_level > 1 {
                    println!("[ADIIS] rejected: E_pred={:14.8} > E0_cur={:14.8}", e_pred, e0_cur);
                }
                if adiis_ok {
                    for i_spin in 0..spin_channel {
                        let mut next_h = MatrixFull::new(self.ediis_fock[0][i_spin].size.clone(), 0.0);
                        for (k, ck) in coeff.iter().enumerate() { next_h.self_scaled_add(&self.ediis_fock[k][i_spin], *ck); }
                        scf.hamiltonian[i_spin] = next_h.to_matrixupper();
                    }
                    if scf.mol.ctrl.print_level > 1 { print!("[ADIIS] coeff:"); for ck in &coeff { print!(" {:.4}", ck); } println!(); }
                    adiis_used = true;
                }
            }
            if !adiis_used && num_diis >= 2 {
                if let Some(coeff) = diis_solver(&self.error_vector, &num_diis) {
                    for i_spin in 0..spin_channel {
                        let mut next_h = MatrixFull::new(self.target_vector[0][i_spin].size.clone(), 0.0);
                        for (k, ck) in coeff.iter().enumerate() { next_h.self_scaled_add(&self.target_vector[k][i_spin], *ck); }
                        scf.hamiltonian[i_spin] = next_h.to_matrixupper();
                    }
                    if scf.mol.ctrl.print_level > 1 { print!("[DIIS] coeff:"); for ck in &coeff { print!(" {:.4}", ck); } println!(); }
                } else {
                    for i_spin in 0..spin_channel {
                        let rd = self.density_matrix[1][i_spin].sub(&self.density_matrix[0][i_spin]).unwrap();
                        scf.density_matrix[i_spin] = self.density_matrix[0][i_spin].scaled_add(&rd, alpha).unwrap();
                    }
                    scf.generate_hf_hamiltonian(mpi_operator);
                    level_shift_applied = false;
                    self.target_vector.clear(); self.error_vector.clear();
                    self.ediis_density.clear(); self.ediis_energy.clear(); self.ediis_fock.clear();
                }
            }
        };

        // now consider if level_shift is applied
        // at present only a constant level shift is implemented for both spin channels and for the whole SCF procedure
        // For the DIIS path, level_shift is applied inside the DIIS block (before target storage)
        // to ensure the DIIS subspace contains level-shifted Fock matrices.
        if scf.mol.ctrl.level_shift.is_some() && !level_shift_applied {
            let level_shift = scf.mol.ctrl.level_shift.unwrap();
            let ovlp = &scf.ovlp;
            let dm_scaling_factor = match scf.scftype {
                SCFType::RHF | SCFType::ROHF=> 0.5,
                SCFType::UHF => 1.0,
                };
                match scf.scftype {
                    SCFType::RHF => {
                        let mut fock = scf.hamiltonian.get_mut(0).unwrap();
                        let dm = scf.density_matrix.get(0).unwrap();
                        level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor);
                    },
                    SCFType::UHF => {
                        for i_spin in 0..scf.mol.spin_channel {
                            let mut fock = scf.hamiltonian.get_mut(i_spin).unwrap();
                            let dm = scf.density_matrix.get(i_spin).unwrap();
                            level_shift_fock(fock, ovlp, level_shift, dm, dm_scaling_factor);
                        }
                    },
                    SCFType::ROHF => {
                        let fock = scf.roothaan_hamiltonian.as_mut().unwrap();
                        let dm = scf.density_matrix[0].clone() + scf.density_matrix[1].clone();
                        level_shift_fock(fock, ovlp, level_shift, &dm, dm_scaling_factor);
                    }
                }
        }
    }

    pub fn refresh(&mut self) {
        self.energy_records.clear();
        self.prev_hamiltonian = vec![[MatrixUpper::empty(), MatrixUpper::empty()]];
        self.eigenvectors = [MatrixFull::new([1, 1], 0.0), MatrixFull::new([1, 1], 0.0)];
        self.eigenvalues = [Vec::<f64>::new(), Vec::<f64>::new()];
        self.density_matrix = [
            vec![MatrixFull::new([1, 1], 0.0), MatrixFull::new([1, 1], 0.0)],
            vec![MatrixFull::new([1, 1], 0.0), MatrixFull::new([1, 1], 0.0)],
        ];
        self.target_vector = Vec::<[MatrixFull<f64>; 2]>::new();
        self.error_vector = Vec::<Vec::<f64>>::new();
        self.ediis_density = Vec::<[MatrixFull<f64>; 2]>::new();
        self.ediis_energy = Vec::<f64>::new();
        self.ediis_fock = Vec::<[MatrixFull<f64>; 2]>::new();
        self.extrapolated_density = false;
        self.needs_fock_rebuild = false;
        self.prev_density_extrapolated = false;
    }
}

pub fn generate_diis_error_vector(hamiltonian: &[MatrixUpper<f64>;2], 
                                ovlp: &MatrixUpper<f64>, 
                                density_matrix: &mut [Vec<MatrixFull<f64>>;2],
                                spin_channel: usize,
                                sqrt_inv_ovlp: &Option<MatrixFull<f64>>) -> (Vec<f64>, [MatrixFull<f64>;2]) {
            let mut cur_error = [
                MatrixFull::new([1,1],0.0),
                MatrixFull::new([1,1],0.0)
            ];
            let mut cur_target = [hamiltonian[0].to_matrixfull().unwrap(),
                 hamiltonian[1].to_matrixfull().unwrap()];

            let mut full_ovlp = ovlp.to_matrixfull().unwrap();

            // now generte the error as the commutator of [fds-sdf]
            (0..spin_channel).into_iter().for_each(|i_spin| {
                cur_error[i_spin] = super::get_grad_dm(&cur_target[i_spin], &full_ovlp, &density_matrix[1][i_spin]);

                // transfer to an orthogonal basis to improve numerical conditioning
                if let Some(sinv) = sqrt_inv_ovlp {
                    let n = cur_error[i_spin].size()[0];
                    let mut tmp = MatrixFull::new([n, n], 0.0);
                    _dgemm_full(&cur_error[i_spin], 'N', sinv, 'N', &mut tmp, 1.0, 0.0);
                    let mut e_orth = MatrixFull::new([n, n], 0.0);
                    _dgemm_full(sinv, 'N', &tmp, 'N', &mut e_orth, 1.0, 0.0);
                    cur_error[i_spin] = e_orth;
                }
            });

            let mut norm = 0.0;
            (0..spin_channel).for_each(|i_spin| {
                let dd = cur_error[i_spin].data.par_iter().fold(|| 0.0, |acc, x| {
                    acc + x*x
                }).sum::<f64>();
                norm += dd
            });

            ([cur_error[0].data.clone(),cur_error[1].data.clone()].concat(),
            cur_target)

}

/// Compute zero-temperature extrapolated energy for EDIIS.
/// E₀ = E(T) − ½·σ·S  eliminates smearing-entropy bias in energy comparison.
pub fn ediis_e0(e_total: f64, smearing_entropy: f64, sigma: Option<f64>) -> f64 {
    match sigma {
        Some(s) if s > 0.0 => e_total - 0.5 * s * smearing_entropy,
        _ => e_total,
    }
}

/// Build the EDIIS penalty matrix B_{ij} = Tr[(D_i−D_j)(F_i−F_j)].
fn generate_ediis_penalty(
    target_vector: &[[MatrixFull<f64>; 2]],
    ediis_density: &[[MatrixFull<f64>; 2]],
    spin_channel: usize,
) -> MatrixFull<f64> {
    let n = target_vector.len();
    let mut bmat = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        for j in i..n {
            let mut tr = 0.0f64;
            for i_spin in 0..spin_channel {
                let dd = ediis_density[i][i_spin].sub(&ediis_density[j][i_spin]).unwrap();
                let df = target_vector[i][i_spin].sub(&target_vector[j][i_spin]).unwrap();
                tr += dd.data.iter().zip(df.data.iter()).map(|(a, b)| a * b).sum::<f64>();
            }
            bmat.set2d([i, j], tr);
            bmat.set2d([j, i], tr);
        }
    }
    bmat
}

/// Solve the EDIIS QP:  min ½cᵀQc + pᵀc  s.t. Σc=1, c≥0.
fn ediis_qp_solver(q: &MatrixFull<f64>, p: &[f64], _eta: f64) -> Vec<f64> {
    let n = p.len();
    if n <= 1 { return vec![1.0f64; n]; }
    let mut c = vec![1.0 / n as f64; n];
    for _iter in 0..200 {
        let mut g = p.to_vec();
        for i in 0..n {
            let row = q.get2d_slice([i, 0], n).unwrap();
            for j in 0..n { g[i] += row[j] * c[j]; }
        }
        let g_mean = g.iter().sum::<f64>() / n as f64;
        let g_proj: Vec<f64> = g.iter().map(|gi| gi - g_mean).collect();
        if g_proj.iter().map(|x| x*x).sum::<f64>().sqrt() < 1e-14 { break; }
        let d: Vec<f64> = g_proj.iter().map(|gi| -gi).collect();
        let mut amax = f64::INFINITY;
        for i in 0..n { if d[i] < -1e-14 { let am = -c[i]/d[i]; if am < amax { amax = am; } } }
        let mut dqd = 0.0f64; let mut gd = 0.0f64;
        for i in 0..n {
            let row = q.get2d_slice([i, 0], n).unwrap();
            let mut qd_i = 0.0f64;
            for j in 0..n { qd_i += row[j] * d[j]; }
            dqd += d[i] * qd_i; gd += d[i] * g[i];
        }
        let alpha = if dqd <= 0.0 { amax } else { (-gd/dqd).min(amax) };
        if alpha < 1e-14 { break; }
        for i in 0..n { c[i] += alpha * d[i]; }
    }
    for ci in c.iter_mut() { *ci = ci.max(0.0); }
    let sum: f64 = c.iter().sum();
    if sum > 0.0 { for ci in c.iter_mut() { *ci /= sum; } } else { c.fill(1.0/n as f64); }
    c
}

/// Build ADIIS penalty: B_{ij} = Tr[(D_i−D_ref)(F_j−F_ref)] using newest point as ref.
/// Positive-semidefinite → convex QP → smooth coefficients. Ref: Hu & Yang JCP 132, 054109 (2010).
fn generate_adiis_penalty(
    densities: &[[MatrixFull<f64>; 2]],
    focks: &[[MatrixFull<f64>; 2]],
    spin_channel: usize,
) -> MatrixFull<f64> {
    let n = densities.len();
    if n == 0 { return MatrixFull::new([0, 0], 0.0); }
    let iref = n - 1;  // newest point
    let mut bmat = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        for j in i..n {
            let mut tr = 0.0f64;
            for i_spin in 0..spin_channel {
                let dd_i = densities[i][i_spin].sub(&densities[iref][i_spin]).unwrap();
                let df_j = focks[j][i_spin].sub(&focks[iref][i_spin]).unwrap();
                tr += dd_i.data.iter().zip(df_j.data.iter()).map(|(a, b)| a * b).sum::<f64>();
            }
            bmat.set2d([i, j], tr);
            bmat.set2d([j, i], tr);
        }
    }
    bmat
}

pub fn diis_solver(em: &Vec<Vec<f64>>,
                   num_vec:&usize) -> Option<Vec<f64>> {

    let dim_vec = em.len();
    let start_dim = if (em.len()>=*num_vec) {em.len()-*num_vec} else {0};
    let dim = if (em.len()>=*num_vec) {*num_vec} else {em.len()};
    let mut coeff = Vec::<f64>::new();
    //let mut norm_rdm = [Vec::<f64>::new(),Vec::<f64>::new()];
    let mut odm = MatrixFull::new([1,1],0.0);
    //let mut inv_opta = MatrixFull::new([dim,dim],0.0);
    //let mut sum_inv_norm_rdm = [0.0,0.0];
    let mut sum_inv_norm_rdm = 0.0_f64;

    // now prepare the norm matrix of the residual density matrix
    let mut opta = MatrixFull::new([dim,dim],0.0);
    (start_dim..dim_vec).into_iter().for_each(|i| {
        (start_dim..dim_vec).into_iter().for_each(|j| {
            let mut inv_norm_rdm = em[i].iter()
                .zip(em[j].iter())
                .fold(0.0,|c,(d,e)| {c + d*e});
            opta.set2d([i-start_dim,j-start_dim],inv_norm_rdm);
            //sum_inv_norm_rdm += inv_norm_rdm;
        })
    });
    let inv_opta = _pinv(&opta, Some(1.0e-12f64)).unwrap();
    let sum_inv = inv_opta.data.iter().sum::<f64>();
    if sum_inv.abs() < 1.0e-6 {
        return None;
    }
    sum_inv_norm_rdm = sum_inv.powf(-1.0f64);
    sum_inv_norm_rdm = inv_opta.data.iter().sum::<f64>().powf(-1.0f64);

    // now prepare the coefficients for the pulay mixing
    coeff = vec![sum_inv_norm_rdm;dim];
    //coeff.iter().for_each(|i| {println!("coeff: {}",i)});
    (0..dim).zip(coeff.iter_mut()).for_each(|(i,value)| {
        //println!("{:?}",*value);
        *value *= inv_opta.get2d_slice([0,i], dim)
                 .unwrap()
                 .iter()
                 .sum::<f64>();
    });

    Some(coeff)

}

pub fn level_shift_fock(fock: &mut MatrixUpper<f64>, ovlp: &MatrixUpper<f64>, level_shift: f64,  dm: &MatrixFull<f64>, dm_scaling_factor: f64) {
    // FC = SCE
    // F' = F + SC \Lambda C^\dagger S
    // F' = F + LF * (S - SDS) 

    let num_basis = dm.size()[0];

    let mut tmp_s = MatrixFull::new([num_basis, num_basis], 0.0);
    tmp_s.iter_matrixupper_mut().unwrap().zip(ovlp.data.iter()).for_each(|(to, from)| {*to = *from});
    let mut tmp_s2 = tmp_s.clone();
    let mut tmp_s3 = tmp_s.clone();
    _dsymm(&tmp_s, dm, &mut tmp_s2, 'L', 'U', -dm_scaling_factor, 0.0);
    _dsymm(&tmp_s, &mut tmp_s2, &mut tmp_s3, 'R', 'U', 1.0, 1.0);
    fock.data.iter_mut().zip(tmp_s3.iter_matrixupper().unwrap()).for_each(|(to, from)| {*to += *from*level_shift});
}
