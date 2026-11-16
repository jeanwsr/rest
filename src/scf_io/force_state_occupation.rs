use std::collections::HashMap;

//use num_complex::ComplexFloat;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tensors::{matrix_blas_lapack::_dgemv, BasicMatrix, MatrixFull, MatrixUpper};

use crate::scf_io::SCFType;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ForceStateOccupation {
    //          file_index, global_start_index, num_basis, num_state, spin_channel
    ref_index: String,
    prev_state: usize,
    prev_spin: usize,
    target_spin: usize,
    force_occ: f64,
    curr_state: usize,
    force_check_min: usize,
    force_check_max: usize,
    ovlp: f64,
    /// Channel of the *current* orbitals in which the overlap search found
    /// `curr_state` (0 for a restricted calculation, and also the fallback when
    /// the requested target channel is empty). Rolled into `prev_spin` by
    /// [`ForceStateOccupation::commit_state`].
    curr_spin: usize,
}

/// A maximum-overlap reference that follows the calculation (`[ctrl]
/// mom_strategy = "step"`, the default).
///
/// `SCF::ref_eigenvectors` holds the wavefunction read from the chkfile/guessfile
/// and never changes, so a geometry optimization keeps comparing every step
/// against the first (Franck-Condon) geometry: that is the *initial* maximum
/// overlap method (IMOM). Committing the converged orbitals of each step as the
/// reference of the next one keeps the overlap comparison local along the path,
/// so an excited-state optimization does not lose the state as the geometry
/// moves away. The struct stands in for the map entry named `key`.
///
/// See docs/dscf_s1_geomopt_plan.md, section A2.
#[derive(Clone, Debug)]
pub struct MomRunningRef {
    /// the `ref_index` (chkfile/guessfile path) this reference stands in for
    pub key: String,
    /// orbitals of the last committed step, channel 0/1 (channel 1 may be empty)
    pub orbitals: [MatrixFull<f64>; 2],
    /// `[global_index, num_basis, num_state, spin_channel]`, same layout as the
    /// values of `SCF::ref_eigenvectors`
    pub dims: [usize; 4],
    /// number of the committed step (diagnostics only)
    pub step: usize,
}

impl MomRunningRef {
    /// Build the reference from a set of orbitals, with the canonical
    /// `[0, num_basis, num_state, spin_channel]` dims.
    pub fn new(key: String, orbitals: [MatrixFull<f64>; 2], num_basis: usize, num_state: usize, spin_channel: usize, step: usize) -> MomRunningRef {
        MomRunningRef {
            key,
            orbitals,
            dims: [0, num_basis, num_state, spin_channel],
            step,
        }
    }
}

impl ForceStateOccupation {
    pub fn init(ref_index: String,
                prev_state: usize, 
                prev_spin: usize,
                target_spin: usize,
                force_occ: f64, 
                force_check_min: usize,
                force_check_max: usize) -> ForceStateOccupation {
        ForceStateOccupation {
            // only the file_index is specified with this initialization
            ref_index,
            prev_state,
            prev_spin,
            target_spin,
            force_occ,
            force_check_min,
            force_check_max,
            curr_state: 0,
            ovlp: 0.0,
            curr_spin: 0,
        }
    }

    /// Fill in ref_index if it was left empty at parse time (no explicit reference given).
    /// When empty, the reference defaults to the file that provides the initial guess.
    pub fn normalize_ref_index(&mut self, default_ref: &str) {
        if self.ref_index.is_empty() {
            self.ref_index = default_ref.to_string();
        }
    }

    pub fn get_force_occ(&self) -> f64 {
        self.force_occ
    }
    pub fn get_occ_spin(&self) -> usize {
        self.prev_spin
    }

    pub fn get_check_window(&self) -> [usize;2] {
        [self.force_check_min,self.force_check_max]
    }

    /// The chkfile/guessfile that provides the reference this entry addresses.
    pub fn ref_index(&self) -> &str {
        &self.ref_index
    }
    pub fn get_prev_state(&self) -> usize {
        self.prev_state
    }
    pub fn get_curr_state(&self) -> usize {
        self.curr_state
    }
    pub fn get_curr_spin(&self) -> usize {
        self.curr_spin
    }
    pub fn get_overlap(&self) -> f64 {
        self.ovlp
    }

    /// Record the result of one overlap projection (which orbital of the current
    /// set carries the constraint, in which channel, with what overlap).
    ///
    /// The projection itself runs on a scratch copy of the constraints, so the
    /// caller writes the outcome back here; `SCF::mom_commit_reference` then rolls
    /// `prev_state` from it. Keeping the two sides in step matters: a commit that
    /// reads a stale `curr_state` moves the tracked orbital to the wrong index and
    /// the state collapses silently.
    pub fn set_selection(&mut self, curr_state: usize, curr_spin: usize, ovlp: f64) {
        self.curr_state = curr_state;
        self.curr_spin = curr_spin;
        self.ovlp = ovlp;
    }

    /// Roll the bookkeeping one step forward (the `step` MOM strategy).
    ///
    /// `curr_state` was found among the *current* orbitals, which have just
    /// become the new reference, so it is also the index to look for in the next
    /// comparison; the channel follows with it. Must be called together with the
    /// reference replacement (`SCF::mom_commit_reference`), otherwise the next
    /// overlap search looks for the wrong orbital.
    pub fn commit_state(&mut self) {
        self.prev_state = self.curr_state;
        self.prev_spin = self.curr_spin;
    }

    pub fn formated_output(&self) -> String {
        let mut output = String::new();
        output = format!(" Prev. State: {}; Prev. Spin: {}; Target Spin: {}; Force Occ. {:16.8}\n", self.prev_state, self.prev_spin, self.target_spin, self.force_occ);
        output = format!("{} State Window: ({}, {})\n", output, self.force_check_min, self.force_check_max);
        output = format!("{} Curr. State: {} with OVLP of {:16.8}", output, self.curr_state, self.ovlp);

        output
    }

    pub fn formated_output_check(&self) -> String {
        let mut output = String::new();
        output = format!(" Prev. State: {}; Prev. Spin: {}; Target Spin: {}; Force Occ. {:16.8}\n", self.prev_state, self.prev_spin, self.target_spin, self.force_occ);
        output = format!("{} State Window: ({}, {})\n", output, self.force_check_min, self.force_check_max);

        output
    }
}

pub fn adapt_occupation_with_force_projection(
    occupation: &mut [Vec<f64>;2], 
    homo: &mut [usize;2],
    lumo: &mut [usize;2],
    force_occ: &mut Vec<ForceStateOccupation>,
    scftype: &SCFType, 
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors: &HashMap<String,([MatrixFull<f64>;2],[usize;4])>,
    running_ref: Option<&MomRunningRef>
) {

    determine_force_projection_v02(force_occ, eigenvectors, ovlp, ref_eigenvectors, running_ref);

    force_occ.iter().for_each(|x| println!("{}", x.formated_output()));

    match scftype {
        SCFType::RHF => {
            let mut occ_s = occupation.get_mut(0).unwrap();
            let num_elec = occ_s.iter().sum::<f64>();
            let mut force_curr = vec![];

            let mut net_force_elec = force_occ.iter()
                .filter(|x| x.prev_spin==0)
                .map(|x| {
                    let original_occ = occ_s[x.curr_state];
                    occ_s[x.curr_state] = x.force_occ;
                    force_curr.push(x.curr_state);
                    original_occ - x.force_occ
                }).sum::<f64>();

            if net_force_elec > 1.0e-6 {
                occ_s.iter_mut().enumerate().for_each(|(i,x)| {
                    if *x<2.0 && net_force_elec > 1.0e-6 {
                        if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                            let tmp_xa = *x + net_force_elec;
                            let tmp_xb = 2.0 - *x;
                            *x = if tmp_xa <=2.0 {
                                net_force_elec = 0.0;
                                tmp_xa
                            } else {
                                net_force_elec -= tmp_xb;
                                2.0
                            }
                        };
                    }
                });
            } else if net_force_elec < -1.0e-6 {
                occ_s.iter_mut().enumerate().rev().for_each(|(i, x)| {
                    if *x>0.0 && net_force_elec < -1.0e-6 {
                        if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                            let tmp_xa = *x + net_force_elec;
                            let tmp_xb = *x;
                            *x = if tmp_xa >= 0.0 {
                                net_force_elec = 0.0;
                                tmp_xa
                            } else {
                                net_force_elec += tmp_xb;
                                0.0
                            }
                        }
                    }

                });
            }
            let mut i_homo = homo.get_mut(0).unwrap();
            *i_homo = occ_s.iter().enumerate()
                .filter(|(i,occ)| **occ >=1.0e-6)
                .map(|(i,occ)| i).max().unwrap();
            let mut i_lumo = lumo.get_mut(0).unwrap();
            //*i_lumo = *i_homo + 1;
            *i_lumo = occ_s.iter().enumerate()
                .filter(|(i,occ)| (**occ-2.0).abs() >=1.0e-6 )
                .map(|(i,occ)| i).min().unwrap();
        },
        _ => {
            for i_spin in (0..2) {
                let mut occ_s = occupation.get_mut(i_spin).unwrap();
                let num_elec = occ_s.iter().sum::<f64>();
                let mut force_curr = vec![];

                let mut net_force_elec = force_occ.iter()
                    .filter(|x| x.target_spin==i_spin)
                    .map(|x| {
                        let original_occ = occ_s[x.curr_state];
                        occ_s[x.curr_state] = x.force_occ;
                        force_curr.push(x.curr_state);
                        original_occ - x.force_occ
                    }).sum::<f64>();

                if net_force_elec > 1.0e-6 {
                    occ_s.iter_mut().enumerate().for_each(|(i,x)| {
                        if *x<1.0 && net_force_elec > 1.0e-6 {
                            if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                                let tmp_xa = *x + net_force_elec;
                                let tmp_xb = 1.0 - *x;
                                *x = if tmp_xa <=1.0 {
                                    net_force_elec = 0.0;
                                    tmp_xa
                                } else {
                                    net_force_elec -= tmp_xb;
                                    1.0
                                }
                            };
                        }
                    });
                } else if net_force_elec < -1.0e-6 {
                    occ_s.iter_mut().enumerate().rev().for_each(|(i, x)| {
                        if *x>0.0 && net_force_elec < -1.0e-6 {
                            if ! force_curr.iter().fold(false, |check, j| check || i==*j) {
                                let tmp_xa = *x + net_force_elec;
                                let tmp_xb = *x;
                                *x = if tmp_xa >= 0.0 {
                                    net_force_elec = 0.0;
                                    tmp_xa
                                } else {
                                    net_force_elec += tmp_xb;
                                    0.0
                                }
                            }
                        }

                    });
                }

                let mut i_homo = homo.get_mut(i_spin).unwrap();
                *i_homo = occ_s.iter().enumerate()
                    .filter(|(i,occ)| **occ >=1.0e-6)
                    .map(|(i,occ)| i).max().unwrap();
                let mut i_lumo = lumo.get_mut(i_spin).unwrap();
                //*i_lumo = *i_homo + 1;
                *i_lumo = occ_s.iter().enumerate()
                    .filter(|(i,occ)| (**occ-1.0).abs() >=1.0e-6 )
                    .map(|(i,occ)| i).min().unwrap();
            }
        },
    }


}

pub fn determine_force_projection(
    force_occ: &mut Vec<ForceStateOccupation>,
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors: &[MatrixFull<f64>;2]) {

        let ovlp_full = ovlp.to_matrixfull().unwrap();

        force_occ.iter_mut().enumerate().for_each(|(i_force_occ,i_obj)| {
            let p_spin = i_obj.prev_spin;
            let p_state = i_obj.prev_state;
            let c_start = i_obj.force_check_min;

            i_obj.ovlp = 0.0;
            i_obj.curr_state = 0;

            let prev_eigenvector = &ref_eigenvectors[p_spin][(..,p_state)];
            let r_states = (i_obj.force_check_min..i_obj.force_check_max);
            eigenvectors[p_spin].iter_columns(r_states).enumerate().for_each(|(il,curr_eigenvector)| {
                let cur_state = il + c_start;
                let mut tmp_vec = vec![0.0;curr_eigenvector.len()];
                _dgemv(&ovlp_full,curr_eigenvector,&mut tmp_vec,'n',1.0,0.0,1,1);

                let temp_value  = tmp_vec.par_iter().zip(prev_eigenvector.par_iter())
                    .fold(|| 0.0,|acc, (a,b)| acc + a*b).sum::<f64>();

                if temp_value.abs()>i_obj.ovlp {
                    i_obj.curr_state = cur_state;
                    i_obj.ovlp = temp_value.abs();
                }
            });
            if i_obj.ovlp < 1.0e-1 {
                println!("WARNNING: MOM overlap is less than 1.0e-1, indicating that the orbital window should be enlarge");
            }
        });

}

pub fn determine_force_projection_v02(
    force_occ: &mut Vec<ForceStateOccupation>,
    eigenvectors: &[MatrixFull<f64>;2],
    ovlp: &MatrixUpper<f64>,
    ref_eigenvectors_map: &HashMap<String,([MatrixFull<f64>;2],[usize;4])>,
    running_ref: Option<&MomRunningRef>
) {

        let ovlp_full = ovlp.to_matrixfull().unwrap();



        force_occ.iter_mut().enumerate().for_each(|(i_force_occ,i_obj)| {

            //println!("{}", i_obj.ref_index);
            //println!("{:?}", ref_eigenvectors_map);
            // The rolling reference (`mom_strategy = "step"`) stands in for the
            // entry named by `key`, and holds the orbitals of the previous
            // geometry step instead of the ones read from the file at startup.
            // Entries that address some other file keep their own reference.
            let (ref_eigenvectors, dims): (&[MatrixFull<f64>; 2], [usize; 4]) = match running_ref {
                Some(rr) if rr.key == i_obj.ref_index => (&rr.orbitals, rr.dims),
                _ => {
                    let (eig, d) = ref_eigenvectors_map.get(&i_obj.ref_index).unwrap();
                    (eig, *d)
                }
            };
            let [global_index, num_basis, _num_state, _spin_channel] = dims;

            let p_spin = i_obj.prev_spin;
            let t_spin = i_obj.target_spin;
            let p_state = i_obj.prev_state;
            let c_start = i_obj.force_check_min;

            // A restricted open-shell (ROHF/ROKS) calculation stores a single
            // common orbital set in channel 0, so a constraint addressed to
            // channel 1 cannot be matched against channel 1 of the current
            // orbitals: fall back to channel 0 for the overlap search. The
            // occupation itself is still written into the requested target
            // channel by adapt_occupation_with_force_projection. The same
            // fallback applies to the reference when it holds only one channel
            // (e.g. a ROHF chkfile used as the MOM reference).
            let cur_spin = if eigenvectors[t_spin].size()[0] == 0 { 0 } else { t_spin };
            let ref_spin = if ref_eigenvectors[p_spin].size()[0] == 0 { 0 } else { p_spin };

            i_obj.ovlp = 0.0;
            i_obj.curr_state = 0;
            i_obj.curr_spin = cur_spin;

            // `global_index` / `num_basis` come from `dims` by value (a `[usize;4]`),
            // so they are plain `usize` here -- the offsets of the map's values are
            // dereferenced at the `match` above.
            let mut prev_eigenvector = vec![0.0;eigenvectors[0].size()[0]];
            prev_eigenvector[global_index..global_index+num_basis].iter_mut().zip(ref_eigenvectors[ref_spin][(..,p_state)].iter()).for_each(|(to, from)| {
                *to = *from
            });

            // The reference orbital was expanded in the basis of *its* geometry
            // (or of the file it was read from), so it is not normalized under the
            // present overlap matrix: <ref|S_now|ref> departs from 1 by O(geometry
            // change). Dividing by its S-norm makes the projection a true cosine in
            // [-1, 1] -- without it the reported overlap grows past 1 as the
            // geometry moves (observed 1.04-1.23 in a geometry optimization) and no
            // threshold on it means anything. The selection itself is unaffected:
            // the factor is common to every candidate of the entry.
            {
                let mut s_ref = vec![0.0; prev_eigenvector.len()];
                _dgemv(&ovlp_full, &prev_eigenvector[..], &mut s_ref, 'n', 1.0, 0.0, 1, 1);
                let norm2: f64 = prev_eigenvector.iter().zip(s_ref.iter()).map(|(a, b)| a * b).sum();
                if norm2 > 1.0e-12 {
                    let inv = 1.0 / norm2.sqrt();
                    prev_eigenvector.iter_mut().for_each(|x| *x *= inv);
                }
            }

            let r_states = (i_obj.force_check_min..i_obj.force_check_max);
            eigenvectors[cur_spin].iter_columns(r_states).enumerate().for_each(|(il,curr_eigenvector)| {
                let cur_state = il + c_start;
                let mut tmp_vec = vec![0.0;curr_eigenvector.len()];
                _dgemv(&ovlp_full,curr_eigenvector,&mut tmp_vec,'n',1.0,0.0,1,1);

                let temp_value  = tmp_vec.par_iter().zip(prev_eigenvector.par_iter())
                    .fold(|| 0.0,|acc, (a,b)| acc + a*b).sum::<f64>();

                if temp_value.abs()>i_obj.ovlp {
                    i_obj.curr_state = cur_state;
                    i_obj.ovlp = temp_value.abs();
                }
            });
            if i_obj.ovlp < 1.0e-1 {
                println!("WARNNING: MOM overlap is less than 1.0e-1, indicating that the orbital window should be enlarge");
            }
        });

}

#[test]
fn test() {
    let dd = MatrixFull::from_vec([5,5], vec![1.0;25]).unwrap();

    dd.iter_columns((2..5)).enumerate().rev().for_each(|(x,y)| {
        println!("{:?}, {:?}",x, y);
    });
}
