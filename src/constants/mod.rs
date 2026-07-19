pub mod vsap;
pub mod c2s;
pub mod solvent;
mod cartesian_gto;
pub mod element;

// use std::collections::HashMap;

// use lazy_static::lazy_static;

pub use crate::constants::vsap::*;
pub use crate::constants::c2s::*;
pub use crate::constants::cartesian_gto::*;
pub use crate::constants::element::*;



// use for the inverse (sqrt inverse) of the auxiliary coulomb matrix
pub const AUXBAS_THRESHOLD: f64 = 1.0e-10;
pub const INVERSE_THRESHOLD: f64 = 1.0e-10;
pub const SQRT_THRESHOLD: f64 = 1.0e-10;


pub const E5: f64 = 1.0e5;
pub const E6: f64 = 1.0e6;
pub const E7: f64 = 1.0e7;
pub const E8: f64 = 1.0e8;
pub const E9: f64 = 1.0e9;


pub struct DMatrix<const L: usize> {
    size:[usize;2],
    indicing:[usize;2],
    data: [f64; L]
}


// =========== libcint ===================================
// for the bas index - libcint
pub const BAS_ATM: usize = 0;
pub const BAS_ANG: usize = 1;
pub const BAS_PRM: usize = 2;
pub const BAS_CTR: usize = 3;
pub const BAS_SLOTS: usize = 6;

// for the atm index - libcint 
pub const ATM_NUC: usize = 0;
pub const ATM_ENV: usize = 1;
pub const ATM_NUC_MOD_OF: usize = 2;
pub const ATM_FRAC_CHARGE_OF: usize = 3;
pub const ATM_SLOTS: usize = 6;

// for ECP - libcint
pub const ECP_LMAX: i32 = 5;
pub const NUC_ECP:  i32 = 4;

// for exp cutoff -libcint
pub const PTR_EXPCUTOFF: i32 = 0;
// for dipole - libcint
pub const PTR_COMMON_ORG: i32 = 1;
// for Gauge origin
pub const PTR_RINV_ORIG: i32 = 4;

pub const NUC_MOD_OF: i32 = 2;

pub const NUC_STAD_CHARGE: i32 = 1;
pub const NUC_GAUS_CHARGE: i32 = 2;
pub const NUC_FRAC_CHARGE: i32 = 3;
// =========== libcint ===================================


//// for the atm index - libcint
//pub const ATM_CHARGE_OF: usize = 0;
//pub const ATM_PRT_COORD: usize = 1;
//pub const ATM_NUC_MOD_OF: usize = 2;
//pub const ATM_PRT_ZETA: usize = 3;

pub const ENV_PRT_START: usize = 20;

// math, physics
// NOTE: these constants come from several different CODATA releases (see per-line sources).
pub const EV: f64 = 27.2113845;                 // Hartree energy in eV, CODATA 2002
pub const HARTREE2KCAL: f64 = 627.509451;       // Hartree -> kcal/mol; CODATA 2002 (Hartree energy in J / (kcal * Avogadro))
pub const HARTREE2WAVENUMBER: f64 = 219474.63;  // Hartree -> cm^-1 (hartree-inverse meter relationship / 100); CODATA (~2.194746313705e7 m^-1)
pub const FQ: f64 = 1822.8884861920776;         // u/m_e ratio (= 1 / electron mass in u 5.48579909070e-4), CODATA 2014
pub const E:  f64 = std::f64::consts::E;         // math constant (std); year-insensitive
pub const PI: f64 = std::f64::consts::PI;        // math constant (std); year-insensitive
//
pub const LIGHT_SPEED: f64 = 137.03599967994;   // inverse fine-structure constant, CODATA 2006; http://physics.nist.gov/cgi-bin/cuu/Value?alph
// BOHR = .529 177 210 92(17) e-10m  // http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
// source: CODATA 2010 (https://physics.nist.gov/cuu/Constants/ArchiveASCII/allascii_2010.txt)
pub const BOHR: f64 = 0.52917721092;  // Angstroms
pub const BOHR_SI: f64 = BOHR * 1e-10;

pub const G_ELECTRON: f64 = 2.00231930436182;  // CODATA 2014; http://physics.nist.gov/cgi-bin/cuu/Value?gem
pub const E_MASS: f64 = 9.10938356e-31;         // kg, CODATA 2014; https://physics.nist.gov/cgi-bin/cuu/Value?me
pub const AVOGADRO: f64 = 6.022140857e23;       // CODATA 2014; https://physics.nist.gov/cgi-bin/cuu/Value?na
pub const PLANCK: f64 = 6.626070040e-34;        // J*s, CODATA 2014; http://physics.nist.gov/cgi-bin/cuu/Value?h
pub const BOLTZMANN: f64 = 1.380649e-23;        // J/K, CODATA 2018 (exact, SI 2019); https://physics.nist.gov/cgi-bin/cuu/Value?k
pub const CLIGHT_CMS: f64 = 2.99792458e10;      // speed of light, cm/s; exact by SI definition (year-insensitive)
pub const R_GAS: f64 = BOLTZMANN * AVOGADRO;    // J/(mol*K) ideal gas constant; derived
pub const E_CHARGE: f64 = 1.6021766208e-19;    // C, CODATA 2014
pub const DEBYE:f64 = 3.335641e-30;            // C*m = 1e-18/LIGHT_SPEED_SI; defined via c, year-insensitive; https://cccbdb.nist.gov/debye.asp
pub const AU2DEBYE:f64 = E_CHARGE * BOHR*1e-10 / DEBYE; // 2.541746; derived


pub const MPI_CHUNK:usize = 134217728; // around 1 GB

