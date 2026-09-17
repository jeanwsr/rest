pub const AU_TIME_TO_FS: f64 = 0.0241888432650856;

pub const AMU_TO_AU_MASS: f64 = 1822.888486209;

pub const KB_HARTREE_PER_K: f64 = 3.166811563e-6;

pub const HARTREE2EV: f64 = 27.211386245988;

pub const EV2KJMOL: f64 = 96.485307;

pub const AU_FORCE2_EV_PER_ANG: f64 = HARTREE2EV / crate::constants::BOHR;

pub fn kinetic_energy_hartree(vel: &[f64], mass_au: &[f64]) -> f64 {
    let mut ekin = 0.0;
    for i_atom in 0..mass_au.len() {
        let vx = vel[3 * i_atom];
        let vy = vel[3 * i_atom + 1];
        let vz = vel[3 * i_atom + 2];
        ekin += 0.5 * mass_au[i_atom] * (vx * vx + vy * vy + vz * vz);
    }
    ekin
}

pub fn masses_amu_to_au(mass_amu: &[f64]) -> Vec<f64> {
    mass_amu.iter().map(|m| m * AMU_TO_AU_MASS).collect()
}
