use serde::{Deserialize, Serialize};
use statrs::function::erf::erfc;
use std::f64::consts::PI;

use super::{SCF, SCFType};
use super::util::occupied_orbital_count;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum SmearingType {
    FERMI,
    GAUSSIAN,
}

pub fn apply_smearing(scf_data: &mut SCF, smearing_type: SmearingType, sigma: f64) {
    let spin_channel = scf_data.mol.spin_channel;

    match &scf_data.scftype {
        SCFType::RHF => {
            let eps = &scf_data.eigenvalues[0];
            let nocc_target = scf_data.mol.num_elec[0] / 2.0;

            let mu = find_mu(eps, nocc_target, sigma, smearing_type);

            let occ_half = match smearing_type {
                SmearingType::FERMI => fermi_occ(eps, mu, sigma),
                SmearingType::GAUSSIAN => gaussian_occ(eps, mu, sigma),
            };

            scf_data.smearing_entropy = match smearing_type {
                SmearingType::FERMI => fermi_entropy(eps, &occ_half, mu, sigma),
                SmearingType::GAUSSIAN => gaussian_entropy(eps, &occ_half, mu, sigma),
            } * 2.0;

            let occ: Vec<f64> = occ_half.iter().map(|&f| f * 2.0).collect();
            scf_data.occupation[0] = occ;

            let nw = occupied_orbital_count(&occ_half);
            let h = nw.saturating_sub(1);
            let l = nw;

            scf_data.homo[0] = h;
            scf_data.lumo[0] = l;
        }

        SCFType::UHF => {
            let n_alpha = scf_data.mol.num_elec[1];
            let n_beta = scf_data.mol.num_elec[2];
            scf_data.smearing_entropy = 0.0;

            for spin in 0..spin_channel {
                let eps = &scf_data.eigenvalues[spin];
                let nocc_target = if spin == 0 { n_alpha } else { n_beta };

                let mu = find_mu(eps, nocc_target, sigma, smearing_type);

                let occ = match smearing_type {
                    SmearingType::FERMI => fermi_occ(eps, mu, sigma),
                    SmearingType::GAUSSIAN => gaussian_occ(eps, mu, sigma),
                };

                let entropy = match smearing_type {
                    SmearingType::FERMI => fermi_entropy(eps, &occ, mu, sigma),
                    SmearingType::GAUSSIAN => gaussian_entropy(eps, &occ, mu, sigma),
                };
                scf_data.smearing_entropy += entropy;

                scf_data.occupation[spin] = occ.clone();

                let nw = occupied_orbital_count(&occ);
                let h = nw.saturating_sub(1);
                let l = nw;

                scf_data.homo[spin] = h;
                scf_data.lumo[spin] = l;
            }
        }

        SCFType::ROHF => {
            panic!("Smearing for ROHF is not implemented!");
        }
    }
}

pub fn fermi_occ(energy: &Vec<f64>, mu: f64, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        panic!("sigma must be positive for fermi smearing");
    }
    energy
        .iter()
        .map(|&e| {
            let x = (e - mu) / sigma;
            if x > 40.0 {
                0.0
            } else if x < -40.0 {
                1.0
            } else {
                1.0 / (1.0 + (x).exp())
            }
        })
        .collect()
}

pub fn gaussian_occ(energy: &Vec<f64>, mu: f64, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        panic!("sigma must be positive for gaussian smearing");
    }
    energy.iter().map(|&e| {
        let x = (e - mu) / sigma;
        0.5 * erfc(x)
    }).collect()
}

pub fn fermi_entropy(_energy: &[f64], occ: &[f64], _mu: f64, _sigma: f64) -> f64 {
    occ.iter()
        .filter(|&&f| f > 0.0 && f < 1.0)
        .map(|&f| -(f * f.ln() + (1.0 - f) * (1.0 - f).ln()))
        .sum()
}

pub fn gaussian_entropy(energy: &[f64], _occ: &[f64], mu: f64, sigma: f64) -> f64 {
    energy
        .iter()
        .map(|&e| (-((e - mu) / sigma).powi(2)).exp())
        .sum::<f64>()
        / (2.0 * PI.sqrt())
}

pub fn find_mu(energy: &[f64], nocc: f64, sigma: f64, smearing_type: SmearingType) -> f64 {
    let n = energy.len() as f64;
    if nocc <= 0.0 {
        return energy[0] - 1.0;
    }
    if nocc >= n {
        return energy[n as usize - 1] + 1.0;
    }

    let nelec_of_mu = |mu: f64| -> f64 {
        match smearing_type {
            SmearingType::FERMI => energy
                .iter()
                .map(|&e| {
                    let x = (e - mu) / sigma;
                    if x > 40.0 {
                        0.0
                    } else if x < -40.0 {
                        1.0
                    } else {
                        1.0 / (1.0 + x.exp())
                    }
                })
                .sum(),
            SmearingType::GAUSSIAN => energy
                .iter()
                .map(|&e| {
                    let x = (e - mu) / sigma;
                    0.5 * erfc(x)
                })
                .sum(),
        }
    };

    let mut lo = energy[0] - 20.0 * sigma;
    let mut hi = energy[energy.len() - 1] + 20.0 * sigma;

    for _iter in 0..200 {
        let mid = (lo + hi) / 2.0;
        let n_mid = nelec_of_mu(mid);
        if (n_mid - nocc).abs() < 1e-12 {
            return mid;
        }
        if n_mid < nocc {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / 2.0
}
