use rayon::prelude::*;
use tensors::{BasicMatrix, MatrixFull};

pub const ORB_OCCUPATION_THRESHOLD: f64 = 1.0e-6;

pub fn occupied_orbital_count_with_threshold(occupation: &[f64], threshold: f64) -> usize {
    occupation
        .iter()
        .rposition(|occ| *occ > threshold)
        .map(|idx| idx + 1)
        .unwrap_or(0)
}

pub fn occupied_orbital_count(occupation: &[f64]) -> usize {
    occupied_orbital_count_with_threshold(occupation, ORB_OCCUPATION_THRESHOLD)
}

pub fn is_aufbau(occupation: &[f64]) -> bool {
    // check if occupation is in descending order
    occupation
        .windows(2)
        .all(|pair| pair[0] + ORB_OCCUPATION_THRESHOLD > pair[1])
}

/// Return the HOMO and LUMO indices that would result from pure (integer)
/// occupation — i.e. ignoring any smearing.  For UHF/ROHF the return is
/// `([homo_α, homo_β], [lumo_α, lumo_β])`; for RHF only the first element
/// of each triplet is meaningful.
pub fn integer_homo_lumo(num_elec: &[f64; 3], spin_channel: usize) -> ([usize; 2], [usize; 2]) {
    let mut homo = [0usize; 2];
    let mut lumo = [0usize; 2];
    if spin_channel == 1 {
        // RHF: all electrons doubly occupied
        let nocc = (num_elec[0] / 2.0).round() as usize;
        homo[0] = nocc.saturating_sub(1);
        lumo[0] = nocc;
    } else {
        // UHF / ROHF: alpha and beta separately
        for s in 0..spin_channel {
            let nocc = num_elec[s + 1].round() as usize; // num_elec[1] = alpha, [2] = beta
            homo[s] = nocc.saturating_sub(1);
            lumo[s] = nocc;
        }
    }
    (homo, lumo)
}

pub fn norm(m: &MatrixFull<f64>, kind: &str) -> f64 {
    let sqsum = m.data.par_iter().map(|x| x * x).sum::<f64>();
    match kind {
        "rms" => {
            let size = m.size();
            let n_elem = (size[0] * size[1]) as f64;
            (sqsum / n_elem).sqrt()
        }
        "l2" => sqsum.sqrt(),
        _ => panic!("unknown norm kind '{}', must be 'rms' or 'l2'", kind),
    }
}

#[cfg(test)]
mod tests {
    use super::{is_aufbau, occupied_orbital_count, occupied_orbital_count_with_threshold};

    #[test]
    fn test_occupied_orbital_count_sparse() {
        let occupation = [1.0, 0.0, 0.5, 0.5, 0.0];
        assert_eq!(occupied_orbital_count(&occupation), 4);
    }

    #[test]
    fn test_occupied_orbital_count_threshold() {
        let occupation = [1.0, 0.0, 9.0e-7, 1.0e-6, 2.0e-6];
        assert_eq!(occupied_orbital_count_with_threshold(&occupation, 1.0e-6), 5);
        assert_eq!(occupied_orbital_count_with_threshold(&occupation, 2.0e-6), 1);
    }

    #[test]
    fn test_is_aufbau() {
        assert!(is_aufbau(&[1.0, 1.0, 0.5, 0.0]));
        assert!(!is_aufbau(&[1.0, 0.5, 0.6, 0.0]));
    }
}
