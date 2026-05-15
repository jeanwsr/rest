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
