//! Element symbol ↔ atomic number, and most-abundant-isotope masses.
//! Masses are from qcelemental's NIST 2011 table (same source Psi4 uses).
//!
//! NOTE: per-atom mass overrides supplied via [`super::Molecule::with_masses`]
//! always take precedence; this table is only a fallback when no mass is
//! supplied. The symmetry test fixtures supply explicit masses.

/// (symbol, Z, most-abundant-isotope mass)
const TABLE: &[(&str, u8, f64)] = &[
    ("H", 1, 1.00782503223),
    ("He", 2, 4.00260325413),
    ("Li", 3, 7.0160034366),
    ("Be", 4, 9.012183065),
    ("B", 5, 11.00930536),
    ("C", 6, 12.0),
    ("N", 7, 14.00307400443),
    ("O", 8, 15.99491461957),
    ("F", 9, 18.99840316273),
    ("Ne", 10, 19.9924401762),
    ("Na", 11, 22.989769282),
    ("Mg", 12, 23.985041697),
    ("Al", 13, 26.98153853),
    ("Si", 14, 27.97692653465),
    ("P", 15, 30.97376199842),
    ("S", 16, 31.9720711744),
    ("Cl", 17, 34.968852682),
    ("Ar", 18, 39.9623831237),
    ("K", 19, 38.9637064864),
    ("Ca", 20, 39.962590863),
    ("Sc", 21, 44.95590828),
    ("Ti", 22, 47.94794198),
    ("V", 23, 50.94395704),
    ("Cr", 24, 51.94050623),
    ("Mn", 25, 54.93804391),
    ("Fe", 26, 55.93493633),
    ("Co", 27, 58.93319429),
    ("Ni", 28, 57.93534241),
    ("Cu", 29, 62.92959772),
    ("Zn", 30, 63.92914201),
    ("Ga", 31, 68.9255735),
    ("Ge", 32, 73.921177761),
    ("As", 33, 74.92159457),
    ("Se", 34, 79.9165218),
    ("Br", 35, 78.9183376),
    ("Kr", 36, 83.9114977282),
    ("I", 53, 126.9044719),
];

/// Normalize a user-supplied symbol to the table form (first letter upper,
/// rest lower), matching Psi4's element lookup. "X" → dummy (Z=0).
fn normalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase(),
        None => String::new(),
    }
}

/// Atomic number for an element symbol, or `None` if unknown. "X" → 0 (dummy).
pub fn symbol_to_z(s: &str) -> Option<u8> {
    let n = normalize(s);
    if n == "X" {
        return Some(0);
    }
    TABLE.iter().find(|(sym, _, _)| *sym == n).map(|(_, z, _)| *z)
}

/// Default most-abundant-isotope mass for atomic number `z`, or `0.0` if
/// unknown (dummy `z==0` → 0.0).
pub fn z_to_mass(z: u8) -> f64 {
    TABLE.iter().find(|(_, zz, _)| *zz == z).map(|(_, _, m)| *m).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups() {
        assert_eq!(symbol_to_z("C"), Some(6));
        assert_eq!(symbol_to_z("c"), Some(6));
        assert_eq!(symbol_to_z("CL"), Some(17));
        assert_eq!(symbol_to_z("br"), Some(35));
        assert_eq!(symbol_to_z("X"), Some(0));
        assert_eq!(symbol_to_z("Uuo"), None);
        assert!((z_to_mass(1) - 1.00782503223).abs() < 1e-12);
        assert!((z_to_mass(8) - 15.99491461957).abs() < 1e-12);
        assert_eq!(z_to_mass(0), 0.0);
    }
}
