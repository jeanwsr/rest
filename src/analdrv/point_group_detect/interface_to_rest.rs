use super::SymmMolecule;

pub fn get_full_point_group_for_vib(
    symbols: &[String],
    masses: &[f64],
    coords: &[[f64; 3]],
    tol: f64,
) -> (String, f64) {
    let mol = SymmMolecule::new(symbols.to_vec(), coords.to_vec()).with_masses(masses.to_vec()).with_tol(tol);
    let pg = mol.detect();
    (pg.full_name, pg.sigma)
}
