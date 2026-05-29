pub fn has_dm(file: &hdf5::File) -> bool {
    let mut has_dm = false;
    if let Ok(_init_guess) = file.dataset("init_guess") {
        has_dm = true;
    }

    has_dm
}

pub fn has_mo_coeff(file: &hdf5::File) -> bool {
    let mut has_mo = false;
    if let Ok(_mo_coeff) = file.dataset("scf/mo_coeff") {
        has_mo = true;
    }

    has_mo
}