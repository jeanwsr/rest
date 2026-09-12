//! Interface of the multipole task to the analdrv driver.

use crate::analdrv::config::{AnalDrvConfig, MultipoleRdm1Relax};
use crate::analdrv::multipole::rmultipole::RMultipoleDH;
use crate::analdrv::response::rgfock_interface::rgfock_dh_interface;
use crate::analdrv::response::rresp_interface::RRespSCF;
use crate::ri_jk::util::get_cint_mol;
use crate::scf_io::{SCFType, SCF};
use crate::utilities::rstsr_util::RestTensorToRstsrTsrAPI;

use rstsr::prelude::*;
use serde::Serialize;

/// Moments of one multipole order, split into contributions.
///
/// Every tensor is flattened row-major (the first Cartesian component the slowest), matching the
/// component labels of the stdout print: `[xx, xy, xz, yx, ...]` for the quadrupole, etc. All
/// values are in atomic units.
#[derive(Debug, Clone, Serialize)]
pub struct MultipoleOrderParts {
    /// Nuclear charge contribution.
    pub nuc: Vec<f64>,
    /// SCF density contribution.
    pub scf: Vec<f64>,
    /// Unrelaxed correlation rdm1 contribution (post-SCF methods only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corr: Option<Vec<f64>>,
    /// Relaxed (Z-vector) response contribution (post-SCF methods, relaxed mode only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resp: Option<Vec<f64>>,
    /// Total moment, the sum of all contributions.
    pub tot: Vec<f64>,
    /// Traceless total (quadrupole order only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceless: Option<Vec<f64>>,
}

/// Output of the electric multipole moment task, in atomic units.
#[derive(Debug, Clone, Serialize)]
pub struct MultipoleOutput {
    /// The origin (Bohr) actually used for the evaluation: the explicit `multipole_origin` if
    /// given, else the center of nuclear mass.
    pub origin: [f64; 3],
    /// Dipole moment (order 1), present if requested in `multipole_orders`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dipole: Option<MultipoleOrderParts>,
    /// Raw (second-moment) quadrupole (order 2), together with its traceless form.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quadrupole: Option<MultipoleOrderParts>,
    /// Octupole moment (order 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub octupole: Option<MultipoleOrderParts>,
    /// Hexadecapole moment (order 4).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hexadecapole: Option<MultipoleOrderParts>,
}

/// Run the electric multipole moment task of the analdrv driver.
///
/// For usual SCF methods (HF/DFT/hybrid-DFT, where the energy functional is the SCF functional),
/// neither the DH composite nor the response object is given to the driver: the moments are the
/// plain nuclear plus SCF-density contractions. For PT2-family post-SCF (fifth-DFA) methods, the
/// unrelaxed correlation rdm1 increment is added, and, when `resp_objs` is passed, the relaxed
/// (Z-vector) increment as well; whether `resp_objs` is passed is decided by the caller from
/// `multipole_rdm1_relax`.
///
/// `resp_objs` is only used for the relaxed DH increments, and requires the DFT grids to be
/// present; the caller ([`crate::analdrv::interface::analdrv_interface`]) regenerates them.
pub fn multipole_interface<'a>(
    scf_data: &'a SCF,
    config: &AnalDrvConfig,
    resp_objs: Option<&mut RRespSCF<'a>>,
) -> MultipoleOutput {
    match scf_data.scftype {
        SCFType::RHF => {},
        _ => panic!("Multipole evaluation is currently only implemented for restricted (RHF/RKS) calculations."),
    }

    let mp_cfg = &config.multipole;
    assert!(!mp_cfg.orders.is_empty(), "multipole_orders cannot be empty");
    for &order in &mp_cfg.orders {
        assert!(
            (1..=4).contains(&order),
            "multipole_orders entries must be 1 (dipole) to 4 (hexadecapole), got {order}"
        );
    }

    let origin = mp_cfg.origin.unwrap_or_else(|| center_of_nuclear_mass(scf_data));

    let device = DeviceBLAS::default();
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let mol_cint = get_cint_mol(&scf_data.mol);

    // DH density increments: PT2-family post-SCF methods only. The remaining requirements (no
    // frozen core, no range separation, rimatr availability) are guarded inside
    // `rgfock_dh_interface`.
    let mut rgfock = if scf_data.mol.xc_data.is_fifth_dfa() {
        match &scf_data.mol.xc_data.dfa_family_pos {
            Some(crate::dft::DFAFamily::PT2) => {},
            other => panic!(
                "Multipole evaluation for post-SCF methods currently supports PT2-family (xDH/BDH/MP2) methods only, got {other:?}."
            ),
        }
        Some(rgfock_dh_interface::<f64>(scf_data))
    } else {
        None
    };

    let mut rmultipole = RMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, origin, rgfock.as_mut(), resp_objs);

    // evaluate the requested orders (each order caches itself; the DH density increments are
    // shared between the orders through the gfock result cache)
    for &order in &mp_cfg.orders {
        match order {
            1 => {
                rmultipole.make_dipole();
            },
            2 => {
                rmultipole.make_quadrupole();
            },
            3 => {
                rmultipole.make_octupole();
            },
            4 => {
                rmultipole.make_hexadecapole();
            },
            _ => unreachable!("multipole order range checked above"),
        }
    }

    let print_level = scf_data.mol.ctrl.print_level;
    rmultipole.print_multipole(print_level);
    if print_level >= 2 {
        for (label, t) in &rmultipole.timing {
            println!("Timing {label}: {t:.3} s");
        }
    }

    // optional: dump the total density (the one contracted for the moments above) into the
    // Gaussian fchk file. Post-SCF (PT2-family: MP2 and double-hybrid) methods dump the total
    // density; SCF-level methods silently ignore the keyword (their total density is the SCF
    // density alone, already in the fchk output).
    if mp_cfg.rdm1_dump && scf_data.mol.xc_data.is_fifth_dfa() {
        let relaxed = matches!(mp_cfg.rdm1_relax, MultipoleRdm1Relax::Relaxed);
        let dm_total_ao = rmultipole.get_total_density_ao(relaxed);

        // the fchk (head + MO coefficients) is regenerated first: with `outputs = ["fchk"]`
        // it was already written before the analdrv tasks in the main driver, so this only
        // recreates the same content when the dump keyword alone requests the file
        scf_data.save_fchk_of_gaussian();

        // pack the density into the fchk layout: Gaussian AO order (consistent with the
        // librest2fch-written MO coefficients), lower triangle by column
        let nbf = dm_total_ao.shape()[0];
        let perm = scf_data.gaussian_ao_permutation();
        assert_eq!(perm.len(), nbf, "AO permutation size mismatch.");
        let mut packed = Vec::with_capacity(nbf * (nbf + 1) / 2);
        for pj in 0..nbf {
            let j = perm[pj];
            for pi in 0..=pj {
                packed.push(dm_total_ao[[perm[pi], j]]);
            }
        }
        scf_data.fchk_append_density_section("Total MP2 Density", &packed);
        println!(
            "    total density: {}",
            if relaxed { "relaxed (SCF + corr. + Z-vector response)" } else { "unrelaxed (SCF + corr.)" }
        );
    }

    MultipoleOutput {
        origin,
        dipole: order_parts(&rmultipole, "dip", false),
        quadrupole: order_parts(&rmultipole, "quad", true),
        octupole: order_parts(&rmultipole, "oct", false),
        hexadecapole: order_parts(&rmultipole, "hex", false),
    }
}

/// Assemble the per-order output from the driver result map; `None` if the order was not
/// evaluated. `traceless` includes the `quad_tot_traceless` entry (quadrupole only).
fn order_parts(driver: &RMultipoleDH, prefix: &str, traceless: bool) -> Option<MultipoleOrderParts> {
    let flat =
        |suffix: &str| driver.result.get(&format!("{prefix}_{suffix}")).map(|v| v.view().into_shape(-1).into_vec());
    Some(MultipoleOrderParts {
        nuc: flat("nuc")?,
        scf: flat("scf")?,
        corr: flat("corr"),
        resp: flat("resp"),
        tot: flat("tot")?,
        traceless: if traceless {
            driver.result.get("quad_tot_traceless").map(|v| v.view().into_shape(-1).into_vec())
        } else {
            None
        },
    })
}

/// Center of nuclear mass (Bohr), from the IUPAC 2021 average atomic weights of REST's element
/// table (Gaussian reports its moments at the center of mass as well, but uses
/// most-abundant-isotope masses, so tiny origin differences are expected in comparisons).
fn center_of_nuclear_mass(scf_data: &SCF) -> [f64; 3] {
    let masses = crate::geom_io::get_mass_charge(&scf_data.mol.geom.elem);
    let mut com = [0.0f64; 3];
    let mut mass_tot = 0.0f64;
    for (xyz, (mass, _)) in scf_data.mol.geom.position.iter_columns_full().zip(masses.iter()) {
        for t in 0..3 {
            com[t] += mass * xyz[t];
        }
        mass_tot += mass;
    }
    com.map(|v| v / mass_tot)
}
