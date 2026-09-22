//! GW / evGW checkpoint (archive) support.
//!
//! REST's GW and evGW runs are expensive: an evGW job may run dozens of full
//! GW passes, and the BSE solve behind it is larger still.  On a queued cluster
//! such jobs get killed by the wall-clock limit and requeued, so the state that
//! crosses the kill boundary has to live on disk.
//!
//! This module writes a single HDF5 archive (`gw_checkpoint_path`, default
//! `./gw_checkpoint.h5`) holding everything that the post-SCF flow needs to
//! continue *without* redoing the work already done:
//!
//! * compatibility metadata (version, number of atoms / basis functions /
//!   states / electrons, charge, spin, spin channel, basis and auxiliary-basis
//!   paths, SCF type, and the geometry as JSON);
//! * the converged SCF arrays (MO energies, MO coefficients, density matrices,
//!   occupations, HOMO/LUMO indices, the SCF and nuclear energies and the Fock
//!   matrices);
//! * the GW quasiparticle energies (`scf_data.gwqp`, both the G and W vectors,
//!   plus the spin-resolved `gwqp_spin` arrays);
//! * the renormalized-singles particle energies when `renormalized_singles`
//!   is active (the RS *orbitals* are already reflected in the saved MO
//!   coefficients, so restoring the eigenvectors restores the RS basis too);
//! * the evGW outer-loop progress: the round counter, the convergence flag,
//!   the last residual / |dG| metrics, the DIIS history and error vectors, and
//!   the iterate itself (which is `scf_data.gwqp`).
//!
//! Two stages are distinguished:
//!
//! * [`GwCheckpointStage::GwFinished`] -- a complete GW (or evGW) calculation
//!   has finished.  Resuming restores the state and goes straight into the
//!   BSE/response step.
//! * [`GwCheckpointStage::EvgwRound`] -- evGW was interrupted after a round.
//!   Resuming restores the state and continues the evGW loop from that round.
//!
//! **What resuming skips.**  With `resume_from_checkpoint = true` the driver
//! skips both `scf_without_build` (the SCF *iteration loop*) and the whole
//! GW/evGW calculation, installing the archived electronic structure instead.
//! The molecule, the basis and the RI integrals are *still* built from the
//! input file: the two-/three-/four-centre machinery in
//! `SCF::initialize_scf` -> `prepare_necessary_integrals` feeds every
//! downstream GW/BSE step and is deliberately not archived.  So "skip the SCF"
//! here means "skip the SCF *iterations*", not "skip the setup".
//!
//! Writing is "atomic-ish": the archive is written to `<path>.tmp` and only
//! renamed onto `<path>` once it is closed, so a job killed mid-write leaves
//! the previous (valid) checkpoint intact instead of a truncated file.
//!
//! Everything lives under the single HDF5 group `rest_gw_checkpoint`; the
//! sub-groups `meta`, `scf`, `gw` and `diis` mirror the four blocks above.

use std::path::Path;
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use hdf5::types::VarLenUnicode;
use tensors::{MatrixFull, MatrixUpper};

use crate::fileop::chkfile::{geom_from_json, geom_to_json};
use crate::scf_io::{SCF, SCFType};

/// Format version of the archive.  Bump whenever the layout changes so that an
/// old checkpoint is rejected with a clear message instead of being misread.
pub const GW_CHECKPOINT_VERSION: usize = 1;

/// Name of the root HDF5 group holding the archive.
const ROOT_GROUP: &str = "rest_gw_checkpoint";

/// Position tolerance (Bohr) when comparing the stored geometry with the
/// current input.  Coordinates are written with full `f64` precision, so this
/// only absorbs the round trip through the JSON representation.
const GEOM_TOL: f64 = 1.0e-8;

/// Which point of the post-SCF flow the archive represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwCheckpointStage {
    /// A complete GW (or a converged/exhausted evGW) calculation has finished.
    GwFinished,
    /// evGW was interrupted; `rounds_done` rounds have completed.
    EvgwRound,
}

impl GwCheckpointStage {
    pub fn as_str(self) -> &'static str {
        match self {
            GwCheckpointStage::GwFinished => "gw_finished",
            GwCheckpointStage::EvgwRound => "evgw_round",
        }
    }

    pub fn from_str(value: &str) -> Result<Self> {
        match value {
            "gw_finished" => Ok(GwCheckpointStage::GwFinished),
            "evgw_round" => Ok(GwCheckpointStage::EvgwRound),
            other => Err(anyhow!(
                "gw_checkpoint: unknown checkpoint stage '{}' (expected 'gw_finished' or 'evgw_round')",
                other
            )),
        }
    }
}

/// Everything about an interrupted GW/evGW run that is *not* already carried by
/// the plain `SCF` struct.
#[derive(Clone)]
pub struct GwCheckpointState {
    pub stage: GwCheckpointStage,
    /// Number of completed evGW rounds (0 for a plain G0W0 checkpoint).
    pub rounds_done: usize,
    /// `evgw_rounds` that was in effect when the archive was written.
    pub total_rounds: usize,
    /// Did the last completed round satisfy `evgw_conv_tol`?
    pub converged: bool,
    pub last_residual: f64,
    pub last_dg: f64,
    /// DIIS history / error vectors of the evGW outer loop, oldest first.
    pub diis_history: Vec<Vec<f64>>,
    pub diis_errors: Vec<Vec<f64>>,
    pub diis_space: usize,
    pub diis_min_history: usize,
    pub diis_safeguard: bool,

    // ---- compatibility metadata (checked on load) ----
    pub natm: usize,
    pub num_basis: usize,
    pub num_state: usize,
    pub num_elec: [f64; 3],
    pub charge: f64,
    pub spin: f64,
    pub spin_channel: usize,
    pub basis_path: String,
    pub auxbas_path: String,
    pub scf_type: SCFType,
}

impl GwCheckpointState {
    /// A metadata-only state describing the *current* SCF object.  Callers then
    /// set `stage` / the evGW progress fields before saving.
    pub fn from_scf(scf_data: &SCF) -> Self {
        GwCheckpointState {
            stage: GwCheckpointStage::GwFinished,
            rounds_done: 0,
            total_rounds: 0,
            converged: false,
            last_residual: f64::INFINITY,
            last_dg: f64::INFINITY,
            diis_history: Vec::new(),
            diis_errors: Vec::new(),
            diis_space: 0,
            diis_min_history: 0,
            diis_safeguard: false,
            natm: scf_data.mol.geom.elem.len(),
            num_basis: scf_data.mol.num_basis,
            num_state: scf_data.mol.num_state,
            num_elec: scf_data.mol.num_elec,
            charge: scf_data.mol.ctrl.charge,
            spin: scf_data.mol.ctrl.spin,
            spin_channel: scf_data.mol.spin_channel,
            basis_path: scf_data.mol.ctrl.basis_path.clone(),
            auxbas_path: scf_data.mol.ctrl.auxbas_path.clone(),
            scf_type: scf_data.scftype,
        }
    }

    /// A short human-readable description used by the resume/save log lines.
    pub fn describe(&self) -> String {
        match self.stage {
            GwCheckpointStage::GwFinished => "finished GW calculation".to_string(),
            GwCheckpointStage::EvgwRound => format!(
                "evGW round {} of {} (converged={}, max|dE_qp|={:.4e} Ha, |dG|={:.4e})",
                self.rounds_done,
                self.total_rounds,
                self.converged,
                self.last_residual,
                self.last_dg
            ),
        }
    }
}

fn scf_type_as_str(scf_type: SCFType) -> &'static str {
    match scf_type {
        SCFType::RHF => "rhf",
        SCFType::ROHF => "rohf",
        SCFType::UHF => "uhf",
    }
}

fn scf_type_from_str(value: &str) -> Result<SCFType> {
    match value.to_ascii_lowercase().as_str() {
        "rhf" => Ok(SCFType::RHF),
        "rohf" => Ok(SCFType::ROHF),
        "uhf" => Ok(SCFType::UHF),
        other => Err(anyhow!(
            "gw_checkpoint: unknown SCF type '{}' stored in the checkpoint",
            other
        )),
    }
}

// ---------------------------------------------------------------------------
// low-level HDF5 helpers
// ---------------------------------------------------------------------------

fn write_usize(group: &hdf5::Group, name: &str, value: usize) -> Result<()> {
    group
        .new_dataset_builder()
        .with_data(&[value])
        .create(name)
        .with_context(|| format!("gw_checkpoint: cannot write dataset '{}'", name))?;
    Ok(())
}

fn write_f64(group: &hdf5::Group, name: &str, value: f64) -> Result<()> {
    group
        .new_dataset_builder()
        .with_data(&[value])
        .create(name)
        .with_context(|| format!("gw_checkpoint: cannot write dataset '{}'", name))?;
    Ok(())
}

fn write_vec_f64(group: &hdf5::Group, name: &str, values: &[f64]) -> Result<()> {
    // HDF5 cannot hold a zero-length dataset through this builder, and an empty
    // array simply means "not present" -- record that fact by omitting it.
    if values.is_empty() {
        return Ok(());
    }
    group
        .new_dataset_builder()
        .with_data(values)
        .create(name)
        .with_context(|| format!("gw_checkpoint: cannot write dataset '{}'", name))?;
    Ok(())
}

fn write_vec_usize(group: &hdf5::Group, name: &str, values: &[usize]) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    group
        .new_dataset_builder()
        .with_data(values)
        .create(name)
        .with_context(|| format!("gw_checkpoint: cannot write dataset '{}'", name))?;
    Ok(())
}

fn write_string(group: &hdf5::Group, name: &str, value: &str) -> Result<()> {
    let unicode_string = VarLenUnicode::from_str(value)
        .map_err(|e| anyhow!("gw_checkpoint: cannot encode string dataset '{}': {:?}", name, e))?;
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create(name)
        .with_context(|| format!("gw_checkpoint: cannot create string dataset '{}'", name))?;
    ds.write_scalar(&unicode_string)
        .with_context(|| format!("gw_checkpoint: cannot write string dataset '{}'", name))?;
    Ok(())
}

fn read_usize(group: &hdf5::Group, name: &str) -> Result<usize> {
    let ds = group
        .dataset(name)
        .with_context(|| format!("gw_checkpoint: required dataset '{}' is missing", name))?;
    let values = ds
        .read_raw::<usize>()
        .with_context(|| format!("gw_checkpoint: dataset '{}' is not a usize array", name))?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("gw_checkpoint: dataset '{}' is empty", name))
}

fn read_f64(group: &hdf5::Group, name: &str) -> Result<f64> {
    let ds = group
        .dataset(name)
        .with_context(|| format!("gw_checkpoint: required dataset '{}' is missing", name))?;
    let values = ds
        .read_raw::<f64>()
        .with_context(|| format!("gw_checkpoint: dataset '{}' is not an f64 array", name))?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("gw_checkpoint: dataset '{}' is empty", name))
}

fn read_string(group: &hdf5::Group, name: &str) -> Result<String> {
    let ds = group
        .dataset(name)
        .with_context(|| format!("gw_checkpoint: required dataset '{}' is missing", name))?;
    let value = ds
        .read_scalar::<VarLenUnicode>()
        .with_context(|| format!("gw_checkpoint: dataset '{}' is not a string", name))?;
    Ok(value.as_str().to_string())
}

fn read_vec_f64(group: &hdf5::Group, name: &str) -> Result<Vec<f64>> {
    match group.dataset(name) {
        Ok(ds) => ds
            .read_raw::<f64>()
            .with_context(|| format!("gw_checkpoint: dataset '{}' is not an f64 array", name)),
        // Absent == empty (see `write_vec_f64`).
        Err(_) => Ok(Vec::new()),
    }
}

fn read_vec_usize(group: &hdf5::Group, name: &str) -> Result<Vec<usize>> {
    match group.dataset(name) {
        Ok(ds) => ds
            .read_raw::<usize>()
            .with_context(|| format!("gw_checkpoint: dataset '{}' is not a usize array", name)),
        Err(_) => Ok(Vec::new()),
    }
}

fn write_matrix(group: &hdf5::Group, name: &str, matrix: &MatrixFull<f64>) -> Result<()> {
    let shape = matrix.size;
    write_vec_usize(group, &format!("{}_shape", name), &shape)?;
    write_vec_f64(group, &format!("{}_data", name), &matrix.data)?;
    Ok(())
}

fn read_matrix(group: &hdf5::Group, name: &str) -> Result<MatrixFull<f64>> {
    let shape = read_vec_usize(group, &format!("{}_shape", name))?;
    let data = read_vec_f64(group, &format!("{}_data", name))?;
    if shape.len() != 2 {
        bail!(
            "gw_checkpoint: matrix '{}' has a malformed shape record ({:?})",
            name,
            shape
        );
    }
    let expected = shape[0] * shape[1];
    if expected != data.len() {
        bail!(
            "gw_checkpoint: matrix '{}' is corrupt: shape {:?} needs {} values but {} were stored",
            name,
            shape,
            expected,
            data.len()
        );
    }
    MatrixFull::from_vec([shape[0], shape[1]], data)
        .ok_or_else(|| anyhow!("gw_checkpoint: cannot rebuild matrix '{}'", name))
}

// ---------------------------------------------------------------------------
// input-card accessors
// ---------------------------------------------------------------------------

fn qp_control(scf_data: &SCF) -> Option<crate::ctrl_io::quasiparticle_methods::QuasiParticle> {
    scf_data.mol.ctrl.quasiparticle_methods.clone()
}

/// Is `save_gw_checkpoint` enabled *and* are we allowed to write (rank 0)?
pub fn checkpoint_enabled(scf_data: &SCF) -> bool {
    if let Some(mpi_data) = &scf_data.mol.mpi_data {
        if mpi_data.rank != 0 {
            return false;
        }
    }
    qp_control(scf_data)
        .map(|qp| qp.save_gw_checkpoint)
        .unwrap_or(false)
}

/// The configured checkpoint path (only meaningful when the feature is on).
pub fn checkpoint_path(scf_data: &SCF) -> String {
    qp_control(scf_data)
        .map(|qp| qp.gw_checkpoint_path)
        .unwrap_or_else(|| String::from("./gw_checkpoint.h5"))
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// Serialise the current post-SCF state to `gw_checkpoint_path`.
///
/// The archive is first written to `<path>.tmp` and renamed onto `<path>` after
/// it is closed, so a process killed while writing leaves the previous
/// checkpoint untouched.
pub fn save_gw_checkpoint(scf_data: &SCF, state: &GwCheckpointState) -> Result<()> {
    let path = checkpoint_path(scf_data);
    let tmp_path = format!("{}.tmp", path);
    if Path::new(&tmp_path).exists() {
        std::fs::remove_file(&tmp_path).with_context(|| {
            format!(
                "gw_checkpoint: cannot remove the stale temporary file '{}'",
                tmp_path
            )
        })?;
    }

    {
        let file = hdf5::File::create(&tmp_path)
            .with_context(|| format!("gw_checkpoint: cannot create '{}'", tmp_path))?;
        let root = file
            .create_group(ROOT_GROUP)
            .with_context(|| format!("gw_checkpoint: cannot create group '{}'", ROOT_GROUP))?;

        write_usize(&root, "version", GW_CHECKPOINT_VERSION)?;
        write_string(&root, "stage", state.stage.as_str())?;
        write_usize(&root, "evgw_rounds_done", state.rounds_done)?;
        write_usize(&root, "evgw_total_rounds", state.total_rounds)?;
        write_usize(&root, "evgw_converged", usize::from(state.converged))?;
        write_f64(&root, "evgw_last_residual", state.last_residual)?;
        write_f64(&root, "evgw_last_dg", state.last_dg)?;
        write_usize(&root, "diis_space", state.diis_space)?;
        write_usize(&root, "diis_min_history", state.diis_min_history)?;
        write_usize(&root, "diis_safeguard", usize::from(state.diis_safeguard))?;

        // ---- metadata ----
        let meta = root.create_group("meta").context("gw_checkpoint: meta group")?;
        write_usize(&meta, "natm", state.natm)?;
        write_usize(&meta, "num_basis", state.num_basis)?;
        write_usize(&meta, "num_state", state.num_state)?;
        write_vec_f64(&meta, "num_elec", &state.num_elec)?;
        write_f64(&meta, "charge", state.charge)?;
        write_f64(&meta, "spin", state.spin)?;
        write_usize(&meta, "spin_channel", state.spin_channel)?;
        write_f64(&meta, "scf_energy", scf_data.scf_energy)?;
        write_f64(&meta, "nuc_energy", scf_data.nuc_energy)?;
        write_string(&meta, "scf_type", scf_type_as_str(state.scf_type))?;
        write_string(&meta, "basis_path", &state.basis_path)?;
        write_string(&meta, "auxbas_path", &state.auxbas_path)?;
        write_string(&meta, "geom", &geom_to_json(&scf_data.mol.geom))?;

        // ---- SCF arrays ----
        let spin_channel = scf_data.mol.spin_channel.min(2);
        let scf = root.create_group("scf").context("gw_checkpoint: scf group")?;
        write_usize(&scf, "spin_channel", spin_channel)?;
        write_vec_usize(&scf, "homo", &scf_data.homo)?;
        write_vec_usize(&scf, "lumo", &scf_data.lumo)?;
        for i_spin in 0..spin_channel {
            write_vec_f64(
                &scf,
                &format!("eigenvalues_{}", i_spin),
                &scf_data.eigenvalues[i_spin],
            )?;
            write_vec_f64(
                &scf,
                &format!("occupation_{}", i_spin),
                &scf_data.occupation[i_spin],
            )?;
            write_matrix(
                &scf,
                &format!("eigenvectors_{}", i_spin),
                &scf_data.eigenvectors[i_spin],
            )?;
            let hamiltonian = scf_data.hamiltonian[i_spin]
                .to_matrixfull()
                .unwrap_or_else(MatrixFull::empty);
            write_matrix(&scf, &format!("hamiltonian_{}", i_spin), &hamiltonian)?;
        }
        let n_dm = scf_data.density_matrix.len().min(2);
        write_usize(&scf, "num_density_matrices", n_dm)?;
        for i_spin in 0..n_dm {
            write_matrix(
                &scf,
                &format!("density_matrix_{}", i_spin),
                &scf_data.density_matrix[i_spin],
            )?;
        }

        // ---- GW quasiparticle data ----
        let gw = root.create_group("gw").context("gw_checkpoint: gw group")?;
        write_vec_f64(&gw, "gwqp_g", &scf_data.gwqp.0)?;
        write_vec_f64(&gw, "gwqp_w", &scf_data.gwqp.1)?;
        for i_spin in 0..2 {
            write_vec_f64(
                &gw,
                &format!("gwqp_spin_g_{}", i_spin),
                &scf_data.gwqp_spin.0[i_spin],
            )?;
            write_vec_f64(
                &gw,
                &format!("gwqp_spin_w_{}", i_spin),
                &scf_data.gwqp_spin.1[i_spin],
            )?;
        }
        write_vec_f64(
            &gw,
            "renormalized_singles_particles",
            &scf_data.renormalized_singles_particles,
        )?;

        // ---- evGW DIIS history ----
        let diis = root.create_group("diis").context("gw_checkpoint: diis group")?;
        write_usize(&diis, "n_vectors", state.diis_history.len())?;
        for (i_vec, (history, error)) in state
            .diis_history
            .iter()
            .zip(state.diis_errors.iter())
            .enumerate()
        {
            write_vec_f64(&diis, &format!("history_{}", i_vec), history)?;
            write_vec_f64(&diis, &format!("errors_{}", i_vec), error)?;
        }

        file.close()
            .with_context(|| format!("gw_checkpoint: cannot close '{}'", tmp_path))?;
    }

    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("gw_checkpoint: cannot move '{}' onto '{}'", tmp_path, path))?;

    println!(
        "[gw_checkpoint] wrote '{}' (version {}, stage={}, {})",
        path,
        GW_CHECKPOINT_VERSION,
        state.stage.as_str(),
        state.describe()
    );
    Ok(())
}

/// Convenience hook for "a complete GW pass just finished".
///
/// Does nothing when `save_gw_checkpoint` is off, when we are not on rank 0, or
/// when the stored quasiparticle vector is empty (e.g. the HOMO/LUMO-only or
/// spectrum-test shortcuts, whose state is not a resumable GW result).
pub fn save_gw_finished_checkpoint(scf_data: &SCF) {
    if !checkpoint_enabled(scf_data) {
        return;
    }
    if scf_data.gwqp.0.is_empty() {
        println!(
            "[gw_checkpoint] skipped: no quasiparticle energies are available \
             (this GW variant does not produce a resumable spectrum)"
        );
        return;
    }
    let mut state = GwCheckpointState::from_scf(scf_data);
    state.stage = GwCheckpointStage::GwFinished;
    state.total_rounds = qp_control(scf_data).map(|qp| qp.evgw_rounds).unwrap_or(0);
    if let Err(e) = save_gw_checkpoint(scf_data, &state) {
        println!(
            "[gw_checkpoint] WARNING: could not write the checkpoint: {:#}",
            e
        );
    }
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

/// Read `gw_checkpoint_path`, validate it against the current input and write
/// the stored state back onto `scf_data`.
///
/// Any incompatibility (version, basis, electron count, spin, geometry, array
/// dimensions) is a hard error: silently continuing with mismatched data would
/// produce a physically meaningless BSE spectrum.
pub fn load_gw_checkpoint(scf_data: &mut SCF) -> Result<GwCheckpointState> {
    let path = checkpoint_path(scf_data);
    if !Path::new(&path).exists() {
        bail!(
            "gw_checkpoint: 'resume_from_checkpoint = true' but the checkpoint file '{}' does not exist.\n\
             Hint: run once with 'save_gw_checkpoint = true' to create it, or set \
             'resume_from_checkpoint = false' to recompute the SCF/GW/evGW work from scratch.",
            path
        );
    }

    let file = hdf5::File::open(&path)
        .with_context(|| format!("gw_checkpoint: cannot open '{}'", path))?;
    let root = file.group(ROOT_GROUP).with_context(|| {
        format!(
            "gw_checkpoint: '{}' is not a REST GW checkpoint (missing group '{}').\n\
             Delete the file or set 'resume_from_checkpoint = false'.",
            path, ROOT_GROUP
        )
    })?;

    let version = read_usize(&root, "version")?;
    if version != GW_CHECKPOINT_VERSION {
        bail!(
            "gw_checkpoint: '{}' was written in format version {} but this build expects version {}.\n\
             Delete the checkpoint file (or set 'resume_from_checkpoint = false') and rerun.",
            path,
            version,
            GW_CHECKPOINT_VERSION
        );
    }

    let stage = GwCheckpointStage::from_str(&read_string(&root, "stage")?)?;

    // ---- metadata + validation ----
    let meta = root
        .group("meta")
        .context("gw_checkpoint: missing 'meta' group")?;
    let natm = read_usize(&meta, "natm")?;
    let num_basis = read_usize(&meta, "num_basis")?;
    let num_state = read_usize(&meta, "num_state")?;
    let num_elec = read_vec_f64(&meta, "num_elec")?;
    let charge = read_f64(&meta, "charge")?;
    let spin = read_f64(&meta, "spin")?;
    let spin_channel = read_usize(&meta, "spin_channel")?;
    let scf_type = scf_type_from_str(&read_string(&meta, "scf_type")?)?;
    let basis_path = read_string(&meta, "basis_path")?;
    let auxbas_path = read_string(&meta, "auxbas_path")?;

    let mismatch = |what: &str, saved: String, current: String| -> anyhow::Error {
        anyhow!(
            "gw_checkpoint: '{}' is incompatible with the current input: {} differs \
             (checkpoint: {}, current input: {}).\n\
             The checkpoint was produced by a different calculation. Fix the input so that it \
             matches, or set 'resume_from_checkpoint = false' (and delete '{}') to recompute \
             everything from scratch.",
            path,
            what,
            saved,
            current,
            path
        )
    };

    if natm != scf_data.mol.geom.elem.len() {
        return Err(mismatch(
            "the number of atoms",
            natm.to_string(),
            scf_data.mol.geom.elem.len().to_string(),
        ));
    }
    if num_basis != scf_data.mol.num_basis {
        return Err(mismatch(
            "the number of basis functions",
            num_basis.to_string(),
            scf_data.mol.num_basis.to_string(),
        ));
    }
    if num_state != scf_data.mol.num_state {
        return Err(mismatch(
            "the number of molecular orbitals",
            num_state.to_string(),
            scf_data.mol.num_state.to_string(),
        ));
    }
    if spin_channel != scf_data.mol.spin_channel {
        return Err(mismatch(
            "the spin channel",
            spin_channel.to_string(),
            scf_data.mol.spin_channel.to_string(),
        ));
    }
    if num_elec.len() != 3
        || num_elec
            .iter()
            .zip(scf_data.mol.num_elec.iter())
            .any(|(a, b)| (a - b).abs() > 1.0e-8)
    {
        return Err(mismatch(
            "the number of electrons",
            format!("{:?}", num_elec),
            format!("{:?}", scf_data.mol.num_elec),
        ));
    }
    if (charge - scf_data.mol.ctrl.charge).abs() > 1.0e-8 {
        return Err(mismatch(
            "the total charge",
            charge.to_string(),
            scf_data.mol.ctrl.charge.to_string(),
        ));
    }
    if (spin - scf_data.mol.ctrl.spin).abs() > 1.0e-8 {
        return Err(mismatch(
            "the spin",
            spin.to_string(),
            scf_data.mol.ctrl.spin.to_string(),
        ));
    }
    if basis_path != scf_data.mol.ctrl.basis_path {
        return Err(mismatch(
            "the basis-set path",
            basis_path,
            scf_data.mol.ctrl.basis_path.clone(),
        ));
    }
    if auxbas_path != scf_data.mol.ctrl.auxbas_path {
        return Err(mismatch(
            "the auxiliary-basis-set path",
            auxbas_path,
            scf_data.mol.ctrl.auxbas_path.clone(),
        ));
    }
    if scf_type_as_str(scf_type) != scf_type_as_str(scf_data.scftype) {
        return Err(mismatch(
            "the SCF type",
            scf_type_as_str(scf_type).to_string(),
            scf_type_as_str(scf_data.scftype).to_string(),
        ));
    }

    // Geometry: same elements, same positions.
    let saved_geom = geom_from_json(&read_string(&meta, "geom")?).ok_or_else(|| {
        anyhow!(
            "gw_checkpoint: the stored geometry in '{}' cannot be decoded",
            path
        )
    })?;
    if saved_geom.elem != scf_data.mol.geom.elem {
        return Err(mismatch(
            "the chemical elements / their order",
            format!("{:?}", saved_geom.elem),
            format!("{:?}", scf_data.mol.geom.elem),
        ));
    }
    if saved_geom.position.size != scf_data.mol.geom.position.size {
        return Err(mismatch(
            "the geometry dimensions",
            format!("{:?}", saved_geom.position.size),
            format!("{:?}", scf_data.mol.geom.position.size),
        ));
    }
    for (i, (saved, current)) in saved_geom
        .position
        .iter()
        .zip(scf_data.mol.geom.position.iter())
        .enumerate()
    {
        if (saved - current).abs() > GEOM_TOL {
            return Err(mismatch(
                &format!(
                    "the geometry (coordinate #{}: {:.10} vs {:.10})",
                    i, saved, current
                ),
                format!("{:.10}", saved),
                format!("{:.10}", current),
            ));
        }
    }

    // ---- restore the SCF arrays ----
    let scf = root.group("scf").context("gw_checkpoint: missing 'scf' group")?;
    let saved_spin_channel = read_usize(&scf, "spin_channel")?;
    let saved_homo = read_vec_usize(&scf, "homo")?;
    let saved_lumo = read_vec_usize(&scf, "lumo")?;
    let mut n_restored_spin = 0_usize;
    for i_spin in 0..saved_spin_channel.min(2) {
        let eigenvalues = read_vec_f64(&scf, &format!("eigenvalues_{}", i_spin))?;
        if eigenvalues.len() != scf_data.mol.num_state {
            bail!(
                "gw_checkpoint: '{}' is corrupt: it stores {} MO energies for spin {} but the \
                 current input has {} orbitals.",
                path,
                eigenvalues.len(),
                i_spin,
                scf_data.mol.num_state
            );
        }
        scf_data.eigenvalues[i_spin] = eigenvalues;
        scf_data.occupation[i_spin] = read_vec_f64(&scf, &format!("occupation_{}", i_spin))?;
        scf_data.eigenvectors[i_spin] = read_matrix(&scf, &format!("eigenvectors_{}", i_spin))?;
        let hamiltonian = read_matrix(&scf, &format!("hamiltonian_{}", i_spin))?;
        if !hamiltonian.data.is_empty() {
            scf_data.hamiltonian[i_spin] =
                MatrixUpper::from_vec(hamiltonian.data.len(), hamiltonian.data)
                    .ok_or_else(|| anyhow!("gw_checkpoint: cannot rebuild the Fock matrix"))?;
        }
        n_restored_spin += 1;
    }
    let n_dm = read_usize(&scf, "num_density_matrices")?;
    for i_spin in 0..n_dm.min(2) {
        scf_data.density_matrix[i_spin] = read_matrix(&scf, &format!("density_matrix_{}", i_spin))?;
    }
    if saved_homo.len() == 2 {
        scf_data.homo = [saved_homo[0], saved_homo[1]];
    }
    if saved_lumo.len() == 2 {
        scf_data.lumo = [saved_lumo[0], saved_lumo[1]];
    }
    scf_data.scf_energy = read_f64(&meta, "scf_energy")?;
    // `nuc_energy` is a pure function of geometry + basis and is rebuilt by
    // `prepare_necessary_integrals`, so a mismatch here also indicates a
    // basis/geometry inconsistency; report it rather than silently overwrite.
    let saved_nuc_energy = read_f64(&meta, "nuc_energy")?;
    if (saved_nuc_energy - scf_data.nuc_energy).abs() > 1.0e-8 {
        println!(
            "[gw_checkpoint] WARNING: the nuclear energy rebuilt from the current geometry/basis \
             is {:18.10} Ha but the checkpoint stores {:18.10} Ha (difference {:.3e} Ha). \
             The electronic structure is restored from the checkpoint, so this may indicate a \
             small inconsistency in the input; continuing with the checkpoint value.",
            scf_data.nuc_energy,
            saved_nuc_energy,
            (saved_nuc_energy - scf_data.nuc_energy).abs()
        );
    }
    scf_data.nuc_energy = saved_nuc_energy;

    // ---- restore the GW state ----
    let gw = root.group("gw").context("gw_checkpoint: missing 'gw' group")?;
    let gwqp_g = read_vec_f64(&gw, "gwqp_g")?;
    let gwqp_w = read_vec_f64(&gw, "gwqp_w")?;
    if gwqp_g.is_empty() {
        bail!(
            "gw_checkpoint: '{}' holds no quasiparticle energies; it cannot be used to resume.",
            path
        );
    }
    if gwqp_g.len() > scf_data.mol.num_state {
        bail!(
            "gw_checkpoint: '{}' is corrupt: {} quasiparticle energies were stored but the \
             current input has only {} orbitals.",
            path,
            gwqp_g.len(),
            scf_data.mol.num_state
        );
    }
    scf_data.gwqp = (gwqp_g, gwqp_w);
    for i_spin in 0..2 {
        scf_data.gwqp_spin.0[i_spin] = read_vec_f64(&gw, &format!("gwqp_spin_g_{}", i_spin))?;
        scf_data.gwqp_spin.1[i_spin] = read_vec_f64(&gw, &format!("gwqp_spin_w_{}", i_spin))?;
    }
    scf_data.renormalized_singles_particles =
        read_vec_f64(&gw, "renormalized_singles_particles")?;

    // ---- restore the evGW outer-loop state ----
    let rounds_done = read_usize(&root, "evgw_rounds_done")?;
    let total_rounds = read_usize(&root, "evgw_total_rounds")?;
    let converged = read_usize(&root, "evgw_converged")? != 0;
    let last_residual = read_f64(&root, "evgw_last_residual")?;
    let last_dg = read_f64(&root, "evgw_last_dg")?;
    let diis_space = read_usize(&root, "diis_space")?;
    let diis_min_history = read_usize(&root, "diis_min_history")?;
    let diis_safeguard = read_usize(&root, "diis_safeguard")? != 0;
    let mut diis_history: Vec<Vec<f64>> = Vec::new();
    let mut diis_errors: Vec<Vec<f64>> = Vec::new();
    if let Ok(diis) = root.group("diis") {
        let n_vectors = read_usize(&diis, "n_vectors").unwrap_or(0);
        for i_vec in 0..n_vectors {
            let history = read_vec_f64(&diis, &format!("history_{}", i_vec))?;
            let error = read_vec_f64(&diis, &format!("errors_{}", i_vec))?;
            if history.len() != error.len() {
                bail!(
                    "gw_checkpoint: '{}' is corrupt: DIIS vector {} has {} entries but {} errors.",
                    path,
                    i_vec,
                    history.len(),
                    error.len()
                );
            }
            diis_history.push(history);
            diis_errors.push(error);
        }
    }

    file.close()
        .with_context(|| format!("gw_checkpoint: cannot close '{}'", path))?;

    let state = GwCheckpointState {
        stage,
        rounds_done,
        total_rounds,
        converged,
        last_residual,
        last_dg,
        diis_history,
        diis_errors,
        diis_space,
        diis_min_history,
        diis_safeguard,
        natm,
        num_basis,
        num_state,
        num_elec: [
            num_elec.first().copied().unwrap_or(0.0),
            num_elec.get(1).copied().unwrap_or(0.0),
            num_elec.get(2).copied().unwrap_or(0.0),
        ],
        charge,
        spin,
        spin_channel,
        basis_path,
        auxbas_path,
        scf_type,
    };

    println!("--------------------------------------------------------------------------------");
    println!("[gw_checkpoint] resuming from '{}'", path);
    println!(
        "[gw_checkpoint] restored SCF state: {} spin channel(s) ({} MO energy set(s), {} MO coefficient set(s)), \
         {} density matrix/matrices, E_SCF = {:18.10} Ha, E_nuc = {:18.10} Ha",
        n_restored_spin,
        n_restored_spin,
        n_restored_spin,
        n_dm.min(2),
        scf_data.scf_energy,
        scf_data.nuc_energy
    );
    println!(
        "[gw_checkpoint] restored GW state: {} quasiparticle energies (G) / {} (W), {} renormalized-singles particles",
        scf_data.gwqp.0.len(),
        scf_data.gwqp.1.len(),
        scf_data.renormalized_singles_particles.len()
    );
    println!("[gw_checkpoint] checkpoint represents: {}", state.describe());
    if state.stage == GwCheckpointStage::EvgwRound {
        println!(
            "[gw_checkpoint] => the evGW loop will continue from round {} ({} DIIS vector(s) restored).",
            state.rounds_done + 1,
            state.diis_history.len()
        );
    } else {
        println!("[gw_checkpoint] => the GW/evGW work is complete; it will NOT be recomputed.");
    }
    println!("--------------------------------------------------------------------------------");

    Ok(state)
}

/// Log lines used by the driver when the SCF iterations are skipped because a
/// checkpoint was restored.
pub fn announce_scf_skipped(scf_data: &SCF) {
    println!("--------------------------------------------------------------------------------");
    println!(
        "[gw_checkpoint] 'resume_from_checkpoint = true': the SCF iterations are SKIPPED -- the \
         converged MO energies, MO coefficients, density matrix and occupations are read from \
         '{}' instead.",
        checkpoint_path(scf_data)
    );
    println!(
        "[gw_checkpoint] note: the molecule, the basis and the integrals are still built from the \
         input file (they are not stored in the archive)."
    );
    println!("--------------------------------------------------------------------------------");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
    use crate::molecule_io::Molecule;

    fn tmp_path(tag: &str) -> String {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rest_gw_ckpt_{}_{}_{}.h5",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path.to_string_lossy().to_string()
    }

    /// A minimal SCF object with hand-filled MO data (no real molecule/basis).
    fn synthetic_scf(path: &str, n: usize, charge: f64) -> SCF {
        let mut mol = Molecule::init_mol();
        mol.num_basis = n;
        mol.num_state = n;
        mol.num_elec = [2.0, 1.0, 1.0];
        mol.spin_channel = 1;
        mol.ctrl.charge = charge;
        mol.ctrl.spin = 0.0;
        let mut qp = QuasiParticle::default();
        qp.save_gw_checkpoint = true;
        qp.gw_checkpoint_path = path.to_string();
        mol.ctrl.quasiparticle_methods = Some(qp);

        let mut scf = SCF::init_scf(&mol);
        scf.eigenvalues[0] = (0..n).map(|i| -0.5 + 0.1 * i as f64).collect();
        scf.occupation[0] = (0..n).map(|i| if 2 * i < n { 2.0 } else { 0.0 }).collect();
        scf.eigenvectors[0] =
            MatrixFull::from_vec([n, n], (0..n * n).map(|i| (i as f64) * 0.01).collect()).unwrap();
        scf.density_matrix[0] =
            MatrixFull::from_vec([n, n], vec![0.25; n * n]).unwrap();
        scf.homo = [n / 2 - 1, 0];
        scf.lumo = [n / 2, 0];
        scf.scf_energy = -1.25;
        // `nuc_energy` is rebuilt by `prepare_necessary_integrals` in a real
        // run; here it must match what the loader recomputes (0.0 for the empty
        // default molecule) for the comparison warning not to fire.
        scf.nuc_energy = 0.0;
        scf.gwqp = (
            (0..n).map(|i| -0.4 + 0.2 * i as f64).collect(),
            (0..n).map(|i| -0.3 + 0.2 * i as f64).collect(),
        );
        scf.renormalized_singles_particles = vec![0.1, 0.2];
        scf
    }

    #[test]
    fn checkpoint_round_trips_scf_and_evgw_state() {
        let path = tmp_path("roundtrip");
        let source = synthetic_scf(&path, 4, 0.0);

        let mut state = GwCheckpointState::from_scf(&source);
        state.stage = GwCheckpointStage::EvgwRound;
        state.rounds_done = 3;
        state.total_rounds = 10;
        state.converged = false;
        state.last_residual = 1.5e-3;
        state.last_dg = 2.5e-4;
        state.diis_history = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.11, 0.21, 0.31, 0.41]];
        state.diis_errors = vec![vec![0.01, 0.02, 0.03, 0.04], vec![0.011, 0.021, 0.031, 0.041]];
        save_gw_checkpoint(&source, &state).expect("the checkpoint must be written");

        // The temporary file must be gone (it was renamed onto the target).
        assert!(!Path::new(&format!("{}.tmp", path)).exists());
        assert!(Path::new(&path).exists());

        let mut target = synthetic_scf(&path, 4, 0.0);
        target.eigenvalues[0] = vec![0.0; 4];
        let loaded = load_gw_checkpoint(&mut target).expect("the checkpoint must load");

        assert_eq!(loaded.stage, GwCheckpointStage::EvgwRound);
        assert_eq!(loaded.rounds_done, 3);
        assert_eq!(loaded.total_rounds, 10);
        assert!(!loaded.converged);
        assert!((loaded.last_residual - 1.5e-3).abs() < 1e-15);
        assert_eq!(loaded.diis_history.len(), 2);
        assert_eq!(loaded.diis_errors[1], vec![0.011, 0.021, 0.031, 0.041]);

        assert_eq!(target.eigenvalues[0], source.eigenvalues[0]);
        assert_eq!(target.occupation[0], source.occupation[0]);
        assert_eq!(target.eigenvectors[0].size, [4, 4]);
        assert_eq!(target.eigenvectors[0].data, source.eigenvectors[0].data);
        assert_eq!(target.density_matrix[0].data, source.density_matrix[0].data);
        assert_eq!(target.homo, source.homo);
        assert_eq!(target.lumo, source.lumo);
        assert!((target.scf_energy - source.scf_energy).abs() < 1e-15);
        assert_eq!(target.gwqp.0, source.gwqp.0);
        assert_eq!(target.gwqp.1, source.gwqp.1);
        assert_eq!(
            target.renormalized_singles_particles,
            source.renormalized_singles_particles
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn checkpoint_rejects_mismatched_input() {
        let path = tmp_path("mismatch");
        let source = synthetic_scf(&path, 4, 0.0);
        let state = GwCheckpointState::from_scf(&source);
        save_gw_checkpoint(&source, &state).expect("the checkpoint must be written");

        // Same molecule but a different total charge -> must be refused.
        let mut target = synthetic_scf(&path, 4, 1.0);
        let error = match load_gw_checkpoint(&mut target) {
            Ok(_) => panic!("a charge mismatch must abort the resume"),
            Err(error) => error,
        };
        let message = format!("{:#}", error);
        assert!(
            message.contains("total charge") && message.contains("resume_from_checkpoint = false"),
            "unexpected error message: {}",
            message
        );

        // A different number of basis functions -> must be refused as well.
        let mut target = synthetic_scf(&path, 5, 0.0);
        assert!(load_gw_checkpoint(&mut target).is_err());

        std::fs::remove_file(&path).ok();
    }
}
