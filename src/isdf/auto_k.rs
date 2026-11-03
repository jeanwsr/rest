//! ISDF's empirical k estimator. RDKit runs in the embedded Python interpreter;
//! no Python subprocess or external Python script is launched at runtime.

use anyhow::{bail, ensure, Context, Result};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Deserialize)]
struct AutoKInput {
    ctrl: AutoKControl,
    geom: AutoKGeometry,
}

#[derive(Deserialize)]
struct AutoKControl {
    basis_path: String,
}

#[derive(Deserialize)]
struct AutoKGeometry {
    position: Positions,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Positions {
    Text(String),
    Lines(Vec<String>),
}

impl Positions {
    fn xyz(self) -> Result<(String, usize)> {
        let text = match self {
            Self::Text(text) => text,
            Self::Lines(lines) => lines.join("\n"),
        };
        // Match set_k_auto.py: use the input coordinates verbatim, retaining
        // four-field atom records. Do not rebuild from Molecule's Bohr geometry.
        let atoms: Vec<_> = text
            .split('\n')
            .filter(|line| line.split_whitespace().count() == 4)
            .map(str::trim_start)
            .collect();
        ensure!(
            !atoms.is_empty(),
            "No four-field atom records in geom.position"
        );
        Ok((
            format!("{}\n\n{}", atoms.len(), atoms.join("\n")),
            atoms.len(),
        ))
    }
}

fn obabel_executable() -> PathBuf {
    if let Some(prefix) = std::env::var_os("REST_OBABEL_PREFIX").filter(|p| !p.is_empty()) {
        return PathBuf::from(prefix).join("bin/obabel");
    }
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/isdf/obabel_minimal/obabel_build/bin/obabel");
    if bundled.is_file() {
        bundled
    } else {
        PathBuf::from("obabel")
    }
}

fn xyz_to_smiles(xyz: &str) -> Result<String> {
    let executable = obabel_executable();
    let mut child = Command::new(&executable)
        .args(["-ixyz", "-osmi"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "Cannot start Open Babel ({}); install obabel or set REST_OBABEL_PREFIX",
                executable.display()
            )
        })?;
    // Close stdin before waiting, and reap the child even if writing failed.
    let write_result = child.stdin.take().unwrap().write_all(xyz.as_bytes());
    let output = child
        .wait_with_output()
        .context("Cannot collect Open Babel output")?;
    ensure!(
        output.status.success(),
        "Open Babel failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    write_result.context("Cannot send geometry to Open Babel")?;
    let smiles = String::from_utf8(output.stdout)
        .context("Open Babel returned non-UTF-8 SMILES")?
        .trim()
        .to_owned();
    ensure!(
        !smiles.is_empty(),
        "Open Babel returned empty SMILES; check its xyz/smi plugins: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(smiles)
}

fn atom_label(atomic_number: usize) -> Option<f64> {
    // The reference script's element table ends at Rn. Keep that domain rather
    // than silently extending its otherwise unreachable 87..118 branch.
    match atomic_number {
        1..=2 => Some(1.0),
        9 => Some(2.5),
        3..=10 => Some(2.0),
        11..=18 => Some(3.0),
        19..=36 => Some(4.0),
        37..=54 => Some(5.0),
        55..=86 => Some(6.0),
        _ => None,
    }
}

fn molecular_label(smiles: &str, input_atom_count: usize) -> Result<f64> {
    // Safe to call repeatedly, including when REST is imported from Python.
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<f64> {
        let chem = py.import("rdkit.Chem")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("sanitize", false)?;
        let molecule = chem.call_method("MolFromSmiles", (smiles,), Some(&kwargs))?;
        if molecule.is_none() {
            return Err(PyValueError::new_err(
                "RDKit could not parse Open Babel's SMILES",
            ));
        }
        let flags = chem.getattr("SANITIZE_ALL")?.extract::<u32>()?
            ^ chem.getattr("SANITIZE_CLEANUP")?.extract::<u32>()?
            ^ chem.getattr("SANITIZE_PROPERTIES")?.extract::<u32>()?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("sanitizeOps", flags)?;
        chem.call_method("SanitizeMol", (&molecule,), Some(&kwargs))?;

        let mut total_label = 0.0;
        for atom in molecule.call_method0("GetAtoms")?.try_iter()? {
            let atom = atom?;
            let z = atom.call_method0("GetAtomicNum")?.extract::<usize>()?;
            let label = atom_label(z).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Atomic number {z} is outside the ISDF automatic-k element table"
                ))
            })?;
            let implicit_h = atom.call_method0("GetNumImplicitHs")?.extract::<usize>()?;
            let in_ring = atom.call_method0("IsInRing")?.extract::<bool>()?;
            total_label += label + implicit_h as f64 + if in_ring { 1.0 } else { 0.0 };
        }
        // Deliberately use the input atom count, not the SMILES atom count.
        Ok(total_label / input_atom_count as f64)
    })
    .context("ISDF automatic k requires RDKit in the Python environment used to build REST")
}

fn k_from_label(label: f64, basis_path: &str) -> Result<usize> {
    let k = if label >= 3.0 {
        3
    } else if label >= 2.5 {
        4
    } else if label >= 2.0 {
        5
    } else {
        6
    };
    // Preserve the reference's basename handling and integer truncation.
    let basis = basis_path.rsplit('/').next().unwrap().to_ascii_lowercase();
    match basis.as_str() {
        "cc-pvdz" | "def2-sv(p)" | "def2-svp" | "6-31gs" => Ok(k),
        "cc-pvtz" | "def2-tzvp" => Ok((k as f64 * 1.8 + 1.0) as usize),
        _ => bail!("No ISDF automatic-k rule for basis {basis_path:?}; set isdf_k explicitly"),
    }
}

/// Estimate ISDF k from a REST input file using Open Babel and embedded RDKit.
///
/// The empirical rules match `set_k_auto.py`. Open Babel's executable is found
/// via `REST_OBABEL_PREFIX`, the legacy source installation, then `PATH`.
pub fn estimate_isdf_k(ctrl_file: &Path) -> Result<usize> {
    let text = std::fs::read_to_string(ctrl_file)
        .with_context(|| format!("Cannot read ISDF input {}", ctrl_file.display()))?;
    let input: AutoKInput = toml::from_str(&text)
        .with_context(|| format!("Cannot parse ISDF input {}", ctrl_file.display()))?;
    let (xyz, atom_count) = input.geom.position.xyz()?;
    let smiles = xyz_to_smiles(&xyz)?;
    let label = molecular_label(&smiles, atom_count)?;
    k_from_label(label, &input.ctrl.basis_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_rule_boundaries() {
        for (label, dz, tz) in [
            (1.999999, 6, 11),
            (2.0, 5, 10),
            (2.499999, 5, 10),
            (2.5, 4, 8),
            (2.999999, 4, 8),
            (3.0, 3, 6),
            (6.0, 3, 6),
        ] {
            for basis in ["cc-pvdz", "def2-sv(p)", "def2-svp", "6-31gs"] {
                assert_eq!(k_from_label(label, basis).unwrap(), dz);
            }
            for basis in ["cc-pvtz", "def2-tzvp"] {
                assert_eq!(k_from_label(label, basis).unwrap(), tz);
            }
        }
        assert_eq!(k_from_label(2.5, "/basis/DEF2-TZVP").unwrap(), 8);
        assert!(k_from_label(2.0, "unsupported").is_err());
    }

    #[test]
    fn geometry_formats_preserve_reference_records() {
        let rows = vec!["  O 0 0 0,".to_owned(), "H 0 0 1".to_owned()];
        let string_xyz = Positions::Text(format!("\n{}\nignored\n", rows.join("\n")))
            .xyz()
            .unwrap();
        assert_eq!(string_xyz, Positions::Lines(rows).xyz().unwrap());
        assert_eq!(string_xyz, ("2\n\nO 0 0 0,\nH 0 0 1".to_owned(), 2));
        assert!(Positions::Text("".to_owned()).xyz().is_err());
    }

    #[test]
    fn element_table_matches_reference_domain() {
        for (z, expected) in [
            (1, 1.0),
            (8, 2.0),
            (9, 2.5),
            (18, 3.0),
            (36, 4.0),
            (54, 5.0),
            (86, 6.0),
        ] {
            assert_eq!(atom_label(z), Some(expected));
        }
        assert_eq!(atom_label(0), None);
        assert_eq!(atom_label(87), None);
    }

    #[test]
    #[ignore = "requires RDKit in the embedded Python environment"]
    fn rdkit_labels_preserve_hydrogens_rings_and_sanitization() {
        for (smiles, input_atoms, expected) in [
            ("N", 4, 1.25),
            ("O", 3, 4.0 / 3.0),
            ("c1ccccc1", 12, 2.0),
            ("FF", 2, 2.5),
            ("ClCl", 2, 3.0),
            ("FS(F)(F)(F)(F)F", 7, 18.0 / 7.0),
            // Explicit bracket H is deliberately not counted as implicit H.
            ("[NH4+]", 5, 2.0 / 5.0),
        ] {
            let actual = molecular_label(smiles, input_atoms).unwrap();
            assert!(
                (actual - expected).abs() < 1e-14,
                "{smiles}: {actual} != {expected}"
            );
        }
        assert!(molecular_label("[Rn]", 1).is_ok());
        assert!(molecular_label("[Fr]", 1).is_err());
        assert!(molecular_label("invalid_smiles", 1).is_err());
    }
}
