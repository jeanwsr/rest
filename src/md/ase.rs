use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const ASE_TIME_PER_FS: f64 = 10.180512809629851;

const PY_ASE_MD: &str = r#"
import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.constraints import FixAtoms

class _RestAseMD:
    def __init__(self, symbols, masses, positions_ang, velocities_ang_fs,
                 temperature_k, friction, friction_units, dt_fs, ensemble,
                 seed, fixed):
        atoms = Atoms(symbols=symbols,
                      positions=np.asarray(positions_ang, dtype=float).reshape(-1, 3),
                      masses=masses)
        self.fixed = list(fixed)
        if self.fixed:
            atoms.set_constraint(FixAtoms(indices=self.fixed))
        rng = np.random.default_rng(seed)
        if velocities_ang_fs is None:
            if temperature_k > 0.0:
                MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
                Stationary(atoms)
        else:
            atoms.set_velocities(np.asarray(velocities_ang_fs, dtype=float).reshape(-1, 3) / units.fs)
        timestep = dt_fs * units.fs
        if ensemble == 'nvt':
            fric = friction if friction_units == 'ase' else friction / units.fs
            self.dyn = Langevin(atoms, timestep=timestep, temperature_K=temperature_k,
                                friction=fric, fixcm=True, rng=rng)
        else:
            self.dyn = VelocityVerlet(atoms, timestep=timestep)
        self.atoms = atoms

    def velocities_ang_fs(self):
        v = self.atoms.get_velocities()
        return [c for row in (v * float(units.fs)) for c in row]

    def step(self, forces_ev_ang):
        f = np.asarray(forces_ev_ang, dtype=float).reshape(-1, 3)
        for i in self.fixed:
            f[i] = 0.0
        self.dyn.step(forces=f)
        p = self.atoms.get_positions()
        return [c for row in p for c in row], self.velocities_ang_fs()

class _RestAseOpt:
    def __init__(self, symbols, masses, positions_ang, fixed):
        atoms = Atoms(symbols=symbols,
                      positions=np.asarray(positions_ang, dtype=float).reshape(-1, 3),
                      masses=masses)
        self.fixed = list(fixed)
        if self.fixed:
            atoms.set_constraint(FixAtoms(indices=self.fixed))
        from ase.optimize import FIRE
        self.dyn = FIRE(atoms, logfile=None)
        self.atoms = atoms

    def step(self, forces_ev_ang):
        f = np.asarray(forces_ev_ang, dtype=float).reshape(-1, 3)
        for i in self.fixed:
            f[i] = 0.0
        self.dyn.step(f=f)
        p = self.atoms.get_positions()
        return [c for row in p for c in row]
"#;

pub struct AseMd {
    driver: Py<PyAny>,
}

impl AseMd {

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbols: &[String],
        masses_amu: &[f64],
        pos_ang: &[f64],
        velocities_ang_fs: Option<&[f64]>,
        temperature_k: f64,
        friction: f64,
        friction_units: &str,
        dt_fs: f64,
        ensemble: &str,
        seed: u64,
        fixed_rows: &[usize],
    ) -> anyhow::Result<Self> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<Self> {
            let code = std::ffi::CString::new(PY_ASE_MD).unwrap();
            let locals = PyDict::new(py);

            py.run(&code, Some(&locals), Some(&locals))?;
            let cls = locals
                .get_item("_RestAseMD")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_RestAseMD missing"))?;
            let driver = cls.call1((
                symbols.to_vec(),
                masses_amu.to_vec(),
                pos_ang.to_vec(),
                velocities_ang_fs.map(|v| v.to_vec()),
                temperature_k,
                friction,
                friction_units.to_string(),
                dt_fs,
                ensemble.to_string(),
                seed,
                fixed_rows.to_vec(),
            ))?;
            Ok(AseMd { driver: driver.into() })
        })
        .map_err(|e| anyhow::anyhow!("ASE MD engine construction failed: {}", e))
    }

    pub fn velocities_ang_fs(&self) -> Vec<f64> {
        Python::with_gil(|py| -> Vec<f64> {
            self.driver
                .bind(py)
                .call_method0("velocities_ang_fs")
                .and_then(|r| r.extract())
                .unwrap_or_default()
        })
    }

    pub fn step(&self, forces_ev_ang: &[f64]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        Python::with_gil(|py| -> PyResult<(Vec<f64>, Vec<f64>)> {
            let res = self.driver.bind(py).call_method1("step", (forces_ev_ang.to_vec(),))?;
            let pos: Vec<f64> = res.get_item(0)?.extract()?;
            let vel: Vec<f64> = res.get_item(1)?.extract()?;
            Ok((pos, vel))
        })
        .map_err(|e| anyhow::anyhow!("ASE MD step failed: {}", e))
    }
}

pub struct AseOpt {
    driver: Py<PyAny>,
}

impl AseOpt {
    pub fn new(
        symbols: &[String],
        masses_amu: &[f64],
        pos_ang: &[f64],
        frozen: &[usize],
    ) -> anyhow::Result<Self> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<Self> {
            let code = std::ffi::CString::new(PY_ASE_MD).unwrap();
            let locals = PyDict::new(py);
            py.run(&code, Some(&locals), Some(&locals))?;
            let cls = locals
                .get_item("_RestAseOpt")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_RestAseOpt missing"))?;
            let driver = cls.call1((
                symbols.to_vec(),
                masses_amu.to_vec(),
                pos_ang.to_vec(),
                frozen.to_vec(),
            ))?;
            Ok(AseOpt { driver: driver.into() })
        })
        .map_err(|e| anyhow::anyhow!("ASE opt engine construction failed: {}", e))
    }

    pub fn step(&self, forces_ev_ang: &[f64]) -> anyhow::Result<Vec<f64>> {
        Python::with_gil(|py| -> PyResult<Vec<f64>> {
            self.driver
                .bind(py)
                .call_method1("step", (forces_ev_ang.to_vec(),))
                .and_then(|r| r.extract())
        })
        .map_err(|e| anyhow::anyhow!("ASE optimization step failed: {}", e))
    }
}
