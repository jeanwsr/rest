use std::cell::RefCell;

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const ASE_TIME_PER_FS: f64 = 10.180512809629851;

type EvalFn = dyn FnMut(&[f64], Option<&[f64]>) -> anyhow::Result<(f64, Vec<f64>)>;

#[pyclass(unsendable)]
struct PyForceHook {
    slot: RefCell<Option<Box<EvalFn>>>,
}

#[pymethods]
impl PyForceHook {
    #[pyo3(signature = (positions, cell=None))]
    fn __call__(
        &self,
        positions: Vec<f64>,
        cell: Option<Vec<f64>>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let mut slot = self.slot.borrow_mut();
        let f = slot.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("REST MD: force hook not initialized")
        })?;
        f(&positions, cell.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

const PY_ASE_MD: &str = r#"
import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

class _RestCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, hook):
        Calculator.__init__(self)
        self._hook = hook

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        Calculator.calculate(self, atoms)
        have_cell = getattr(self.atoms, 'cell', None) is not None and self.atoms.cell.rank == 3
        cell = [float(self.atoms.cell[i, i]) for i in range(3)] if have_cell else None
        energy, forces = self._hook(list(atoms.get_positions().flatten()), cell)
        f = np.asarray(forces, dtype=float).reshape(-1, 3)
        self.results = {'energy': energy, 'forces': f}

class _RestAseMD:
    def __init__(self, symbols, masses, positions_ang, velocities_ang_fs,
                 temperature_k, friction, friction_units, dt_fs, ensemble,
                 seed, fixed, hook, cell_ang=None, pressure_bar=1.01325,
                 taut_fs=100.0, taup_fs=2000.0, compressibility_au=None,
                 baro_interval=25, baro_scale=0.02,
                 xtc_path='', xtc_interval=0):
        atoms = Atoms(symbols=symbols,
                      positions=np.asarray(positions_ang, dtype=float).reshape(-1, 3),
                      masses=masses)
        if cell_ang is not None:
            atoms.set_cell(np.diag(np.asarray(cell_ang, dtype=float)))
            atoms.pbc = True
        if fixed:
            atoms.set_constraint(FixAtoms(indices=list(fixed)))
        atoms.calc = _RestCalculator(hook)
        self._hook = hook
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._temp_k = temperature_k
        self._xtc_path = str(xtc_path)
        self._xtc_interval = int(xtc_interval)
        self._xtc_fh = None
        self._n = 0
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
        elif ensemble == 'npt':
            fric = friction if friction_units == 'ase' else friction / units.fs
            self.dyn = Langevin(atoms, timestep=timestep, temperature_K=temperature_k,
                                friction=fric, fixcm=True, rng=rng)
            self._baro = {'interval': max(1, int(baro_interval)),
                          'p_bar': float(pressure_bar),
                          'delta': float(baro_scale),
                          'hist': [], 'n': 0}
        else:
            self.dyn = VelocityVerlet(atoms, timestep=timestep)
        self.atoms = atoms

    def cell_ang(self):
        if getattr(self.atoms, 'cell', None) is not None and self.atoms.cell.rank == 3:
            return [float(self.atoms.cell[i, i]) for i in range(3)]
        return None

    def velocities_ang_fs(self):
        v = self.atoms.get_velocities()
        return [c for row in (v * float(units.fs)) for c in row]

    def step(self):
        self.dyn.step()
        b = getattr(self, '_baro', None)
        if b:
            b['n'] += 1
            if b['n'] % b['interval'] == 0:
                self._volume_mc(b)
        self._n += 1
        if self._xtc_interval > 0 and self._xtc_path and self._n % self._xtc_interval == 0:
            self._xtc_write()
        p = self.atoms.get_positions()
        return [c for row in p for c in row], self.velocities_ang_fs()

    def _xtc_write(self):
        if self._xtc_fh is None:
            try:
                from mdtraj.formats import XTCTrajectoryFile
            except ImportError as e:
                raise RuntimeError(
                    'md output "xtc" requires the mdtraj package '
                    '(pip install mdtraj)') from e
            self._xtc_fh = XTCTrajectoryFile(self._xtc_path, 'w')
        atoms = self.atoms
        p = np.asarray(atoms.get_positions()) / 10.0
        have_cell = getattr(atoms, 'cell', None) is not None and atoms.cell.rank == 3
        box = None
        if have_cell:
            edges = np.array([float(atoms.cell[i, i]) for i in range(3)]) / 10.0
            box = np.diag(edges)
        self._xtc_fh.write(p, box=box)

    def close_xtc(self):
        if getattr(self, '_xtc_fh', None) is not None:
            self._xtc_fh.close()
            self._xtc_fh = None

    def _volume_mc(self, b):
        atoms = self.atoms
        V0 = atoms.get_volume()
        n = len(atoms)
        kT = units.kB * self._temp_k
        p_eva3 = b['p_bar'] * units.bar
        s = 1.0 + b['delta'] * (2.0 * self._rng.random() - 1.0)
        E0 = atoms.get_potential_energy()
        cell0 = np.array(atoms.get_cell())
        pos0 = atoms.get_positions().copy()
        atoms.set_cell(np.diag(np.diag(cell0) * s), scale_atoms=True)
        E1 = atoms.get_potential_energy()
        dG = (E1 - E0) + p_eva3 * V0 * (s ** 3 - 1.0) - n * kT * 3.0 * np.log(s)
        ok = dG <= 0.0 or self._rng.random() < np.exp(-dG / kT)
        if not ok:
            atoms.set_cell(cell0, scale_atoms=False)
            atoms.set_positions(pos0)
        b['hist'].append(ok)
        if len(b['hist']) >= 10:
            rate = sum(b['hist']) / float(len(b['hist']))
            if rate > 0.5:
                b['delta'] = min(b['delta'] * 1.1, 0.05)
            elif rate < 0.2:
                b['delta'] = max(b['delta'] * 0.9, 1e-6)
            b['hist'] = []

class _RestAseOpt:
    def __init__(self, symbols, masses, positions_ang, fixed):
        atoms = Atoms(symbols=symbols,
                      positions=np.asarray(positions_ang, dtype=float).reshape(-1, 3),
                      masses=masses)
        self.fixed = list(fixed)
        if self.fixed:
            atoms.set_constraint(FixAtoms(indices=self.fixed))
        self._last_f = np.zeros((len(atoms), 3))
        atoms.calc = _RestCalculator(lambda flat: (0.0, self._last_f.flatten()))
        from ase.optimize import FIRE
        self.dyn = FIRE(atoms, logfile=None)
        self.atoms = atoms

    def step(self, forces_ev_ang):
        f = np.asarray(forces_ev_ang, dtype=float).reshape(-1, 3)
        for i in self.fixed:
            f[i] = 0.0
        self._last_f = f.copy()
        self.dyn.step(f=f)
        p = self.atoms.get_positions()
        return [c for row in p for c in row]
"#;

/// ASE release that first projected the Langevin random displacement and
/// velocity with the sqrt(N/(N-1)) factor for `fixcm=True`.
///
/// ASE 3.22 used an older scheme, ASE 3.23 to 3.24 removed the centre of mass
/// without that factor. Both follow a different trajectory from the same seed
/// and therefore cannot reproduce the MD reference logs committed in
/// `rest_regression`.
pub const MIN_ASE_VERSION: (u32, u32) = (3, 25);

/// Identity of the Python interpreter embedded in REST and of the ASE
/// installation that this interpreter imports.
#[derive(Clone, Debug)]
pub struct AseEnv {
    pub python_version: String,
    pub python_prefix: String,
    pub ase_version: String,
    pub ase_path: String,
}

impl AseEnv {
    pub fn describe(&self) -> String {
        format!(
            "python {} (prefix {}), ase {} ({})",
            self.python_version, self.python_prefix, self.ase_version, self.ase_path
        )
    }

    /// Major and minor version of ASE, when they can be parsed.
    fn ase_major_minor(&self) -> Option<(u32, u32)> {
        let mut parts = self.ase_version.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        Some((major, minor))
    }
}

/// Read the interpreter and ASE identity from the embedded Python.
///
/// The interpreter is fixed when REST is built, so the reported prefix is also
/// the environment whose pip has to provide ASE.
fn probe_ase(py: Python<'_>) -> anyhow::Result<AseEnv> {
    let sys = py.import("sys")?;
    let version: String = sys.getattr("version")?.extract()?;
    let python_version = version.split_whitespace().next().unwrap_or("?").to_string();
    let python_prefix: String = sys.getattr("prefix")?.extract()?;
    let ase = py.import("ase").map_err(|error| {
        anyhow::anyhow!(
            "the Python module 'ase' is not available ({error})\n  interpreter: python {python_version} (prefix {python_prefix})\n  REST MD needs ASE >= {}.{} in that environment, for example:\n    {}/bin/python -m pip install -U \"ase>={}.{}\"",
            MIN_ASE_VERSION.0,
            MIN_ASE_VERSION.1,
            python_prefix,
            MIN_ASE_VERSION.0,
            MIN_ASE_VERSION.1,
        )
    })?;
    Ok(AseEnv {
        python_version,
        python_prefix,
        ase_version: ase.getattr("__version__")?.extract()?,
        ase_path: ase.getattr("__file__")?.extract()?,
    })
}

/// Announce the interpreter and ASE identity, and warn when ASE is old enough
/// to change the Langevin trajectory.
fn report_ase_env(env: &AseEnv) {
    println!("MD engine ASE: {}", env.describe());
    if let Some(version) = env.ase_major_minor() {
        if version < MIN_ASE_VERSION {
            println!(
                "WARNING: ASE {}.{} is older than ASE {}.{}. REST MD uses Langevin(fixcm=True), \
                 whose projection of the random displacement and velocity changed in ASE {}.{}. \
                 Trajectories from the same seed therefore differ from the committed MD reference \
                 logs. Upgrade with:\n    {}/bin/python -m pip install -U \"ase>={}.{}\"",
                version.0,
                version.1,
                MIN_ASE_VERSION.0,
                MIN_ASE_VERSION.1,
                MIN_ASE_VERSION.0,
                MIN_ASE_VERSION.1,
                env.python_prefix,
                MIN_ASE_VERSION.0,
                MIN_ASE_VERSION.1,
            );
        }
    }
}

pub struct AseMd {
    driver: Py<PyAny>,
    _hook: Py<PyForceHook>,
}

impl AseMd {

    #[allow(clippy::too_many_arguments)]
    #[allow(deprecated)]
    pub fn new<F>(
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
        cell_ang: Option<[f64; 3]>,
        pressure_bar: f64,
        taut_fs: f64,
        taup_fs: f64,
        compressibility_bar: f64,
        baro_interval: usize,
        baro_scale: f64,
        xtc_path: String,
        xtc_interval: usize,
        eval: F,
    ) -> anyhow::Result<Self>
    where
        F: FnMut(&[f64], Option<&[f64]>) -> anyhow::Result<(f64, Vec<f64>)> + 'static,
    {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<Self> {
            let env = probe_ase(py)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            report_ase_env(&env);
            let code = std::ffi::CString::new(PY_ASE_MD).unwrap();
            let locals = PyDict::new(py);

            py.run(&code, Some(&locals), Some(&locals))?;
            let hook = Py::new(py, PyForceHook { slot: RefCell::new(Some(Box::new(eval))) })?;
            let cls = locals
                .get_item("_RestAseMD")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("_RestAseMD missing"))?;
            let args = pyo3::types::PyTuple::new(py, vec![
                symbols.to_vec().into_py(py),
                masses_amu.to_vec().into_py(py),
                pos_ang.to_vec().into_py(py),
                velocities_ang_fs.map(|v| v.to_vec()).into_py(py),
                temperature_k.into_py(py),
                friction.into_py(py),
                friction_units.to_string().into_py(py),
                dt_fs.into_py(py),
                ensemble.to_string().into_py(py),
                seed.into_py(py),
                fixed_rows.to_vec().into_py(py),
                hook.clone_ref(py).into_py(py),
                cell_ang.map(|c| c.to_vec()).into_py(py),
                pressure_bar.into_py(py),
                taut_fs.into_py(py),
                taup_fs.into_py(py),
                compressibility_bar.into_py(py),
                baro_interval.into_py(py),
                baro_scale.into_py(py),
                xtc_path.into_py(py),
                xtc_interval.into_py(py),
            ])?;
            let driver = cls.call(args, None)?;
            Ok(AseMd { driver: driver.into(), _hook: hook })
        })
        .map_err(|e| anyhow::anyhow!("ASE MD engine construction failed: {}", e))
    }

    pub fn finish(&self) {
        Python::with_gil(|py| {
            let _ = self.driver.bind(py).call_method0("close_xtc");
        });
    }

    pub fn cell_ang(&self) -> Option<[f64; 3]> {
        Python::with_gil(|py| -> Option<[f64; 3]> {
            let res = self.driver.bind(py).call_method0("cell_ang").ok()?;
            let v: Option<Vec<f64>> = res.extract().ok()?;
            let v = v?;
            if v.len() == 3 {
                Some([v[0], v[1], v[2]])
            } else {
                None
            }
        })
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

    pub fn step(&self) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        Python::with_gil(|py| -> PyResult<(Vec<f64>, Vec<f64>)> {
            let res = self.driver.bind(py).call_method0("step")?;
            let pos: Vec<f64> = res.get_item(0)?.extract()?;
            let vel: Vec<f64> = res.get_item(1)?.extract()?;
            Ok((pos, vel))
        })
        .map_err(|e| anyhow::anyhow!("ASE MD step failed: {}", e))
    }

    pub fn clear_hook(&self) {
        Python::with_gil(|py| {
            if let Ok(h) = self._hook.bind(py).try_borrow() {
                *h.slot.borrow_mut() = None;
            }
        });
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
            let env = probe_ase(py)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            println!("ASE optimizer: {}", env.describe());
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
