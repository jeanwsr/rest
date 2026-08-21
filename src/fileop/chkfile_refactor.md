# Chkfile Refactor

## Summary

Reworked the chkfile save/load pipeline and initial guess projection logic to eliminate derived-data duplication, fix the `basis_path='chkfile'` stale-geometry bug, and add safety checks (Refuse) to the initial guess decision.

---

## Collect Basis Refactor

**Files**: `molecule_io/mod.rs`, `molecule_io/basis.rs`

**`collect_basis`** split into three phases:

| Phase | Function | Role |
|---|---|---|
| A | inline | Determine `CintType`, download missing basis files |
| B | `read_basis_per_atom` (`pub fn` in `basis.rs`) | Read per-atom basis JSON files, normalize, set `global_index` → `Vec<Basis4Elem>` |
| C | `build_cint` (`pub fn` in `mod.rs`) | Pure function: `&[Basis4Elem] + &GeomCell + &CintType` → `(cint_atm, cint_bas, cint_env, fdqc_bas, cint_fdqc, num_basis, ecpbas)` |

Shared helpers in `molecule_io/basis.rs`:

| Function | Role |
|---|---|
| `shell_nao` | Count AO basis functions from shell descriptors |
| `build_fdqc` | Build `fdqc_bas` + `cint_fdqc` from `cint_bas` + `cint_type` (used by `build_cint` and `set_cint_data` fallback) |

`num_elec` is computed by `Molecule::update_num_elec()` (reads `cint_atm` nuclear charges, accounts for ECP, charge, and spin). Called in `build_native` after `collect_basis` + field assignment, and in `update_basis_from_hdf5chk` after `set_cint_data`. `num_state` is set to `num_basis` in `collect_basis`.

---

## Chkfile Save/Load Redesign

**File**: `fileop/chkfile.rs`

### What was removed

- **`"mol"` JSON string scalar** from `save_chkfile`. This blob mixed `_atm`, `_bas`, `_ecpbas`, and `_env` (a flat array of coordinates + basis exponents/coefficients). It is fully derivable from the structured "ancestors" already saved separately.

### What remains in the chkfile

| HDF5 path | Content | Needed for |
|---|---|---|
| `scf/` | e_tot, mo_coeff, mo_energy, mo_occ, num_basis, num_states, spin_channel, spin, charge | SCF results |
| `molecule/basis4elem` | `Vec<Basis4Elem>` (JSON) | Basis reconstruction |
| `molecule/geom` | GeomCell (JSON: name, elem, unit, position) | Coord reconstruction + element checks |
| `molecule/cinttype` | "spheric" / "cartesian" | Basis reconstruction |
| `molecule/num_elec` | Total electron count (`f64`, JSON) | Electron-count Refuse check |

### What was added

- **`molecule/geom`** — GeomCell serialized as JSON, saved by `save_chkfile`. Used by `load_geom` for element-name checks, and by `reconstruct_cint_data` as the coordinate source when rebuilding `cint_env`.

- **`reconstruct_cint_data(chkfile, geom_override)`** — replaces `load_cint_data` for all callers. Two code paths:

  | Chkfile format | Detection | Behavior |
  |---|---|---|
  | **New** | `molecule/basis4elem` + `molecule/cinttype` exist | Reads `basis4elem` + `geom` (or `geom_override`) + `cint_type` → calls `build_cint` to reconstruct `cint_env`, `fdqc_bas`, `cint_fdqc`. No stale coords. |
  | **Old** | No `molecule/basis4elem` | Falls back to `load_cint_data` (reads legacy `"mol"` JSON). Backward compatible. |

- **`set_cint_data`** (on `Molecule`, `molecule_io/mod.rs`) — replaces `update_from_cint`. Pure field assignment (`cint_atm/bas/env/ecpbas`, optionally `fdqc_bas`/`cint_fdqc`/`basis4elem`/`cint_type`, `natm_real`/`natm_all`, `num_state=num_basis`). When `basis4elem` is set, also recomputes `ecp_electrons`. If `fdqc_bas` is not passed, recomputes it via `build_fdqc` fallback. No `num_elec`/`start_mo` computation — use `update_num_elec()` for the former; `generate_start_mo` for the latter.

### `load_cint_data`

Preserved as private fallback (called only by `reconstruct_cint_data` for old-format chkfiles). No external callers remain.

---

## Initial Guess Decision Refactor

**File**: `initial_guess/proj.rs`

### `decide_guess(chkfile, mol_target) -> GuessAction`

Three-outcome decision pipeline:

- Load data: `load_basic` (per-field `Option`s) + `reconstruct_cint_data` + `load_geom`
- **Refuse checks** (O(1)): atom count, element order (GeomCell.elem or ATM_NUC fallback), CintType, ECP electron count, electron count (`molecule/num_elec`, fallback `scf/mo_occ` sum). Any mismatch → `Refuse(String)`.
- **Geom diff** (O(natoms)): compare positions via GeomCell or cint_env/ATM_ENV fallback.
- **Early exit**: `basis_diff || geom_diff` → `Project(mol_source)`.
- **S21 final gate** (O(nbasis²)): if counts and geometry match, compute `S22 = int_ij_matrixupper("ovlp")`, `S21 = int_cross(&mol_source, "ovlp")`. If ∥S21 − S22∥_F / ∥S22∥_F > 1e-5 → `Project`. Otherwise → `DirectReuse`.
- `mol_source` is built once, shared by Project and S21 paths.

| Tier | Check | Cost | Catches |
|---|---|---|---|
| Refuse | atom count / element / CintType / ECP / electron count | O(1) | Genuinely different molecules |
| Early exit | basis_diff ‖ geom_diff | O(1) / O(natoms) | Different basis / geometry → Project |
| S21 | ‖S21 − S22‖ / ‖S22‖ | O(nbasis²) | Same count + same geometry but different exponents |

### `GuessAction` enum

```rust
pub enum GuessAction {
    DirectReuse,
    Project(Molecule),   // mol_source, ready for proj_mo
    Refuse(String),
}
```

### Basis projection: `basis_projection` keyword + `proj_mo`

New input keyword `basis_projection` (default `"occupied"`):

| Value | Behavior |
|---|---|
| `"occupied"` | Project only occupied MOs from source (nocc from `mol.num_elec`, `round()`) |
| `"full"` | Project all source MOs |

`mo_range` depends on the **target** scftype:

| Target | `"occupied"` mo_range | `"full"` mo_range |
|---|---|---|
| RKS / UKS | `[0..nocc_alpha, 0..nocc_beta]` | `[0..source.num_state, 0..source.num_state]` |
| ROKS / ROHF | `[0..nocc_alpha, 0..0]` (beta empty) | `[0..source.num_state, 0..0]` |

`proj_mo(mol_target, mol_source, mo_source, mo_range: [Range<usize>; 2])`:
- Projects only columns in `mo_range[spin]`, then zero-pads output to `mol_target.num_state`.
- Clamps `end` to `mol_target.num_state` with `warn!` when the range exceeds target (e.g., large→small basis projection).
- Panics only for `start > end` or `end > src_nmo` (nonexistent source columns).

Safety: `decide_guess` Refuse rejects different electron counts before projection, so source and target MO spaces always span the same occupied orbitals.

### MO import mode: `initial_guess_from_raw`

**File**: `initial_guess/mod.rs`

`initial_guess_from_raw` now takes `target_scftype: &SCFType` (replacing the old `spin_channel: usize` heuristic). The **source** SCFType is derived from the chkfile data channel counts:

| Eigenvector channels | Occupation channels | Source SCFType |
|---|---|---|
| 1 | 1 | RHF (RKS) |
| 1 | 2 | ROHF (ROKS) |
| 2 | — | UHF (UKS) |

Mode = `import_mode(source, target)`:

| src \ tgt | RKS (RHF) | ROKS (ROHF) | UKS (UHF) |
|---|---|---|---|
| RKS (RHF) | `R2R` | panic (r2ro) | `R2U` |
| ROKS (ROHF) | panic (ro2r) | `RO2RO` | `RO2U` |
| UKS (UHF) | `U2R` panic | `U2R` panic | `U2U` |

Eigenvector output structure depends on the target: RKS/ROKS targets store the shared spatial set in `eigenvectors[0]` only (ROHF's `eigenvectors[1]` stays empty — REST synthesizes it on the fly), so `R2R`/`RO2RO` return `[spatial, empty]`. UKS targets need both channels, so `R2U`/`RO2U` return `[spatial, spatial]`. `U2U` reads both channels directly. Occupation: RKS source total occ is halved per channel; ROKS source 2-channel occ is read as-is.

`import_guess_from_hdf5chkfile` no longer takes a `spin_channel` parameter (was unused).

---

## Initial Guess Deferred I/O

**File**: `initial_guess/mod.rs`

`import_guess_from_hdf5chkfile` (the HDF5 eigenvector read) moved from before `decide_guess` into the `DirectReuse` / `Project` match arms. Refuse path now pays zero eigenvector I/O.

---

## `basis_path='chkfile'` Fix

**File**: `initial_guess/mod.rs:268` (`update_basis_from_hdf5chk`)

Previously: loaded chkfile's `cint_env` (which contains stale coordinates) → SCF used wrong geometry.

Now: calls `reconstruct_cint_data(&chkfile, Some(&scf_data.mol.geom))` → basis exponents/coefficients from chkfile, **coordinates from current input**. Then `set_cint_data` with `fdqc_bas`/`cint_fdqc` from `reconstruct_cint_data` (no recomputation), followed by `start_mo` recomputation from the updated `ecp_electrons`.

### Behavior change for this feature

| Chkfile format | Before | After |
|---|---|---|
| **New** (has `molecule/basis4elem`) | `cint_env` from chkfile's `"mol"` JSON → SCF at chkfile's stale geometry | `cint_env` reconstructed from chkfile's `basis4elem` + **current** `mol.geom` → SCF at input geometry |
| **Old** (legacy `"mol"` JSON) | `cint_env` from `"mol"` JSON → SCF at chkfile's stale geometry, `fdqc_bas` recomputed in-place | `cint_env` from `"mol"` JSON → SCF at chkfile's geometry (unchanged), `fdqc_bas` recomputed by `set_cint_data` fallback |

---

## Data flow summary

- **Save**: Bifurcated — structured ancestors and SCF results.
  - `mol.geom` → `molecule/geom`
  - `mol.basis4elem` → `molecule/basis4elem`
  - `mol.cint_type` → `molecule/cinttype`
  - SCF results → `scf/` (e_tot, mo_coeff, mo_energy, mo_occ, num_basis, num_states, spin_channel, spin, charge)

- **Load**: `reconstruct_cint_data` reads `basis4elem` + `geom` + `cint_type` → calls `build_cint` → reconstructs `cint_atm`, `cint_bas`, `cint_env`, `fdqc_bas`, `cint_fdqc`, `ecpbas`.

No derived `cint_env` is stored. `build_cint` is the canonical reconstruction path for both new chkfiles and the `basis_path='chkfile'` feature.

---

## Program Behavior Change

**Old behavior**: `nbasis` + `nmo` match → direct MO reuse (no projection). Otherwise → project with `check_proj_sanity` assert.

**New behavior**:

| Situation | Old | New |
|---|---|---|
| Different element/atom count in chkfile | Projected (silently wrong) | **Refuse** — panics with message |
| Different electron count | Projected (silently wrong MO space) | **Refuse** — panics with message |
| Different CintType / ECP | Assert failure | **Refuse** — panics with message |
| Same basis, different geometry | Direct MO reuse (poor guess) | **Project** via cross-overlap S21 |
| Same total nbasis, different basis set (e.g., reordered atoms, different exponents) | Direct MO reuse (wrong) | **Project** via S21 |
| Identical basis + geometry | Direct MO reuse | Direct MO reuse (unchanged) |

---

## TODO: Projection for `inherit` initial guess on geometry change

When `initial_guess = "inherit"` and geometry has changed (geom opt), project old eigenvectors to the new geometry via `proj_mo` instead of using them directly. Relies on the chkfile saved after each SCF convergence (chkfile contains `molecule/geom` with the geometry the eigenvectors were computed at).

### Planned steps

1. In `initial_guess/mod.rs` `inherit` branch: check if chkfile exists and `load_geom` returns a `GeomCell`.
2. Compare `prev_geom.positions` with current `mol.geom.positions` (tol 1e-6).
3. If geometry changed: `build_cint(&mol.basis4elem, &prev_geom, &mol.cint_type)` → build `mol_source`, call `proj::proj_mo`, replace eigenvectors.
4. Proceed with `generate_occupation()` + `generate_density_matrix()` as before.

### Needed imports

`std::path::Path`, `crate::molecule_io::build_cint`, `crate::molecule_io::Molecule`, `crate::fileop::chkfile::load_geom`.

### Behavior

| Scenario | Result |
|---|---|
| Geom opt step, geometry changed | Projected to new geometry |
| Geom opt step, same geometry | Skip projection (geom matches) |
| Restart / same-geometry inherit | Skip projection |
| First SCF (no chkfile yet) | Skip (no chkfile) |
| Old chkfile (no `molecule/geom`) | Skip (`load_geom` returns None) |

No new struct fields, no API changes.
