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

`num_elec` is computed by `collect_basis` after `build_cint` returns (from the final `cint_atm` nuclear charges, accounting for ECP, charge, and spin). `num_state` is set to `num_basis` in `collect_basis`.

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

### What was added

- **`molecule/geom`** — GeomCell serialized as JSON, saved by `save_chkfile`. Used by `load_geom` for element-name checks, and by `reconstruct_cint_data` as the coordinate source when rebuilding `cint_env`.

- **`reconstruct_cint_data(chkfile, geom_override)`** — replaces `load_cint_data` for all callers. Two code paths:

  | Chkfile format | Detection | Behavior |
  |---|---|---|
  | **New** | `molecule/basis4elem` + `molecule/cinttype` exist | Reads `basis4elem` + `geom` (or `geom_override`) + `cint_type` → calls `build_cint` to reconstruct `cint_env`, `fdqc_bas`, `cint_fdqc`. No stale coords. |
  | **Old** | No `molecule/basis4elem` | Falls back to `load_cint_data` (reads legacy `"mol"` JSON). Backward compatible. |

- **`set_cint_data`** (on `Molecule`, `molecule_io/mod.rs`) — replaces `update_from_cint`. Pure field assignment (`cint_atm/bas/env/ecpbas`, optionally `fdqc_bas`/`cint_fdqc`/`basis4elem`/`cint_type`, `natm_real`/`natm_all`, `num_state=num_basis`). When `basis4elem` is set, also recomputes `ecp_electrons`. If `fdqc_bas` is not passed, recomputes it via `build_fdqc` fallback. No `num_elec`/`start_mo` computation.

### `load_cint_data`

Preserved as private fallback (called only by `reconstruct_cint_data` for old-format chkfiles). No external callers remain.

---

## Initial Guess Decision Refactor

**File**: `initial_guess/proj.rs`

### `decide_guess(chkfile, mol_target) -> GuessAction`

Three-outcome decision pipeline:

- Load data: `load_basic` + `reconstruct_cint_data` + `load_geom`
- **Refuse checks** (O(1)): atom count, element order (GeomCell.elem or ATM_NUC fallback), CintType, ECP electron count. Any mismatch → `Refuse(String)`.
- **Geom diff** (O(natoms)): compare positions via GeomCell or cint_env/ATM_ENV fallback.
- **Early exit**: `basis_diff || geom_diff` → `Project(mol_source)`.
- **S21 final gate** (O(nbasis²)): if counts and geometry match, compute `S22 = int_ij_matrixupper("ovlp")`, `S21 = int_cross(&mol_source, "ovlp")`. If ∥S21 − S22∥_F / ∥S22∥_F > 1e-5 → `Project`. Otherwise → `DirectReuse`.
- `mol_source` is built once, shared by Project and S21 paths.

| Tier | Check | Cost | Catches |
|---|---|---|---|
| Refuse | atom count / element / CintType / ECP | O(1) | Genuinely different molecules |
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
| Different CintType / ECP | Assert failure | **Refuse** — panics with message |
| Same basis, different geometry | Direct MO reuse (poor guess) | **Project** via cross-overlap S21 |
| Same total nbasis, different basis set (e.g., reordered atoms, different exponents) | Direct MO reuse (wrong) | **Project** via S21 |
| Identical basis + geometry | Direct MO reuse | Direct MO reuse (unchanged) |
