# PR !195 Review: improve basisproj and chkbasis

**Date**: 2026-08-21
**Branch**: `restart-opt` → `RESTGroup:master`
**Commits**: 10 (0ad1e39..26705a7)
**Files changed**: 10 (+1318, -639)

---

## Fix applied: `num_elec` regression in `basis_path='chkfile'`

### Problem

For `basis_path='chkfile'`:
1. `Molecule::build_native` skips `collect_basis` (`chkbasis=true`), so `mol.num_elec` stays `[0.0;3]`.
2. The old `update_from_cint` recomputed `num_elec` from `cint_atm` (removed in this PR).
3. The new `set_cint_data` explicitly does not compute `num_elec` (docstring: "No num_elec/start_mo computation").
4. The caller `update_basis_from_hdf5chk` only recomputes `start_mo`, not `num_elec`.

Consequences:
- `decide_guess` electron-count Refuse always fires (`|0 - N| > 0.5`).
- `scftype` detection reads stale `[0,0,0]` → RHF regardless of input.
- `basis_path='chkfile'` feature completely broken.

### Changes

| File | Line | Change |
|---|---|---|
| `src/molecule_io/mod.rs` | ~348 | Added `Molecule::update_num_elec(&mut self)` method |
| `src/molecule_io/mod.rs` | 857 | Removed `[f64;3]` from `collect_basis` return type |
| `src/molecule_io/mod.rs` | 900 | Removed inline `num_elec` computation from `collect_basis` |
| `src/molecule_io/mod.rs` | 193-194 | Updated destructuring in `build_native` (9-element tuple) |
| `src/molecule_io/mod.rs` | 208 | Replaced `mol.num_elec = num_elec` with `mol.update_num_elec()` |
| `src/initial_guess/mod.rs` | 296-297 | Added `scf_data.mol.update_num_elec()` after `set_cint_data` |
| `src/fileop/chkfile_refactor.md` | 28, 61 | Updated doc to reflect new `update_num_elec` method |

`grad/rhf.rs:686` only uses `.0` (basis4elem) from `collect_basis` — no change needed.

---

## Remaining review findings (TODO)

### High priority

**Ghost basis atoms lost in chkfile round-trip**
- `save_chkfile` writes geom JSON with only `name/elem/unit/position`.
- `load_geom` hard-codes `ghost_bs_elem: vec![]`.
- `reconstruct_cint_data` → `build_cint(basis4elem, ghostless_geom)` builds `atm` without ghost atoms, but `bas` references `atm_index` beyond `atm` for ghost entries in `basis4elem`.
- Ghost-basis chkfiles (QM/MM, CP correction) produce corrupted cint data.
- Old format preserved ghosts via raw `_atm/_bas/_env` JSON.
- **Fix needed**: save ghost fields in geom JSON, or handle `ghost_bs_elem` in `load_geom`.

### Medium priority

**`load_geom` drops pbc/lattice/ghost_pc_chrg/ghost_ep_path**
- Periodic, charged-ghost, and external-potential ghost systems lose state on chkfile round-trip.
- Affects `decide_guess` geometry comparison and `reconstruct_cint_data` for PBC systems.

**`import_guess_from_hdf5chkfile` reads only `mo_occ`**
- `initial_guess/mod.rs:368` reads `scf.dataset("mo_occ")` only.
- Old save wrote both `mo_occupation` and `mo_occ`.
- External tools writing only `mo_occupation` would cause `.unwrap()` panic.
- **Fix needed**: fallback to `mo_occupation` if `mo_occ` absent.

**`proj_mo` `mo_range` panic when nocc > source num_state**
- ROHF/UKS `nocc` derived from *target* `num_elec` can exceed source `num_state` when input spin/charge differ from chkfile's.
- `proj.rs:91-92`: `if end > src_nmo { panic! }`.
- **Fix needed**: clamp `end` to `min(end, src_nmo)` with warning, similar to target-side clamp.

### Low priority

**DirectReuse S21-vs-S22 gate threshold**
- Threshold `1e-5` at `proj.rs:283` is a heuristic.
- Untested for float noise in large or periodic systems.
- **Suggested**: add test coverage or document the heuristic's limitations.

---

## Note: pre-existing scftype detection ordering

`scftype` detection (`scf_io/mod.rs:175`) runs in `init_scf` *before* `initialize_scf` (which calls `update_basis_from_hdf5chk`). So a UHF/ROHF `basis_path='chkfile'` run would still misdetect `scftype=RHF` unless `spin_polarization` is set. This was already the case in master — same ordering. Out of scope for the `num_elec` fix but worth noting as a separate issue.
