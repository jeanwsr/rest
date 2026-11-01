#!/usr/bin/env python3
"""
Build a SrTiO₃(001)-SrO / Ir₄O₈ cluster model for Ti-O probe VSE study.

STO perovskite, a = 3.905 Å, (001) surface (z = normal), 2×2 in-plane.

Stack (bottom → top):
  TiO₂ (bottom cap, z = -1.5A)
  SrO  (z = -A)
  TiO₂ (PROBE Ti layer, z = -A/2)
  SrO  (SURFACE, z = 0, 50% Sr vacancies)
  Ir₄O₈ cluster (z ≈ 2 Å above surface)

Charge neutrality via +2 point charges at Sr vacancy sites.
Embedding via O²⁻ point charges at cluster periphery.
"""

import os, numpy as np

A = 3.905       # SrTiO₃ lattice constant (Å)
D = A / 2.0     # half lattice (≈ 1.9525 Å)

# ─── 1. STO CLUSTER ─────────────────────────────────────────────────────────
# Perovskite (001) layers (z normal, 2×2 in xy):
#   SrO layer:  Sr at (ox,oy)  O at (ox+D,oy) & (ox,oy+D)  per 1×1 cell
#   TiO₂ layer: Ti at (ox+D,oy+D)  O at (ox,oy)             per 1×1 cell

sto = []

def add(elem, x, y, z):
    sto.append((elem, x, y, z))

# Layer 0: TiO₂ (bottom cap, z = -1.5A)
for ox in (0.0, A):
    for oy in (0.0, A):
        add("Ti", ox + D, oy + D, -1.5 * A)
        add("O",  ox,      oy,      -1.5 * A)

# Layer 1: SrO (z = -A)
for ox in (0.0, A):
    for oy in (0.0, A):
        add("Sr", ox,      oy,      -A)
        add("O",  ox + D,  oy,      -A)
        add("O",  ox,      oy + D,  -A)

# Layer 2: TiO₂ (PROBE Ti at z = -D = -A/2)
for ox in (0.0, A):
    for oy in (0.0, A):
        add("Ti", ox + D, oy + D, -D)
        add("O",  ox,      oy,      -D)

# Layer 3: SrO SURFACE (z = 0) with 50% Sr vacancies
all_sr_surf = [(0.0, 0.0), (A, 0.0), (0.0, A), (A, A)]
keep_sr = {(0.0, 0.0), (A, A)}   # diagonal → maintain symmetry
vacancy_pos = [p for p in all_sr_surf if p not in keep_sr]

for ox, oy in all_sr_surf:
    if (ox, oy) in keep_sr:
        add("Sr", ox, oy, 0.0)
for ox in (0.0, A):
    for oy in (0.0, A):
        add("O", ox + D, oy,      0.0)
        add("O", ox,      oy + D, 0.0)

# Deduplicate
def dedup(atoms, tol=0.01):
    seen = []
    for e, x, y, z in atoms:
        if not any(abs(x - sx) < tol and abs(y - sy) < tol and abs(z - sz) < tol
                   for _, sx, sy, sz in seen):
            seen.append((e, x, y, z))
    return seen

sto = dedup(sto)

# ─── 2. Ir₄O₈ CLUSTER ───────────────────────────────────────────────────────
# 4 Ir in a square above the surface (atop the D,D positions between surface O)
# 8 O:  4 bridging O connecting Ir to surface, 4 planar O between Ir pairs
# Result: edge-sharing IrO₄ squares (like rutile (110) chains)

irox = []
z_ir = 2.05    # Ir height above surface
z_ob = 0.95    # bridging O height
z_op = z_ir    # planar O height (same plane as Ir)

ir_xy = [(D, D), (A + D, D), (D, A + D), (A + D, A + D)]

for ix, (x, y) in enumerate(ir_xy):
    irox.append(("Ir", x, y, z_ir))
    # 2 bridging O per Ir → connecting to surface O below
    # Surface O directly below Ir: O at (x, y - D) and (x - D, y) [in SrO layer]
    # Actually the surface O is at (ox+D, oy) and (ox, oy+D)
    # Below Ir at (ox+D, oy+D): surface O at (ox+D, oy) and (ox, oy+D)
    # So bridging O at midpoint: (ox+D, oy+0.5D) and (ox+0.5D, oy+D)
    irox.append(("O", x,      y - 0.5*D, z_ob))  # toward -y surface O
    irox.append(("O", x - 0.5*D, y,      z_ob))  # toward -x surface O

irox = dedup(irox)
# After dedup, bridging O at shared positions merge → fewer than 8

# Count and adjust to exactly Ir₄O₈
n_ir = sum(1 for e, *_ in irox if e == "Ir")
n_o  = sum(1 for e, *_ in irox if e == "O")
need = 2 * n_ir - n_o

if need > 0:
    # Add terminal O in the +z direction (apical O for each Ir)
    for x, y in ir_xy[:need]:
        irox.append(("O", x, y, z_ir + 0.95))
elif need < 0:
    # Remove excess O (keep those closest to Ir)
    irox_o = [(i, e, x, y, z) for i, (e, x, y, z) in enumerate(irox) if e == "O"]
    irox_o.sort(key=lambda a: (a[4], a[3], a[2]))  # by z then y then x
    # Keep the ones with lowest z first (bridging O more important)
    drop = set(item[0] for item in irox_o[:-need])  # drop excess with lowest z
    irox = [a for i, a in enumerate(irox) if not (a[0] == "O" and i in drop)]

irox = dedup(irox)
n_ir = sum(1 for e, *_ in irox if e == "Ir")
n_o  = sum(1 for e, *_ in irox if e == "O")

# Fallback: if still not Ir₄O₈, pad/trim directly
while n_o < 2 * n_ir:
    idx = n_o % n_ir
    x, y = ir_xy[idx]
    irox.append(("O", x + 0.3, y + 0.3, z_ir + 1.0 + 0.1 * n_o))
    n_o += 1
while n_o > 2 * n_ir:
    # Remove the last added O
    for i in range(len(irox) - 1, -1, -1):
        if irox[i][0] == "O":
            irox.pop(i)
            n_o -= 1
            break

print(f"Ir₄O₈: Ir={n_ir}, O={n_o}")

# ─── 3. CHARGE ANALYSIS ─────────────────────────────────────────────────────
all_atoms = sto + irox
charge = 0
for e, *_ in all_atoms:
    if e == "Sr":   charge += 2
    elif e == "Ti": charge += 4
    elif e == "O":  charge -= 2
    elif e == "Ir": charge += 4

n_sr = sum(1 for e, *_ in all_atoms if e == "Sr")
n_ti = sum(1 for e, *_ in all_atoms if e == "Ti")
n_o  = sum(1 for e, *_ in all_atoms if e == "O")
n_ir = sum(1 for e, *_ in all_atoms if e == "Ir")

print(f"Cluster: Sr{n_sr}Ti{n_ti}O{n_o}Ir{n_ir}  charge={charge:+d}")

# ─── 4. COMPENSATING POINT CHARGES ──────────────────────────────────────────
pc_comp = []
for ox, oy in vacancy_pos:
    pc_comp.append((+2.0, ox, oy, 0.0))  # +2 at each Sr vacancy

# ─── 5. EMBEDDING POINT CHARGES ─────────────────────────────────────────────
pc_embed = []

# Below cluster: Sr²⁺ charges mimicking bulk continuation
for ox in (D, A + D):
    for oy in (D, A + D):
        pc_embed.append((+2.0, ox, oy, -2.5 * A))

# Above Ir₄O₈: O²⁻ screening
for ox in (D, A + D):
    for oy in (D, A + D):
        pc_embed.append((-2.0, ox, oy, z_ir + A))

# Edge screening: O²⁻ at xy periphery at each z layer
edge_xy = [(-D, D), (-D, A + D), (2*A + D, D), (2*A + D, A + D),
           (D, -D), (A + D, -D), (D, 2*A + D), (A + D, 2*A + D)]
for z in (-1.5*A, -A, -D, 0.0):
    for x, y in edge_xy:
        pc_embed.append((-2.0, x, y, z))

# ─── 6. WRITE INPUT FILE ────────────────────────────────────────────────────
all_sorted = sorted(all_atoms, key=lambda a: (a[3], a[2], a[1]))

pos_lines = []
for e, x, y, z in all_sorted:
    pos_lines.append(f'"{e}  {x:14.8f}  {y:14.8f}  {z:14.8f}"')

ghost_lines = []
for chg, x, y, z in pc_comp + pc_embed:
    ghost_lines.append(
        f"point charge, {chg:8.4f},  {x:14.8f}  {y:14.8f}  {z:14.8f}"
    )

# Identify a probe Ti
probe_candidates = [(x, y, z) for e, x, y, z in all_sorted
                     if e == "Ti" and abs(z - (-D)) < 0.01]
probe = probe_candidates[0] if probe_candidates else (D, D, -D)

ghost_block = "\n".join(ghost_lines)
pos_block = ",\n".join(f"         {l}" for l in pos_lines)

toml = f'''# ─── SrTiO₃(001)-SrO / Ir₄O₈ Cluster (Ti-O VSE Probe) ───
# Sr{n_sr}Ti{n_ti}O{n_o}Ir{n_ir}  |  formal charge = {charge:+d}
# Sr vacancies: {len(vacancy_pos)} Sr²⁺ removed → compensated by +2 point charges
# Ti-O probe at: ({probe[0]:.2f}, {probe[1]:.2f}, {probe[2]:.2f})

[ctrl]
     print_level           = 1
     job_type              = "normal_modes"
     xc                    = "pbe"
     basis_path            = "basis-set-pool/def2-SVP"
     basis_type            = "spheric"
     auxbas_path           = "basis-set-pool/def2-SVP-JKFIT"
     auxbas_type           = "spheric"
     eri_type              = "ri-v"
     initial_guess         = "sap"
     chkfile               = "none"
     charge                = {charge}.0
     spin                  = 1.0
     spin_polarization     = false
     mixer                 = "diis"
     num_max_diis          = 8
     start_diis_cycle      = 1
     mix_param             = 0.5
     max_scf_cycle         = 200
     scf_acc_rho           = 1.0e-5
     scf_acc_eev           = 1.0e-5
     scf_acc_etot          = 1.0e-8
     numerical_force       = false
     nhessian_displacement = 0.005
     num_threads           = 16

[geom]
     name              = "STO_Ir4O8"
     unit              = "angstrom"
     position = [
{pos_block}
     ]
     ghost             = """
{ghost_block}
     """
#    ext_field_dipole  = [0.0, 0.0, 0.000]   # uncomment for Stark scan
'''

with open("sto_irox_base.in.toml", "w") as f:
    f.write(toml)

print(f"Base input → sto_irox_base.in.toml")
print(f"  Real atoms: {len(all_atoms)}  |  Point charges: {len(pc_comp)+len(pc_embed)}")
print(f"  Ti-O probe: ({probe[0]:.3f}, {probe[1]:.3f}, {probe[2]:.3f})")

# ─── 7. STARK SCAN INPUTS ───────────────────────────────────────────────────
os.makedirs("vse_scan_inputs", exist_ok=True)
fields = np.linspace(-0.003, 0.003, 13)

for i, ez in enumerate(fields):
    ft = toml.replace(
        '#    ext_field_dipole  = [0.0, 0.0, 0.000]   # uncomment for Stark scan',
        f'     ext_field_dipole  = [0.0, 0.0, {ez:.6f}]'
    )
    with open(f"vse_scan_inputs/field_{i:02d}_ez_{ez:+.6f}.in.toml", "w") as f:
        f.write(ft)

print(f"Stark scan: {len(fields)} inputs in vse_scan_inputs/")
print(f"  E_z: [{fields[0]:+.4f}, {fields[-1]:+.4f}] a.u.")
print(f"     ≈ [{fields[0]*5.142e9:.1e}, {fields[-1]*5.142e9:.1e}] V/cm")
