#!/usr/bin/env python3
"""
Verification script: Compare REST finite-difference Hessian against PySCF analytical Hessian.

System: H2O, B3LYP/def2-SVP, RI approximation with def2-universal-jkfit.

Usage:
    python3 verify_fd_hessian.py

Requires:
    - PySCF (pip install pyscf)
    - REST binary at $REST_HOME/target/debug/rest

The script:
    1. Computes PySCF analytical Hessian (B3LYP/def2-SVP, RI-J) as reference
    2. Generates REST input file (Hessian via finite difference)
    3. Runs REST
    4. Parses REST output Hessian and compares element-wise and via frequencies
"""

import subprocess
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# ─── Configuration ───
H2O_GEOM = """
O   0.0000000000   0.0000000000   0.1170790000
H   0.0000000000   0.7569500000  -0.4683160000
H   0.0000000000  -0.7569500000  -0.4683160000
"""

REST_HOME = os.environ.get("REST_HOME", "/home/igor/Documents/Package-Pool/rest_workspace")
REST_BIN = os.path.join(REST_HOME, "target", "debug", "rest")
REST_BASIS_DIR = os.path.join(REST_HOME, "rest", "basis-set-pool")

FD_DISPLACE_ANG = 0.005  # Angstrom

# H2O test geometry (approximate equilibrium)
H2O_GEOM = """
O   0.0000000000   0.0000000000   0.1170790000
H   0.0000000000   0.7569500000  -0.4683160000
H   0.0000000000  -0.7569500000  -0.4683160000
"""
    # de2 shape: (natm, natm, 3, 3)
    natm = mol.natm
    hess_flat = de2.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    return hess_flat, mol.atom_charges(), mf.mo_energy


def compute_frequencies(hessian, masses_amu):
    """Mass-weight Hessian → diagonalize → frequencies in cm⁻¹."""
    natm = hessian.shape[0] // 3
    mw = np.zeros_like(hessian)
    for i in range(natm * 3):
        mi = masses_amu[i // 3]
        for j in range(natm * 3):
            mj = masses_amu[j // 3]
            mw[i, j] = hessian[i, j] / np.sqrt(mi * mj)

    evals, evecs = np.linalg.eigh(mw)
    # Convert: √(λ in Hartree/Bohr²·amu) * 5140.487 → cm⁻¹
    FREQ_CONV = 5140.487
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * FREQ_CONV
    return freqs


def generate_rest_input(output_dir):
    """Write REST input file for NormalModes job."""
    ctrl = f"""
[ctrl]
     print_level =          3
     xc =                   "b3lyp"
     basis_path =           "{REST_BASIS_DIR}/def2-SVP"
     auxbas_path =          "{REST_BASIS_DIR}/def2-universal-jkfit"
     basis_type =           "spheric"
     eri_type =             "ri-v"
     auxbas_type =          "spheric"
     guessfile =            "none"
     chkfile =              "none"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     auxbasis_response =    true
     external_grids =       "none"
     initial_guess =        "sad"
     mixer =                "diis"
     num_max_diis =         8
     start_diis_cycle =     3
     mix_param =            0.8
     max_scf_cycle =        100
     scf_acc_rho =          1.0e-9
     scf_acc_eev =          1.0e-9
     scf_acc_etot =         1.0e-10
     num_threads =          16
     job_type =             "Normal_Modes"
     nhessian_displacement = {FD_DISPLACE_ANG}

[geom]
    name = "H2O-B3LYP-def2SVP"
    unit = "Angstrom"
    position = \"\"\"
{H2O_GEOM.strip()}
    \"\"\"
"""
    input_path = os.path.join(output_dir, "ctrl.in")
    with open(input_path, "w") as f:
        f.write(ctrl.strip())
    return input_path


def parse_rest_hessian(output_text):
    """Extract raw Cartesian Hessian from REST output.

    REST prints at print_level >= 1 the full Hessian.
    At print_level > 1, it prints "=== Raw Cartesian Hessian (a.u.) ==="
    followed by one line per row in format "   i: val_0 val_1 ... val_{dim-1}".

    We'll parse this section.
    """
    lines = output_text.splitlines()
    in_hessian = False
    hessian_rows = []
    dim = None

    for line in lines:
        if "=== Raw Cartesian Hessian (a.u.) ===" in line:
            in_hessian = True
            continue
        if in_hessian:
            if line.startswith("===") or line.startswith("Cartesian normal mode"):
                break
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                parts = stripped.split(":")
                if len(parts) == 2:
                    row_idx = int(parts[0])
                    vals = list(map(float, parts[1].split()))
                    if dim is None:
                        dim = len(vals)
                    hessian_rows.append(vals)

    if not hessian_rows:
        # Fallback: look for lines starting with "0:", "1:", etc.
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and ":" in stripped[:10]:
                parts = stripped.split(":", 1)
                try:
                    row_idx = int(parts[0])
                    vals = list(map(float, parts[1].split()))
                    if dim is None:
                        dim = len(vals)
                    if len(vals) == dim:
                        hessian_rows.append(vals)
                except (ValueError, IndexError):
                    pass

        # Keep only contiguous sequences
        if len(hessian_rows) >= dim:
            hessian_rows = hessian_rows[:dim] if dim else hessian_rows

    if not hessian_rows:
        return None

    hessian = np.array(hessian_rows)
    return hessian


def parse_rest_frequencies(output_text):
    """Extract frequencies from REST output."""
    lines = output_text.splitlines()
    freqs = []
    in_freq_table = False

    for line in lines:
        if "Mode" in line and "Eigenvalue" in line and "Freq" in line:
            in_freq_table = True
            continue
        if in_freq_table:
            if line.startswith("==="):
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                freqs.append(float(parts[-1]))

    return np.array(freqs) if freqs else None


def main():
    print("=" * 70)
    print("  REST FD Hessian Verification: H2O, B3LYP/def2-SVP")
    print("=" * 70)

    # ─── Step 1: PySCF reference Hessian ───
    print("\n[1/4] Computing PySCF analytical Hessian as reference...")
    pyscf_hess, atom_charges, mo_energy = compute_pyscf_hessian()
    pyscf_dim = pyscf_hess.shape[0]
    print(f"  PySCF Hessian dimension: {pyscf_dim}")
    print(f"  PySCF Hessian max element: {np.max(np.abs(pyscf_hess)):.6f}")

    # Masses for frequency comparison
    mass_map = {8: 15.99491461956, 1: 1.00782503223}
    masses_amu = np.array([mass_map.get(int(c), 1.0) for c in atom_charges])

    # PySCF frequencies
    pyscf_freqs = compute_frequencies(pyscf_hess, masses_amu)
    print(f"  PySCF frequencies (cm⁻¹):")
    for i, f in enumerate(pyscf_freqs):
        flag = " ← near zero" if abs(f) < 100 else ""
        print(f"    Mode {i+1:2d}: {f:12.4f}{flag}")

    # ─── Step 2: Generate REST input ───
    print("\n[2/4] Generating REST input file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = generate_rest_input(tmpdir)
        print(f"  Input file: {input_path}")

        # ─── Step 3: Run REST ───
        print("\n[3/4] Running REST finite-difference Hessian...")
        if not os.path.exists(REST_BIN):
            raise FileNotFoundError(f"REST binary not found at {REST_BIN}. Build first: cd $REST_HOME/rest && cargo build")

        env = os.environ.copy()
        result = subprocess.run(
            [REST_BIN, "--input_file", input_path],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            env=env,
            timeout=600,  # 10 minutes for 18 SCF jobs
        )

        if result.returncode != 0:
            print(f"\nREST STDERR:\n{result.stderr}")
            print(f"\nREST STDOUT:\n{result.stdout}")
            raise RuntimeError(f"REST exited with code {result.returncode}")

        rest_output = result.stdout
        print(f"  REST output length: {len(rest_output)} chars")

        # ─── Step 4: Parse and compare ───
        print("\n[4/4] Parsing REST output and comparing...")

        rest_hess = parse_rest_hessian(rest_output)

        if rest_hess is None:
            print("  WARNING: Could not parse Hessian from REST output.")
            print("  Printing full REST output (last 200 lines):")
            for line in rest_output.splitlines()[-200:]:
                print(line)
        else:
            print(f"  REST Hessian dimension: {rest_hess.shape[0]} x {rest_hess.shape[1]}")

            # Symmetrize for safer comparison
            rest_hess = 0.5 * (rest_hess + rest_hess.T)

            # Element-wise comparison
            diff = rest_hess - pyscf_hess
            max_abs_diff = np.max(np.abs(diff))
            rms_diff = np.sqrt(np.mean(diff ** 2))
            max_rel_diff = np.max(np.abs(diff) / (np.abs(pyscf_hess) + 1e-10))
            print(f"\n  Element-wise comparison (REST - PySCF):")
            print(f"    Max absolute difference:  {max_abs_diff:.6e}")
            print(f"    RMS difference:           {rms_diff:.6e}")
            print(f"    Max relative difference:  {max_rel_diff:.6e}")

            # Frequencies comparison
            rest_freqs = compute_frequencies(rest_hess, masses_amu)
            print(f"\n  Frequencies comparison:")
            print(f"  {'Mode':>5}  {'REST (cm⁻¹)':>14}  {'PySCF (cm⁻¹)':>14}  {'Diff (cm⁻¹)':>14}")
            for i in range(len(rest_freqs)):
                diff_freq = rest_freqs[i] - pyscf_freqs[i]
                print(f"  {i+1:5d}  {rest_freqs[i]:14.4f}  {pyscf_freqs[i]:14.4f}  {diff_freq:14.4f}")

            freq_diff = np.abs(rest_freqs - pyscf_freqs)
            print(f"\n  Max frequency difference: {np.max(freq_diff):.2f} cm⁻¹")
            print(f"  Mean frequency difference: {np.mean(freq_diff):.2f} cm⁻¹")

            # Verdict
            print("\n" + "=" * 70)
            if max_abs_diff < 1e-4 and np.max(freq_diff) < 50:
                print("  RESULT: PASS ✓ — REST FD Hessian agrees with PySCF analytical")
            else:
                print("  RESULT: CHECK — significant differences found")
                print(f"    Max Hessian diff = {max_abs_diff:.2e}")
                print(f"    Max freq diff = {np.max(freq_diff):.1f} cm⁻¹")
            print("=" * 70)

        # Also parse frequencies from REST output text
        rest_freqs_parsed = parse_rest_frequencies(rest_output)
        if rest_freqs_parsed is not None and len(rest_freqs_parsed) > 0:
            print(f"\n  REST-reported frequencies: {rest_freqs_parsed}")


if __name__ == "__main__":
    main()
