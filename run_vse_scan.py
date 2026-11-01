#!/usr/bin/env python3
"""
Stark Effect Scan Driver for Ti-O Probe (Vibrational Stark Effect).

Workflow:
  1. For each external electric field Ez (from vse_scan_inputs/):
     - Run REST with the corresponding input file
     - Parse the frequency output to extract Ti-O mode
     - Record frequency and IR intensity
  2. Compute Stark tuning rate: d(nu) / d(Ez)
  3. Optional: plot frequency vs field

Usage:
  ./run_vse_scan.py [--rest-bin ./target/release/rest] [--parse-only] [--plot]
"""

import os, re, sys, subprocess, argparse, glob

REST_BIN = "./target/release/rest"
SCAN_DIR = "vse_scan_inputs"


def run_rest(input_file: str, rest_bin: str) -> str:
    """Run REST on an input file, return stdout."""
    cmd = [rest_bin, "--input_file", input_file]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR: REST exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[-500:]}")
    return result.stdout


def parse_frequencies(output: str) -> list[dict]:
    """
    Parse the REST frequency output.
    Expected format:
      Mode   Freq (cm⁻¹)   IR Int (km/mol)
      ---------------------------------------------------------
        1      123.4567         0.123456
        ...
    Returns list of {'mode': int, 'freq': float, 'ir_int': float}
    """
    modes = []
    in_table = False
    for line in output.splitlines():
        line = line.strip()
        if "Freq (cm⁻¹)" in line and "IR Int" in line:
            in_table = True
            continue
        if in_table and line.startswith("==="):
            break
        if in_table and line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    mode = int(parts[0])
                    freq_str = parts[1]
                    ir_int = float(parts[2])
                    # Handle imaginary frequencies (e.g., "123.4567i")
                    is_imag = freq_str.endswith("i")
                    freq = float(freq_str.rstrip("i"))
                    if is_imag:
                        freq = -freq
                    modes.append({"mode": mode, "freq": freq, "ir_int": ir_int})
                except (ValueError, IndexError):
                    continue
    return modes


def identify_ti_o_mode(modes: list[dict]) -> dict | None:
    """
    Identify the Ti-O stretching mode (~600-1000 cm⁻¹).
    Preference: mode with highest IR intensity in the 600-1000 cm⁻¹ range.
    """
    candidates = [m for m in modes if 400 < abs(m["freq"]) < 1200]
    if not candidates:
        # fallback: look for strong IR modes anywhere
        candidates = [m for m in modes if m["ir_int"] > 1.0]
    if not candidates:
        # last resort: highest IR mode overall (excluding near-zero)
        candidates = [m for m in modes if abs(m["freq"]) > 100]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m["ir_int"])


def main():
    parser = argparse.ArgumentParser(description="VSE Stark Scan Driver")
    parser.add_argument("--rest-bin", default=REST_BIN,
                        help=f"Path to REST binary (default: {REST_BIN})")
    parser.add_argument("--parse-only", action="store_true",
                        help="Parse existing output files without re-running REST")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a quick ASCII plot of nu vs Ez")
    parser.add_argument("--scan-dir", default=SCAN_DIR,
                        help=f"Directory with scan input files (default: {SCAN_DIR})")
    args = parser.parse_args()

    # Find input files
    input_files = sorted(glob.glob(f"{args.scan_dir}/field_*.in.toml"))
    if not input_files:
        print(f"No input files found in {args.scan_dir}/")
        print("Run build_sto_irox.py first to generate inputs.")
        sys.exit(1)

    print(f"Found {len(input_files)} scan inputs")
    print(f"REST binary: {args.rest_bin}")
    print()

    results = []  # list of (ez, freq, ir_int)

    for infile in input_files:
        # Extract Ez from filename: field_00_ez_+0.000000.in.toml
        m = re.search(r'ez_([+-]?\d+\.\d+)', infile)
        ez = float(m.group(1)) if m else 0.0

        outfile = infile.replace(".in.toml", ".out")
        print(f"[Ez = {ez:+.6f} a.u.]  input: {os.path.basename(infile)}")

        if args.parse_only and os.path.exists(outfile):
            with open(outfile) as f:
                output = f.read()
        else:
            if not args.parse_only:
                output = run_rest(infile, args.rest_bin)
                with open(outfile, "w") as f:
                    f.write(output)
            else:
                print(f"  Skipping (no .out file)")
                continue

        modes = parse_frequencies(output)
        if not modes:
            print(f"  WARNING: No frequencies parsed")
            continue

        ti_o_mode = identify_ti_o_mode(modes)
        if ti_o_mode:
            results.append((ez, ti_o_mode["freq"], ti_o_mode["ir_int"]))
            print(f"  Ti-O mode #{ti_o_mode['mode']:3d}: "
                  f"nu = {ti_o_mode['freq']:10.4f} cm⁻¹  "
                  f"IR = {ti_o_mode['ir_int']:10.4f} km/mol")
        else:
            print(f"  WARNING: Could not identify Ti-O mode")
            print(f"  All modes above 100 cm⁻¹:")
            for m in modes:
                if abs(m["freq"]) > 100:
                    print(f"    Mode {m['mode']:3d}: {m['freq']:10.4f} cm⁻¹  "
                          f"IR = {m['ir_int']:10.4f} km/mol")

    if len(results) < 2:
        print("\nNot enough results for Stark analysis.")
        sys.exit(1)

    # ─── Fit Stark tuning rate ───────────────────────────────────────────
    results.sort()  # sort by Ez
    ez_vals = [r[0] for r in results]
    nu_vals = [r[1] for r in results]

    # Linear fit: nu = nu0 + sigma * Ez
    import numpy as np
    coeffs = np.polyfit(ez_vals, nu_vals, 1)
    nu0, sigma = coeffs[1], coeffs[0]

    # R²
    nu_pred = np.polyval(coeffs, ez_vals)
    ss_res = np.sum((np.array(nu_vals) - nu_pred) ** 2)
    ss_tot = np.sum((np.array(nu_vals) - np.mean(nu_vals)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print()
    print("=" * 60)
    print("  Vibrational Stark Effect Analysis")
    print("=" * 60)
    print(f"  Ti-O zero-field frequency:  nu₀ = {nu0:.4f} cm⁻¹")
    print(f"  Stark tuning rate:          σ   = {sigma:.4f} cm⁻¹ / a.u.")
    print(f"  In practical units:         σ   = {sigma / 5.142e9:.4e} cm⁻¹ / (V/cm)")
    print(f"  R² (linear fit):            R²  = {r2:.6f}")
    print()
    print(f"  nu(Ez) = {nu0:.4f} + {sigma:.4f} · Ez")
    print()
    print(f"  Comparison with experiment (manuscript Fig 2d):")
    print(f"    The manuscript reports Ti-O red shift with increasing potential.")
    print(f"    Positive Ez → Ti-O frequency decreases → negative sigma expected.")
    print(f"    Observed sigma = {sigma:.4f} cm⁻¹/a.u.")
    if sigma < 0:
        print(f"    ✓ Matches experimental trend (red shift with field)")
    else:
        print(f"    ✗ Opposite to experimental trend (check field direction)")

    # ─── Save results ────────────────────────────────────────────────────
    import json
    report = {
        "nu0_cm1": nu0,
        "sigma_cm1_per_au": sigma,
        "sigma_cm1_per_Vcm": sigma / 5.142e9,
        "r_squared": r2,
        "data": [{"Ez_au": ez, "nu_cm1": nu, "ir_km_mol": ir}
                 for ez, nu, ir in results],
    }
    with open("vse_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to vse_results.json")

    # ─── ASCII plot (optional) ───────────────────────────────────────────
    if args.plot:
        try:
            plot_ascii(ez_vals, nu_vals, nu0, sigma)
        except Exception as e:
            print(f"Plot error: {e}")


def plot_ascii(ez_vals, nu_vals, nu0, sigma):
    """Simple ASCII plot of frequency vs electric field."""
    print()
    print("  nu (cm⁻¹)")
    print("  ^")
    nu_min, nu_max = min(nu_vals), max(nu_vals)
    nu_range = nu_max - nu_min
    ez_min, ez_max = min(ez_vals), max(ez_vals)

    n_rows = 20
    n_cols = 60

    # Map data points to grid
    grid = [[" " for _ in range(n_cols)] for _ in range(n_rows)]

    for ez, nu in zip(ez_vals, nu_vals):
        col = int((ez - ez_min) / (ez_max - ez_min) * (n_cols - 1))
        row = n_rows - 1 - int((nu - nu_min) / nu_range * (n_rows - 1))
        col = max(0, min(n_cols - 1, col))
        row = max(0, min(n_rows - 1, row))
        grid[row][col] = "*"

    # Fit line
    for col in range(n_cols):
        ez = ez_min + col * (ez_max - ez_min) / (n_cols - 1)
        nu_fit = nu0 + sigma * ez
        row = n_rows - 1 - int((nu_fit - nu_min) / nu_range * (n_rows - 1))
        row = max(0, min(n_rows - 1, row))
        if grid[row][col] == " ":
            grid[row][col] = "."

    for i, row in enumerate(grid):
        nu = nu_max - i * nu_range / (n_rows - 1)
        label = f"{nu:8.1f} |" if i % 4 == 0 else "         |"
        print(f"  {label}{''.join(row)}")
    print(f"         +{'-' * n_cols}")
    print(f"         {ez_min:+.4f}{' ' * (n_cols - 16)}{ez_max:+.4f}  Ez (a.u.)")


if __name__ == "__main__":
    main()
