# H2O RI-4c Timing Notes

Date: 2026-05-20

## Temporary RI Reconstruction Benchmark

Command:

```bash
python3 rest/src/lib_rint/tmp/bench_h2o_ri4c_vs_libcint.py --cases sto-3g --repeat 1
python3 rest/src/lib_rint/tmp/bench_h2o_ri4c_vs_libcint.py --cases cc-pVDZ,def2-TZVP --repeat 1
```

Full output:

- `rest/src/lib_rint/tmp/ri4c_vs_libcint_sto3g_run.log`
- `rest/src/lib_rint/tmp/ri4c_vs_libcint_ccpvdz_def2tzvp_run.log`

Measured output, using REST's default auxiliary basis `def2-universal-jkfit`:

| basis | nao | naux | full nint | libcint direct / s | lib_rint RI setup / s | RI reconstruct / s | RI total / s | RI total / libcint | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sto-3g | 7 | 133 | 2401 | 0.001718 | 0.397975 | 0.000535 | 0.398510 | 231.936 | 1.543995253002e-4 |
| cc-pVDZ | 25 | 133 | 390625 | 0.019360 | 4.079226 | 0.103739 | 4.182964 | 216.067 | 5.223015365385e-3 |
| def2-TZVP | 48 | 133 | 5308416 | 0.118268 | 4.545465 | 1.682792 | 6.228257 | 52.662 | 5.182302624269e-3 |

This benchmark compares libcint direct four-center ERI tensor generation against lib_rint RI factor construction plus explicit four-center reconstruction from the RI factor.

## Existing Direct 4c Scalar Report

Source: `work_pool/h2o_lib_rint_vs_libcint_int2e_cart_report.md`.

| basis | libcint int2e_cart / s | lib_rint direct scalar 4c / s | lib_rint / libcint |
|---|---:|---:|---:|
| sto-3g | 0.005072 | 0.031242 | 6.160 |
| cc-pVDZ | 0.032393 | 9.041764 | 279.127 |
| def2-TZVP | 0.189344 | 12.589776 | 66.492 |

The direct report compares `lib_rint::eri_ao_4c_r` against `libcint int2e_cart`; it is not an RI reconstruction benchmark.

## Release Binary RI-r2 Run, Not The Requested 4c Benchmark

Command:

```bash
python3 rest/src/lib_rint/tmp/run_release_h2o_rir2.py --cases sto-3g,cc-pVDZ,def2-TZVP --threads 4
```

Full output was saved to `rest/src/lib_rint/tmp/release_h2o_rir2_last_run.log`.

This uses the already-built `target/release/rest` binary.  The `SCF` column is REST's standard `eri_type = "ri_v"` path, which uses libcint through REST's normal RI J/K machinery.  The `RI-r2` column is the additional `run_lib_rint = true` timing block.  This is not the requested direct comparison between libcint four-center integrals and lib_rint four-center RI reconstruction.

| basis | SCF standard RI / s | lib_rint RI-r2 / s | RI-r2 / SCF |
|---|---:|---:|---:|
| sto-3g | 0.161 | 0.009 | 0.056 |
| cc-pVDZ | 0.540 | 0.076 | 0.141 |
| def2-TZVP | 0.498 | 0.124 | 0.249 |
