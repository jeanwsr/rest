# Temporary lib_rint RI 4c Benchmark

This directory holds temporary benchmark helpers only.  The main benchmark is:

```bash
python3 rest/src/lib_rint/tmp/bench_h2o_ri4c_vs_libcint.py --cases sto-3g --repeat 1
```

It compares, for H2O with Cartesian basis functions:

- `libcint`: direct full four-center ERI tensor generation through `Molecule::int_ijkl_erifull()`.
- `lib_rint`: RI-r matrix construction with `prepare_rimatr_for_r_sync()`, using the default auxiliary basis loaded by `Molecule::build`, followed by explicit four-center reconstruction from the RI factor.

The script temporarily injects one ignored Rust unit test into `rest/src/lib_rint/mod.rs`, runs it, and restores the original file in a `finally` block.  Persistent files are kept under `rest/src/lib_rint/tmp`.

The output line starts with `TMP_RI4C_BENCH` and includes wall times in seconds:

- `libcint_direct_s`
- `lib_rint_ri_setup_s`
- `lib_rint_ri_reconstruct_s`
- `lib_rint_ri_total_s`
- `total_over_libcint`

For the older direct-scalar four-center comparison, see `work_pool/h2o_lib_rint_vs_libcint_int2e_cart_report.md`.

Larger cases can be requested explicitly, for example:

```bash
python3 rest/src/lib_rint/tmp/bench_h2o_ri4c_vs_libcint.py --cases cc-pVDZ --repeat 1
```

For this specific question, use the temporary test-harness benchmark above.  It times direct libcint four-center ERI generation against lib_rint RI four-center reconstruction.

For exact lib_rint four-center ERI only, use the release benchmark below.  It compares the old scalar `eri_ao_4c_r` loop with the shell-quartet batched `int4c_r_shell_block` path and restores `mod.rs` after the run:

```bash
python3 rest/src/lib_rint/tmp/bench_exact4c_scalar_vs_shellblock.py --cases sto-3g --repeat 1
```

To compare libcint's direct exact four-center tensor generation against lib_rint's exact shell-block path, use:

```bash
python3 rest/src/lib_rint/tmp/bench_exact4c_vs_libcint.py --cases sto-3g --repeat 1
```

The first run injects a temporary ignored Rust test, builds the release test
binary, restores `mod.rs`, then executes the built binary.  To build once and
reuse the release binary for later repeats without another `cargo build
--release`, run:

```bash
python3 rest/src/lib_rint/tmp/bench_exact4c_vs_libcint.py --cases cc-pVDZ --build-only
python3 rest/src/lib_rint/tmp/bench_exact4c_vs_libcint.py --reuse-release --cases cc-pVDZ --repeat 3
```

For shell-quartet timing diagnostics, add `--profile-shells`.  This prints
`TMP_EXACT4C_SHELL_PROFILE` rows grouped by `(la, lb, lc, ld, nroots)` before
the main `TMP_EXACT4C_LIBCINT_BENCH` line:

```bash
python3 rest/src/lib_rint/tmp/bench_exact4c_vs_libcint.py --cases cc-pVDZ --repeat 1 --profile-shells
```

The release-binary runner below is a different, coarser diagnostic.  It compares REST's normal SCF timing against the extra `run_lib_rint` RI-r2 block, so it is not the direct libcint-4c-vs-lib_rint-RI-4c comparison:

```bash
python3 rest/src/lib_rint/tmp/run_release_h2o_rir2.py --cases sto-3g --threads 4
```
