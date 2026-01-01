#!/bin/bash
# ============================================================
# non0tab correctness benchmark v2
#   2 × H₂O @ 15 Å, cc-pVDZ, B3LYP
#   Dense (ao_cutoff=0) vs Sparse (ao_cutoff=1e-14 for debug)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REST_HOME=/home/igor/Documents/Package-Pool/rest_workspace
REST_BIN="$REST_HOME/target/release/rest"
cd "$SCRIPT_DIR"

POS_STR='O     0.000000    0.000000    0.000000
H     0.758602    0.000000    0.504284
H     0.758602    0.000000   -0.504284
O     0.000000    0.000000   15.000000
H     0.758602    0.000000   15.504284
H     0.758602    0.000000   14.495716'

build_ctrl() {
    local AO_CUT=$1
    local TAG=$2
    cat > "ctrl_${TAG}.toml" << EOFCTRL
[ctrl]
xc = "b3lyp"
basis_path = "cc-pvdz"
eri_type = "ri-v"
basis_type = "Spheric"
spin = 1
charge = 0
print_level = 2
num_threads = 4
max_scf_cycle = 300
scf_acc_rho = 1.0e-7
scf_acc_eev = 1.0e-5
ao_cutoff = ${AO_CUT}
drop_dense_ao = false
vxc_screen_threshold = 0.0
use_dm_only = false
[geom]
elem = "O,H,H,O,H,H"
position = """
${POS_STR}"""
EOFCTRL
}

# Build release if needed
if [ ! -f "$REST_BIN" ]; then
    echo "=== Building REST release ==="
    source ~/.bash_profile
    cd "$REST_HOME/rest" && cargo build --release 2>&1 | tail -3
    cd "$SCRIPT_DIR"
fi

source ~/.bash_profile

# Dense
echo "=== RUN 1: DENSE (ao_cutoff=0.0) ==="
build_ctrl 0.0 dense
"$REST_BIN" -i ctrl_dense.toml 2>&1 | tee log_dense.txt
E_DENSE=$(grep "SCF energy" log_dense.txt | tail -1 | awk '{print $(NF-1)}')
echo "DENSE  E = $E_DENSE"

# Sparse (debug cutoff)
echo "=== RUN 2: SPARSE (ao_cutoff=1.0e-8) ==="
build_ctrl 1.0e-8 sparse
"$REST_BIN" -i ctrl_sparse.toml 2>&1 | tee log_sparse.txt
E_SPARSE=$(grep "SCF energy" log_sparse.txt | tail -1 | awk '{print $(NF-1)}')
echo "SPARSE E = $E_SPARSE"

# Compare
echo ""
echo "=== COMPARISON ==="
echo "E_dense  = $E_DENSE"
echo "E_sparse = $E_SPARSE"
echo ""
grep "DEBUG-non0tab" log_sparse.txt

if [ -n "$E_DENSE" ] && [ -n "$E_SPARSE" ]; then
    DELTA=$(awk "BEGIN { d = $E_DENSE - $E_SPARSE; if (d < 0) d = -d; printf \"%.2e\", d }")
    echo "|ΔE| = $DELTA"
fi
