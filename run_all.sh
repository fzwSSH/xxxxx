#!/bin/bash
# Runs every check on ONE GPU and writes logs/<name>.log. Usage:
#   bash run_all.sh                 # checks (~5 min): reference kernels, sandbox-rule check
#   T1=1 bash run_all.sh            # + T1 tile sweep, quick grid (full grid: SWEEP_GRID=full)
#   T2=1 bash run_all.sh            # + T2 warp-specialization timing
# GPU: CUDA_VISIBLE_DEVICES (default 0).
cd "$(dirname "$0")/moe_gu"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p ../logs
PY=${PY:-python}
run() {  # name timeout_s cmd...
  local n=$1 t=$2; shift 2
  echo "== $n: $*"
  timeout $t "$@" > ../logs/$n.log 2>&1
  local rc=$?
  [ $rc = 124 ] && echo "   TIMEOUT after $t s (a hang?)"
  grep -E "^(PASS|FAIL|SKIP|INFO|REF|BEST|\[|\|)" ../logs/$n.log | tail -40
  echo "   exit $rc, full log logs/$n.log"
}
$PY -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, triton.__file__); print('GPU', torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
run test_ref 900 $PY test_ref.py
if $PY -c "import RestrictedPython" 2>/dev/null; then
  run rp_ref 600 $PY rp_check.py ref_gu.py skip ../logs/rp
  run rp_dn 600 $PY rp_check.py dn_ref.py skip ../logs/rp
  run rp_ws 600 $PY rp_check.py ws_gu.py skip ../logs/rp
else
  echo "== rp_check skipped (pip install RestrictedPython)"
fi
if [ "${T1:-0}" = 1 ]; then
  run sweep 7200 $PY sweep.py --grid ${SWEEP_GRID:-quick} --json ../results/T1_sweep.json --md ../results/T1_sweep.md
fi
if [ "${T2:-0}" = 1 ]; then
  run bench_ws 3600 $PY bench.py --json ../results/T2_ws.json --md ../results/T2_ws.md
fi
