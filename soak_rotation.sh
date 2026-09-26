#!/bin/bash
# 45-min soak: random rotation of benches against the deployed box, no restart.
# Each bench exits non-zero on failure; we collect results and fail loudly.
BASE="http://192.168.1.238:8000/v1"
LOG=/tmp/soak_results.log
: > "$LOG"
run() {
  local name="$1"; shift
  echo "=== $(date +%H:%M:%S) START $name ===" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 uv run "$name" "$@" --base-url "$BASE"
  local rc=$?
  echo "=== $(date +%H:%M:%S) DONE $name rc=$rc ===" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then echo "SOAK_FAIL $name rc=$rc" | tee -a "$LOG"; fi
}

run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000
run needle-test --lengths 50000,100000,200000
run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000
run cache-pressure --kv-size 460000
run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000
run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000
run needle-test --lengths 50000,100000,200000
run agent-sim --sessions 4 --main-tokens 150000 --sub-tokens 40000

echo "=== SOAK COMPLETE $(date +%H:%M:%S) ===" | tee -a "$LOG"
if grep -q SOAK_FAIL "$LOG"; then echo "SOAK RESULT: FAIL"; exit 1; fi
echo "SOAK RESULT: PASS"
