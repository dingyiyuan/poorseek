#!/bin/bash
# fetch_v41_loop.sh DEST REPO [WAIT_PID] - keep re-running fetch_v41.sh (same env: ONLY/FIT/MARGIN_GB/JOBS) until every
# file is verified.  Each run resumes partial files (curl -C -) and skips verified ones (.ok), so a dropped connection,
# a killed script or a reboot (re-run this) just continues.  WAIT_PID: an already-running fetch to let finish first
# (never two fetches on one DEST at once - they would write the same file).  Stops after MAX_RUNS (default 100).
set -u
DEST=$1; REPO=$2; WAIT=${3:-}; MAX=${MAX_RUNS:-100}; HERE=$(dirname "$(readlink -f "$0")")
[ -n "$WAIT" ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
for run in $(seq 1 "$MAX"); do
  bash "$HERE/fetch_v41.sh" "$DEST" "$REPO" && { echo "$(date '+%F %T') loop: ALL VERIFIED after run $run"; exit 0; }
  rc=$?
  [ -f "$DEST/.space-stop" ] && { echo "$(date '+%F %T') loop: out of space - free some and re-run"; exit 2; }
  echo "$(date '+%F %T') loop: run $run ended rc=$rc - retrying in 60 s"; sleep 60
done
echo "loop: gave up after $MAX runs"; exit 1
