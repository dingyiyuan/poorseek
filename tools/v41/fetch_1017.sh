#!/bin/bash
# fetch_1017.sh [WAIT_PID] - the rest of deepseek-ai/DeepSeek-V4.1-Flash-1017 after the Engram shards, ONE download at a
# time (Mal), split over the drives that have room (2026-10-18):
#   Team CX2   shards 47-48 (Engram, 2 x 101.5 GB)              - fetched first, separately (fetch-engram.log there)
#   Crucial1TB the small files (index, tokenizer, config) FIRST, then shards 1-39   = MAIN (config/index/tokenizer)
#   Crucible   shards 40-46 (layers 37-39, head/norm, mtp.0-2; 31.4 GB)
# Every tool finds them through  V41_SRC="$V41_SRC_1017"  (below; same list in docs/ds4/V41_1017_QUANT_PLAN.md).
# Resumable: re-run any time (partial files resume, verified .ok files are skipped).  WAIT_PID = a running fetch to
# let finish first.  Log: <dir>/fetch-rest.log in each folder.
set -u
HERE=$(dirname "$(readlink -f "$0")"); REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
CRU=/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017
TCX="/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017"
CRB=/media/mal/Crucible/models/DeepSeek-V4.1-Flash-1017
V41_SRC_1017="$CRU:$TCX:$CRB"
log() { echo "$(date '+%F %T') fetch_1017: $*"; }
[ -n "${1:-}" ] && { log "waiting for pid $1"; while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
for f in model-00047-of-00048 model-00048-of-00048; do
  [ -f "$TCX/$f.safetensors.ok" ] || log "WARNING: $f not verified on Team CX2 yet (re-run fetch-engram there)"; done
pass() {   # dir, env ONLY/SKIP: loop until verified; a full drive waits (re-checks every 5 min)
  local d=$1; shift; mkdir -p "$d"
  while :; do
    env FIT=1 MARGIN_GB=10 JOBS=1 "$@" bash "$HERE/fetch_v41_loop.sh" "$d" "$REPO" >> "$d/fetch-rest.log" 2>&1; rc=$?
    [ $rc = 2 ] || return $rc
    log "$d full - waiting for space"; sleep 300
  done
}
log "1/3 small files -> $CRU";  pass "$CRU" SKIP='^model-000' || { log "small files failed"; exit 1; }
log "2/3 shards 1-39 -> $CRU";  pass "$CRU" ONLY='^model-000(0[1-9]|[1-3][0-9])-of-00048' || { log "shards 1-39 failed"; exit 1; }
log "3/3 shards 40-46 -> $CRB"; pass "$CRB" ONLY='^model-0004[0-6]-of-00048' || { log "shards 40-46 failed"; exit 1; }
log "ALL 1017 FILES VERIFIED - V41_SRC=\"$V41_SRC_1017\""
