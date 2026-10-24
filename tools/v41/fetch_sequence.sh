#!/bin/bash
# fetch_sequence.sh - one file at a time: let the running main fetch finish its current shard, then the two Engram
# shards to their own SSDs, then resume the main fetch without them.  Log: $MAIN/sequence.log
set -u
F=~/AI/Xolotl/tools/v41/fetch_v41.sh
MAIN=/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
E47=/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
E48="/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
CUR=${1:?current shard name or -}; PGID=${2:?main fetch pgid or -}; FIRST=${FIRST:-}   # FIRST=regex: fetch these into MAIN before the Engram shards
log() { echo "$(date '+%F %T') $*"; }
if [ "$PGID" != - ]; then
  log "waiting for $CUR to verify, then pausing main fetch (pgid $PGID)"
  until [ -f "$MAIN/$CUR.ok" ]; do
    kill -0 -- "-$PGID" 2>/dev/null || { log "main fetch already gone"; break; }
    sleep 20
  done
  kill -- "-$PGID" 2>/dev/null; sleep 3; kill -9 -- "-$PGID" 2>/dev/null
  log "main fetch paused"
fi
if [ -n "$FIRST" ]; then
  log "fetch $FIRST -> main first"
  ( cd "$MAIN" && JOBS=1 FIT=1 MARGIN_GB=10 ONLY="$FIRST" "$F" "$MAIN" ) || log "WARN $FIRST fetch returned $?"
fi
for pair in "model-00047-of|$E47" "model-00048-of|$E48"; do
  pat=${pair%%|*}; dst=${pair#*|}; mkdir -p "$dst"
  log "fetch $pat -> $dst"
  ( cd "$dst" && JOBS=1 FIT=1 MARGIN_GB=10 ONLY="$pat" "$F" "$dst" ) || log "WARN $pat fetch returned $?"
done
log "resume main fetch (skipping the Engram shards)"
( cd "$MAIN" && JOBS=1 FIT=1 MARGIN_GB=10 SKIP='model-0004[78]-of' "$F" "$MAIN" )
log "sequence done rc=$?"
