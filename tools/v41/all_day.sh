#!/bin/bash
# all_day.sh - the whole unattended V4.1 chain, one step after another, each resumable (rerun = skips what is done):
#   1. calibration layers 0-39 + head gate  (calib_layers.sh; GPU ~11.5 min/layer)
#   2. tier map  v41_alloc.py --per-proj -> alloc_full.json  (CPU, minutes)
#   3. lay out the final GGUF on SSD NVME  (CPU ~10 min; experts are holes until step 4 fills them)
#   4. GSQ quantize pass, mixed map, written in place  (quant_layers.sh; GPU ~4-5 h est.) + its head gate
#   (the all-Q2_0 control pass is skipped - Mal 2026-10-10; run it by hand if the GSQ head numbers look doubtful)
# Stops at the first failure (see $W/allday.log).  Started/stopped via:  v41allday start|status|stop|log
set -u
cd /home/mal/AI/Xolotl
W=/media/Crucial1TB/v41-calib
MAP=$W/alloc_full.json
GDIR="/media/mal/SSD NVME/AI/V41-GSQ"; GGUF="$GDIR/V41-GSQ.gguf"
QMIX=/media/Crucial1TB/v41-quant-mixed; QQ2=/media/Crucial1TB/v41-quant-q2
LOG=$W/allday.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { echo "$*" > "$W/allday.step"; log "== $*"; }
die() { log "FAILED: $*"; echo "FAILED: $*" > "$W/allday.step"; exit 1; }
rm -f "$W/allday.STOP" "$QMIX/STOP" "$QQ2/STOP"
chk() { [ -f "$W/allday.STOP" ] && { log "stop requested - exiting"; echo "stopped" > "$W/allday.step"; exit 0; }; }
log "all_day start (pid $$)"

# 1. calibration ------------------------------------------------------------------------------------------------------
step "1/4 calibration"
if [ ! -f "$W/stats/head_L40.json" ]; then
  if [ -f "$W/driver.pid" ] && kill -0 "$(cat "$W/driver.pid")" 2>/dev/null; then
    read -r L st < "$W/current" 2>/dev/null || st=""
    if [ "$st" = waiting-gpu ]; then        # an armed start-when-idle driver: replace it, no 15-min idle wait
      tools/v41/calibctl.sh stop-now >> "$LOG" 2>&1
    fi
  fi
  if ! { [ -f "$W/driver.pid" ] && kill -0 "$(cat "$W/driver.pid")" 2>/dev/null; }; then
    IDLE_MIN=0 tools/v41/calibctl.sh start >> "$LOG" 2>&1
  fi
  for att in 1 2 3; do
    pid=$(cat "$W/driver.pid")
    while kill -0 "$pid" 2>/dev/null; do chk; sleep 60; done
    { [ -f "$W/stats/L39.json" ] && [ -f "$W/stats/head_L40.json" ]; } && break
    chk; [ "$att" = 3 ] && break
    log "calibration driver exited early (attempt $att: $(tail -1 "$W/calib.log")) - restarting now"; sleep 5
    IDLE_MIN=0 tools/v41/calibctl.sh start >> "$LOG" 2>&1
  done
  [ -f "$W/stats/L39.json" ] || die "calibration stopped before layer 39 ($(tail -1 "$W/calib.log"))"
  [ -f "$W/stats/head_L40.json" ] || die "calibration head gate missing (logs/head.log)"
fi
log "calibration complete (40/40 + head)"; chk

# 2. tier map ---------------------------------------------------------------------------------------------------------
step "2/4 tier map"
if [ ! -f "$MAP" ]; then
  nice -n 15 python3 tools/v41/v41_alloc.py --layers 0-39 --per-proj --out "$MAP" > "$W/logs/alloc_full.log" 2>&1 \
    || die "v41_alloc.py (logs/alloc_full.log)"
fi
log "tier map: $(tail -3 "$W/logs/alloc_full.log" 2>/dev/null | tr '\n' ' ')"; chk

# 3. GGUF layout ------------------------------------------------------------------------------------------------------
step "3/4 GGUF layout"
mkdir -p "$GDIR"
if [ ! -f "$GGUF.experts.json" ]; then
  nice -n 15 python3 tools/v41/v41_gguf.py --out "$GGUF" --rtn-map "$MAP" --experts-in-place --no-engram --threads 4 \
    > "$W/logs/gguf_layout.log" 2>&1 || die "v41_gguf.py layout (logs/gguf_layout.log)"
fi
log "layout: $(ls -la "$GGUF" 2>&1)"; chk

# 4. GSQ quantize pass (mixed) ---------------------------------------------------------------------------------------
step "4/4 GSQ quantize pass (mixed map) -> $GGUF"
if [ ! -f "$QMIX/stats/head_L40.json" ]; then
  for att in 1 2 3; do
    bash tools/v41/quant_layers.sh "$QMIX" 0 39 -- --map "$MAP" --gguf "$GGUF" >> "$LOG" 2>&1 && break
    chk; [ "$att" = 3 ] && die "quant pass (mixed), 3 attempts: $(tail -1 "$QMIX/quant.log")"
    log "quant pass exited early (attempt $att: $(tail -1 "$QMIX/quant.log")) - restarting now"; sleep 5
  done
  [ -f "$QMIX/stats/head_L40.json" ] || { chk; die "quant pass (mixed) ended without head_L40.json"; }
fi
log "GSQ mixed pass complete"; chk

step "DONE - GSQ file $GGUF; compare head gates: teacher $W/stats/head_L40.json vs GSQ $QMIX/stats/head_L40.json"
