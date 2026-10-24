#!/bin/bash
# calibctl.sh - start / stop / check the V4.1 per-layer calibration run (alias: v41calib).
#   v41calib start [FROM TO]  start (or resume) in the background, detached from the terminal; default layers 0..39.
#                             Done layers are skipped, so `start` after any stop simply continues.
#   v41calib start-when-idle  same, but the first layer waits until the GPU lock + VRAM stay free 15 min (IDLE_MIN)
#   v41calib stop             graceful: finish the layer that is running, then exit (or exit at once if waiting)
#   v41calib stop-now         kill the running layer immediately; that layer reruns from scratch on the next start
#   v41calib status           running? which layer, progress, layers done, time per layer, ETA
#   v41calib log              follow the run log (Ctrl-C to leave; the run keeps going)
set -u
W=${V41_WORK:-/media/Crucial1TB/v41-calib}
REPO=/home/mal/AI/Xolotl
PIDF="$W/driver.pid"
mkdir -p "$W/logs"

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
done_layers() { for f in "$W"/stats/L[0-9][0-9].json; do [ -f "$f" ] || continue; n=${f##*/L}; echo $((10#${n%.json})); done | sort -n | tr '\n' ' '; }

case "${1:-status}" in
start|start-when-idle)
  [ "$1" = start-when-idle ] && export IDLE_MIN=${IDLE_MIN:-15}
  if running; then echo "already running (driver pid $(cat "$PIDF")) - see: v41calib status"; exit 1; fi
  rm -f "$W/STOP"
  setsid nohup bash "$REPO/tools/v41/calib_layers.sh" "${2:-0}" "${3:-39}" >> "$W/logs/driver.log" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 1
  echo "started driver pid $(cat "$PIDF") (layers ${2:-0}..${3:-39}); done so far: $(done_layers)"
  ;;
stop)
  if ! running; then echo "not running"; exit 0; fi
  touch "$W/STOP"
  echo "stop requested: the current layer finishes, then the driver exits (v41calib status to watch;"
  echo "v41calib stop-now to abort the layer instead)"
  ;;
stop-now)
  touch "$W/STOP"
  if running; then
    pid=$(cat "$PIDF")
    for u in $(systemctl --user list-units --type=scope --no-legend 'ds4-guard-*' 2>/dev/null | awk '{print $1}'); do
      # only scopes that belong to this run (their command line is v41_calib.py)
      if systemctl --user status "$u" 2>/dev/null | grep -q v41_calib.py; then
        systemctl --user kill -s TERM "$u"; echo "killed $u"
      fi
    done
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    sleep 2
    echo "$(date '+%F %T') stop-now: driver $pid killed; current layer will rerun on next start" >> "$W/calib.log"
  fi
  rm -f "$W/current" "$PIDF"
  echo "stopped. done layers: $(done_layers)"
  ;;
status)
  if running; then echo "RUNNING (driver pid $(cat "$PIDF"))$([ -f "$W/STOP" ] && echo ' - stop requested, exits after this layer')"
  else echo "STOPPED"; fi
  if [ -f "$W/current" ]; then
    read -r L state < "$W/current"
    N=$(printf %02d "$L")
    echo "layer $L: $state"
    [ "$state" = running ] && tail -1 "$W/logs/L$N.log" 2>/dev/null | sed 's/^/  last: /'
  fi
  d=$(done_layers); n=$(echo $d | wc -w)
  echo "done: $n/40 layers  [${d}]"
  python3 - "$W" "$n" <<'EOF'
import json, glob, sys
w, n = sys.argv[1], int(sys.argv[2])
done = {int(p[-7:-5]): json.load(open(p))["summary"].get("seconds", 0) for p in sorted(glob.glob(f"{w}/stats/L[0-9][0-9].json"))}
ENGRAM = (1, 14)   # these layers also read the Engram tables: ~2x a normal layer
normal = sorted(v for k, v in done.items() if k not in ENGRAM) or sorted(done.values())
if normal:
    m = normal[len(normal) // 2]
    rest = sum((2 if L in ENGRAM else 1) * m for L in range(40) if L not in done)
    print(f"time per layer ~{m / 60:.1f} min (Engram layers 1, 14 ~2x); remaining {40 - len(done)} layers ~{rest / 3600:.1f} h of GPU time")
EOF
  tail -3 "$W/calib.log" | sed 's/^/  log: /'
  ;;
log)
  tail -n 20 -f "$W/calib.log"
  ;;
*)
  sed -n 2,10p "$0"; exit 2
  ;;
esac
