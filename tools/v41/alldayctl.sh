#!/bin/bash
# alldayctl.sh (alias v41allday) start|status|stop|log - the unattended chain in all_day.sh
#   start   run calibration -> tier map -> GGUF layout -> GSQ quant pass, detached (safe to rerun: resumes)
#   status  which step, plus that step's own status
#   stop    stop after the current GPU layer (sets the STOP files of all_day + the running driver)
#   log     follow allday.log
W=/media/Crucial1TB/v41-calib; PIDF=$W/allday.pid
running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "${1:-status}" in
start)
  if running; then echo "already running (pid $(cat "$PIDF"))"; exit 1; fi
  setsid nohup bash /home/mal/AI/Xolotl/tools/v41/all_day.sh >> "$W/allday.out" 2>&1 < /dev/null &
  echo $! > "$PIDF"; sleep 3; echo "started all_day pid $(cat "$PIDF")"; tail -3 "$W/allday.log" ;;
stop)
  touch "$W/allday.STOP" "$W/STOP" /media/Crucial1TB/v41-quant-mixed/STOP /media/Crucial1TB/v41-quant-q2/STOP 2>/dev/null
  echo "stop requested: the running layer finishes, then everything exits.  Resume later: v41allday start"
  echo "(before resuming, nothing to clean: start removes the STOP files it owns; quant dirs: rm <qwork>/STOP)" ;;
log) tail -n 30 -f "$W/allday.log" ;;
*)
  running && echo "RUNNING (pid $(cat "$PIDF"))" || echo "NOT RUNNING"
  echo "step: $(cat "$W/allday.step" 2>/dev/null)"
  tail -4 "$W/allday.log" 2>/dev/null | sed 's/^/  log: /'
  case "$(cat "$W/allday.step" 2>/dev/null)" in
    1/4*) /home/mal/AI/Xolotl/tools/v41/calibctl.sh status 2>/dev/null | head -4 ;;
    4/4*) tail -2 /media/Crucial1TB/v41-quant-mixed/quant.log ;;
  esac ;;
esac
