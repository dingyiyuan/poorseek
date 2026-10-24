#!/bin/bash
# run_dprof.sh - DS41_DPROF measurement run for the S4 step-3 dense accounting (see
# docs/xolotl-sycl-s4-step3-dense-dprof-report.md).  The established 3-token command + the env gate,
# MemAvailable-watched and timeout-guarded: the card wedged once on an expert launch, so never leave a
# process behind.
#!/bin/bash
# DS41_DPROF measurement run (S4 step 3): the established 3-token command, DS41_DPROF=1.  Watchdog on MemAvailable and
# a hard timeout: the card wedged once on a bad expert launch, so never leave a process behind.
TAG="$1"; shift
BIN=/home/dyy/deepseek-work/s4/step2/ds41_generate_gpuexp
OUT=/home/dyy/deepseek-work/s4/step2
cd /home/dyy/deepseek-work/llm-for-glm-flash/models/dsv41-gsq || exit 9
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || { echo "setvars failed"; exit 8; }
export ONEAPI_DEVICE_SELECTOR=level_zero:0
unset STRATA_ROUTED_HOST
LIMIT_KB=$((4*1024*1024))
start=$(date +%s)
timeout -k 10 300 stdbuf -oL -eL "$BIN" -m V41-GSQ-1017.gguf --backend cuda --experts gpu \
  --engram engram-layers.1:engram-layers.14 --rank rank.bin \
  --ids ${IDS:-100,200,300} -n 1 --ctx 2048 --temp 0 --slots 1200 --prompt-batch 0 --prefill-chunk 1 "$@" \
  > "$OUT/dprof-$TAG.log" 2>&1 &
PID=$!
MIN_AVAIL=999999999
while kill -0 $PID 2>/dev/null; do
  AVAIL=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  [ -n "$AVAIL" ] && [ "$AVAIL" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$AVAIL
  if [ -n "$AVAIL" ] && [ "$AVAIL" -lt "$LIMIT_KB" ]; then
    echo "WATCHDOG: MemAvailable ${AVAIL} kB < ${LIMIT_KB} kB -- killing" >> "$OUT/dprof-$TAG.log"
    kill -9 $PID 2>/dev/null; break
  fi
  sleep 2
done
wait $PID 2>/dev/null; RC=$?
echo "TAG=$TAG WALL=$(( $(date +%s) - start ))s EXIT=$RC MIN_AVAIL_KB=$MIN_AVAIL"
