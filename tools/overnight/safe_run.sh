#!/bin/bash
# 单次实验：gate → 停服务 → 跑引擎 → 记录 → 恢复服务。任何失败立刻收手。
# 用法: safe_run.sh <标签> <引擎参数...>   (环境变量由调用方提供)
set -u
TAG=$1; shift
LOG=/tmp/ov_${TAG}.log
echo "=== [$TAG] stop service first ==="
systemctl --user stop dsv41.service 2>/dev/null; sleep 5
echo "=== [$TAG] gate ==="
/data/strata-new/tools/overnight/gate.sh 90 || { echo "[$TAG] gate 未过，放弃本次"; exit 10; }
echo "=== [$TAG] run ==="
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
timeout 900 /data/strata-new/build-sycl-jit/strata-dsv41 "$@" >/dev/null 2>"$LOG" </tmp/ov_ids.txt
rc=$?
date +%s > /tmp/overnight_last_engine_end
echo "[$TAG] rc=$rc" | tee -a /tmp/overnight_results.txt
# 失败即收手：不当场重启服务，交给调用方判断
if [ $rc -ne 0 ]; then
  echo "[$TAG] 引擎非零退出，检查日志：$LOG"
  grep -qiE 'DEVICE_LOST|OUT_OF_DEVICE_MEMORY' "$LOG" && echo "[$TAG] **显存/设备错误，进入冷却，不自动重启**" 
  exit $rc
fi
echo "=== [$TAG] 恢复服务 ==="
/data/strata-new/tools/overnight/gate.sh 60 && systemctl --user start dsv41.service
sleep 5; systemctl --user is-active dsv41.service
