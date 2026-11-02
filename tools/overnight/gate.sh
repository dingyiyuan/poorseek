#!/bin/bash
# 健康门：任何引擎启动前必须返回 0。用法: gate.sh [最短等待秒数]
WANT=${1:-90}
LASTF=/tmp/overnight_last_engine_end
now=$(date +%s)
if [ -f "$LASTF" ]; then
  last=$(cat "$LASTF"); d=$(( now - last ))
  if [ "$d" -lt "$WANT" ]; then echo "gate: 距上次引擎结束仅 ${d}s，等 $((WANT-d))s"; sleep $((WANT-d)); fi
fi
# 1) 残留进程
# GPU 独占锁：别人持有锁时拒绝启动；持有者通过 GPU_LOCK_OWNER 环境变量声明自己
if [ -e /tmp/gpu.lock ] && [ "${GPU_LOCK_OWNER:-}" != "$(sed -n 's/^owner=//p' /tmp/gpu.lock)" ]; then
  echo "gate: FAIL GPU 被占用 →"; cat /tmp/gpu.lock; exit 1
fi
n=$(ps -eo cmd | grep -cE '(^|/)strata-dsv41( |$)')   # 只算真正的引擎进程：命令行里 strata-dsv41 前必须是行首或斜杠（避免误匹配 ninja/icpx 的构建目标名）
[ "$n" -gt 0 ] && { echo "gate: FAIL 还有 $n 个引擎进程"; exit 1; }
# 2) 显卡在
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
c=$(sycl-ls 2>/dev/null | grep -c 'Arc(TM) Pro B70')
[ "$c" -lt 1 ] && { echo "gate: FAIL Arc 不在 sycl-ls 里（可能 wedged）"; exit 2; }
# 3) 近 10 分钟无新 reset
r=$(journalctl -k --since "10 min ago" --no-pager 2>/dev/null | grep -cEi 'xe .*(declared device|wedged)')
[ "$r" -gt 0 ] && { echo "gate: FAIL 10 分钟内有 wedged 声明"; exit 3; }
echo "gate: OK"; exit 0
