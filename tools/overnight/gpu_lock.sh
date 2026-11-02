#!/bin/bash
# GPU 独占锁：任何要起引擎的一方先 'gpu_lock.sh take <owner>'，用完 'gpu_lock.sh release <owner>'。
# gate.sh 会检查这个锁；锁存在时拒绝启动引擎。
LOCK=/tmp/gpu.lock
case "$1" in
  take)
    if [ -e "$LOCK" ]; then cat "$LOCK"; exit 1; fi
    printf "owner=%s\nsince=%s\n" "${2:-unknown}" "$(date -Is)" > "$LOCK"; echo "GPU 锁已取得: ${2:-unknown}";;
  release)
    if [ -e "$LOCK" ] && grep -q "owner=${2:-unknown}" "$LOCK"; then rm -f "$LOCK"; echo "GPU 锁已释放: ${2:-unknown}";
    elif [ -e "$LOCK" ]; then cat "$LOCK"; echo "锁不属于 ${2:-unknown}，未释放"; exit 1
    else echo "当前没有锁"; fi;;
  status) [ -e "$LOCK" ] && cat "$LOCK" || echo "无锁";;
  *) echo "用法: $0 {take <owner>|release <owner>|status}"; exit 2;;
esac
