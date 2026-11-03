#!/bin/bash
# 明早一条命令：专家内核三条通路一次跑完（微基准 → 交叉点表 → 张量核 A/B）。
#
#   tools/overnight/morning.sh            # 全部
#   tools/overnight/morning.sh fast       # 跳过最慢的张量核 A/B（约 6 分钟）
#
# 前提：服务是 inactive（脚本会检查，不会替你 systemctl start 生产单元）。
# 每一步都会把输出同时写到 /tmp/morning-<时间戳>/ 下，跑完打印汇总。
set -uo pipefail
MODE=${1:-all}
ROOT=/data/strata-new
LOCK=$ROOT/tools/overnight/gpu_lock.sh
OUT=$(mktemp -d "/tmp/morning-$(date +%m%d-%H%M).XXXXXX")
cd "$ROOT" || exit 2
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
export GPU_LOCK_OWNER=morning
say () { echo; echo "──────────────── $* ────────────────"; }

ulimit -c 0   # 崩溃时别写 core：11-03 早上一个 core dump 把作业挂了 31 分钟，还连带把卡搞掉了

say "0 前置检查"
systemctl --user is-active dsv41.service | sed 's/^/  服务: /'
if pgrep -f '(^|/)strata-dsv41( |$)' >/dev/null; then echo "  还有引擎进程在跑，先 ./START.sh stop"; exit 2; fi
"$LOCK" status | sed 's/^/  锁: /'
"$LOCK" take morning || { echo "  GPU 被占用，先看 gpu_lock.sh status"; exit 2; }
trap '"$LOCK" release morning' EXIT

say "1 解量化算术 + 瓦片构建（隐含 TFLOPS）"
ninja -C build-sycl-jit iq_dequant_bench >/dev/null || exit 3
./build-sycl-jit/iq_dequant_bench 131072 20 4 80 128 2>&1 | tee "$OUT/1-dequant.txt"

say "2 融合 vs 两段式的 rows/expert 交叉点表（真实形状 2560/1280）"
ninja -C build-sycl-jit xmx_gemm_bench >/dev/null || exit 3
./build-sycl-jit/xmx_gemm_bench 16 both 30 sweep real 2>&1 | tee "$OUT/2-cross.txt"

say "3 张量核专家内核：正确性对拍 + 瓦片吞吐（不启引擎，几十秒就有结论）"
TCB=build-sycl-tc2/ds4_iq_test_tc
if [ -x "$TCB" ]; then
  echo "--- 对拍：F32 基线（STRATA_DS4_NO_TC=1）"
  STRATA_DS4_NO_TC=1 "$TCB" 2>&1 | tail -18 | tee "$OUT/3a-parity-f32.txt"
  echo "--- 对拍：张量核（STRATA_DS4_TC=1）——worst 列与上一组比，FAIL 就是直译错了"
  STRATA_DS4_TC=1 "$TCB" 2>&1 | tail -18 | tee "$OUT/3b-parity-tc.txt"
  echo "--- 瓦片吞吐（IQ2_XXS，6 专家，rows/expert = 96/256/900）"
  for v in NO_TC TC; do
    if [ "$v" = NO_TC ]; then E="STRATA_DS4_NO_TC=1"; else E="STRATA_DS4_TC=1"; fi
    echo "  [$v]"; env $E "$TCB" --bench-tiles 16 6 96,256,900 2>&1 | tail -8 \
      | sed "s/^/    /" | tee -a "$OUT/3c-tiles-$v.txt"
  done
else
  echo "  没有 $TCB（先 ninja -C build-sycl-tc2 ds4_iq_test_tc）"
fi

say "4 两段式专家 GEMM：数值对拍（DS4_EXP_2PHASE=1，几秒钟，不启引擎）"
if [ -x "$TCB" ]; then
  # 对拍用的集合只有 37 行，而两段式的行数门槛默认 96 —— 必须把门槛压到 1 才会真的接管，
  # 否则它会静默回退到融合路径，数字和第 3 步一模一样，看起来"通过"其实一次没跑。
  # 先只跑 IQ2_XXS、60 s 超时：11-03 早上全量跑第一次就把卡弄掉了，这次从最小样本开始
  echo "  [4a] 单类型 IQ2_XXS，60 s 超时"
  STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 DS4_EXP_2PHASE_MINROWS=1 \
      timeout 60 "$TCB" --type 16 2>&1 | tail -10 | tee "$OUT/4a-twophase-iq2.txt"
  rc=${PIPESTATUS[0]:-$?}
  if [ "$rc" -ne 0 ]; then
    echo "  ⚠ 单类型就失败了（rc=$rc）：不要继续全量，先看 $OUT/4a-twophase-iq2.txt"
  else
    echo "  [4b] 全量（同样的开关）"
    STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 DS4_EXP_2PHASE_MINROWS=1 \
        timeout 120 "$TCB" 2>&1 | tail -14 | tee "$OUT/4-twophase-parity.txt"
  fi
  echo "  对比上面第 3 步的 3a-parity-f32.txt：set gate_up / set down 的 worst 不应明显变差（容差 3e-3）"
  echo "  ⚠ 若与本文件里的 3a 完全逐位相同，先怀疑没接管（行数门槛/类型/激活指针）"
else
  echo "  没有 $TCB"
fi

say "4.5 RM=128（同一份权重少解一半）：单类型对拍 + 计时"
if [ -x "$TCB" ]; then
  echo "  [4.5a] 基线 RM=64"
  STRATA_DS4_NO_TC=1 timeout 120 "$TCB" --type 16 2>&1 | grep -E "set gate_up|set down|ALL OK|FAIL" | head -4 | tee "$OUT/45a-rm64.txt"
  echo "  [4.5b] DS4_IQ_RM128=1（数值应一致，瓦片微基准预期 1.75x）"
  STRATA_DS4_NO_TC=1 DS4_IQ_RM128=1 timeout 120 "$TCB" --type 16 2>&1 | grep -E "set gate_up|set down|ALL OK|FAIL" | head -4 | tee "$OUT/45b-rm128.txt"
  echo "  ⚠ 两组的 worst 应当在同一量级；若 RM128 组明显变差或 FAIL，说明 128 行块的边界处理有问题"
else
  echo "  没有 $TCB"
fi

if [ "$MODE" = fast ]; then
  say "跳过第 5 步（fast）"
else
  say "5 张量核 A/B（release-own-lock，让 tc_ab.sh 自己拿锁）"
  "$LOCK" release morning
  tools/overnight/tc_ab.sh bench/prompts/p45k.txt 48 2>&1 | tee "$OUT/3-tc-ab.txt"
  "$LOCK" take morning >/dev/null 2>&1 || true
fi

say "汇总"
grep -E "隐含|Gval/s|逐位一致|SIMD 版 vs|16B 向量存 vs|2ph/fus|^ *[0-9]+ " \
     "$OUT/1-dequant.txt" "$OUT/2-cross.txt" 2>/dev/null | head -30
[ -f "$OUT/4-twophase-parity.txt" ] && grep -E "set gate_up|set down" "$OUT/4-twophase-parity.txt" | head -4
[ -f "$OUT/3-tc-ab.txt" ] && sed -n '/对照/,$p' "$OUT/3-tc-ab.txt" | head -12
echo
echo "完整输出: $OUT/"
echo "结论按 docs/EXPERT_KERNEL_ALU.md 第五节的判据走（判据是昨晚预先写死的）。"
