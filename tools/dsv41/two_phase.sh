#!/bin/bash
# 两段式预填充：预填充用混合 MXFP4 包（快 + 精度高），解码用 IQ2 包（显存友好）。
# 用法: two_phase.sh <prompt_file> [max_tokens]
# 依赖: dsv41.service (systemd --user)，配置 /data/models/dsv41-iq2xxs-pack-real/strata-server.json
set -uo pipefail
PACK_FAST=/data/models/dsv41-mixed-mxfp4-l0-19
PACK_DEC=/data/models/dsv41-iq2xxs-pack-real
CFG=$PACK_DEC/strata-server.json
LOCK=/data/strata-new/tools/overnight/gpu_lock.sh
PROMPT_FILE=${1:?用法: two_phase.sh <prompt_file> [max_tokens]}
MAXTOK=${2:-64}
API=http://127.0.0.1:8160
WORK=$(mktemp -d)

export GPU_LOCK_OWNER=two-phase
$LOCK take two-phase || { echo "GPU 被占用"; exit 1; }
trap 'systemctl --user stop dsv41.service 2>/dev/null; $LOCK release two-phase' EXIT

set_pack () {   # $1 = pack dir
  python3 - "$CFG" "$1" <<'PY'
import json,sys
cfg,pack=sys.argv[1],sys.argv[2]
c=json.load(open(cfg)); a=c['args']
a[a.index('--pack')+1]=pack
json.dump(c,open(cfg,'w'),indent=2,ensure_ascii=False)
PY
}
now_ms () { date +%s%3N; }

wait_engine_gone () {   # 轮询直到引擎进程消失
  local t0; t0=$(now_ms)
  for _ in $(seq 1 240); do
    ps -eo cmd | grep -qE '(^|/)strata-dsv41( |$)' || { echo "  [t] 引擎退出 $(( $(now_ms) - t0 )) ms"; return 0; }
    sleep 0.25
  done
  echo "  [t] 警告: 引擎 60 s 未退出"; return 1
}

wait_vram_free () {     # 轮询直到显存回落到阈值以下（读不到就退化为 3 s）
  local t0; t0=$(now_ms)
  local f; f=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
  if [ -z "$f" ]; then local ms=${TWO_PHASE_SETTLE_MS:-1000}; sleep "$(awk "BEGIN{print $ms/1000}")"; echo "  [t] 显存不可读, 固定等 ${ms} ms (TWO_PHASE_SETTLE_MS)"; return 0; fi
  for _ in $(seq 1 240); do
    local used; used=$(cat "$f" 2>/dev/null || echo 0)
    if [ "${used:-0}" -lt 2147483648 ]; then echo "  [t] 显存释放 $(( $(now_ms) - t0 )) ms (used $(( used/1048576 )) MiB)"; return 0; fi
    sleep 0.25
  done
  echo "  [t] 警告: 显存 60 s 未释放"; return 1
}

wait_health () {        # 轮询直到 HTTP 就绪
  local t0; t0=$(now_ms)
  for _ in $(seq 1 480); do
    curl -sf --max-time 2 http://127.0.0.1:8160/health >/dev/null 2>&1 && { echo "  [t] 服务就绪 $(( $(now_ms) - t0 )) ms"; return 0; }
    sleep 0.25
  done
  echo "  [t] 警告: 服务 120 s 未就绪"; return 1
}

restart () {            # 轮询真实条件, 不再固定 sleep 60。
  # 实测: 引擎退出 14-56 ms, 服务就绪 0.8-2.2 s, 单次切换 4.5-6 s (曾用固定 60 s 冷却)。
  # TWO_PHASE_SETTLE_MS 是"显存不可读"时的保险等待, 默认 1000; 实测 500 也安全。
  local t0; t0=$(now_ms)
  systemctl --user stop dsv41.service 2>/dev/null
  wait_engine_gone; wait_vram_free
  /data/strata-new/tools/overnight/gate.sh 0 || { echo "gate 未过"; exit 1; }
  systemctl --user start dsv41.service
  wait_health
  systemctl --user is-active dsv41.service >/dev/null || { echo "服务没起来"; exit 1; }
  echo "  [t] 切换总耗时 $(( $(now_ms) - t0 )) ms"
}

ask () {  # $1 = messages json, $2 = tag
  python3 - "$1" "$MAXTOK" "$2" <<'PY'
import json,urllib.request,time,sys
msgs=json.loads(sys.argv[1]); mx=int(sys.argv[2]); tag=sys.argv[3]
body={"model":"deepseek-v4.1-flash-abliterated-iq2xxs","messages":msgs,"max_tokens":mx,"temperature":0}
rq=urllib.request.Request("http://127.0.0.1:8160/v1/chat/completions",
    data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.time(); d=json.loads(urllib.request.urlopen(rq,timeout=7200).read()); wall=time.time()-t0
m=d["choices"][0]["message"]; txt=(m.get("content") or m.get("reasoning_content") or "")
print(f"[{tag}] wall {wall:.1f}s | out {len(txt)} chars | {txt[:80]!r}")
PY
}

echo "=== ① 预填充：混合 MXFP4 包 ==="
set_pack "$PACK_FAST"; restart
PROMPT=$(cat "$PROMPT_FILE")
MSGS1=$(python3 -c 'import json,sys; print(json.dumps([{"role":"user","content":sys.argv[1]}]))' "$PROMPT")
ask "$MSGS1" "phase-A-prefill"
echo "=== ② 换 IQ2 包（KV 靠 slot 承接）==="
set_pack "$PACK_DEC"; restart
echo "=== ③ 同一段对话继续（应命中 slot，跳过预填充）==="
MSGS2=$(python3 -c 'import json,sys; p=json.loads(sys.argv[1]); p.append({"role":"assistant","content":"ok"}); p.append({"role":"user","content":"Continue."}); print(json.dumps(p))' "$MSGS1")
ask "$MSGS2" "phase-B-decode"
echo "=== 日志里应有 'reused' 计数：命中则 KV 成功跨 pack 承接 ==="
grep -oE 'prompt [0-9]+ tokens = [0-9]+ reused \+ [0-9]+ read' /data/strata-new/serve-engine.log | tail -3
