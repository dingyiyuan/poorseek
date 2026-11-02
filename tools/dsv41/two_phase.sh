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
restart () {
  systemctl --user stop dsv41.service 2>/dev/null; sleep 60
  /data/strata-new/tools/overnight/gate.sh 0 || { echo "gate 未过"; exit 1; }
  systemctl --user start dsv41.service; sleep 15
  systemctl --user is-active dsv41.service >/dev/null || { echo "服务没起来"; exit 1; }
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
