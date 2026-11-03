#!/bin/bash
# 两段式 A/B：同一个二进制，只差 DS4_EXP_2PHASE=1
#
#   基线     STRATA_DS4_NO_TC=1     -> F32 标量专家内核（等于生产现在的行为）
#   张量核   STRATA_DS4_TC=1        -> dpct 翻译过来的 set_tc_kernel（joint_matrix / XMX）
#   张量核+  STRATA_DS4_TC=1 STRATA_DS4_TC_MINROWS=16  -> 预填充走张量核、解码留 F32
#
# 为什么需要这个脚本：服务器的引擎路径写在 strata-server.json 的 exe 里，环境变量改不了它；
# 所以这里临时改该配置（改前备份、退出时无条件还原），跑完两次后打印对照表。
#
# 用法: tc_ab.sh <长文本提示文件> [max_tokens]
#   （提示要够长才有意义：≥15k token 才看得出预填充的差别）
set -uo pipefail
PROMPT_FILE=${1:?用法: tc_ab.sh <提示文本文件> [max_tokens]}
MAXTOK=${2:-48}
TC_BIN=/data/strata-new/build-sycl-tc2/strata-dsv41
CFG=/data/models/dsv41-iq2xxs-pack-real/strata-server.json
UNIT=dsv41.service
LOG=/data/strata-new/serve-engine.log
LOCK=/data/strata-new/tools/overnight/gpu_lock.sh
GATE=/data/strata-new/tools/overnight/gate.sh
API=http://127.0.0.1:8160
BAK=$(mktemp /tmp/tc-ab-cfg.XXXXXX.json)
OUT=$(mktemp -d /tmp/tc-ab.XXXXXX)
export GPU_LOCK_OWNER=tp-ab

[ -x "$TC_BIN" ] || { echo "缺 TC 二进制: $TC_BIN（先 ninja -C build-sycl-tc2 strata-dsv41）"; exit 2; }
systemctl --user is-active --quiet "$UNIT" && { echo "服务在跑：先 ./START.sh stop"; exit 2; }

cp -p "$CFG" "$BAK"
restore () {
  systemctl --user stop "$UNIT" 2>/dev/null
  cp -p "$BAK" "$CFG" && echo "[还原] $CFG"
  "$LOCK" release tp-ab 2>/dev/null
}
trap restore EXIT
"$LOCK" take tp-ab || { echo "GPU 被占用"; exit 1; }

set_cfg () {   # $1 = exe, $2 = 额外环境变量（JSON 片段，可为空）
  /usr/bin/python3 - "$CFG" "$1" "$2" <<'PY'
import json,sys
cfg=sys.argv[1]; exe=sys.argv[2]; extra=sys.argv[3]
c=json.load(open(cfg)); c['exe']=exe
c['env'].pop('STRATA_DS4_TC',None); c['env'].pop('STRATA_DS4_NO_TC',None); c['env'].pop('STRATA_DS4_TC_MINROWS',None)
if extra: c['env'].update(json.loads(extra))
json.dump(c,open(cfg,'w'),indent=2,ensure_ascii=False)
PY
}
now_ms () { date +%s%3N; }
wait_health () {
  local t0; t0=$(now_ms)
  for _ in $(seq 1 480); do
    curl -sf --max-time 2 "$API/health" >/dev/null 2>&1 && { echo "  [t] 就绪 $(( $(now_ms) - t0 )) ms"; return 0; }
    sleep 0.25
  done
  echo "  [t] 120 s 未就绪"; return 1
}

run_one () {   # $1 = tag, $2 = 额外 env JSON
  local tag=$1 extra=${2-}
  echo "=== $tag ==="
  set_cfg "$TC_BIN" "$extra"
  systemctl --user stop "$UNIT" 2>/dev/null; sleep 3
  rm -rf /data/models/dsv41-iq2xxs-pack-real/slots/* 2>/dev/null
  : > /tmp/tc-ab-mark; LINES0=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  systemctl --user start "$UNIT" || { echo "起不来"; return 1; }
  wait_health || { journalctl --user -u "$UNIT" -n 20 --no-pager; return 1; }
  DS4_PROF=1 /usr/bin/python3 - "$PROMPT_FILE" "$MAXTOK" "$tag" "$OUT" <<'PY'
import json,sys,time,urllib.request
pf,mx,tag,out=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
txt=open(pf,encoding='utf-8',errors='replace').read()
# 每次加一个唯一的尾巴：token 序列不同 -> 服务端的 KV/slot 缓存绝不可能命中。
# （只清 slots/ 不够：slot 在停止时异步保存，清完又被上一次的存档落回来。）
txt += "\n\n[run nonce %d]" % int(time.time()*1000 % 100000000)
body={"model":"deepseek-v4.1-flash-abliterated-iq2xxs","messages":[{"role":"user","content":txt}],
      "max_tokens":mx,"temperature":0}
key=open('/home/dyy/.config/dsv41/api-key').read().strip()
rq=urllib.request.Request("http://127.0.0.1:8160/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
t0=time.time()
try:
    d=json.loads(urllib.request.urlopen(rq,timeout=7200).read())
except Exception as e:
    print(f"[{tag}] 请求失败: {e}"); sys.exit(1)
wall=time.time()-t0
m=d["choices"][0]["message"]; t=(m.get("content") or m.get("reasoning_content") or "")
u=d.get("usage") or {}
pt=u.get("prompt_tokens") or 0; ct=u.get("completion_tokens") or 0
res={"tag":tag,"wall_s":round(wall,2),"prompt_tokens":pt,"completion_tokens":ct,
     "out_chars":len(t),"head":t[:120]}
json.dump(res,open(f"{out}/{tag}.json","w"),ensure_ascii=False,indent=1)
print(f"[{tag}] wall {wall:.1f}s | prompt {pt} tok | out {ct} tok | {t[:90]!r}")
if ct: print(f"[{tag}] 解码 {ct/(wall-0):.2f} tok/s（含预填充，仅作粗看）")
PY
  systemctl --user stop "$UNIT" 2>/dev/null; sleep 1
  tail -n +$((LINES0+1)) "$LOG" > "$OUT/$tag.engine.log" 2>/dev/null
  grep -E "prompt [0-9]+ tokens|tok/s|experts|TFLOPS|gate/up|down" "$OUT/$tag.engine.log" | tail -12
}

run_one base '{"STRATA_DS4_NO_TC":"1"}'
run_one tp   '{"STRATA_DS4_NO_TC":"1","DS4_EXP_2PHASE":"1"}'

echo
echo "================ 对照 ================"
/usr/bin/python3 - "$OUT" <<'PY'
import json,glob,sys,os
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"*.json"))):
    d=json.load(open(f)); rows.append(d)
print(f"{'tag':8s} {'wall s':>8s} {'prompt tok':>10s} {'prefill tok/s':>14s} {'out tok':>8s}")
base=None
for d in rows:
    pt=d['prompt_tokens']; w=d['wall_s']
    tf=pt/w if w>0 else 0
    if base is None: base=tf
    print(f"{d['tag']:8s} {w:8.1f} {pt:10d} {tf:14.1f} {d['completion_tokens']:8d}"
          + (f"   {tf/base:.2f}x" if base else ""))
print("\n判据: tp（两段式）的 wall 比 base 小就是赚；输出必须正常（两段的数值对拍已过 3e-4）")
PY
echo "原始引擎日志: $OUT/*.engine.log"
