#!/bin/bash
# 把"张量核专家内核"落地到生产（或撤回）。只改 strata-server.json 的两个字段，不覆盖任何二进制。
#
#   tc_promote.sh status            # 现在用的是哪个二进制、TC 开没开
#   tc_promote.sh promote [minrows] # exe 指向 build-sycl-tc2 + env STRATA_DS4_TC=1（minrows 可选，默认 0）
#   tc_promote.sh revert            # 还原到 promote 之前的那份配置
#
# 为什么只改配置：TC 二进制在 build-sycl-tc2/ 里独立存在，build-sycl-jit/strata-dsv41 这份"已知可用"的
# 二进制全程不动，回滚就是把 exe 指回去——没有任何不可逆动作。
set -uo pipefail
CFG=/data/models/dsv41-iq2xxs-pack-real/strata-server.json
BAK=$CFG.bak-pre-tc
TC_BIN=/data/strata-new/build-sycl-tc2/strata-dsv41
UNIT=dsv41.service

show () {
  /usr/bin/python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
e=c.get('env') or {}
print("exe :", c.get('exe'))
print("TC  :", "STRATA_DS4_TC=" + e['STRATA_DS4_TC'] if 'STRATA_DS4_TC' in e else "(没开)")
if 'STRATA_DS4_TC_MINROWS' in e: print("行下限:", e['STRATA_DS4_TC_MINROWS'])
if 'STRATA_DS4_NO_TC' in e: print("强制关闭:", e['STRATA_DS4_NO_TC'])
PY
}

case "${1:-status}" in
  status)
    show
    echo "服务: $(systemctl --user is-active "$UNIT")"
    [ -f "$BAK" ] && echo "回滚备份: $BAK 存在" || echo "回滚备份: 无（还没 promote 过）"
    ;;
  promote)
    MINROWS=${2:-0}
    [ -x "$TC_BIN" ] || { echo "缺 TC 二进制: $TC_BIN"; exit 2; }
    systemctl --user is-active --quiet "$UNIT" && { echo "服务在跑：先 ./START.sh stop"; exit 2; }
    [ -f "$BAK" ] || cp -p "$CFG" "$BAK" || exit 1
    /usr/bin/python3 - "$CFG" "$TC_BIN" "$MINROWS" <<'PY'
import json,sys
cfg,exe,mr=sys.argv[1],sys.argv[2],int(sys.argv[3])
c=json.load(open(cfg)); c['exe']=exe
env=c.setdefault('env',{})
env.pop('STRATA_DS4_NO_TC',None)
env['STRATA_DS4_TC']='1'
if mr>0: env['STRATA_DS4_TC_MINROWS']=str(mr)
else: env.pop('STRATA_DS4_TC_MINROWS',None)
json.dump(c,open(cfg,'w'),indent=2,ensure_ascii=False)
print("已写入:", cfg)
PY
    show
    echo
    echo "验证: /data/models/dsv41-iq2xxs-pack-real/START.sh start"
    echo "回滚: $0 revert"
    ;;
  revert)
    [ -f "$BAK" ] || { echo "没有备份可回滚"; exit 2; }
    systemctl --user is-active --quiet "$UNIT" && { echo "服务在跑：先 ./START.sh stop"; exit 2; }
    cp -p "$BAK" "$CFG" && echo "已还原 $CFG" && show
    ;;
  *) echo "用法: $0 {status|promote [minrows]|revert}"; exit 2;;
esac
