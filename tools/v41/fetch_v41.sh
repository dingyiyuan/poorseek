#!/bin/bash
# fetch_v41.sh DEST [REPO] - resumable download of a DeepSeek-V4.1-Flash safetensors checkpoint with sha256 checks.
#   default REPO = dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 (510.3 GB; experts/engram byte-identical to the base)
#   Re-run any time: finished+verified files are skipped (.ok markers), partial files resume (curl -C -).
#   Shards 47/48 are the two Engram tables (101.5 GB each) - keep them on a fast SSD after quantization.
#   Env: JOBS=1 (files at a time; 1 = sequential, steadiest), ORDER=name|size (default name = shard order),
#        FIT=1 (download what fits now: skip the whole-repo space check, stop cleanly before a file that would leave
#        < MARGIN_GB free; re-run after freeing space), MARGIN_GB=10, HF_TOKEN (default ~/.cache/huggingface/token).
set -u
DEST=${1:?usage: fetch_v41.sh DEST [REPO]}; REPO=${2:-dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8}; JOBS=${JOBS:-1}; ORDER=${ORDER:-name}; FIT=${FIT:-0}; MARGIN_GB=${MARGIN_GB:-10}
TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
mkdir -p "$DEST" && cd "$DEST" || exit 1
curl -sSf -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/models/$REPO/tree/main?recursive=1" > .tree.json || { echo "tree fetch failed"; exit 1; }
python3 - "$REPO" > .files.tsv <<'PY'
import json,sys
for f in json.load(open('.tree.json')):
    if f.get('type')!='file': continue
    oid=(f.get('lfs') or {}).get('oid','')
    print(f"{f['path']}\t{f.get('size',0)}\t{oid}")
PY
[ -n "${ONLY:-}" ] && { grep -E -- "$ONLY" .files.tsv > .files.f || true; mv .files.f .files.tsv; }
[ -n "${SKIP:-}" ] && { grep -v -E -- "$SKIP" .files.tsv > .files.f || true; mv .files.f .files.tsv; }
need=$(awk -F'\t' '{s+=$2} END{printf "%.0f", s}' .files.tsv)
have=$(du -sb . | cut -f1); free=$(df -B1 --output=avail . | tail -1)
echo "$(date '+%F %T') $REPO: $(wc -l < .files.tsv) files, $((need/1000000000)) GB total, $((have/1000000000)) GB here, $((free/1000000000)) GB free"
if [ "$FIT" != 1 ] && [ $((need - have)) -gt $((free - MARGIN_GB*1000000000)) ]; then
  echo "NOT ENOUGH SPACE (need $(((need-have)/1000000000)) GB + ${MARGIN_GB} GB margin) - free space or use FIT=1"; exit 2; fi
one() {   # path size sha
  local p=$1 s=$2 h=$3
  [ -f "$p.ok" ] && return 0
  if [ "$FIT" = 1 ]; then   # would this file (rest of it) leave less than the margin free?
    local cur free; cur=$(stat -c %s "$p" 2>/dev/null || echo 0); free=$(df -B1 --output=avail . | tail -1)
    if [ $((s - cur)) -gt $((free - MARGIN_GB*1000000000)) ]; then echo "$(date '+%T') SPACE: stop before $p ($(((s-cur)/1000000000)) GB needed, $((free/1000000000)) GB free)"; touch .space-stop; return 1; fi
  fi
  mkdir -p "$(dirname "$p")"
  for try in 1 2 3 4 5; do
    curl -sS -L --retry 5 --retry-delay 10 -C - -H "Authorization: Bearer $TOKEN" -o "$p" "https://huggingface.co/$REPO/resolve/main/$p" 2>>.curl.err
    [ "$(stat -c %s "$p" 2>/dev/null)" = "$s" ] && break
    [ "$(stat -c %s "$p" 2>/dev/null || echo 0)" -gt "$s" ] && rm -f "$p"   # corrupt overshoot: restart
  done
  [ "$(stat -c %s "$p" 2>/dev/null)" = "$s" ] || { echo "FAIL size $p"; return 1; }
  if [ -n "$h" ]; then
    local got; got=$(dd if="$p" iflag=direct bs=16M status=none | sha256sum | cut -d' ' -f1)   # O_DIRECT: page cache lies on this box
    [ "$got" = "$h" ] || { echo "MISMATCH $p ($got != $h) - deleting"; rm -f "$p"; return 1; }
  fi
  touch "$p.ok"; echo "$(date '+%T') ok $p"
}
export -f one; export REPO TOKEN FIT MARGIN_GB
rm -f .space-stop
# shard order by default (ORDER=size: biggest first); xargs runs JOBS at a time; FIT=1 stops at the first that won't fit
if [ "$ORDER" = size ]; then SORT=(sort -t$'\t' -k2,2nr); else SORT=(sort -t$'\t' -k1,1); fi
"${SORT[@]}" .files.tsv | tr '\t\n' '\0\0' | xargs -0 -n3 -P "$JOBS" bash -c '[ -f .space-stop ] && exit 0; one "$0" "$1" "$2" || [ -f .space-stop ] && exit 0 || true'
bad=$(awk -F'\t' '{print $1}' .files.tsv | while read -r p; do [ -f "$p.ok" ] || echo "$p"; done)
[ -z "$bad" ] && echo "ALL VERIFIED OK" || { echo "MISSING/BAD:"; echo "$bad"; exit 3; }
