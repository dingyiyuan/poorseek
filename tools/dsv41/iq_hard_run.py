#!/usr/bin/env python3
"""在本地引擎上跑 iq_hard.json，答案与 --gold 文件比对（gold 由全精度参照填写）。"""
import argparse, json, re, time, urllib.request
INSTR=("Solve the problem carefully. Then end your reply with a final line exactly of the form "
       "'ANSWER: <value>' where <value> is only the requested number/name/fraction.")
def extract(t):
    if not t: return ""
    m=re.findall(r'ANSWER:\s*([^\n]{1,40})',t,re.I)
    return (m[-1] if m else t.split()[-1] if t.split() else "").strip().lower().strip('.').replace(' ','')
a=argparse.ArgumentParser(); a.add_argument('--api',default='http://127.0.0.1:8160')
a.add_argument('--model',default='deepseek-v4.1-flash-abliterated-iq2xxs')
a.add_argument('--out',default='/tmp/iq_hard_local.json'); A=a.parse_args()
qs=json.load(open('/data/strata-new/tools/dsv41/iq_hard.json'))
rows=[]
for it in qs:
    body={"model":A.model,"messages":[{"role":"user","content":INSTR+"\n\n"+it["q"]}],"max_tokens":1200,"temperature":0}
    rq=urllib.request.Request(A.api.rstrip('/')+"/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        d=json.loads(urllib.request.urlopen(rq,timeout=3600).read())
        m=d["choices"][0]["message"]; txt=m.get("content") or m.get("reasoning_content") or ""
    except Exception as e:
        txt=f"<err {e}>"
    got=extract(txt); rows.append({"id":it["id"],"got":got,"wall":round(time.time()-t0,1),"tail":txt[-200:]})
    print(f"[local] q{it['id']:2d} got={got!r} {rows[-1]['wall']}s", flush=True)
json.dump(rows,open(A.out,'w'),ensure_ascii=False,indent=1); print("→",A.out)
