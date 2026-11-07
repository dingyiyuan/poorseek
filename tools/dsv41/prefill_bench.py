import subprocess,os,sys,time,re
env=dict(os.environ)
base_env={"ONEAPI_DEVICE_SELECTOR":"level_zero:0","DS4_RING_DEP":"host","SYCL_PROGRAM_COMPILE_OPTIONS":"-O3",
 "DS4_AHEAD":"0","DS4_ATTN_SLAB":"2048","DS4_YOCO_KEEP":"160","DS4_EXP_2PHASE":"1",
 "DS4_EXP_2PHASE_MINROWS":"96","DS4_IX_SLAB_MB":"256"}
base=[294,8760,344,11111,16,455,6102,7213,1482,7065,2359,436,1750,2173,983,270]
N=int(sys.argv[1]); arm=sys.argv[2]; out=sys.argv[3]
ids=[base[i%16] for i in range(N)]+[99999]
e=dict(env); e.update(base_env)
if arm=="nosync": e["DS4_PROMPT_SYNC"]="0"
cmd=["/data/strata-new/build-sycl-jit/strata-dsv41","--serve","--pack","/data/models/dsv41-iq2xxs-pack-real",
 "--max-context","393216","--gpus","0","--experts","6","--threads","on","--prefill-chunk","40960",
 "--mmap-experts","auto","--vram-experts","0.75","--dspark","none","--window","16"]
logf=open(out,"w")
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=logf,stderr=logf,env=e,text=True,bufsize=1)
t_load=time.time()
for _ in range(900):
    time.sleep(1)
    try:
        if "READY" in open(out).read(): break
    except Exception: pass
load=time.time()-t0 if False else time.time()-t_load
print(f"[{arm}] loaded in {load:.0f}s",flush=True)
t0=time.time()
p.stdin.write("GEN 1 "+",".join(map(str,ids))+"\n"); p.stdin.flush()
done=False
for _ in range(1200):
    time.sleep(2)
    try: txt=open(out).read()
    except Exception: continue
    if "DONE 1" in txt or "stopped" in txt: done=True; break
wall=time.time()-t0
p.stdin.write("QUIT\n"); p.stdin.flush()
try: p.wait(timeout=120)
except Exception: p.terminate()
logf.close()
txt=open(out).read()
m=re.search(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms \(([\d.]+) tok/s\)",txt)
sync = "every launch is waited on" in txt
if m:
    print(f"[{arm}] tokens={m.group(1)} {int(m.group(2))/1000:.1f}s = {m.group(3)} tok/s | wall {wall:.0f}s | sync={sync}",flush=True)
else:
    print(f"[{arm}] NO RESULT wall={wall:.0f}s sync={sync}",flush=True)
