#!/usr/bin/env python3
"""serve smoke: temp 0 / temp 1 code requests, a continuation turn, a ~6K-token prompt. Prints per-request wall + tokens."""
import json, sys, time, urllib.request
U = sys.argv[1]
def chat(msgs, temp, mt=256, top_p=0.95):
    b = {"model": "x", "messages": msgs, "max_tokens": mt, "reasoning_effort": "none", "stream": False, "temperature": temp}
    if temp > 0: b["top_p"] = top_p
    r = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); j = json.load(urllib.request.urlopen(r, timeout=3600)); return j, time.time() - t
q = [{"role": "user", "content": "Write a Python function that parses an ISO-8601 duration like 'P3DT4H5M' into seconds, with tests."}]
q1 = [{"role": "user", "content": "Write a Python class implementing an LRU cache with O(1) get/put, with a short usage example."}]
for name, temp, qq in (("code_t0", 0, q), ("code_t1", 1.0, q1)):
    j, dt = chat(qq, temp); m = j["choices"][0]["message"]["content"] or ""
    if name == "code_t0": m0 = m
    print(f"{name}: {j['usage']['completion_tokens']} tok in {dt:.1f}s | {m[:120]!r}", flush=True)
msgs = q + [{"role": "assistant", "content": m0}, {"role": "user", "content": "Now make it also accept weeks (P2W). Show only the changed function."}]
j, dt = chat(msgs, 0); print(f"continue_t0: {j['usage']['completion_tokens']} tok in {dt:.1f}s | {(j['choices'][0]['message']['content'] or '')[:120]!r}", flush=True)
import os
big = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../v41/engine/ds41_moe.cpp')).read()[:22000]
j, dt = chat([{"role": "user", "content": "Summarise what this C++ file does in 5 bullet points:\n\n" + big}], 0, 160)
print(f"long_prompt: prompt {j['usage']['prompt_tokens']} tok, {j['usage']['completion_tokens']} out in {dt:.1f}s | {(j['choices'][0]['message']['content'] or '')[:160]!r}", flush=True)
