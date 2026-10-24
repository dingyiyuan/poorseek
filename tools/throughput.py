#!/usr/bin/env python3
"""throughput.py - one model's headline speeds on a served engine, the same probe for every model.

  throughput.py <url> [--prompt 8000] [--decode 512] [--turn 200] [--json out.json]

1. prefill: a fresh ~--prompt-token user message (repo source code, distinct from anything cached), 1 token out ->
   the server's prompt tok/s on a long prompt (the HumanEval prompts are too short to show it).
2. decode: the same conversation + "continue" asking for --decode tokens, thinking off, ignore_eos-like prompt
   (write code) -> the server's decode tok/s.
3. agent turn: + a ~--turn-token user message on top of that context, 1 token out -> the seconds a small agent step
   costs with prefix reuse.
All numbers are the server's own timings (prompt_n / prompt_ms, predicted_n / predicted_ms).
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turnbench import corpus  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def post(url, msgs, max_tokens):
    body = {"model": "x", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none",
            "stream": False}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(req, timeout=3600))
    j["_wall"] = time.time() - t0
    return j


def main():
    url = sys.argv[1]
    n_prompt, n_dec, n_turn = int(arg("--prompt", 8000)), int(arg("--decode", 512)), int(arg("--turn", 200))
    text = corpus()
    # ~3.3 chars/token for code; skip the first 40K chars so the text is not a prefix anything else has cached
    cur = 40000 + int(time.time()) % 20000
    take = lambda n: text[cur:cur + int(n * 3.3)]
    msgs = [{"role": "user", "content": "Read this code carefully:\n" + take(n_prompt)}]
    cur += int(n_prompt * 3.3)
    a = post(url, msgs, 1)
    ta = a.get("timings", {})
    msgs.append({"role": "assistant", "content": (a["choices"][0]["message"].get("content") or "ok").strip() or "ok"})
    msgs.append({"role": "user", "content": f"Now write a long, complete Python module (at least {n_dec} tokens) that "
                                            "re-implements the main ideas of that code with docstrings. Only code."})
    b = post(url, msgs, n_dec)
    tb = b.get("timings", {})
    msgs.append({"role": "assistant", "content": b["choices"][0]["message"].get("content") or ""})
    msgs.append({"role": "user", "content": "More code:\n" + take(n_turn)})
    c = post(url, msgs, 1)
    tc = c.get("timings", {})
    rate = lambda n, ms: round(1000 * n / ms, 1) if ms else None
    row = {"prefill_tokens": ta.get("prompt_n"), "prefill_tok_s": rate(ta.get("prompt_n", 0), ta.get("prompt_ms", 0)),
           "decode_tokens": tb.get("predicted_n"), "decode_tok_s": rate(tb.get("predicted_n", 0), tb.get("predicted_ms", 0)),
           "turn_new_tokens": tc.get("prompt_n"), "turn_reused": tc.get("cache_n"),
           "turn_prefill_s": round(tc.get("prompt_ms", 0) / 1000, 2), "turn_wall_s": round(c["_wall"], 2)}
    print(f"prefill {row['prefill_tokens']} tok at {row['prefill_tok_s']} tok/s | decode {row['decode_tokens']} tok at "
          f"{row['decode_tok_s']} tok/s | +{row['turn_new_tokens']}-token turn on {row['turn_reused']} reused: "
          f"{row['turn_prefill_s']} s", flush=True)
    if "--json" in sys.argv:
        json.dump(row, open(arg("--json", "throughput.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
