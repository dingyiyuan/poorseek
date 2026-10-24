#!/usr/bin/env python3
"""Export DeepSeek-V4.1-Flash's tokenizer (the HF checkpoint's tokenizer.json) into the directory layout Strata's
server reads: vocab.json, merges.txt, token_type.json (llama.cpp's: 3 = control (added, special), 4 = user-defined
(added, not special: <｜System｜>, <｜User｜>, <｜Assistant｜> ... - HF matches them whole), 1 = normal), and a marker
chat_template.jinja (serve_ds41.py renders with DeepSeek's own encoder, never this file).  The same tables as the
GGUF's tokenizer.ggml.* keys (v41_gguf_meta.py), so the server and the engine agree on every id.

  export_tokenizer.py [HF_DIR] OUT_DIR          (needs the `tokenizers` package: export time only)
  then: test_serve_ds41.py checks Strata's BPE (pre "joyai-llm") == HF tokenizers on a corpus sample
"""
import json
import os
import sys

HF = "/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"


def main():
    hf, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (HF, sys.argv[1])
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(hf, "tokenizer.json"))
    tj = json.load(open(os.path.join(hf, "tokenizer.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(hf, "config.json")))
    n = cfg["text_config"]["vocab_size"]
    if tj["model"]["type"] != "BPE":
        raise SystemExit("unsupported tokenizer")
    # special tokens are in both the BPE vocabulary and added_tokens: the parsed id table resolves the overlap
    tokens = [tk.id_to_token(i) for i in range(n)]
    if any(t is None for t in tokens):
        raise SystemExit("tokenizer has missing ids")
    special = {a["id"] for a in tj["added_tokens"] if a["special"]}
    added = {a["id"] for a in tj["added_tokens"]}
    merges = [" ".join(p) if isinstance(p, list) else p for p in tj["model"]["merges"]]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(tokens)}, f, ensure_ascii=False)
    with open(os.path.join(out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges))
    with open(os.path.join(out, "token_type.json"), "w") as f:
        json.dump([3 if i in special else 4 if i in added else 1 for i in range(n)], f)
    with open(os.path.join(out, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write("{# DeepSeek-V4.1: rendered by serve_ds41.py with DeepSeek's encoder (dsv41_encoding.py) #}\n")
    print(f"{n} tokens ({len(special)} special), {len(merges)} merges -> {out}")


if __name__ == "__main__":
    main()
