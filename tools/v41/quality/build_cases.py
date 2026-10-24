#!/usr/bin/env python3
"""Build the DwarfStar V4.1 official-fixture cases for our engine's quality scorer.

For each case of a manifest (default: deepseek-v4.1-flash-20260919-router/manifest.tsv): the prompt rendered with
DeepSeek's own encoder (dsv41_encoding, thinking OFF, no system message - as collected), tokenized with DeepSeek's
published tokenizer.json, followed by the API's sampled continuation tokens (teacher forcing).  The continuation is
mapped token by token from the API's logprob bytes; the API's top-20 alternatives likewise.

Writes <out>/cases.jsonl: {id, group, ids (prompt + continuation), n_prompt, ref (continuation ids),
api_top1, api_top20 [[id, logprob]...] per position, aligned (False: a token's bytes did not map - the
case is scored for NLL only, like DwarfStar's case_047 rule)}.
"""
import argparse, json, os, sys
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # tools/v41: dsv41_encoding.py
import dsv41_encoding as E  # noqa: E402

FIX = os.path.expanduser('~/AI/ds4-ref/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router')
TOK = '/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json'


def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.join(FIX, 'manifest.tsv'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tk = Tokenizer.from_file(TOK)
    bd = byte_decoder()
    by_bytes = {}
    for tok, i in sorted(tk.get_vocab().items(), key=lambda kv: kv[1]):
        try:
            b = bytes(bd[c] for c in tok)
        except KeyError:   # added / special tokens are not byte-level strings
            continue
        by_bytes.setdefault(b, i)
    root = os.path.dirname(os.path.dirname(os.path.dirname(FIX)))   # the ds4-ref checkout (manifest paths are relative to it)
    os.makedirs(a.out, exist_ok=True)
    n_ok = n_bad = 0
    with open(a.out + '/cases.jsonl', 'w') as out:
        for line in open(a.manifest):
            if line.startswith('#') or not line.strip():
                continue
            cid, pf, cf, rf = line.rstrip('\n').split('\t')[:4]
            prompt = open(os.path.join(root, pf)).read()
            resp = json.load(open(os.path.join(root, rf)))
            text = E.encode_messages([{'role': 'user', 'content': prompt}], thinking_mode='chat')
            pids = tk.encode(text, add_special_tokens=False).ids
            lp = resp['choices'][0]['logprobs']['content']
            ref, top1, top20, aligned = [], [], [], True
            for p in lp:
                i = by_bytes.get(bytes(p['bytes']))
                if i is None:
                    aligned = False; i = -1
                ref.append(i)
                alts = [[by_bytes.get(bytes(t['bytes']), -1), t['logprob']] for t in p['top_logprobs']]
                top20.append(alts)
                top1.append(alts[0][0] if alts else -1)
            # cross-check: our tokenizer on the continuation text gives the same ids (DwarfStar's alignment rule)
            cont_ids = tk.encode(resp['choices'][0]['message']['content'], add_special_tokens=False).ids
            if cont_ids != ref:
                aligned = False
            n_ok += aligned; n_bad += not aligned
            out.write(json.dumps({'id': cid, 'ids': pids + [r for r in ref if r >= 0], 'n_prompt': len(pids), 'ref': ref,
                                  'api_top1': top1, 'api_top20': top20, 'aligned': aligned}) + '\n')
    print(f'{n_ok} aligned, {n_bad} not (NLL only) -> {a.out}/cases.jsonl')


if __name__ == '__main__':
    main()
