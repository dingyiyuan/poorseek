# decode tok/s measured DURING the real e2e tasks (smoke + HumanEval + tools + agentic), from the server's per-request
# "[strata] done: N tokens in T s (R tok/s)" lines: token-weighted (sum tokens / sum decode seconds), plus the median.
import re, sys, statistics as st
for f in sys.argv[1:]:
    toks, secs, rates = 0, 0.0, []
    for m in re.finditer(r"done: (\d+) tokens in [\d.]+ s \(([\d.]+) tok/s\)", open(f, errors="ignore").read()):
        n, r = int(m.group(1)), float(m.group(2))
        if n >= 8 and r > 0: toks += n; secs += n / r; rates.append(r)
    print(f"{f}: {len(rates)} requests, {toks} tokens, decode {toks/secs:.1f} tok/s token-weighted, median {st.median(rates):.1f}")
