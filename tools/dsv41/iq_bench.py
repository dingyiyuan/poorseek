#!/usr/bin/env python3
"""阶段感知混合精度的精度基准：同一套题，跑不同 (预填充精度, 解码精度) 组合。

用法:
  iq_bench.py --api http://127.0.0.1:8160 --tag C-iq2 --out /tmp/iq_C.json [--repeat 1]
评分: 每题给一个唯一可提取的答案（数字或字母），贪心解码，严格提取比对。
"""
import argparse, json, re, sys, time, urllib.request

# 24 题：数学(10) / 逻辑(7) / 计数与推理(7)。答案都可自动提取。
QS = [
    # --- 数学（GSM8K 风格）---
    ("math", "A shop sells pencils at 3 for $2. How much do 15 pencils cost, in dollars?", "10"),
    ("math", "John has 48 apples. He gives 1/4 to his sister and 1/3 of the rest to his brother. How many apples does he have left?", "24"),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?", "72"),
    ("math", "If 5 workers build a wall in 12 days, how many days would 15 workers take at the same rate?", "4"),
    ("math", "A rectangle has perimeter 36 and length twice its width. What is its area?", "72"),
    ("math", "What is 15% of 80?", "12"),
    ("math", "A book costs $24 after a 20% discount. What was the original price in dollars?", "30"),
    ("math", "Sam is 3 times as old as his son. In 12 years Sam will be twice as old. How old is Sam now?", "36"),
    ("math", "A jar contains 40 marbles: 15 red, 10 blue, and the rest green. What fraction of the marbles are green? Answer as a percent.", "37.5"),
    ("math", "The sum of three consecutive even integers is 78. What is the largest of them?", "28"),
    # --- 逻辑 ---
    ("logic", "All roses are flowers. Some flowers fade quickly. Which MUST be true? Answer with one letter: (A) All roses fade quickly (B) Some roses fade quickly (C) All roses are flowers (D) No flowers are roses", "C"),
    ("logic", "If A > B and B > C, and D < C, which is largest? Answer with a letter.", "A"),
    ("logic", "Five people sit in a row. Ann is left of Ben. Ben is left of Cara. Dan is between Ben and Cara. Who is in the middle? Answer with a name.", "Dan"),
    ("logic", "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost in dollars?", "0.05"),
    ("logic", "Which does not belong? Answer one: square, triangle, cube, circle", "cube"),
    ("logic", "If it takes 5 machines 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets, in minutes?", "5"),
    ("logic", "Mary's father has five daughters: Nana, Nene, Nini, Nono, and ? Give the missing name.", "Mary"),
    # --- 计数 / 推理 ---
    ("count", "How many times does the digit 7 appear in the numbers from 1 to 100 inclusive?", "20"),
    ("count", "How many rectangles are formed by a 3x3 grid of squares (all axis-aligned, any size)? Give a number.", "36"),
    ("count", "In how many ways can the letters of the word LEVEL be arranged? Give a number.", "30"),
    ("count", "A cube is painted on all faces and cut into 27 equal smaller cubes. How many small cubes have exactly two painted faces?", "12"),
    ("count", "How many trailing zeros are in 100!?", "24"),
    ("count", "What is the next number? 2, 6, 12, 20, 30, ?", "42"),
    ("count", "What is the next number? 1, 1, 2, 3, 5, 8, 13, ?", "21"),
]

INSTRUCTION = ("Answer the question. Work briefly if needed, then end your reply with a final line of the form "
               "'ANSWER: <value>' where <value> is only the requested number, letter, or name.")

def norm(s):
    s = s.strip().lower().replace('$','').replace('%','').replace(',','').rstrip('.')
    s = re.sub(r'\s+',' ',s)
    return s

def extract(text):
    if not text: return ""
    m = re.findall(r'ANSWER:\s*([^\n]{1,40})', text, re.I)
    if m: return norm(m[-1])
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums: return norm(nums[-1])
    return norm(text.split()[-1]) if text.split() else ""

def ask(api, q, model, timeout=3600):
    body = {"model": model, "messages":[{"role":"user","content": INSTRUCTION+"\n\n"+q}],
            "max_tokens": 900, "temperature": 0}
    rq = urllib.request.Request(api.rstrip('/')+"/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(rq, timeout=timeout).read())
    msg = d["choices"][0]["message"]; txt = msg.get("content") or msg.get("reasoning_content") or ""
    return txt, time.time()-t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://127.0.0.1:8160')
    ap.add_argument('--model', default='deepseek-v4.1-flash-abliterated-iq2xxs')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    qs = QS[:a.limit] if a.limit else QS
    rows=[]; ok=0
    for i,(cat,q,gold) in enumerate(qs,1):
        try:
            txt, wall = ask(a.api, q, a.model)
            got = extract(txt)
        except Exception as e:
            txt, wall, got = f"<error {e}>", 0.0, ""
        hit = (got == norm(gold)) or (got.rstrip('0').rstrip('.') == norm(gold).rstrip('0').rstrip('.'))
        ok += hit
        rows.append({"i":i,"cat":cat,"q":q,"gold":gold,"got":got,"hit":hit,"wall":round(wall,1),"text_tail":txt[-300:]})
        print(f"[{a.tag}] {i}/{len(qs)} {cat:5s} {'✓' if hit else '✗'} gold={gold} got={got!r} {wall:.0f}s", flush=True)
    acc = ok/len(qs)
    json.dump({"tag":a.tag,"acc":acc,"ok":ok,"n":len(qs),"rows":rows}, open(a.out,'w'), ensure_ascii=False, indent=1)
    print(f"\n[{a.tag}] accuracy {ok}/{len(qs)} = {acc:.3f}  → {a.out}")

if __name__ == '__main__':
    main()
