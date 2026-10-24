#!/usr/bin/env python3
"""docs/brand/gen_openrouter.py - render the Xolotl brand set through OpenRouter image models (no local GPU).

    OPENROUTER_API_KEY=... python3 docs/brand/gen_openrouter.py [--model google/gemini-3-pro-image] [names...]
Writes docs/brand/out/<name>.<model-tag>.<n>.png.  Prompts: the same art direction as XOLOTL_BRAND.md.
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
STYLE = ("Art direction: an artistic minimalism - few elements, strong silhouette and negative space, but rich in "
         "craft: hand-inked line quality, subtle grain, metallic leaf accents, like a premium tattoo flash or a museum "
         "poster, never flat clip-art and never busy. Palette: obsidian black (#0B0A0F) ground, bone white (#E8E1D3) "
         "linework, one ember accent (#E2552B) and one cold violet glow (#7C5CFF). Subject: the dark twin of "
         "Quetzalcoatl - a WINGED FEATHERED SERPENT, sleek and menacing, black-feathered wings, scales like obsidian, "
         "a single violet eye; subtle Aztec step-fret geometry in the feathers. Elegant, dangerous, timeless.")
ASSETS = {
    "logo": ("1:1", "A logo mark: the dark winged feathered serpent coiled into a near-circular emblem, head turned in "
             "profile with one violet eye, wings swept back as two sharp black-feather arcs. Bold and legible at 64 px. "
             "Centred on black, no text."),
    "logo-mono": ("1:1", "The same dark winged serpent emblem as a single-colour mark: bone-white linework only on "
                  "black, no gradients, clean enough for a favicon or a sticker. No text."),
    "wordmark-banner": ("21:9", "A wide README banner: the word XOLOTL spelled exactly X-O-L-O-T-L in an elegant, "
                        "sharp, custom display typeface in bone white, with the dark winged serpent's wings spreading "
                        "from behind the letters and its tail curling under them; one small ember spark. Below the "
                        "word, small clean text: 'DeepSeek-V4.1-Flash on one gaming PC'. Generous black space. No "
                        "other text."),
    "hero": ("16:9", "Key art: the dark winged feathered serpent in flight against a black night sky, wings fully "
             "spread, body forming a long elegant S-curve, a faint evening star behind it, ember sparks trailing "
             "from the wingtips. Lots of negative space on one side for a headline. No text."),
    "social-card": ("16:9", "A social preview card: the dark winged serpent on the right third, wings raised, on the "
                    "left in bone-white type the word XOLOTL and beneath it smaller 'Frontier MoE on a gaming PC'. "
                    "Spell XOLOTL exactly. Minimal, high contrast."),
    "twins": ("16:9", "Two winged feathered serpents facing each other in perfect mirror symmetry: on the left "
              "Quetzalcoatl in gold and teal under the morning star, on the right his dark twin in obsidian, bone and "
              "ember with a violet eye under the evening star; their tails intertwine at the bottom centre. Same "
              "artistic minimal style for both. No text."),
}


def gen(model, name, n):
    aspect, prompt = ASSETS[name]
    body = {"model": model, "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": f"{prompt}\n\n{STYLE}\n\nAspect ratio {aspect}."}],
            "image_config": {"aspect_ratio": aspect}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                                  "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    imgs = r["choices"][0]["message"].get("images") or []
    tag = model.split("/")[-1]
    saved = []
    for i, im in enumerate(imgs):
        url = im["image_url"]["url"]
        data = base64.b64decode(url.split(",", 1)[1])
        p = os.path.join(OUT, f"{name}.{tag}.{n}{'' if i == 0 else f'-{i}'}.png")
        open(p, "wb").write(data)
        saved.append(p)
    if not saved:
        print(f"{name}: no image ({str(r)[:300]})", file=sys.stderr)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(ASSETS))
    ap.add_argument("--model", default="google/gemini-3-pro-image")
    ap.add_argument("-n", type=int, default=1, help="takes per asset")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in a.names:
        for n in range(1, a.n + 1):
            try:
                for p in gen(a.model, name, n):
                    print(p)
            except Exception as e:   # one failure must not stop the set
                print(f"{name} #{n}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
