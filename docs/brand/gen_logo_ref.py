#!/usr/bin/env python3
"""gen_logo_ref.py - minimal logo takes in the style of the chosen banner (wordmark-banner.gemini-3-pro-image.2.png,
passed as the reference image).  python3 gen_logo_ref.py [takes]"""
import base64, json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "out", "wordmark-banner.gemini-3-pro-image.2.png")
PROMPT = ("Using this banner as the exact style reference (the same black-feathered winged serpent, same ink texture, "
          "single glowing violet eye, tiny ember spark, dark textured background), design a MINIMAL SQUARE LOGO MARK: "
          "only the winged serpent, centred, wings raised in a clean symmetric V above a simple S-curve body. Reduce it "
          "to a bold, simple silhouette with very few details - it must read instantly at 32-64 px, like an app icon. "
          "No text, no letters, no frame, no circle, nothing else in the image. Aspect ratio 1:1.")
img = "data:image/png;base64," + base64.b64encode(open(REF, "rb").read()).decode()
for n in range(1, int(sys.argv[1] if len(sys.argv) > 1 else 3) + 1):
    body = {"model": "google/gemini-3-pro-image", "modalities": ["image", "text"], "image_config": {"aspect_ratio": "1:1"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": img}}]}]}
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
              {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
        ims = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"].get("images") or []
        for im in ims[:1]:
            p = os.path.join(HERE, "out", f"logo-min.{n}.png")
            open(p, "wb").write(base64.b64decode(im["image_url"]["url"].split(",", 1)[1])); print(p)
    except Exception as e:
        print(n, e, file=sys.stderr)
