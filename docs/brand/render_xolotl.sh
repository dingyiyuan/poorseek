#!/usr/bin/env bash
# render_xolotl.sh - the Xolotl brand images through klein (Flux-2 Klein in ComfyUI).  GPU: run ONLY when the DS4
# engine is stopped (ds41-engine.sh refuses to start while ComfyUI holds :8188, and both want the 4090).
# Each prompt is rendered SEEDS times; pick the best into docs/brand/out/<name>.png.
set -euo pipefail
SEEDS="${SEEDS:-4}"
if pgrep -x ds41_generate >/dev/null; then echo "the DS4 engine is running - stop it first" >&2; exit 1; fi

STYLE="dark mythic techno-Aztec, obsidian black background, ember orange and storm violet lightning, bone white \
highlights, a single white-violet evening star, crisp vector-like edges, high contrast, sleek and professional, \
not cartoonish, no text artifacts"

declare -A P W H
P[logo]="minimalist emblem of Xolotl the Aztec dog-headed god of fire and lightning, skeletal dog skull in profile \
built from geometric obsidian step-fret shapes, a forked lightning bolt for the eye, ember glow, centred, flat icon"
W[logo]=1024; H[logo]=1024
P[wordmark-banner]="wide banner, the word XOLOTL in a heavy geometric display typeface carved from obsidian with \
ember-lit edges, left of it a small dog-skull emblem, faint Aztec step-fret border, violet lightning running along \
the bottom like circuit traces"
W[wordmark-banner]=1536; H[wordmark-banner]=512
P[hero]="Xolotl the dark twin of Quetzalcoatl rising from a glowing underworld staircase, dog-headed skeletal god \
wreathed in fire, carrying glowing cubes of light up from the depths, lightning arcs shaped like circuit traces, \
the evening star above, cinematic wide shot"
W[hero]=1920; H[hero]=1080
P[tiers]="cross-section of a descending Aztec temple staircase into an underworld, three levels: a bright top \
chamber glowing violet (VRAM), a middle chamber of warm ember light lined with stacked memory sticks (RAM), a deep \
dark cavern of obsidian slabs that look like NVMe drives (Mictlan), a dog-headed spirit carrying a glowing cube \
upward between levels, clean infographic composition with empty space for labels"
W[tiers]=1536; H[tiers]=864
P[axolotl]="cute but cool axolotl mascot with feathery ember-orange gills and obsidian black skin, tiny lightning \
bolt markings, glowing violet eyes, sitting on a stepped Aztec stone, simple flat sticker style, dark background"
W[axolotl]=1024; H[axolotl]=1024
P[twins]="two divine twins facing each other: on the left Quetzalcoatl the feathered serpent in gold and teal under \
the morning star, on the right Xolotl the dog-headed god of fire and lightning in obsidian ember and violet under \
the evening star, symmetrical composition, Aztec codex geometry fused with circuit boards, epic"
W[twins]=1920; H[twins]=1080

for name in "${@:-logo wordmark-banner hero tiers axolotl twins}"; do
    for name1 in $name; do
        for s in $(seq 1 "$SEEDS"); do
            klein "${P[$name1]}. Style: $STYLE" -w "${W[$name1]}" -h "${H[$name1]}" -p "Xolotl/${name1}-$s"
        done
    done
done
echo "outputs: /media/mal/Crucible/AI-ART/ComfyUI/output/Xolotl/ - copy the picks to docs/brand/out/<name>.png"
