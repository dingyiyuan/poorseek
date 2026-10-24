# Xolotl - brand kit (DRAFT, not yet applied to the repo / Hub)

**Xolotl** is the proposed public name of this engine (today: Strata-DSV4.1).  Status: draft for Mal's sign-off.
Nothing has been renamed; `render_xolotl.sh` renders the images once the GPU is free (ComfyUI and the engine cannot
run together).

## Why Xolotl

- **Quetzalcoatl's twin.**  In Aztec myth Xolotl is the feathered serpent's dark twin - the god of fire, lightning,
  death and misfortune, dog-headed, the Evening Star to Quetzalcoatl's Morning Star.  Mal's agent is **Quetza**
  (QuetzaCodetl); the engine that runs frontier models on scrap hardware is its shadow.  Same family, opposite light.
- **The one who goes down into the underworld.**  Xolotl guides souls through Mictlan and brings back what was buried
  (he carried the bones of the dead up to make the new humans).  That is literally what the engine does: 552B
  parameters live in the underworld of your NVMe and RAM, and Xolotl drags the ones each token needs up into VRAM.
  The three tiers map onto the myth: VRAM = the living world, the RAM arena = the threshold, the NVMe file tier =
  Mictlan.
- **The axolotl** is named after him (the "water Xolotl").  A free, friendly secondary mascot for social posts,
  emoji and the favicon - the cute face of a dark god.
- Short, unusual, searchable, pronounceable once heard ("SHO-lotl").  No collision in the inference-engine space.

**Name usage:** `Xolotl` (the engine) · repo `Xolotl` or `xolotl-engine` · tagline line "built on Strata" stays as
credit to Niko1221's Strata, which the engine grew out of.  Recommendation: keep "Strata" as the family name
("Xolotl - a Strata engine for DeepSeek-V4.1") - ai-muninn and others already write about Strata, and that
recognition is worth keeping.

## Taglines

- **Xolotl - frontier MoE from the underworld.**
- *552 billion parameters. One gaming PC. The rest stays buried until it's needed.*
- *Quetza's dark twin: the engine that brings the dead weights back to life.*
- *What the router calls, Xolotl carries up.*

## Palette (the mirror of Quetza's)

| Role | Quetza (light twin) | Xolotl (dark twin) |
| --- | --- | --- |
| Ground | navy `#0F172A` | obsidian `#0B0A0F` |
| Primary | gold `#D4A843` | ember `#E2552B` |
| Accent | teal `#2DD4BF` | storm violet `#7C5CFF` (lightning) |
| Highlight | amber `#F59E0B` | bone `#E8E1D3` |
| Glow | - | Venus white-violet `#C9B8FF` (the Evening Star) |

Type: a heavy geometric display face for the wordmark (e.g. "Unbounded" / "Syne"), a mono for numbers
("JetBrains Mono").  Motifs: obsidian glyphs, Aztec step-frets, a dog/skull head in profile, lightning that runs like
PCIe traces, a descending staircase into Mictlan (the tiers), the evening star.

## Image set (prompts in `render_xolotl.sh`)

| File | Size | Use |
| --- | --- | --- |
| `logo` | 1024x1024 | GitHub org/repo avatar, HF avatar |
| `wordmark-banner` | 1536x512 | README header |
| `hero` | 1920x1080 | Reddit post / social card |
| `tiers` | 1536x864 | "how it works" figure: VRAM / RAM / NVMe as the descent into Mictlan |
| `axolotl` | 1024x1024 | friendly mascot, favicon / emoji source |
| `twins` | 1920x1080 | Quetza and Xolotl, morning star and evening star (the family shot) |

## README header draft

```html
<p align="center"><img src="docs/brand/out/wordmark-banner.png" alt="Xolotl" width="100%"></p>
<h1 align="center">Xolotl</h1>
<p align="center"><b>DeepSeek-V4.1-Flash (552B MoE) on one gaming PC.</b><br>
The experts live in the underworld - VRAM, pinned RAM and NVMe - and Xolotl carries up the ones each token needs.<br>
A <a href="https://github.com/Niko1221/Strata">Strata</a> engine · Linux · NVIDIA · MIT</p>
```

## Open decisions for Mal

1. Final name: **Xolotl** vs Mantle (Mantle is clean but generic and collides with AMD's old Mantle API).
2. GitHub: rename `Indras-Mirror/Strata-DSV4.1` -> `Indras-Mirror/Xolotl` IN PLACE (Settings -> Rename), never a new
   repo.  A rename keeps the stars (20 as of 2026-10-15), watchers, forks, issues and history, and redirects every old
   URL (web, git clone/pull/push, API) - this repo already went Strata-DS4 -> Strata-DSV4.1 that way.  The one thing
   that breaks the redirect: creating a new repo with the old name.  Search findability lives in the description,
   topics and README, so keep "DeepSeek-V4.1-Flash", "Strata", "MoE offload", "RTX 4090" there (description draft:
   "Xolotl - DeepSeek-V4.1-Flash (552B MoE) on one gaming PC: a Strata engine. VRAM + RAM + NVMe expert tiers,
   DSpark speculative decoding, 512K context."; topics: deepseek, deepseek-v4, moe, llm-inference, offloading,
   speculative-decoding, cuda, local-llm, strata).
3. Hugging Face: the model repo name can stay (it describes the weights), only its card gets the brand.
4. Keep "a Strata engine" in the subtitle (recommended) or drop it.

## FINAL (2026-10-17, Mal's picks) - the sources are in `/Assets`

| source (`Assets/`) | role | composed into (`docs/brand/assets/`) |
| --- | --- | --- |
| `circle.1.png` | **the logo** - the coiled winged serpent in a ring | `logo-circle-400.png` (README), `avatar-512/256.png` (GitHub / Hugging Face), `favicon.ico` + `favicon-64.png` |
| `wordmark-banner.gemini-3-pro-image.2.png` | **the banner** | `readme-header-1200.png`, `github-social-preview-1280x640.png`, `og-card-1200x630.png` |
| `hero.gemini-3-pro-image.2.png` | hero / post image | `readme-hero-1600.png` |

Composed locally with PIL (no image-model credits). `logo-min.*` (below) is superseded by the circle logo.

## Chosen direction + final assets (2026-10-15, Mal's picks)

Direction: the **dark winged feathered serpent** (Quetza's dark twin) - ink linework, black feathers, one violet eye,
an ember spark, textured black ground.  Mal's favourites: `out/wordmark-banner.gemini-3-pro-image.2.png` (banner) and
`out/hero.gemini-3-pro-image.2.png` (hero).  Minimal logo in the banner's style: `out/logo-min.{1..4}.png`
(`gen_logo_ref.py`, the banner as reference; pick: logo-min.3).  `assets/` is composed locally from those (PIL, no
image-model credits):

| file | where it goes |
| --- | --- |
| `assets/github-social-preview-1280x640.png` | GitHub repo Settings -> Social preview |
| `assets/readme-header-1200.png` | top of README.md (GitHub) and the Hugging Face model card |
| `assets/readme-hero-1600.png` | README "how it works" / Reddit post image |
| `assets/og-card-1200x630.png` | link card for Reddit / X / Discord posts |
| `assets/avatar-512.png`, `avatar-256.png` | GitHub org (Indras-Mirror) + Hugging Face avatar |
| `assets/favicon.ico`, `favicon-64.png` | docs site / artifact pages |

Caveat: the logo is black on dark grey (low contrast) - fine as an avatar on its own square; for a light page use a
bone-white mono variant (`out/logo-mono.*`).  Generating more costs OpenRouter credits (~Gemini 3 Pro Image) - prefer
re-composing `assets/` from the chosen renders.
