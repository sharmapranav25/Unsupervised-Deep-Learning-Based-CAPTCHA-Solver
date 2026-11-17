# Coverage survey: CAPTCHA visual features vs captcha-effects (2026-10-03)

**Sources:**
1. the 33 PHP CAPTCHA libraries of the hybrid-v3 feature bank (adapter parameters, and what they draw, measured from
   4,000 intercepted renders: `ranges/library_ranges.json`);
2. the text-CAPTCHA security literature:
   - Bursztein et al., *Text-based CAPTCHA Strengths and Weaknesses*, CCS 2011 — anti-segmentation vs anti-recognition;
   - Bursztein et al., *The End is Nigh: Generic Solving of Text-based CAPTCHAs*, WOOT 2014;
   - Gao et al., *A Simple Generic Attack on Text Captchas*, NDSS 2016, Table I — character-isolated / hollow / "crowding
     characters together" (CCT) schemes; rotation, distortion, varied fonts and sizes, varied lengths, overlap,
     background clutter, noise arcs;
   - Ye et al., *Yet Another Text Captcha Solver*, CCS 2018;
   - Moradi Vastegani et al. (?), *CAPTCHA Types and Breaking Techniques*, ACM Computing Surveys (arXiv 2307.10239) —
     font tricks, case, deformation, twists, scaling; clutter shapes (circles, squares, rectangles), lines (vertical,
     diagonal, horizontal), colours, dots; overlap; fragmentation; multiple background/foreground colours; 3D text;
     hollow;
   - two-layer schemes (Microsoft, 2015) and "pasting conspicuous patches" from the CNN-attack literature.
   - Author details marked (?) are to be checked before citing.

Two categories, following Bursztein et al. 2011:
- **anti-segmentation:** makes it hard to split the image into characters;
- **anti-recognition:** makes single characters hard to read.

| Feature | Category | Where it comes from | captcha-effects |
|---|---|---|---|
| Lines / straight interference | anti-seg | 16 libraries; all surveys | `lines` |
| Wide bands across the text | anti-seg | City-style schemes | `bands` |
| Noise arcs / curves | anti-seg | Gao 2016 (Sina); 6 libraries | `curves`, `strike` |
| Scribbles / squiggles | anti-seg | freeCap, SVG libs | `scribbles`, `scribbles2` |
| Dots / salt-and-pepper | anti-rec | 10 libraries | `dots` |
| Blobs / circles / shapes clutter | anti-seg | survey (circles, squares, rectangles) | `blobs`, `circles`, `blocks`, `occluders` |
| Background clutter: blocks, grid, texture, gradient, patterns | anti-seg | Gao 2016; libraries | `blocks`, `grid`, `texture`, `gradient`, `pattern` |
| Decoy characters | anti-seg | 5 libraries | `decoys` |
| Pasted patches over glyphs | anti-seg | CNN-attack literature | `occluders` |
| Interference in the text's own colour | anti-seg (vs colour filtering) | survey (colour-filling segmentation attacks) | palette `match_text` |
| Fragmented / broken characters | anti-rec | survey | `cuts`, `thin` |
| Hollow characters | anti-seg + anti-rec | Gao 2016 (Yahoo, QQ); 4 libraries | `hollow` |
| Outline, shadow, 3D | anti-rec | survey (3D text); libraries | `outline`, `shadow`, `extrude` |
| Bold / thin strokes | anti-rec | font tricks | `bold`, `thin` |
| Textured / gradient-filled glyphs | anti-rec | libraries' themes | `fill_texture` |
| Translucent text | anti-rec | 3 libraries | `translucent` |
| Global warp / waving | anti-rec | all surveys | `wave` |
| Local distortion / twists | anti-rec | survey ("twists") | `morph`, `swirl` |
| Perspective / random scaling | anti-rec | survey | `perspective` (scaling: text side) |
| Blur | anti-rec | 4 libraries | `blur` |
| Pixel noise, JPEG, pixelation, low resolution, few colours | anti-rec | multi-captcha (JPEG), P-Captcha (`pixlized`), resampling libraries | `grain`, `jpeg`, `pixelate`, `resample`, `quantize` |
| Inverted regions | anti-seg | negative-band schemes | `invert` |
| Caption / watermark | — | Fuel, freeCap | `caption` |
| Border / frame | — | 2 libraries | `border` |

## Not in captcha-effects: belongs to the text renderer (the CAPTCHA libraries / text stage)
- Character layout: isolated vs **crowded/connected (CCT)**, overlap, spacing, **two-layer (stacked rows)**, varied
  length.
- Character geometry: rotation, vertical jitter, size variation, text on a curve, case, character set.
- Fonts: varied fonts, font mixing.

The libraries cover all of these except **two-layer text**: no training library draws two rows. → noted for the text side.

## Out of scope (not single-image text CAPTCHAs)
Animated (multi-frame GIF), interactive 3D (DotCHA), audio, image-selection and behavioural CAPTCHAs.

## Not included on purpose
Photo backgrounds. Natural-image crops need a licensed image source; procedural `texture` / `pattern` / `gradient`
stand in. Adding a licensed photo set is possible later.
