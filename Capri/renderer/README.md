# captcha-effects

Parameterised CAPTCHA **background, interference, glyph and deformation effects** and a **glyph-by-glyph text renderer
from real font files**, both with **exact per-pixel labels**. Every effect has a **calibrated 0–10 intensity**, every
glyph knows its font, category and weight class, and text-level properties (rotation, jitter, spacing, overlap, size
variation, curve, slant, width, two-layer...) are measured. One call (`render_captcha`) gives a complete labelled
CAPTCHA. The effects also work with any other text renderer (e.g. real CAPTCHA libraries).

Built for training and controlling a CAPTCHA *feature describer* (which effects an image has, how strong, in which
style), and for rendering datasets back from such descriptions.

- **39 effects → 37 feature heads** sampled by default, each with a style schema (layout, palette, stroke, opacity...).
  `caption` (fixed text such as "enter the code") is still available but is not sampled and has no head: extra
  unlabelled text would confuse the text reader.
- **Labels:** every op gets an id; `opaque` / `translucent` id maps record which op owns each pixel. Warps move the
  labels with the pixels.
- **Intensity:** measured from what is *visible* (pixels, contrast, displacement...), never copied from the request.
  The monotone 0–10 scale is calibrated per effect (`calibration.json`, 39/39 pass).
- **Coverage:** checked against the 33 PHP CAPTCHA libraries and the text-CAPTCHA security literature (`SURVEY.md`).

Version 0.5.0. Dependencies: numpy, Pillow (fontTools for `tools/build_font_meta.py`).

## Quick start

```python
import numpy as np
import captcha_effects as ce

rng = np.random.default_rng(0)
spec = {"background": [255, 255, 255], "effects": [
    {"name": "blocks", "level": 5, "style": {"palette": "pastel", "layout": "random"}},
    {"name": "grid", "level": 4},
    {"name": "curves", "level": 6},
    {"name": "shadow", "level": 3},
    {"name": "wave", "level": 2}]}
cv = ce.new_canvas(200, 60, spec, rng)
log = ce.draw_under(cv, spec, rng)                    # backgrounds + interference
box = ce.draw_text(cv, "qNyTy", rng, "font.ttf")      # or: a CAPTCHA library draws the text here
log += ce.apply_glyph(cv, spec, rng)                  # effects on the glyph pixels
log += ce.draw_over(cv, spec, rng, box)               # strike / noise / occluders / cuts over the text
log += ce.apply_post(cv, spec, rng)                   # warps and degradations
img, opaque, translucent = cv.image(), cv.opaque, cv.translucent
# log: [{name, head, level_requested, params, style}, ...]; cv.records: one record per op (id, effect, head, colour...)
```

**Labelling a rendered image:**
- intensity: `ce.visible_amount(name, cv.records, cv.opaque, cv.translucent, cv.background, params.get("measured"))`
  → `calibrate.level_of(table, name, value)`;
- style: `ce.style_labels(name, params)`.

**Random balanced specs:** `ce.sample_spec(rng)`. Each effect is on with probability 0.15 (default) at a uniform level
1–10, and mutually exclusive effects are kept apart.

## Text renderer (`captcha_effects.text`)

- **Fonts:** `fonts/font_meta.json`: **2,186 entries, 1,872 families.**
  - Contents:
    - the 175-family hybrid-v3 bank (`tools/build_font_meta.py`; its train/heldout split is unchanged);
    - **1,697 Google Fonts families** (`tools/build_google_fonts.py`), from github.com/google/fonts at commit `9710da1e`
      (OFL / Apache / UFL, licence file + sha256 per family; checkout in `../font_data/google-fonts`, not in this repo).
  - Kept only with the latin subset and all 62 alphanumerics. Skipped: symbol families and non-Latin-only families.
    Excluded after a render test of every font: 3 handwriting-guideline fonts.
  - **Split:** 1,873 train / 310 held out. New families are ranked by sha256 inside their category, 15% held out.
  - **Categories** (Google Fonts metadata): Sans Serif 909, Display 486, Serif 464, Handwriting 253, Monospace 72;
    plus stroke class (sans / serif / slab).
  - **703 variable fonts** (weight / width axes driven) and **316 true italics.**
  - Measured per font: stroke_rel, x_height, width_rel. Caps-only fonts detected.
- **Each glyph** is rasterised alone (4× supersampled), x-scaled, slanted and rotated, then placed as its own labelled
  op (head `text`).
  - Per-glyph labels: char, group, font key / family / category / stroke class, size, angle, weight axis, synthetic
    stroke, measured stroke width (→ weight class thin / regular / bold / heavy), colour, box (after all warps).
  - **Font heads per glyph** (0–10, ranges = 2nd–98th percentile over the font table): weight (measured stroke
    width), width (font advance × applied x-scale), x_height, slant_level; italic (true italic) yes/no.
  - Per image: dominant category and stroke class, and median weight / width / x-height / slant.
- **Layouts:** line, baseline curve, two-layer (interlocking rows). Also: spacing from wide to crowded/overlapping,
  vertical jitter, size variation, font mixing, multicolour, charsets (digits / upper / lower / mixed …), pronounceable
  words.
- **Text heads** (`text_labels`): 0–10 levels **measured from the placed glyphs** (applied offsets, rotations, sizes —
  never the request) for:
  - per_char_rotation, vertical_jitter, size_variation, text_on_curve, slant, overlap_tight_spacing, wide_spacing,
    condensed, expanded;
  - presence for font_mixture, multicolor_text, two_layer;
  - content heads (digits / upper / lower / mixed, short ≤ 4, long ≥ 7).
- **Legibility** (`render_captcha`): each glyph keeps ≥ 60% of its pixels after over-text and post effects, and the
  median glyph contrast is ≥ 22 (luminance). `out["text"]["usable"]` is False otherwise, so the generator redraws.
  - Text colours keep ≥ 60 luminance from the background.
  - About 72% of random CAPTCHAs pass; ~0.09 s per image.

Sheets: `sheets/08_text_renderer.png` (categories, layouts, transforms) and `sheets/07_complete_captchas.png` (complete
CAPTCHAs with their labels).

## Effects

| Pass | Effects (feature head) |
|---|---|
| under the text | `gradient` (bg_gradient), `texture` (bg_image_texture), `blocks` (bg_blocks), `grid` (bg_grid), `circles` (bg_circles), `decoys` (bg_decoy_chars), `border` (border_frame), `pattern` (bg_pattern), `caption` (fixed_caption), `lines` (straight_lines), `bands` (bands), `curves` (curves), `scribbles` / `scribbles2` (scribbles), `dots` (dot_noise), `blobs` (blob_noise) |
| on the glyphs | `shadow` (glyph_shadow), `extrude` (extrude_3d), `outline` (glyph_outline), `hollow` (hollow_outline_glyphs), `bold` (stroke_bold), `thin` (stroke_thin), `fill_texture` (glyph_fill_texture), `translucent` (translucent_text) |
| over the text | `strike` (strike_through), `noise_over` (noise_over_text), `occluders` (occlusion), `cuts` (glyph_cuts) |
| on everything | `wave` (wave_warp), `morph` (morph_distortion), `perspective` (perspective), `swirl` (local_warp), `blur` (blur), `invert` (inverted_regions), `grain` (pixel_noise), `jpeg` (compression), `pixelate` (pixelation), `resample` (low_resolution), `quantize` (colour_quantization) |

**Styles:** 10 palettes (random, pastel, muted, dark, grey, contrast, near_background, saturated, shades of the
background hue, match_text = the text's own colour); clean / rough / dashed strokes; anti-aliased or crisp edges;
solid / semi / faint opacity; block layouts random / tiled / strips / overlapping / dense; speckle; and more per effect
(`STYLE_SCHEMA`).

**Sheets** (every effect at levels 0, 2, 5, 8, 10 in four random styles): `sheets/01_backgrounds.png` …
`sheets/06_degradations.png`. Combined demo: `sheets/00_capabilities_demo.png`.

## Calibration

```
python -m captcha_effects.calibrate --out calibration.json --seeds 24 --sheet sheets --font <font.ttf>
```

1. Each effect is drawn at levels 0–10 over 24 seeds and 5 canvas sizes, and the visible amount is measured
   (`captcha_effects.measure`).
2. The per-level medians get an isotonic (monotone) fit. Bins sit halfway between neighbouring fitted medians.
3. An effect passes when:
   - level 0 draws nothing;
   - the requested level and the visible amount have Spearman rank correlation ≥ 0.7;
   - level 10 is at least 2× level 1.

## Tests

```
python -m pytest tests -q
```

- every effect runs and is deterministic;
- level 0 is a no-op;
- labels sit on pixels of the op's (opacity-blended) colour;
- calibration is complete and monotone;
- the measure agrees with the calibration;
- style labels follow the schema;
- 39 effects; 37 heads in `render_captcha` output (no `fixed_caption`).

## Tools

- `tools/measure_libraries.py`: what the original PHP libraries draw (ranges for each effect), from intercepted
  renders.
- `tools/demo_sheet.py`: the capabilities demo.

## Notes
- **Text is not part of this package.** Character layout, fonts, rotation, overlap and two-layer text belong to the text
  renderer. `apply_glyph` works on whatever glyph pixels the caller marks (records with head "text", or ids passed in).
- **No code from the CAPTCHA libraries is included.** Ranges are measured from their output only.
- **Licence:** to be chosen before publishing.
