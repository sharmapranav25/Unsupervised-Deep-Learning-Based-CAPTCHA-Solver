"""Active hybrid-v3 feature taxonomy (33 text renderers).

Each head is a visual property a detector can be asked about. For every head we record
which libraries can produce it and how:
  param  - the adapter samples a setting that switches/strengthens it (counterfactual pairs possible)
  random - the library decides internally (present in some outputs, not controllable)
  always - part of that library's fixed style
Assignments come from reading each renderer's drawing code plus sampled visual review
(runs/bank02_smoke_*). They are capability claims, NOT per-image visible labels: per-image
labels still need instrumented draw logs and visibility checks.

Usage: python -m feature_bank.bank02_taxonomy  -> writes docs/feature_bank/FEATURE_BUCKETS_02.{json,md}
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = {
    "background": {
        "bg_solid": "single flat background colour",
        "bg_gradient": "smooth colour gradient across the canvas",
        "bg_image_texture": "photographic or bitmap texture behind the text",
        "bg_blocks": "rectangular/morphed patches of contrasting colour",
        "bg_grid": "regular grid or lattice lines",
        "bg_circles": "concentric or scattered circle/ellipse fields",
        "bg_dark": "dark background with light text (inverted polarity)",
        "bg_decoy_chars": "small or faint non-answer characters scattered in the background",
        "border_frame": "frame or border around the image",
    },
    "interference": {
        "straight_lines": "straight line segments across the image",
        "curves": "sine/bezier/arc curves crossing the image",
        "strike_through": "a line deliberately through the text body",
        "dot_noise": "isolated pixels or small dots",
        "blob_noise": "larger filled spots/circles",
        "scribbles": "dense random strokes/polylines",
        "noise_over_text": "interference drawn on top of glyphs rather than behind",
    },
    "glyph_appearance": {
        "filled_glyphs": "solid filled TrueType glyphs",
        "hollow_outline_glyphs": "outlined/hollow glyph strokes",
        "bitmap_font": "GD built-in or GDF bitmap font",
        "vector_stroke_glyphs": "fontless glyphs drawn as polylines/paths",
        "sprite_glyphs": "glyphs cut from a raster sprite sheet",
        "pixelated_dotted_glyphs": "glyph strokes broken into dots/pixels",
        "glyph_shadow": "offset shadow copy of the text",
        "multicolor_text": "different colours per character",
        "translucent_text": "partially transparent or faint glyphs",
        "emboss_bevel": "embossed/bevelled glyph effect",
        "font_mixture": "different fonts within one image",
    },
    "glyph_geometry": {
        "per_char_rotation": "characters rotated independently",
        "vertical_jitter": "per-character baseline offsets",
        "overlap_tight_spacing": "touching/overlapping neighbouring glyphs",
        "wide_spacing": "characters spread with large gaps",
        "size_variation": "per-character size changes",
        "text_on_curve": "baseline follows a curve",
        "edge_clipping": "glyphs partly cut by the image border",
    },
    "global_deformation": {
        "wave_warp": "sinusoidal/multi-wave pixel displacement",
        "morph_distortion": "non-sinusoidal morphing of glyph shapes",
        "blur": "gaussian or smoothing blur",
        "greyscale": "colour removed",
    },
    "text_content": {
        "digits_only": "answer uses only digits",
        "uppercase_only": "answer uses only capitals",
        "lowercase_only": "answer uses only lowercase",
        "mixed_case_alnum": "mixed case and digits",
        "pronounceable_word": "dictionary or vowel/consonant word",
        "length_short_le4": "four or fewer characters",
        "length_long_ge7": "seven or more characters",
    },
    "watermark_footer": {
        "fixed_caption": "constant library caption/watermark strip",
    },
}

P, R, A = "param", "random", "always"
LIB = {
    # ---- feature bank 02 text libraries (two arithmetic sources excluded) ----
    "arsovskidev_php-captcha": {"bg_solid": A, "straight_lines": A, "dot_noise": A, "filled_glyphs": A, "lowercase_only": A, "length_long_ge7": A},
    "hnhx_captcha": {"bg_solid": A, "bg_dark": A, "straight_lines": R, "noise_over_text": A, "bitmap_font": A, "vertical_jitter": A, "wide_spacing": R, "edge_clipping": R, "uppercase_only": R, "length_long_ge7": R},
    "LordDashMe_php-simple-captcha": {"bg_image_texture": A, "strike_through": A, "filled_glyphs": A, "glyph_shadow": P, "overlap_tight_spacing": A, "mixed_case_alnum": A, "length_long_ge7": P},
    "eusonlito_captcha": {"bg_solid": A, "straight_lines": P, "blob_noise": P, "filled_glyphs": A, "hollow_outline_glyphs": R, "per_char_rotation": A, "uppercase_only": A, "length_long_ge7": P},
    "josecl_cool-php-captcha": {"bg_solid": A, "strike_through": P, "filled_glyphs": A, "font_mixture": R, "per_char_rotation": P, "overlap_tight_spacing": A, "wave_warp": P, "blur": P, "lowercase_only": A, "pronounceable_word": R, "length_long_ge7": P},
    "lifei6671_php-captcha": {"bg_solid": A, "bg_decoy_chars": P, "curves": P, "straight_lines": P, "filled_glyphs": A, "hollow_outline_glyphs": R, "multicolor_text": A, "font_mixture": R, "per_char_rotation": A, "size_variation": R, "edge_clipping": R, "mixed_case_alnum": A},
    "lovefc_captcha": {"bg_solid": A, "bg_decoy_chars": A, "straight_lines": P, "filled_glyphs": A, "per_char_rotation": A, "wide_spacing": A, "uppercase_only": A, "length_short_le4": P},
    "dubashi_simple-captcha-php": {"bg_solid": A, "bg_gradient": P, "dot_noise": P, "vector_stroke_glyphs": A, "emboss_bevel": P, "vertical_jitter": P, "morph_distortion": P, "digits_only": A},
    "omranjamal_RealCaptcha": {"bg_solid": P, "filled_glyphs": A, "font_mixture": R, "text_on_curve": A, "overlap_tight_spacing": A, "morph_distortion": A, "mixed_case_alnum": A, "length_long_ge7": P},
    "sibche2013_pure-captcha": {"bg_gradient": A, "bg_grid": P, "bg_decoy_chars": P, "curves": P, "straight_lines": P, "dot_noise": P, "noise_over_text": A, "filled_glyphs": A, "multicolor_text": A, "per_char_rotation": P, "wave_warp": P, "digits_only": P, "uppercase_only": P, "lowercase_only": P},
    "tarunk04_Captcha_Generator": {"bg_solid": A, "straight_lines": A, "noise_over_text": A, "filled_glyphs": A, "multicolor_text": A, "font_mixture": A, "per_char_rotation": A, "size_variation": A, "mixed_case_alnum": A, "length_short_le4": A},
    "php-quickorm_Captcha": {"bg_solid": A, "straight_lines": P, "filled_glyphs": A, "multicolor_text": A, "translucent_text": R, "per_char_rotation": A, "size_variation": A, "wide_spacing": A, "mixed_case_alnum": A, "length_short_le4": A},
    "BSN4_P-Captcha": {"bg_dark": R, "bg_grid": P, "bg_circles": P, "straight_lines": P, "dot_noise": P, "pixelated_dotted_glyphs": P, "filled_glyphs": A, "per_char_rotation": P, "edge_clipping": R, "digits_only": A},
    "sjbr_sr-freecap": {"bg_grid": P, "bg_blocks": P, "bg_dark": R, "scribbles": P, "bitmap_font": A, "multicolor_text": P, "morph_distortion": A, "blur": P, "vertical_jitter": A, "lowercase_only": A, "pronounceable_word": A, "fixed_caption": A},
    "NikolaiT_SVG-Captcha": {"bg_solid": A, "scribbles": P, "noise_over_text": A, "vector_stroke_glyphs": A, "hollow_outline_glyphs": A, "per_char_rotation": A, "size_variation": A, "overlap_tight_spacing": R, "morph_distortion": P, "edge_clipping": R, "mixed_case_alnum": A},
    "isszz_svg-captcha": {"bg_solid": P, "curves": P, "noise_over_text": A, "vector_stroke_glyphs": A, "multicolor_text": P, "per_char_rotation": A, "mixed_case_alnum": A},
    "takayukister_really-simple-captcha": {"bg_solid": P, "filled_glyphs": A, "font_mixture": A, "wide_spacing": A, "uppercase_only": A},
    "sameer-shelavale_multi-captcha": {"bg_solid": A, "bg_grid": P, "bg_decoy_chars": P, "filled_glyphs": A, "multicolor_text": A, "per_char_rotation": A, "size_variation": A, "mixed_case_alnum": A, "length_long_ge7": P},
    "igoshev_laravel-captcha": {"bg_solid": P, "curves": P, "strike_through": R, "filled_glyphs": A, "overlap_tight_spacing": P, "wave_warp": A, "mixed_case_alnum": A, "length_long_ge7": P},
    "mortezaashrafi_filament-shield-captcha": {"bg_solid": A, "bg_dark": P, "straight_lines": P, "dot_noise": P, "filled_glyphs": A, "per_char_rotation": A, "vertical_jitter": A, "wide_spacing": A, "digits_only": P, "mixed_case_alnum": P, "length_long_ge7": P},
    "kolanos_kohana-captcha": {"bg_gradient": P, "bg_dark": P, "bg_circles": P, "bg_decoy_chars": P, "scribbles": P, "straight_lines": P, "filled_glyphs": A, "hollow_outline_glyphs": P, "multicolor_text": P, "translucent_text": P, "per_char_rotation": A, "size_variation": A, "uppercase_only": A, "pronounceable_word": P, "length_long_ge7": P},
    "top-think_think-captcha": {"bg_solid": A, "bg_image_texture": P, "bg_decoy_chars": P, "curves": P, "strike_through": P, "filled_glyphs": A, "font_mixture": R, "per_char_rotation": A, "wide_spacing": A, "edge_clipping": R, "mixed_case_alnum": A},
    "zerosdev_laravel-captcha": {"bg_solid": A, "straight_lines": A, "dot_noise": A, "noise_over_text": R, "filled_glyphs": A, "glyph_shadow": A, "per_char_rotation": A, "size_variation": A, "mixed_case_alnum": A, "length_long_ge7": P},
    "Dev-3bdulrahman_Laravel-Captcha": {"bg_solid": A, "straight_lines": A, "dot_noise": A, "bitmap_font": P, "filled_glyphs": P, "multicolor_text": A, "per_char_rotation": A, "vertical_jitter": A, "wide_spacing": A, "uppercase_only": P, "mixed_case_alnum": P},
    "yiisoft_yii-captcha": {"bg_solid": P, "filled_glyphs": A, "per_char_rotation": A, "overlap_tight_spacing": P, "size_variation": A, "lowercase_only": A, "pronounceable_word": A, "length_long_ge7": P},
    "innocead_CaptchaBundle": {"bg_solid": A, "border_frame": P, "straight_lines": P, "dot_noise": P, "bg_circles": P, "noise_over_text": P, "filled_glyphs": A, "hollow_outline_glyphs": R, "multicolor_text": P, "translucent_text": P, "font_mixture": A, "per_char_rotation": P, "vertical_jitter": P, "wide_spacing": P, "size_variation": P, "blur": P, "greyscale": P, "edge_clipping": R, "uppercase_only": A},
    "carlcraig_fuel-captcha": {"bg_solid": A, "sprite_glyphs": A, "vertical_jitter": A, "wave_warp": P, "blur": P, "lowercase_only": A, "fixed_caption": P},
    "splitbrain_dokuwiki-plugin-captcha": {"bg_solid": A, "straight_lines": P, "filled_glyphs": P, "vector_stroke_glyphs": P, "multicolor_text": P, "font_mixture": A, "per_char_rotation": P, "vertical_jitter": P, "size_variation": A, "uppercase_only": A},
    # ---- original 5 (feature bank 01; Laminas is a training source) ----
    "Gregwar_Captcha": {"bg_solid": A, "bg_image_texture": R, "straight_lines": P, "noise_over_text": P, "filled_glyphs": A, "per_char_rotation": P, "vertical_jitter": P, "wave_warp": P, "mixed_case_alnum": A},
    "dapphp_securimage": {"bg_solid": A, "bg_image_texture": R, "curves": P, "dot_noise": P, "filled_glyphs": A, "font_mixture": A, "per_char_rotation": A, "wave_warp": P, "mixed_case_alnum": A},
    "mobicms_captcha": {"bg_solid": A, "filled_glyphs": A, "multicolor_text": A, "font_mixture": P, "per_char_rotation": A, "size_variation": A, "mixed_case_alnum": A},
    "yasirmturk_simple-php-captcha": {"bg_image_texture": R, "filled_glyphs": A, "glyph_shadow": P, "per_char_rotation": P, "edge_clipping": R, "mixed_case_alnum": A},
    "laminas_laminas-captcha": {"bg_solid": A, "straight_lines": P, "dot_noise": P, "filled_glyphs": A, "wave_warp": A, "lowercase_only": A, "pronounceable_word": A},
}
WITHHELD = {"NikolaiT_SVG-Captcha", "yiisoft_yii-captcha", "hnhx_captcha", "sameer-shelavale_multi-captcha"}
OLD_HEADS = {  # the hybrid's original 15 heads -> new heads covering them
    "straight_interference": ["straight_lines", "strike_through"], "curved_interference": ["curves"],
    "dot_scatter": ["dot_noise", "blob_noise"], "decorative_shapes": ["bg_circles", "scribbles"],
    "periodic_grid": ["bg_grid"], "background_texture": ["bg_image_texture"], "background_gradient": ["bg_gradient"],
    "background_blocks": ["bg_blocks"], "hollow_lettering": ["hollow_outline_glyphs"], "glyph_border": ["hollow_outline_glyphs", "emboss_bevel"],
    "glyph_shadow": ["glyph_shadow"], "nonlinear_text_deformation": ["wave_warp", "morph_distortion", "text_on_curve"],
    "blur": ["blur"], "glyph_rotation": ["per_char_rotation"], "multicolor_text": ["multicolor_text"],
}


def build():
    heads = {h: (b, d) for b, hs in BUCKETS.items() for h, d in hs.items()}
    for lib, feats in LIB.items():
        unknown = set(feats) - set(heads)
        assert not unknown, (lib, unknown)
    table = []
    for h, (bucket, desc) in heads.items():
        srcs = {lib: how for lib, f in LIB.items() for k, how in f.items() if k == h}
        train = {l: v for l, v in srcs.items() if l not in WITHHELD}
        table.append(dict(head=h, bucket=bucket, description=desc,
                          sources=len(train), controllable=sum(v == P for v in train.values()),
                          libraries=srcs,
                          coverage=("strong" if len(train) >= 6 and any(v == P for v in train.values())
                                    else "thin" if len(train) >= 2 else "single-source" if train else "none")))
    return table


def main():
    table = build()
    out = ROOT / "docs" / "feature_bank"
    (out / "FEATURE_BUCKETS_02.json").write_text(json.dumps(
        dict(buckets=BUCKETS, libraries=LIB, withheld=sorted(WITHHELD), old_head_mapping=OLD_HEADS, heads=table), indent=1), encoding="utf-8")
    cov = Counter(r["coverage"] for r in table)
    lines = [
        "# Feature buckets — feature bank 02",
        "",
        f"{len(table)} heads in {len(BUCKETS)} buckets, mapped over {len(LIB)} renderers "
        f"({len(LIB) - len(WITHHELD)} training sources; four whole-library holdouts). Generated by `feature_bank/bank02_taxonomy.py`.",
        "",
        "Capability claims from code reading plus sampled visual review — not per-image visible labels. "
        "`param` = adapter-controlled (counterfactual pairs possible), `random` = library decides, `always` = fixed style. "
        "Coverage: **strong** ≥6 training sources incl. ≥1 controllable; **thin** 2–5; **single-source** 1.",
        "",
        f"Coverage summary: {dict(cov)}",
        "",
    ]
    for bucket in BUCKETS:
        lines += [f"## {bucket.replace('_', ' ').title()}", "", "| Head | Sources | Controllable | Coverage | Libraries (P=param, R=random, A=always) |", "|---|---:|---:|---|---|"]
        for r in (r for r in table if r["bucket"] == bucket):
            libs = ", ".join(f"{l.split('_', 1)[-1]}({v[0].upper()})" for l, v in sorted(r["libraries"].items()))
            lines.append(f"| `{r['head']}` — {r['description']} | {r['sources']} | {r['controllable']} | {r['coverage']} | {libs} |")
        lines.append("")
    lines += ["## Mapping from the hybrid's original 15 heads", "", "| Old head | New head(s) |", "|---|---|"]
    lines += [f"| `{o}` | {', '.join('`%s`' % n for n in ns)} |" for o, ns in OLD_HEADS.items()]
    (out / "FEATURE_BUCKETS_02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in sorted(table, key=lambda r: r["sources"]):
        print(f"{r['bucket']:18} {r['head']:24} src={r['sources']:2} ctrl={r['controllable']:2} {r['coverage']}")


if __name__ == "__main__":
    main()
