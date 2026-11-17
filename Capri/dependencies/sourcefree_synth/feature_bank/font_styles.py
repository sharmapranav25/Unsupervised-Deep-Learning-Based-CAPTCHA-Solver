"""Measured glyph style of every font file the renderers can use (bank + library-bundled).

Style is measured from the font's own raster, not guessed from its name:
  hollow_score  = ink pixels / pixels after filling holes, over letters with no counters
                  (E F H I K L M N S T U V W X Y Z). Solid faces ~1.0; outline faces << 1.
  dotted_score  = mean connected components per such glyph. Solid faces ~1; dotted/stencil > 2.
Style tags: "hollow" if hollow_score < 0.72; "dotted" if dotted_score >= 2.5; else "filled".
Cached in resources/font_bank_02/font_styles.json keyed by file sha256.

Usage: python -m feature_bank.font_styles   (measures all fonts, prints the hollow/dotted ones)
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "resources" / "font_bank_02" / "font_styles.json"
LETTERS = "EFHIKLMNSTUVWXYZ"
HOLLOW_BELOW = 0.72
DOTTED_FROM = 2.5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(path):
    font = ImageFont.truetype(str(path), 160)
    ratios, comps = [], []
    for ch in LETTERS + LETTERS.lower():
        im = Image.new("L", (320, 320))
        ImageDraw.Draw(im).text((40, 20), ch, font=font, fill=255)
        ink = np.asarray(im) >= 32  # low threshold: hairline outlines stay closed
        if ink.sum() < 200:
            continue
        filled = ndimage.binary_fill_holes(ink)
        ratios.append(ink.sum() / filled.sum())
        comps.append(ndimage.label(ink)[1])
    if not ratios:
        return dict(hollow_score=None, dotted_score=None, style=None)
    h, d = float(np.median(ratios)), float(np.mean(comps))
    style = "hollow" if h < HOLLOW_BELOW else "dotted" if d >= DOTTED_FROM else "filled"
    return dict(hollow_score=round(h, 3), dotted_score=round(d, 2), style=style)


# Visual review of the measured non-filled fonts (2026-10-02 specimen sheet). Measurement flags
# strokes broken into pieces; only true dotted/segmented faces count as pixelated_dotted_glyphs.
# Distressed/grunge faces are broken too, but not "dotted": their dotted label is unknown (null).
REVIEWED = {
    "ballet": "filled",                       # connected script with loops, not an outline face
    "codystar": "dotted", "Codystar-Regular.ttf": "dotted", "ralewaydots": "dotted",
    "bitcountpropsingleink": "dotted", "digitalnumbers": "dotted", "silkscreen": "dotted",
    "subway.ttf": "dotted", "workbench": "dotted", "stardosstencil": "dotted",
    "monoton": "hollow", "trainone": "hollow",
}


def reviewed_style(entry):
    """Style after review: 'filled' | 'hollow' | 'dotted' | 'distressed' | None."""
    f = entry.get("file") or ""
    key = Path(f).parent.name if f.endswith("font.ttf") else Path(f).name
    if key in REVIEWED:
        return REVIEWED[key]
    return "distressed" if entry.get("style") == "dotted" else entry.get("style")


@lru_cache(maxsize=1)
def _cache():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


@lru_cache(maxsize=4096)
def style_of(path):
    """{'hollow_score','dotted_score','style'} for a font file; measured once and cached by hash."""
    p = Path(path)
    if not p.exists():
        return dict(hollow_score=None, dotted_score=None, style=None)
    digest = sha256(p)
    c = _cache()
    if digest not in c:
        try:
            c[digest] = dict(measure(p), file=p.name)
        except OSError:
            c[digest] = dict(hollow_score=None, dotted_score=None, style=None, file=p.name)
    e = dict(c[digest])
    e["reviewed_style"] = reviewed_style(e)
    return e


def main():
    paths = sorted({p for p in (ROOT / "resources").rglob("*") if p.suffix.lower() in (".ttf", ".otf")}
                   | {ROOT / r["path"] for r in json.loads((ROOT / "resources/font_bank_02/fonts.json").read_text())})
    out = {}
    for p in paths:
        digest = sha256(p)
        if digest in out:
            continue
        try:
            out[digest] = dict(measure(p), file=str(p.relative_to(ROOT)).replace("\\", "/"))
        except OSError as e:
            out[digest] = dict(hollow_score=None, dotted_score=None, style=None, file=str(p.relative_to(ROOT)), error=str(e))
    CACHE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for v in sorted(out.values(), key=lambda v: (v["style"] or "", v["file"])):
        if v["style"] != "filled":
            print(v["style"], v["hollow_score"], v["dotted_score"], v["file"])
    print(len(out), "fonts;", {s: sum(1 for v in out.values() if v["style"] == s) for s in ("filled", "hollow", "dotted", None)})


if __name__ == "__main__":
    main()
