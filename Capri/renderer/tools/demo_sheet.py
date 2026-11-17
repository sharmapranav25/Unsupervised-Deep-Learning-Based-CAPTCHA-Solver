"""Capability demo: family-style specs (hand-written) + random balanced specs, with plain text standing in for the
CAPTCHA library's text."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import captcha_effects as ce  # noqa: E402

FONT = "E:/pranav/Final sem/sourcefree_synth/resources/font_bank_02/fonts/acme/font.ttf"
STYLED = {
    "CryptBB-like": dict(size=(200, 60), background=[255, 255, 255], text="qNyTy", effects=[
        dict(name="blocks", level=5, style=dict(palette="pastel", layout="random", opacity=0.6)),
        dict(name="grid", level=4, style=dict(palette="pastel")),
        dict(name="curves", level=6, style=dict(palette="pastel")),
        dict(name="dots", level=3, style=dict(palette="pastel"))]),
    "DarkBay-like": dict(size=(200, 50), background=[120, 104, 128], text="KA1TM8", effects=[
        dict(name="blocks", level=9, style=dict(palette="shades", layout="dense", opacity=1.0, speckle=False))]),
    "City-like": dict(size=(324, 81), background=[255, 255, 255], text="93249", effects=[
        dict(name="bands", level=6, style=dict(palette="grey"))]),
}


def render(spec, seed, size, text):
    rng = np.random.default_rng(seed)
    cv = ce.new_canvas(size[0], size[1], spec, rng)
    ce.draw_under(cv, spec, rng)
    tb = ce.draw_text(cv, text, rng, FONT)
    ce.apply_glyph(cv, spec, rng)
    ce.draw_over(cv, spec, rng, tb)
    ce.apply_post(cv, spec, rng)
    return cv


def main(out):
    W, H = 300, 85
    rows = []
    for name, s in STYLED.items():
        rows.append((name, [render(s, k, s["size"], s["text"]) for k in range(4)], ""))
    rng = np.random.default_rng(11)
    for r in range(6):
        cvs, descs = [], []
        for k in range(4):
            spec = ce.sample_spec(rng)
            cvs.append(render(spec, 100 * r + k, (240, 80), "Xk7pQ"))
            descs.append(", ".join(f"{e['name']} {e['level']}" for e in spec["effects"]))
        rows.append(("random mix", cvs, " | ".join(descs)))
    S = Image.new("RGB", (120 + 4 * (W + 8), len(rows) * (H + 30) + 10), "white")
    d = ImageDraw.Draw(S)
    for i, (name, cvs, desc) in enumerate(rows):
        y = 10 + i * (H + 30)
        d.text((6, y + H // 2), name, fill="black")
        for k, cv in enumerate(cvs):
            im = Image.fromarray(cv.image())
            im.thumbnail((W, H))
            S.paste(im, (120 + k * (W + 8), y))
        d.text((120, y + H + 4), desc[:230], fill=(20, 20, 140))
    S.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
