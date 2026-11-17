"""Complete-CAPTCHA demo: random effects + our own text, with the labels printed under each image."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import captcha_effects as ce  # noqa: E402


def main(out, n=20, seed=21):
    bank = ce.FontBank()
    rng = np.random.default_rng(seed)
    W, H = 280, 84
    S = Image.new("RGB", (4 * (W + 8), (n // 4) * (H + 56)), "white")
    d = ImageDraw.Draw(S)
    k = 0
    while k < n:
        size = [(200, 60), (240, 80), (324, 81), (200, 50), (280, 84)][k % 5]
        r = ce.render_captcha(rng, bank, size)
        if r is None or not r["text"]["usable"]:
            continue
        x, y = (k % 4) * (W + 8), (k // 4) * (H + 56)
        im = Image.fromarray(r["image"]); im.thumbnail((W, H)); S.paste(im, (x, y))
        eff = ", ".join(f"{h} {v['level']}" for h, v in r["effects"].items() if v["present"])
        th = r["text"]["heads"]
        tx = ", ".join(f"{h} {v['level']}" for h, v in th.items() if v["present"] and v.get("level"))
        cats = "/".join(c[:5] for c in r["text"]["style"]["categories"])
        lines = [f"'{r['text']['text']}' {r['text']['style']['layout']} {cats} {r['text']['style']['weight_class']} | vis {r['text']['legibility']['min_visible']} con {r['text']['legibility']['median_contrast']}",
                 eff[:60], eff[60:120], tx[:60]]
        for i, ln in enumerate(lines):
            d.text((x, y + H + 2 + 12 * i), ln, fill=(20, 20, 140) if i else (0, 0, 0))
        k += 1
    S.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
