"""Text renderer sheet: rows = font category / layout / transform, 5 random samples each (plain backgrounds)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import captcha_effects as ce  # noqa: E402
from captcha_effects.text import FontBank, sample_text_spec, draw_text_spec, text_labels  # noqa: E402

ROWS = [("Sans Serif", {}), ("Serif", {}), ("Display", {}), ("Handwriting", {}), ("Monospace", {}),
        ("two-layer", dict(layout="two_layer")), ("baseline curve", dict(layout="curve")),
        ("overlapping (crowded)", dict(spacing=-0.3)), ("wide spacing", dict(spacing=0.8)),
        ("rotation + jitter", dict(rotation=35.0, jitter=0.18)), ("slant + condensed", dict(slant=0.35, x_scale=0.7)),
        ("font mix + multicolour", dict(mix=3, multicolour=True)), ("bold (stroke)", dict(bold_stroke=2, weight=1.0)),
        ("size variation", dict(size_jitter=0.4))]


def main(out):
    bank = FontBank()
    rng = np.random.default_rng(8)
    W, H = 240, 80
    S = Image.new("RGB", (190 + 5 * (W + 6), len(ROWS) * (H + 18)), "white")
    d = ImageDraw.Draw(S)
    for r, (name, over) in enumerate(ROWS):
        y = r * (H + 18)
        d.text((6, y + H // 2), name, fill="black")
        k = 0
        while k < 5:
            cat = name if name in ("Sans Serif", "Serif", "Display", "Handwriting", "Monospace") else None
            spec = sample_text_spec(rng, bank, layout=over.get("layout"), font_mix=over.get("mix"))
            if cat:
                pool = bank.pool("train", category=cat, caseless=True)
                spec["fonts"] = [str(rng.choice([f["key"] for f in pool]))]
                if all(bank.by_key[f]["caps_only"] for f in spec["fonts"]):
                    spec["text"] = spec["text"].upper()
                spec.update(rotation=0.0, jitter=0.0, slant=0.0, x_scale=1.0, spacing=0.08, size_jitter=0.0)
            spec.update({kk: v for kk, v in over.items() if kk not in ("layout", "mix")})
            cv = ce.new_canvas(W, H, dict(background=[250, 250, 248]), rng)
            res = draw_text_spec(cv, spec, rng, bank)
            if res is None or res["measures"]["max_clipped"] > 0.2:
                continue
            x = 190 + k * (W + 6)
            S.paste(Image.fromarray(cv.image()), (x, y))
            fams = "/".join(sorted({g["family"] for g in res["glyphs"]}))[:40]
            d.text((x, y + H + 2), fams, fill=(20, 20, 140))
            k += 1
    S.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
