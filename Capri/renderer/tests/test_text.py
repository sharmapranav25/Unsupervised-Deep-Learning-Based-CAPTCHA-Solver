"""Text renderer tests: glyph labels, font splits, measured text heads, complete CAPTCHAs."""
import numpy as np
import pytest

import captcha_effects as ce
from captcha_effects.text import FontBank, sample_text_spec, draw_text_spec, text_labels

BANK = FontBank()


def draw(seed, size=(240, 80), **kw):
    rng = np.random.default_rng(seed)
    spec = sample_text_spec(rng, BANK, **kw)
    cv = ce.new_canvas(size[0], size[1], dict(background=[245, 245, 245]), rng)
    return cv, spec, draw_text_spec(cv, spec, rng, BANK)


@pytest.mark.parametrize("seed", range(12))
def test_every_glyph_labelled_and_inside(seed):
    cv, spec, res = draw(seed)
    assert res["text"] == "".join(g["char"] for g in res["glyphs"])
    for g in res["glyphs"]:
        assert g["box"] is not None and g["visible_px"] > 0
        x0, y0, x1, y1 = g["box"]
        assert 0 <= x0 <= x1 < cv.w and 0 <= y0 <= y1 < cv.h
        assert g["font"] in BANK.by_key and g["category"] == BANK.by_key[g["font"]]["category"]
    assert res["measures"]["max_clipped"] <= 0.2


def test_deterministic():
    a, _, ra = draw(3)
    b, _, rb = draw(3)
    assert np.array_equal(a.image(), b.image()) and ra["text"] == rb["text"]


def test_font_split_respected():
    for seed in range(20):
        _, spec, res = draw(seed, split="heldout")
        assert all(BANK.by_key[g["font"]]["split"] == "heldout" for g in res["glyphs"])
        _, spec, res = draw(100 + seed)
        assert all(BANK.by_key[g["font"]]["split"] == "train" for g in res["glyphs"])


def test_caps_only_fonts_get_uppercase():
    for seed in range(40):
        _, spec, res = draw(seed)
        for g in res["glyphs"]:
            if BANK.by_key[g["font"]]["caps_only"]:
                assert not g["char"].islower()


def test_text_heads_measured_not_requested():
    _, spec, res = draw(5, layout="line")
    spec2 = dict(spec, rotation=0.0, jitter=0.0, curve=0.0, size_jitter=0.0)
    rng = np.random.default_rng(5)
    cv = ce.new_canvas(240, 80, dict(background=[245, 245, 245]), rng)
    lab = text_labels(draw_text_spec(cv, spec2, rng, BANK))
    for h in ("per_char_rotation", "vertical_jitter", "text_on_curve", "size_variation"):
        assert lab["heads"][h]["level"] == 0, h


def test_curve_layout_is_detected():
    hits = 0
    for seed in range(20):
        _, spec, res = draw(seed, layout="curve", length=7)
        hits += text_labels(res)["heads"]["text_on_curve"]["present"]
    assert hits >= 15


def test_complete_captcha_labels():
    rng = np.random.default_rng(11)
    usable = 0
    for _ in range(30):
        r = ce.render_captcha(rng, BANK, (200, 60))
        assert set(v["present"] for v in r["effects"].values()) <= {0, 1}
        assert len(r["effects"]) == 37 and "fixed_caption" not in r["effects"]
        usable += r["text"]["usable"]
    assert usable >= 15
