"""captcha-effects tests: every effect runs, is deterministic, labels align with pixels, level 0 is a no-op,
the calibrated scale is monotone and the style labels follow the schema.

    python -m pytest tests -q
"""
import json
from pathlib import Path

import numpy as np
import pytest

import captcha_effects as ce
from captcha_effects.calibrate import level_of, measure_effect

ROOT = Path(__file__).resolve().parents[1]


def render(name, level, seed, size=(200, 60)):
    rng = np.random.default_rng(seed)
    cv = ce.new_canvas(size[0], size[1], dict(background=[250, 250, 250]), rng)
    spec = dict(effects=[dict(name=name, level=level)])
    out = ce.draw_under(cv, spec, rng)
    tb = ce.draw_text(cv, "Ab3k", rng, colour=(10, 10, 10))
    out += ce.apply_glyph(cv, spec, rng)
    out += ce.draw_over(cv, spec, rng, tb)
    out += ce.apply_post(cv, spec, rng)
    return cv, out


@pytest.mark.parametrize("name", ce.ALL_EFFECTS)
def test_runs_and_is_deterministic(name):
    a, _ = render(name, 7, 123)
    b, _ = render(name, 7, 123)
    assert a.image().shape == (60, 200, 3)
    assert np.array_equal(a.image(), b.image())
    assert np.array_equal(a.opaque, b.opaque)


@pytest.mark.parametrize("name", ce.ALL_EFFECTS)
def test_level_zero_draws_nothing(name):
    cv0, out = render(name, 0, 5)
    ref, _ = render("gradient", 0, 5)
    assert out == []
    assert np.array_equal(cv0.image(), ref.image())


@pytest.mark.parametrize("name", [n for n in ce.ALL_EFFECTS if ce.REGISTRY[n]["where"] in ("under", "over")])
def test_labels_align_with_pixels(name):
    """Pixels labelled with an effect's op differ from the plain canvas (the op really drew there)."""
    cv, _ = render(name, 8, 9)
    ids = [r["id"] for r in cv.records if r.get("effect") == name and r.get("colour")]
    lab = np.isin(cv.opaque, ids)
    if lab.sum() < 20:
        pytest.skip("effect drew too little visible area for this seed")
    rec = {r["id"]: r for r in cv.records}
    bgc = np.array(cv.background, np.float32)
    # expected colour of an op pixel: its colour blended at its opacity over (approximately) the background
    cols = np.array([np.array(rec[i]["colour"], np.float32) * rec[i]["opacity"] + bgc * (1 - rec[i]["opacity"])
                     for i in cv.opaque[lab]], np.float32)
    px = cv.image()[lab].astype(np.float32)
    d_op = np.linalg.norm(px - cols, axis=1)
    d_bg = np.linalg.norm(px - np.array(cv.background, np.float32), axis=1)
    contrast = np.linalg.norm(cols - np.array(cv.background, np.float32), axis=1)
    keep = contrast > 30  # ops coloured almost like the background cannot be told apart by colour
    if keep.sum() < 20:
        pytest.skip("op colours too close to the background for this seed")
    # labelled pixels (coverage >= 50%) must look more like the op's colour than like the background
    assert np.mean(d_op[keep] <= d_bg[keep] + 1e-3) > 0.9, name


def test_calibration_is_monotone_and_complete():
    t = json.loads((ROOT / "calibration.json").read_text(encoding="utf-8"))
    assert set(t) == set(ce.ALL_EFFECTS)
    for name, e in t.items():
        assert e["passed"], name
        assert all(b >= a for a, b in zip(e["fitted"][:-1], e["fitted"][1:])), name
        lv = [level_of(t, name, v) for v in e["fitted"]]
        assert lv == sorted(lv) and lv[0] == 0, name


def test_measure_matches_calibration_roughly():
    t = json.loads((ROOT / "calibration.json").read_text(encoding="utf-8"))
    for name in ["blocks", "lines", "dots", "blur", "hollow"]:
        lows = [level_of(t, name, measure_effect(name, 2, s)[0]) for s in range(6)]
        highs = [level_of(t, name, measure_effect(name, 9, s)[0]) for s in range(6)]
        assert np.median(highs) > np.median(lows), name


@pytest.mark.parametrize("name", list(ce.STYLE_SCHEMA))
def test_style_labels_follow_schema(name):
    _, out = render(name, 6, 11)
    for o in out:
        for field, v in o["style"].items():
            assert v == "na" or v in ce.STYLE_SCHEMA[name][field], (name, field, v)


def test_heads():
    heads = {ce.REGISTRY[n]["head"] for n in ce.ALL_EFFECTS}
    assert len(ce.ALL_EFFECTS) == 39 and len(heads) == 38
