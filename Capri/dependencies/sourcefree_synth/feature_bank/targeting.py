"""Per-epoch batches for the hybrid v3 curriculum (docs/HYBRID_V3_TRAINING_HANDOFF.md §3).

A batch of n images (split "epochNN", indices 0..n-1):
  * indices [0, n_base): base mix, kinds by exact quota of cfg["mix"] (Python share <= the cap);
  * indices [n_base, n): targeted, one slot spec per index (head, polarity, python allowed), from targets.json.
    With no targets (epoch 1, or every head passed / flagged) these indices are base mix too.
n_base = n - round(n * targeted_fraction) does not depend on the targets, so the base half can be rendered
while the GPU trains and the targeted half once the steering scores exist.

A targeted slot only chooses *settings*; labels still come from the final image. The generator's accept()
redraws a slot until the head's measured presence matches the polarity (hard positive: present = 1; hard
negative: present = 0, drawn as a positive of a look-alike head), and its last `target_fallback_attempts`
attempts drop the target (base settings) so no slot stays empty; the row records target_met.

targets.json: {"epoch": k, "heads": {head: share of the targeted half}, ...} (shares are normalised here).
"""
import json
import random
from functools import lru_cache

# Hard negatives: the head absent, with a look-alike effect present.
LOOKALIKE = {
    "bg_solid": ["bg_gradient", "bg_image_texture", "bg_dark"],
    "bg_gradient": ["bg_solid", "bg_image_texture"],
    "bg_image_texture": ["bg_gradient", "bg_blocks", "dot_noise"],
    "bg_blocks": ["bg_grid", "bg_image_texture"],
    "bg_grid": ["bg_blocks", "straight_lines", "border_frame"],
    "bg_circles": ["blob_noise", "curves", "dot_noise"],
    "bg_dark": ["bg_solid", "bg_gradient", "bg_image_texture"],
    "bg_decoy_chars": ["font_mixture", "size_variation", "noise_over_text"],
    "border_frame": ["straight_lines", "bg_grid"],
    "straight_lines": ["strike_through", "curves", "bg_grid"],
    "curves": ["straight_lines", "scribbles", "wave_warp"],
    "strike_through": ["straight_lines", "noise_over_text", "curves"],
    "dot_noise": ["blob_noise", "pixelated_dotted_glyphs", "bg_image_texture"],
    "blob_noise": ["dot_noise", "bg_circles"],
    "scribbles": ["curves", "straight_lines"],
    "noise_over_text": ["straight_lines", "dot_noise", "curves"],
    "filled_glyphs": ["hollow_outline_glyphs", "vector_stroke_glyphs"],
    "hollow_outline_glyphs": ["filled_glyphs", "emboss_bevel", "glyph_shadow"],
    "bitmap_font": ["pixelated_dotted_glyphs", "sprite_glyphs"],
    "vector_stroke_glyphs": ["hollow_outline_glyphs", "filled_glyphs"],
    "sprite_glyphs": ["bitmap_font", "font_mixture"],
    "pixelated_dotted_glyphs": ["dot_noise", "bitmap_font"],
    "glyph_shadow": ["emboss_bevel", "blur", "translucent_text"],
    "multicolor_text": ["font_mixture", "bg_decoy_chars"],
    "translucent_text": ["glyph_shadow", "blur"],
    "emboss_bevel": ["glyph_shadow", "hollow_outline_glyphs"],
    "font_mixture": ["size_variation", "multicolor_text"],
    "per_char_rotation": ["vertical_jitter", "text_on_curve", "wave_warp"],
    "vertical_jitter": ["per_char_rotation", "text_on_curve"],
    "overlap_tight_spacing": ["wide_spacing", "size_variation"],
    "wide_spacing": ["overlap_tight_spacing", "length_short_le4"],
    "size_variation": ["vertical_jitter", "font_mixture"],
    "text_on_curve": ["vertical_jitter", "wave_warp"],
    "edge_clipping": ["wide_spacing", "length_long_ge7"],
    "wave_warp": ["morph_distortion", "curves", "text_on_curve"],
    "morph_distortion": ["wave_warp", "blur"],
    "blur": ["translucent_text", "morph_distortion", "glyph_shadow"],
    "greyscale": ["bg_solid", "bg_dark"],
    "digits_only": ["uppercase_only", "mixed_case_alnum"],
    "uppercase_only": ["mixed_case_alnum", "digits_only"],
    "lowercase_only": ["mixed_case_alnum", "pronounceable_word"],
    "mixed_case_alnum": ["uppercase_only", "lowercase_only"],
    "pronounceable_word": ["lowercase_only"],
    "length_short_le4": ["length_long_ge7"],
    "length_long_ge7": ["length_short_le4"],
    "fixed_caption": ["border_frame", "bg_decoy_chars"],
}
# Text-content heads controlled through the generator's own text draw: (positive, hard-negative) text overrides.
TEXT_OVERRIDES = {
    "length_short_le4": (dict(length=[2, 4]), dict(length=[5, 5])),
    "length_long_ge7": (dict(length=[7, 8]), dict(length=[6, 6])),
    "pronounceable_word": (dict(pronounceable_fraction=1.0), dict(pronounceable_fraction=0.0)),
}
SHAPE_OF = {}  # head -> python_shapes layer kinds


def _shape_of():
    if not SHAPE_OF:
        from feature_bank.py_sources import SHAPE_HEAD
        for k, h in SHAPE_HEAD.items():
            SHAPE_OF.setdefault(h, []).append(k)
    return SHAPE_OF


@lru_cache(maxsize=1)
def taxonomy_sources():
    from feature_bank.heads import TAXONOMY
    return {h["head"]: dict(h["libraries"]) for h in TAXONOMY["heads"]}


def recipes_for(head):
    from feature_bank.generate_corpus import approved_recipes
    recs = approved_recipes()
    php = sorted(n for n, r in recs.items() if r["feature"] == head and r.get("source", "php") == "php")
    py = sorted(n for n, r in recs.items() if r["feature"] == head and r.get("source", "php") != "php")
    return php, py


def has_python_route(head):
    return bool(recipes_for(head)[1] or _shape_of().get(head))


# ------------------------------------------------------------------ slot plan

def n_base_of(n, cfg):
    return n - int(round(n * float(cfg["curriculum"]["targeted_fraction"])))


def quota_kinds(n, mix, rng):
    """Exactly-apportioned kinds (largest remainder), shuffled; Python never above its share."""
    keys = list(mix)
    tot = sum(float(mix[k]) for k in keys)
    raw = {k: n * float(mix[k]) / tot for k in keys}
    cnt = {k: int(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: raw[k] - cnt[k], reverse=True)[: n - sum(cnt.values())]:
        if k != "python":
            cnt[k] += 1
    while sum(cnt.values()) < n:  # remainder never goes to python
        cnt[max((k for k in keys if k != "python"), key=lambda k: raw[k])] += 1
    out = [k for k in keys for _ in range(cnt[k])]
    rng.shuffle(out)
    return out


def slot_plan(cfg, split, n, targets):
    """Per index: dict(kind=...) for base slots, dict(target=...) for targeted ones."""
    cur = cfg["curriculum"]
    nb = n_base_of(n, cfg)
    nt = n - nb
    base = quota_kinds(nb, cfg["mix"], random.Random(f"{cfg['seed']}|{split}|base"))
    heads = dict((targets or {}).get("heads") or {})
    if not heads or nt == 0:
        rest = quota_kinds(nt, cfg["mix"], random.Random(f"{cfg['seed']}|{split}|rest"))
        return [dict(kind=k) for k in base] + [dict(kind=k) for k in rest]
    shares = normalise_shares(heads, float(cur["head_min_share"]), float(cur["head_max_share"]))
    raw = {h: nt * s for h, s in shares.items()}
    cnt = {h: int(raw[h]) for h in shares}
    for h in sorted(shares, key=lambda h: raw[h] - cnt[h], reverse=True)[: nt - sum(cnt.values())]:
        cnt[h] += 1
    specs = []
    for h in sorted(cnt):
        pf = float((targets.get("pos_fraction") or {}).get(h, cur["hard_positive_fraction"]))
        npos = int(round(cnt[h] * pf))
        specs += [dict(head=h, polarity="pos")] * npos + [dict(head=h, polarity="neg")] * (cnt[h] - npos)
    specs = [dict(s) for s in specs]
    # Python budget for the targeted half: cap * n minus the base half's python, given to heads with a Python route
    budget = int(float(cfg["mix"].get("python", 0)) * n) - base.count("python")
    rng = random.Random(f"{cfg['seed']}|{split}|targeted")
    def py_route(sp):
        if sp["polarity"] == "pos":
            return has_python_route(sp["head"])
        return any(has_python_route(l) for l in LOOKALIKE.get(sp["head"], []))
    elig = [i for i, sp in enumerate(specs) if py_route(sp)]
    rng.shuffle(elig)
    for i in elig[:max(0, budget)]:
        specs[i]["python"] = True
    rng.shuffle(specs)
    return [dict(kind=k) for k in base] + [dict(target=s) for s in specs]


def normalise_shares(heads, lo, hi):
    """Shares proportional to the given weights, each clipped to [lo, hi] and renormalised (iteratively)."""
    w = {h: max(float(v), 1e-9) for h, v in heads.items()}
    if len(w) * lo > 1:  # too many heads for the floor: equal shares
        return {h: 1 / len(w) for h in w}
    fixed = {}
    for _ in range(50):
        free = {h: v for h, v in w.items() if h not in fixed}
        mass = 1 - sum(fixed.values())
        tot = sum(free.values())
        s = {h: mass * v / tot for h, v in free.items()}
        viol = {h: (lo if v < lo else hi) for h, v in s.items() if v < lo - 1e-12 or v > hi + 1e-12}
        if not viol:
            out = dict(fixed, **s)
            break
        fixed.update(viol)
        if len(fixed) == len(w):
            out = fixed
            break
    tot = sum(out.values())
    return {h: v / tot for h, v in out.items()}


# ------------------------------------------------------------------ settings for one targeted slot

def targeted_settings(cfg, split, index, attempt, spec, rng_base):
    """Settings for a targeted slot at an attempt; mechanisms in proportion to what exists for the head."""
    from feature_bank.generate_corpus import (training_libraries, layer_libraries, font_params, make_text, stage_seed,
                                              approved_recipes, pick)
    from feature_bank.render_chain import BG_LIBRARIES
    rng = random.Random(f"{cfg['seed']}|{split}|{index}|{attempt}|tgt")
    libs = training_libraries(cfg)
    head, pol = spec["head"], spec["polarity"]
    py_ok = bool(spec.get("python"))
    # a hard negative is a positive of a look-alike head (or the head's own recipe at level 0)
    want = head
    if pol == "neg":
        php_r, py_r = recipes_for(head)
        cands = list(LOOKALIKE.get(head, []))
        if php_r:
            cands.append("__level0__")
        want = rng.choice(cands) if cands else None
    tcfg = cfg
    s = dict(kind="targeted", stratum="seen", font_split="train", text_mode="random", recipe=None, source="library",
             target=dict(head=head, polarity=pol, via=want, python=py_ok))
    if head in TEXT_OVERRIDES:  # the generator draws the text: override length / word mode, any library
        tcfg = dict(cfg, text=dict(cfg["text"], **TEXT_OVERRIDES[head][0 if pol == "pos" else 1]))
        s["target"]["via"] = head
        return _library(s, tcfg, rng, rng.choice(libs))
    if want == "__level0__":
        name = rng.choice(recipes_for(head)[0])
        return _php_recipe(s, tcfg, rng, name, 0)
    if want is None:
        return _library(s, tcfg, rng, rng.choice(libs))
    php_r, py_r = recipes_for(want)
    src = {l: m for l, m in taxonomy_sources().get(want, {}).items() if l in libs}
    shapes = _shape_of().get(want, []) if py_ok else []
    mech = {}
    if php_r:
        mech["php_recipe"] = 0.4
    if src:
        mech["library"] = 0.35
        layer_src = sorted(set(src) & set(layer_libraries(libs)))
        top_src = sorted(set(src) & set(BG_LIBRARIES))
        if layer_src or top_src:
            mech["chain"] = 0.25
    if py_ok and py_r:
        mech["py_recipe"] = 0.6
    if py_ok and shapes:
        mech["py_shapes"] = 0.3
    if not mech:
        return _library(s, tcfg, rng, rng.choice(libs))
    m = pick(rng, mech)
    s["target"]["mechanism"] = m
    levels = cfg["curriculum"]["hard_positive_levels"]
    if m == "php_recipe":
        return _php_recipe(s, tcfg, rng, rng.choice(php_r), rng.choice(levels))
    if m == "py_recipe":
        from feature_bank import py_sources
        name = rng.choice(py_r)
        r = approved_recipes()[name]
        level = rng.choice(levels)
        st = py_sources.recipe_stage(name, r, level, rng, cfg)
        s.update(kind="python", source=r["source"], recipe=dict(name=name, level=level, feature=r["feature"]), stages=[st])
        return s
    if m == "py_shapes":
        from feature_bank import py_sources
        layers = [py_sources.random_layer(rng, rng.choice(shapes))]
        tops = sorted(set(BG_LIBRARIES) & set(libs))
        shp = dict(source="python_shapes", seed=stage_seed(rng),
                   params=dict(width=rng.randint(150, 300), height=rng.randint(50, 100), layers=layers))
        top = rng.choice(tops)
        p = font_params(rng, top, tcfg, "train")
        text, mode = make_text(rng, top, tcfg, p)
        p["text"] = text
        s.update(kind="python", source="python_shapes", stages=[shp, dict(source="php", library=top, seed=stage_seed(rng), params=p)],
                 text_mode=mode)
        return s
    if m == "chain":
        layer_src = sorted(set(src) & set(layer_libraries(libs)))
        top_src = sorted(set(src) & set(BG_LIBRARIES))
        tops = sorted(set(BG_LIBRARIES) & set(libs))
        if layer_src and (not top_src or rng.random() < 0.5):
            first, top = rng.choice(layer_src), rng.choice(tops)  # the effect as a background layer under text
        else:
            first, top = rng.choice(layer_libraries(libs)), rng.choice(top_src)  # the effect in the text stage over a layer
        p = font_params(rng, top, tcfg, "train")
        text, mode = make_text(rng, top, tcfg, p)
        p["text"] = text
        s.update(kind="chain", stages=[dict(source="php", library=first, seed=stage_seed(rng), params={}),
                                       dict(source="php", library=top, seed=stage_seed(rng), params=p)], text_mode=mode)
        return s
    return _library(s, tcfg, rng, rng.choice(sorted(src)))


def _library(s, cfg, rng, lib):
    from feature_bank.generate_corpus import font_params, make_text, stage_seed
    params = font_params(rng, lib, cfg, "train")
    text, mode = make_text(rng, lib, cfg, params)
    params["text"] = text
    s.update(kind="library_single", text_mode=mode, stages=[dict(source="php", library=lib, seed=stage_seed(rng), params=params)])
    return s


def _php_recipe(s, cfg, rng, name, level):
    from feature_bank.generate_corpus import approved_recipes, make_text, stage_seed
    r = approved_recipes()[name]
    lib = r["library"]
    params = dict(r["params_by_level"][str(level)])
    text, mode = make_text(rng, lib, cfg, params)
    params["text"] = text
    s.update(kind="intensity", recipe=dict(name=name, level=level, feature=r["feature"]), text_mode=mode,
             stages=[dict(source="php", library=lib, seed=stage_seed(rng), params=params)])
    return s


def load_targets(path):
    return json.loads(open(path, encoding="utf-8").read()) if path else None
