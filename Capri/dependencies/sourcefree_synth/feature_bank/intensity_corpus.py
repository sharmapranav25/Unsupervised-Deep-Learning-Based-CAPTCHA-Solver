"""Calibrate visible 0–10 feature levels, then build an approved PHP-rendered corpus.

Examples:
  python -m feature_bank.intensity_corpus calibrate --out runs/intensity_cal_01 --seeds 2
  python -m feature_bank.intensity_corpus build --calibration runs/intensity_cal_01/report.json \
      --out runs/intensity_corpus_01 --cases-per-level 20

No rating is inferred from a PHP knob alone. Calibration renders every level, measures the
target call's visible area, and writes contact sheets. The build command requires an explicit
reviewed=True on a passing recipe in the calibration report. The original library draws all
pixels; this module only sets its existing adapter parameters and audits the result.
"""
import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feature_bank.bank02_smoke import ROOT, active_libraries, run_case
from feature_bank.label_maps import decode


@dataclass(frozen=True)
class Recipe:
    name: str
    library: str
    feature: str
    control: str
    maximum: float
    call_class: str
    site: str
    fixed: tuple = ()
    metric: str = "pixels"
    paired_control: str = ""
    start_value: float = None

    def params(self, level):
        if not 0 <= level <= 10:
            raise ValueError(level)
        if self.start_value is not None:
            value = round(self.start_value + (self.maximum - self.start_value) * level / 10)
        elif isinstance(self.maximum, float):
            value = round(self.maximum * level / 10, 4)
        else:
            value = 0 if level == 0 else max(1, round(self.maximum * level / 10))
        params = {**dict(self.fixed), self.control: value}
        if self.paired_control:
            params[self.paired_control] = value
        return params


# Only knobs with an actual zero setting and an intercepted draw site belong here.
# A recipe is a candidate until its full ladder passes measurement and visual review.
RECIPES = {
    r.name: r for r in (
        Recipe("laminas_lines", "laminas_laminas-captcha", "straight_lines", "line_noise", 20,
               "line", "generateImage", (("dot_noise", 0),)),
        Recipe("laminas_dots", "laminas_laminas-captcha", "dot_noise", "dot_noise", 300,
               "filled_ellipse", "generateImage", (("line_noise", 0),)),
        Recipe("eusonlito_lines", "eusonlito_captcha", "straight_lines", "noise_lines", 18,
               "filled_rectangle", "noiseLines", (("noise_points", 0), ("font_source", "bank"))),
        Recipe("eusonlito_blobs", "eusonlito_captcha", "blob_noise", "noise_points", 320,
               "filled_arc", "noisePoints", (("noise_lines", 0), ("font_source", "bank"))),
        Recipe("dapphp_curves", "dapphp_securimage", "curves", "num_lines", 16,
               "filled_rectangle", "drawLines", (("noise_level", 0), ("perturbation", 0))),
        Recipe("dapphp_dots", "dapphp_securimage", "dot_noise", "noise_level", 12,
               "filled_arc", "drawNoise", (("num_lines", 0), ("perturbation", 0))),
        Recipe("gregwar_lines", "Gregwar_Captcha", "straight_lines", "max_front_lines", 20,
               "line", "drawLine", (("max_behind_lines", 0), ("max_angle", 0), ("max_offset", 0),
                                      ("distortion", False), ("interpolation", False))),
        Recipe("lovefc_lines", "lovefc_captcha", "straight_lines", "interfere_line", 40,
               "line", "extLine", (("font_source", "bank"),)),
        Recipe("morteza_dots", "mortezaashrafi_filament-shield-captcha", "dot_noise", "noise_dots", 3000,
               "pixel", "drawNoise", (("noise_level", 5), ("noise_lines", 0))),
        Recipe("morteza_lines", "mortezaashrafi_filament-shield-captcha", "straight_lines", "noise_lines", 24,
               "line", "drawNoise", (("noise_level", 5), ("noise_dots", 0))),
        Recipe("dapphp_wave", "dapphp_securimage", "wave_warp", "perturbation", 1.5,
               "text", "drawWord", (("num_lines", 0), ("noise_level", 0),
                                      ("use_text_angles", False), ("use_random_baseline", False),
                                      ("use_random_spaces", False), ("use_transparent_text", False)), "difference"),
        Recipe("innocead_dots", "innocead_CaptchaBundle", "dot_noise", "noise_max_px", 8000,
               "pixel", "addNoiseToImage", (("noise_max_lines", 0), ("noise_max_circles", 0),
                                            ("noise_color", 1), ("noise_on_top", True),
                                            ("brush_size", 1), ("char_max_rot_angle", 0),
                                            ("char_transparent", 0), ("char_random_color", False),
                                            ("effect_blur", False)), paired_control="noise_min_px"),
        Recipe("innocead_over_text", "innocead_CaptchaBundle", "noise_over_text", "noise_max_px", 8000,
               "pixel", "addNoiseToImage", (("noise_max_lines", 0), ("noise_max_circles", 0),
                                            ("noise_color", 1), ("noise_on_top", True),
                                            ("brush_size", 1), ("char_max_rot_angle", 0),
                                            ("char_transparent", 0), ("char_random_color", False),
                                            ("effect_blur", False)), "glyph_overlap", "noise_min_px"),
        Recipe("innocead_lines", "innocead_CaptchaBundle", "straight_lines", "noise_max_lines", 100,
               "line", "addNoiseToImage", (("noise_max_px", 0), ("noise_max_circles", 0),
                                           ("noise_color", 1), ("noise_on_top", True),
                                           ("brush_size", 1), ("char_max_rot_angle", 0),
                                           ("char_transparent", 0), ("char_random_color", False),
                                           ("effect_blur", False)), paired_control="noise_min_lines"),
        Recipe("innocead_circles", "innocead_CaptchaBundle", "bg_circles", "noise_max_circles", 50,
               "arc", "addNoiseToImage", (("noise_max_px", 0), ("noise_max_lines", 0),
                                          ("noise_color", 1), ("noise_on_top", True),
                                          ("brush_size", 1), ("char_max_rot_angle", 0),
                                          ("char_transparent", 0), ("char_random_color", False),
                                          ("effect_blur", False)), paired_control="noise_min_circles"),
        Recipe("innocead_translucent", "innocead_CaptchaBundle", "translucent_text", "char_transparent", 110,
               "text", "addLettersToImage", (("noise_max_px", 0), ("noise_max_lines", 0),
                                              ("noise_max_circles", 0), ("char_max_rot_angle", 0),
                                              ("char_random_color", False), ("effect_blur", False),
                                              ("width", 320), ("height", 90)), "alpha"),
        Recipe("innocead_rotation", "innocead_CaptchaBundle", "per_char_rotation", "char_max_rot_angle", 120,
               "text", "addLettersToImage", (("noise_max_px", 0), ("noise_max_lines", 0),
                                              ("noise_max_circles", 0), ("char_transparent", 0),
                                              ("char_random_color", False), ("effect_blur", False)), "angle"),
        Recipe("josecl_strike", "josecl_cool-php-captcha", "strike_through", "line_width", 20,
               "line", "WriteLine", (("blur", False), ("max_rotation", 0),
                                      ("x_amplitude", 0), ("y_amplitude", 0))),
        Recipe("gregwar_jitter", "Gregwar_Captcha", "vertical_jitter", "max_offset", 35,
               "text", "writePhrase", (("max_behind_lines", 0), ("max_front_lines", 0),
                                        ("max_angle", 0), ("distortion", False),
                                        ("interpolation", False), ("height", 140)), "glyph_displacement"),
        Recipe("carl_wave", "carlcraig_fuel-captcha", "wave_warp", "distort_amplitude", 35,
               "text", "draw", (("distort", True), ("distort_period", 30),
                                ("font_gaussian_blur", False), ("message", False),
                                ("height", 90)), "difference"),
        Recipe("igoshev_tight", "igoshev_laravel-captcha", "overlap_tight_spacing", "letter_spacing", -6,
               "text", "render", (("width", 300), ("font_size", 28), ("height_ratio", 2.2),
                                   ("colour", "1d1d1d"), ("background", "ffffff")), "xcompression", start_value=12),
        Recipe("innocead_size_spread", "innocead_CaptchaBundle", "size_variation", "char_max_size", 70,
               "text", "addLettersToImage", (("width", 400), ("height", 120), ("char_min_size", 24),
                                              ("char_max_rot_angle", 0), ("char_transparent", 0),
                                              ("noise_max_px", 0), ("noise_max_lines", 0),
                                              ("noise_max_circles", 0), ("effect_blur", False)),
               "size_range", start_value=24),
        Recipe("gregwar_rotation", "Gregwar_Captcha", "per_char_rotation", "max_angle", 45,
               "text", "writePhrase", (("max_behind_lines", 0), ("max_front_lines", 0),
                                      ("max_offset", 0), ("distortion", False), ("interpolation", False)), "angle"),
        Recipe("josecl_rotation", "josecl_cool-php-captcha", "per_char_rotation", "max_rotation", 45,
               "text", "WriteText", (("line_width", 0), ("blur", False), ("x_amplitude", 0),
                                     ("y_amplitude", 0)), "angle"),
    )
}

# The other taxonomy heads describe categories or text content. They keep their existing
# presence/content targets; an ordinal 10 would have no defensible meaning for them.
ORDINAL_HEADS = frozenset({
    "bg_gradient", "bg_image_texture", "bg_blocks", "bg_grid", "bg_circles", "bg_decoy_chars",
    "straight_lines", "curves", "strike_through", "dot_noise", "blob_noise", "scribbles", "noise_over_text",
    "pixelated_dotted_glyphs", "glyph_shadow", "translucent_text", "emboss_bevel",
    "per_char_rotation", "vertical_jitter", "overlap_tight_spacing", "wide_spacing", "size_variation",
    "text_on_curve", "edge_clipping", "wave_warp", "morph_distortion", "blur",
})


def target_pixels(stem, recipe):
    data = decode(stem)
    wanted = [cid for cid, c in data["calls"].items()
              if c.get("class") == recipe.call_class and recipe.site in c.get("site", "")]
    ids = np.array(wanted, dtype=np.uint32)
    mask = np.isin(data["opaque"], ids) | np.isin(data["translucent"], ids)
    # The target must be visibly drawn. Presence in the call log alone is insufficient.
    return int(mask.sum()), int(mask.size), data


def inspect_case(stem, recipe):
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if not meta.get("ok") or not Path(str(stem) + ".calls.json").exists():
        return dict(ok=False, reason=meta.get("exception") or meta.get("fatal") or meta.get("reason") or meta.get("labels_error"))
    try:
        pixels, area, data = target_pixels(stem, recipe)
    except Exception as exc:
        return dict(ok=False, reason=f"label_decode:{exc!r}")
    boxes = data["boxes"]
    glyph_ok = len(boxes) == len(str(meta["label"])) and all(b is not None for b in boxes)
    if recipe.metric == "angle":
        angles = [min(abs(float(data["calls"][gid].get("angle", 0))) % 360,
                      360 - (abs(float(data["calls"][gid].get("angle", 0))) % 360))
                  for gid in data["glyphs"]]
        value = float(np.mean(angles)) if angles else 0.0
    elif recipe.metric == "alpha":
        alphas = [float(c.get("rgba", [0, 0, 0, 0])[3]) for c in data["calls"].values()
                  if c.get("class") == recipe.call_class and recipe.site in c.get("site", "")]
        value = float(np.mean(alphas) / 127) if alphas else 0.0
    elif recipe.metric == "glyph_overlap":
        ids = [cid for cid, c in data["calls"].items()
               if c.get("class") == recipe.call_class and recipe.site in c.get("site", "")]
        target = np.isin(data["opaque"], ids) | np.isin(data["translucent"], ids)
        region = np.zeros(target.shape, dtype=bool)
        for b in boxes:
            if b is not None:
                region[b["y0"]:b["y1"] + 1, b["x0"]:b["x1"] + 1] = True
        value = float((target & region).sum() / max(1, region.sum()))
    elif recipe.metric == "size_range":
        sizes = [float(data["calls"][gid].get("size", 0)) for gid in data["glyphs"]]
        value = (max(sizes) - min(sizes)) / max(1, min(sizes)) if sizes else 0.0
    else:
        value = pixels / area
    return dict(ok=True, target_pixels=pixels, area=area, target_fraction=round(pixels / area, 6),
                target_value=round(value, 6), metric=recipe.metric,
                glyph_ok=glyph_ok, label=str(meta["label"]),
                sha256=hashlib.sha256(stem.with_suffix(".png").read_bytes()).hexdigest())


def sheet(recipe, rows, out):
    # Same seed along a row; levels across columns make monotonicity inspectable.
    by_seed = {}
    for row in rows:
        if row.get("ok"):
            by_seed.setdefault(row["seed"], []).append(row)
    tile_w, tile_h = 245, 130
    canvas = Image.new("RGB", (tile_w * 11, tile_h * len(by_seed)), "white")
    draw = ImageDraw.Draw(canvas)
    for y, (seed, cases) in enumerate(sorted(by_seed.items())):
        for row in sorted(cases, key=lambda r: r["level"]):
            x = row["level"]
            with Image.open(row["image"]) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w - 8, tile_h - 30))
                canvas.paste(im, (x * tile_w + 4, y * tile_h + 20))
            draw.text((x * tile_w + 4, y * tile_h + 2),
                      f"L{x} measure={row['target_value']:.3f} glyph={int(row['glyph_ok'])}", fill="black")
    canvas.save(out / f"{recipe.name}_sheet.png")


def monotonic_report(rows):
    value = lambda r: r.get("target_value", r.get("target_fraction"))
    by_level = {k: [value(r) for r in rows if r.get("ok") and r["level"] == k] for k in range(11)}
    if any(not v for v in by_level.values()):
        return dict(passes=False, reason="missing_level")
    med = np.array([np.median(by_level[k]) for k in range(11)])
    # Allow local stochastic dips, but require an increasing trend and a visible endpoint.
    ranks = np.argsort(np.argsort(med))
    rho = float(np.corrcoef(np.arange(11), ranks)[0, 1])
    rising = int((np.diff(med) >= 0).sum())
    distinct = len(set(np.round(med, 6)))
    seed_rhos = []
    for seed in sorted({r["seed"] for r in rows if r.get("ok")}):
        seq = [next((value(r) for r in rows if r.get("ok") and r["seed"] == seed and r["level"] == k), None)
               for k in range(11)]
        if all(v is not None for v in seq):
            order = np.argsort(np.argsort(seq))
            seed_rhos.append(float(np.corrcoef(np.arange(11), order)[0, 1]))
    zero_clear = all(value(r) == 0 for r in rows if r.get("ok") and r["level"] == 0)
    glyphs_ok = all(r.get("glyph_ok", False) for r in rows if r.get("ok"))
    passes = bool(zero_clear and glyphs_ok and rho >= .9 and rising >= 7 and distinct >= 10
                  and seed_rhos and min(seed_rhos) >= .9 and med[-1] > med[1])
    return dict(passes=passes, zero_clear=zero_clear, glyphs_ok=glyphs_ok,
                spearman=round(rho, 4), min_seed_spearman=round(min(seed_rhos), 4) if seed_rhos else None,
                nondecreasing_steps=rising, distinct_levels=distinct,
                median_target_value=[round(float(x), 6) for x in med], reviewed=False)


def render_ladder(recipe, seed, out, max_attempts=8):
    # An upstream colour error invalidates a whole paired ladder; redraw with a new seed.
    for attempt in range(max_attempts):
        actual_seed = seed + attempt * 1000003
        rows = []
        failed = False
        for level in range(11):
            d = out / recipe.name / f"seed_{actual_seed}"
            d.mkdir(parents=True, exist_ok=True)
            stem_name = f"level_{level:02d}"
            result = run_case(recipe.library, actual_seed, d, intercept=True,
                              params=recipe.params(level), stem_name=stem_name)
            stem = d / stem_name
            item = inspect_case(stem, recipe) if result["ok"] else dict(ok=False, reason=result.get("reason"))
            item.update(seed=actual_seed, requested_seed=seed, attempt=attempt, level=level,
                        recipe=recipe.name, library=recipe.library, feature=recipe.feature,
                        params=recipe.params(level), image=str(stem.with_suffix(".png")))
            rows.append(item)
            if not item["ok"]:
                failed = True
                break
        if not failed and recipe.metric == "difference":
            with Image.open(rows[0]["image"]) as im:
                baseline = np.asarray(im.convert("RGB"), dtype=np.int16)
            for item in rows:
                with Image.open(item["image"]) as im:
                    changed = np.asarray(im.convert("RGB"), dtype=np.int16)
                if changed.shape != baseline.shape:
                    item.update(ok=False, reason="image_size_changed_with_control")
                    failed = True
                    break
                item["target_value"] = round(float(np.mean(np.abs(changed - baseline)) / 255), 6)
        if not failed and recipe.metric == "glyph_displacement":
            baseline = decode(Path(rows[0]["image"]).with_suffix(""))["boxes"]
            base_y = np.array([(b["y0"] + b["y1"]) / 2 for b in baseline])
            for item in rows:
                d = decode(Path(item["image"]).with_suffix(""))
                boxes = d["boxes"]
                height, width = d["opaque"].shape
                if (len(boxes) != len(base_y) or any(b is None or b["x0"] < 1 or b["y0"] < 1
                    or b["x1"] >= width - 1 or b["y1"] >= height - 1 for b in boxes)):
                    item.update(ok=False, reason="jitter_glyph_clipped_or_missing")
                    failed = True
                    break
                y = np.array([(b["y0"] + b["y1"]) / 2 for b in boxes])
                item["target_value"] = round(float(np.mean(np.abs(y - base_y)) / height), 6)
        if not failed and recipe.metric == "xcompression":
            base_boxes = decode(Path(rows[0]["image"]).with_suffix(""))["boxes"]
            base_centers = [(b["x0"] + b["x1"]) / 2 for b in base_boxes]
            base_span = max(base_centers) - min(base_centers)
            for item in rows:
                boxes = decode(Path(item["image"]).with_suffix(""))["boxes"]
                if len(boxes) != len(base_boxes) or any(b is None for b in boxes):
                    item.update(ok=False, reason="spacing_glyph_missing")
                    failed = True
                    break
                centers = [(b["x0"] + b["x1"]) / 2 for b in boxes]
                span = max(centers) - min(centers)
                item["target_value"] = round(max(0.0, (base_span - span) / max(1.0, base_span)), 6)
        if not failed:
            return rows, attempt
    return rows, max_attempts


def calibrate(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scale={"absent": 0, "present": [1, 10],
                         "ten": "effect is visibly strong enough to make answer text hard to read"},
                  recipes={}, attempts=[])
    names = args.recipes or list(RECIPES)
    for name in names:
        recipe = RECIPES[name]
        if recipe.library not in active_libraries():
            raise ValueError(f"excluded library: {recipe.library}")
        rows = []
        for i in range(args.seeds):
            batch, retries = render_ladder(recipe, args.first_seed + i, out)
            rows.extend(batch)
            report["attempts"].append(dict(recipe=name, requested_seed=args.first_seed + i,
                                           redraws=retries, complete=len(batch) == 11 and all(r["ok"] for r in batch)))
        metric = monotonic_report(rows)
        if len(rows) != 11 * args.seeds or not all(r.get("ok") for r in rows):
            metric["passes"] = False
            metric["reason"] = "incomplete_ladder"
        report["recipes"][name] = dict(library=recipe.library, feature=recipe.feature,
                                       control=recipe.control, maximum=recipe.maximum,
                                       metric=metric, rows=rows)
        sheet(recipe, rows, out)
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(name, metric, flush=True)
    return report


def build(args):
    calibrations = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.calibration]
    entries = {}
    for cal in calibrations:
        entries.update(cal["recipes"])
    taxonomy = json.loads((ROOT / "docs/feature_bank/FEATURE_BUCKETS_02.json").read_text(encoding="utf-8"))
    heads = [h["head"] for h in taxonomy["heads"]]
    assert len(heads) == 46 and ORDINAL_HEADS <= set(heads)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "manifest.jsonl"
    summary = {}
    with manifest.open("w", encoding="utf-8") as fh:
        for name, entry in entries.items():
            if args.recipes and name not in args.recipes:
                continue
            if name not in RECIPES:
                continue  # historical calibration superseded by a corrected recipe
            recipe = RECIPES[name]
            if not entry["metric"].get("passes") or not entry["metric"].get("reviewed"):
                continue
            if recipe.feature not in ORDINAL_HEADS:
                continue
            if (entry["library"], entry["feature"], entry["control"], entry["maximum"]) != (
                    recipe.library, recipe.feature, recipe.control, recipe.maximum):
                raise ValueError(f"recipe changed since calibration: {name}")
            if any(r.get("params") != recipe.params(r["level"]) for r in entry["rows"] if r.get("ok")):
                raise ValueError(f"recipe parameters changed since calibration: {name}")
            endpoint_anchor = float(entry["metric"]["median_target_value"][10])
            counts = {k: 0 for k in range(11)}
            ladders = redraws = failures = rejected_trend = rejected_endpoint = 0
            limit = args.max_ladders_per_recipe or max(20, args.cases_per_level * 20)
            for i in range(limit):
                if all(n >= args.cases_per_level for n in counts.values()):
                    break
                ladders += 1
                rows, retries = render_ladder(recipe, args.first_seed + i, out)
                redraws += retries
                if len(rows) != 11 or not all(r["ok"] and r["glyph_ok"] for r in rows):
                    failures += 1
                    continue
                trend = monotonic_report(rows)
                if not trend["passes"]:
                    rejected_trend += 1
                    continue
                endpoint = rows[-1]["target_value"]
                # A seed whose level ten is far weaker/stronger than the reviewed anchor
                # cannot inherit that visual rating, even if its own ladder rises.
                if not (.75 * endpoint_anchor <= endpoint <= 1.4 * endpoint_anchor):
                    rejected_endpoint += 1
                    continue
                for r in rows:
                    targets = {h: None for h in heads}
                    verified_level = r["level"]
                    if counts[verified_level] >= args.cases_per_level:
                        continue
                    targets[recipe.feature] = verified_level
                    fh.write(json.dumps({**r, "redraws": retries, "verified_level": verified_level,
                                         "endpoint_value": endpoint, "endpoint_anchor": endpoint_anchor,
                                         "intensity_targets": targets}) + "\n")
                    counts[verified_level] += 1
            summary[name] = dict(counts=counts, complete=all(n >= args.cases_per_level for n in counts.values()),
                                 ladders=ladders, redraws=redraws, failed_ladders=failures,
                                 rejected_trend=rejected_trend, rejected_endpoint=rejected_endpoint)
            fh.flush()
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("manifest", manifest)
    print("summary", out / "summary.json")
    if not summary:
        raise RuntimeError("no reviewed and passing recipes were supplied")
    if not all(v["complete"] for v in summary.values()):
        raise RuntimeError("some observed intensity levels did not reach quota; see summary.json")


def verify(args):
    corpus = Path(args.corpus)
    rows = [json.loads(line) for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((corpus / "summary.json").read_text(encoding="utf-8"))
    heads = {h["head"] for h in json.loads(
        (ROOT / "docs/feature_bank/FEATURE_BUCKETS_02.json").read_text(encoding="utf-8"))["heads"]}
    counts = {}
    for row in rows:
        name, level, feature = row["recipe"], row["verified_level"], row["feature"]
        assert name in RECIPES and RECIPES[name].feature == feature
        assert 0 <= level <= 10 and row["level"] == level and row["glyph_ok"]
        assert row["params"] == RECIPES[name].params(level)
        assert set(row["intensity_targets"]) == heads
        assert sum(v is not None for v in row["intensity_targets"].values()) == 1
        assert row["intensity_targets"][feature] == level
        assert hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() == row["sha256"]
        counts[(name, level)] = counts.get((name, level), 0) + 1
    assert summary and all(info["complete"] for info in summary.values())
    assert all(counts.get((name, level)) == info["counts"][str(level)]
               for name, info in summary.items() for level in range(11))
    assert set(summary) == {name for name, _ in counts}
    print(f"verified {len(rows)} images, {len(summary)} recipes, {len(set(RECIPES[n].feature for n in summary))} heads")


def merge(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    combined = {}
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as dst:
        for source_index, source in enumerate(args.corpus):
            source = Path(source)
            summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
            omitted = set(args.exclude) | (set(args.exclude_first) if source_index == 0 else set())
            summary = {name: info for name, info in summary.items() if name not in omitted}
            overlap = set(combined) & set(summary)
            if overlap:
                raise ValueError(f"duplicate recipes across corpora: {sorted(overlap)}")
            combined.update(summary)
            with (source / "manifest.jsonl").open(encoding="utf-8") as src:
                for line in src:
                    if json.loads(line)["recipe"] in summary:
                        dst.write(line)
    (out / "summary.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    verify(argparse.Namespace(corpus=str(out)))


def pack(args):
    source = Path(args.corpus)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    suffixes = (".png", ".json", ".calls.json", ".labels.png", ".labels_translucent.png")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            old_stem = Path(row["image"]).with_suffix("")
            new_stem = out / row["recipe"] / f"seed_{row['seed']}" / f"level_{row['level']:02d}"
            new_stem.parent.mkdir(parents=True, exist_ok=True)
            for suffix in suffixes:
                old, new = Path(str(old_stem) + suffix), Path(str(new_stem) + suffix)
                if not old.exists():
                    raise FileNotFoundError(old)
                try:
                    os.link(old, new)
                except OSError:
                    shutil.copy2(old, new)
            row["source_image"] = row["image"]
            row["image"] = str(new_stem.with_suffix(".png"))
            fh.write(json.dumps(row) + "\n")
    shutil.copy2(source / "summary.json", out / "summary.json")
    verify(argparse.Namespace(corpus=str(out)))


def review(args):
    path = Path(args.calibration)
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in args.recipes:
        if not data["recipes"][name]["metric"]["passes"]:
            raise ValueError(f"metric gate failed: {name}")
        data["recipes"][name]["metric"].update(reviewed=True, review_note=args.note)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("reviewed", ", ".join(args.recipes))


def revoke(args):
    path = Path(args.calibration)
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in args.recipes:
        data["recipes"][name]["metric"].update(reviewed=False, review_note=args.note)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("revoked", ", ".join(args.recipes))


def reassess(args):
    path = Path(args.calibration)
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, entry in data["recipes"].items():
        old = entry["metric"]
        metric = monotonic_report(entry["rows"])
        metric["reviewed"] = bool(old.get("reviewed")) and metric["passes"]
        if old.get("review_note") and metric["reviewed"]:
            metric["review_note"] = old["review_note"]
        entry["metric"] = metric
        print(name, metric["passes"], metric.get("distinct_levels"), metric.get("min_seed_spearman"))
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def gallery(args):
    corpus = Path(args.corpus)
    rows = [json.loads(line) for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for name in sorted({r["recipe"] for r in rows}):
        selected = [r for r in rows if r["recipe"] == name]
        sheet(RECIPES[name], selected, corpus)
        print("sheet", corpus / f"{name}_sheet.png")


def inventory(args):
    taxonomy = json.loads((ROOT / "docs/feature_bank/FEATURE_BUCKETS_02.json").read_text(encoding="utf-8"))
    approved = set()
    for path in args.calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        approved |= {name for name, entry in data["recipes"].items()
                     if name in RECIPES and entry["metric"].get("passes") and entry["metric"].get("reviewed")}
    rows = []
    for h in taxonomy["heads"]:
        head = h["head"]
        available = [lib for lib, mode in h["libraries"].items()
                     if mode == "param" and lib not in taxonomy["withheld"]]
        reviewed = [name for name in approved if RECIPES[name].feature == head]
        rows.append(dict(head=head, bucket=h["bucket"], ordinal=head in ORDINAL_HEADS,
                         controllable_training_libraries=sorted(available),
                         reviewed_recipes=sorted(reviewed),
                         status="reviewed" if reviewed else "unmapped" if head in ORDINAL_HEADS else "categorical"))
    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("ordinal", sum(r["ordinal"] for r in rows), "reviewed", sum(r["status"] == "reviewed" for r in rows),
          "unmapped", sum(r["status"] == "unmapped" for r in rows), "inventory", out)


def export_map(args):
    selected = {}
    for path in args.calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, entry in data["recipes"].items():
            if name in RECIPES and entry["metric"].get("passes") and entry["metric"].get("reviewed"):
                selected[name] = (str(path), entry)
    result = {}
    for name, (path, entry) in sorted(selected.items()):
        recipe = RECIPES[name]
        if any(r.get("params") != recipe.params(r["level"]) for r in entry["rows"] if r.get("ok")):
            raise ValueError(f"recipe changed since calibration: {name}")
        result[name] = dict(library=recipe.library, feature=recipe.feature, control=recipe.control,
                            paired_control=recipe.paired_control or None,
                            metric=recipe.metric, calibration=path,
                            calibration_endpoint=entry["metric"]["median_target_value"][10],
                            review_note=entry["metric"].get("review_note"),
                            params_by_level={str(k): recipe.params(k) for k in range(11)})
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("mapped", len(result), "recipes", len({x["feature"] for x in result.values()}), "heads", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--recipes", nargs="*", choices=list(RECIPES))
    c.add_argument("--seeds", type=int, default=2)
    c.add_argument("--first-seed", type=int, default=14000)
    b = sub.add_parser("build")
    b.add_argument("--calibration", required=True, nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--cases-per-level", type=int, default=20)
    b.add_argument("--first-seed", type=int, default=20000)
    b.add_argument("--max-ladders-per-recipe", type=int)
    b.add_argument("--recipes", nargs="*", choices=list(RECIPES))
    r = sub.add_parser("review")
    r.add_argument("--calibration", required=True)
    r.add_argument("--recipes", nargs="+", required=True, choices=list(RECIPES))
    r.add_argument("--note", required=True)
    v = sub.add_parser("revoke")
    v.add_argument("--calibration", required=True)
    v.add_argument("--recipes", nargs="+", required=True)
    v.add_argument("--note", required=True)
    u = sub.add_parser("reassess")
    u.add_argument("--calibration", required=True)
    g = sub.add_parser("gallery")
    g.add_argument("--corpus", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--calibration", nargs="+", required=True)
    i.add_argument("--out", required=True)
    z = sub.add_parser("verify")
    z.add_argument("--corpus", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--corpus", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--exclude", nargs="*", default=[])
    m.add_argument("--exclude-first", nargs="*", default=[])
    e = sub.add_parser("export-map")
    e.add_argument("--calibration", nargs="+", required=True)
    e.add_argument("--out", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "calibrate":
        calibrate(args)
    elif args.command == "review":
        review(args)
    elif args.command == "revoke":
        revoke(args)
    elif args.command == "reassess":
        reassess(args)
    elif args.command == "gallery":
        gallery(args)
    elif args.command == "inventory":
        inventory(args)
    elif args.command == "verify":
        verify(args)
    elif args.command == "merge":
        merge(args)
    elif args.command == "export-map":
        export_map(args)
    elif args.command == "pack":
        pack(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
