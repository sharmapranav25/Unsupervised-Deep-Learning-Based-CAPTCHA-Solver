"""Render executor: one settings dictionary -> one labelled image (single library or chain).

Settings (JSON-able):
{
  "stages": [                       # A1 .. Ak (text off, each feeds the next as background), then B (text)
    {"source": "php", "library": "kolanos_kohana-captcha", "seed": 11, "params": {...}, "disable": ["text"]},
    {"source": "python_shapes", "seed": 12, "params": {...}},          # Python sources: see py_sources.py
    {"source": "php", "library": "Gregwar_Captcha", "seed": 13, "params": {"text": "ab3"}}
  ],
  "font_split": "train" | "heldout",
  "text_mode": "random" | "pronounceable",
  "recipe": null | {"name": "...", "level": k}       # intensity mode (single stage)
}
A single-library image is a one-stage chain. Background stages always run with text disabled;
the last stage draws the text. Each stage after the first receives the previous stage's PNG as
its background through the library's own background input, and its label canvases are seeded
from the previous stage's label maps (ids continue after the previous stage's ids). The merged
call log records the stage and library of every call.

python -m feature_bank.render_chain --settings settings.json --out runs/<new_dir> [--stem name]
"""
import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from feature_bank.bank02_smoke import ROOT, run_case
from feature_bank.label_maps import read_ids

# Libraries whose own background-image input is used as a chain stage input (verified 2026-10-02).
BG_LIBRARIES = {
    "Gregwar_Captcha": "setBackgroundImages: image used at its own size (canvas = background size)",
    "yasirmturk_simple-php-captcha": "backgrounds config: canvas = background size",
    "LordDashMe_php-simple-captcha": "backgrounds config (relative to its folder): canvas = background size",
    "dapphp_securimage": "show(background_image): imagecopyresized to Securimage's own canvas size",
    "kolanos_kohana-captcha": "background file: upstream only copies it when sizes differ, so its own size is kept (resampled)",
    "top-think_think-captcha": "reads .jpg only: given a quality-95 JPEG copy, resampled to its own canvas",
}
# Libraries where the chain sets the canvas to the background's size (no resampling).
NATIVE_SIZE = {"Gregwar_Captcha": ("width", "height"), "dapphp_securimage": ("width", "height")}
EXTRA_BG_PARAMS = {"top-think_think-captcha": {"use_img_bg": True}}


def php_stage(stage, out_dir, stem, font_split, prev=None):
    params = dict(stage.get("params") or {})
    options = {"font_split": font_split}
    if prev:
        options.update(bg_image=str(prev["png"]), chain_labels=str(prev["stem"]), id_offset=prev["next_id"])
        if stage["library"] in NATIVE_SIZE:
            w, h = NATIVE_SIZE[stage["library"]]
            params.setdefault(w, prev["size"][0])
            params.setdefault(h, prev["size"][1])
        for k, v in EXTRA_BG_PARAMS.get(stage["library"], {}).items():
            params.setdefault(k, v)
    r = run_case(stage["library"], int(stage["seed"]), out_dir, intercept=True, disable=stage.get("disable", []),
                 params=params, stem_name=stem, options=options)
    s = out_dir / stem
    if not r["ok"] and not (r.get("reason") == "blank image" and "text" in stage.get("disable", [])):
        return None, r.get("reason") or "render failed"
    if not Path(str(s) + ".calls.json").exists():
        return None, "no label maps"
    meta = json.loads(s.with_suffix(".json").read_text(encoding="utf-8"))
    log = json.loads(Path(str(s) + ".calls.json").read_text(encoding="utf-8"))
    if prev and not log["meta"].get("chain_seeded"):
        return None, "background not loaded through the chain input"
    return dict(stem=s, png=s.with_suffix(".png"), meta=meta, log=log, size=[meta["width"], meta["height"]],
                next_id=int(log["meta"].get("next_id") or (max([c["id"] for c in log["calls"]] or [0]) + 1))), None


def py_stage(stage, out_dir, stem, font_split, prev=None):
    from feature_bank import py_sources
    s = out_dir / stem
    r = py_sources.render(stage["source"], int(stage["seed"]), dict(stage.get("params") or {}), s,
                          text_on="text" not in stage.get("disable", []), font_split=font_split,
                          bg=prev)
    if not r.get("ok"):
        return None, r.get("reason") or "python render failed"
    meta = json.loads(s.with_suffix(".json").read_text(encoding="utf-8"))
    log = json.loads(Path(str(s) + ".calls.json").read_text(encoding="utf-8"))
    return dict(stem=s, png=s.with_suffix(".png"), meta=meta, log=log, size=[meta["width"], meta["height"]],
                next_id=int(log["meta"]["next_id"])), None


def paired_value(stage, recipe, work, font_split):
    """Recipes measured against their own level-0 twin (same seed): vertical jitter, tight spacing."""
    from feature_bank.intensity_corpus import RECIPES
    from feature_bank.label_maps import decode
    r = RECIPES[recipe["name"]]
    if r.metric not in ("glyph_displacement", "xcompression"):
        return None
    zero = dict(stage, params={**stage["params"], **r.params(0)})
    out, err = php_stage(zero, work, "twin0", font_split)
    if err:
        return None
    a, b = decode(out["stem"]), decode(work / "s0")
    if len(a["boxes"]) != len(b["boxes"]) or any(x is None for x in a["boxes"] + b["boxes"]):
        return None
    if r.metric == "glyph_displacement":
        y0 = np.array([(x["y0"] + x["y1"]) / 2 for x in a["boxes"]])
        y1 = np.array([(x["y0"] + x["y1"]) / 2 for x in b["boxes"]])
        return float(np.mean(np.abs(y1 - y0)) / b["opaque"].shape[0])
    c0 = [(x["x0"] + x["x1"]) / 2 for x in a["boxes"]]
    c1 = [(x["x0"] + x["x1"]) / 2 for x in b["boxes"]]
    s0, s1 = max(c0) - min(c0), max(c1) - min(c1)
    return max(0.0, (s0 - s1) / max(1.0, s0))


def render(settings, out_stem, keep_stages=False, allow_empty_stage=False):
    """Render settings to <out_stem>.{png,json,labels.npz,calls.json.gz}. Returns a result dict with
    ok/reason, labels (feature_bank.heads.label_image output) and per-stage records."""
    from feature_bank.heads import label_image
    out_stem = Path(out_stem).resolve()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    font_split = settings.get("font_split", "train")
    stages = settings["stages"]
    with tempfile.TemporaryDirectory(dir=str(out_stem.parent)) as td:
        work = Path(td)
        prev, done, merged, intended = None, [], [], {}
        for k, st in enumerate(stages):
            is_last = k == len(stages) - 1
            st = dict(st)
            if not is_last:
                st["disable"] = sorted(set(st.get("disable", [])) | {"text"})
            if prev and st["source"] == "php" and st["library"] not in BG_LIBRARIES:
                return dict(ok=False, reason=f"stage {k}: {st['library']} has no background input")
            fn = php_stage if st["source"] == "php" else py_stage
            rec, err = fn(st, work, f"s{k}", font_split, prev)
            if err:
                return dict(ok=False, reason=f"stage {k} ({st.get('library', st['source'])}): {err}")
            lib = st.get("library") or st["source"]
            for c in rec["log"]["calls"]:
                if c["id"] >= (prev["next_id"] if prev else 0):
                    merged.append(dict(c, stage=k, library=lib))
            done.append(dict(library=lib, source=st["source"], seed=st["seed"], meta=rec["meta"], size=rec["size"],
                             disable=st.get("disable", []), chain_seeded=rec["log"]["meta"].get("chain_seeded", 0)))
            if not is_last:
                intended[str(k)] = own_layers(rec, lib, prev["next_id"] if prev else 0)
                if not intended[str(k)] and not allow_empty_stage:
                    return dict(ok=False, reason=f"stage {k} ({lib}): no visible background layer")
            prev = rec
        # final image = last stage; write merged call log next to it
        last = prev
        final_log = dict(meta=dict(last["log"]["meta"], chain=[dict(library=d["library"], seed=d["seed"]) for d in done]),
                         calls=merged)
        for suffix in (".png", ".json", ".labels.png", ".labels_translucent.png", ".labels_classes.png"):
            shutil.copyfile(str(last["stem"]) + suffix, str(out_stem) + suffix)
        Path(str(out_stem) + ".calls.json").write_text(json.dumps(final_log), encoding="utf-8")
        recipe = settings.get("recipe")
        if recipe and len(stages) == 1 and stages[0]["source"] != "php":
            from feature_bank import python_recipes
            r = python_recipes.RECIPES[recipe["name"]]
            if r.metric.startswith("paired"):
                pv = python_recipes.twin_value(r, out_stem, stages[0], work)
                if pv is not None:
                    done[0]["meta"]["_paired_value"] = {recipe["feature"]: pv}
        if recipe and len(stages) == 1 and stages[0]["source"] == "php":
            pv = paired_value(stages[0], recipe, work, font_split)
            if pv is not None:
                done[0]["meta"]["_paired_value"] = {recipe["feature"]: pv}
        try:
            lab = label_image(out_stem, stages=[dict(library=d["library"], meta=d["meta"]) for d in done],
                              text_mode=settings.get("text_mode", "random"))
        except Exception as e:  # noqa: BLE001 - reported as a failed render
            return dict(ok=False, reason=f"labelling failed: {e!r}"[:300])
        stage_layers = lab["layers"]["by_stage"]
        if keep_stages:
            for k in range(len(done)):
                for suffix in (".png", ".labels.png"):
                    shutil.copyfile(str(work / f"s{k}") + suffix, str(out_stem) + f".stage{k}{suffix}")
    return dict(ok=True, labels=lab, stages=done, stage_layers=stage_layers, intended_layers=intended)


def own_layers(rec, lib, first_id):
    """Layer heads a background stage visibly drew in its own output (>= tau): its intended layers."""
    from feature_bank.heads import call_head, flat_params, thresholds, LAYER_HEADS
    opaque = read_ids(str(rec["stem"]) + ".labels.png")
    trans = read_ids(str(rec["stem"]) + ".labels_translucent.png")
    ids, cnt = np.unique(np.concatenate([opaque.ravel(), trans.ravel()]), return_counts=True)
    vis = dict(zip(ids.tolist(), cnt.tolist()))
    params = flat_params(rec["meta"])
    px = {}
    for c in rec["log"]["calls"]:
        if c["id"] < first_id or c.get("class") in ("text", "filter"):
            continue
        h = call_head(c, lib, params)
        if h in LAYER_HEADS:
            px[h] = px.get(h, 0) + int(vis.get(c["id"], 0))
    tau = thresholds()
    return {h: v for h, v in px.items() if v >= tau.get(h, 12)}


def compress(stem):
    """Corpus storage: label maps -> one compressed .labels.npz, call log -> .calls.json.gz."""
    stem = Path(stem)
    maps = {k: read_ids(str(stem) + suf) for k, suf in
            (("opaque", ".labels.png"), ("translucent", ".labels_translucent.png"), ("classes", ".labels_classes.png"))}
    np.savez_compressed(str(stem) + ".labels.npz", **{k: v.astype(np.uint32) for k, v in maps.items()})
    raw = Path(str(stem) + ".calls.json").read_bytes()
    Path(str(stem) + ".calls.json.gz").write_bytes(gzip.compress(raw, 6))
    for suf in (".labels.png", ".labels_translucent.png", ".labels_classes.png", ".calls.json"):
        Path(str(stem) + suf).unlink()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, help="JSON file with one settings dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default="render")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path(a.settings).read_text(encoding="utf-8"))
    r = render(settings, out / a.stem, keep_stages=True)
    print(json.dumps(dict(ok=r["ok"], reason=r.get("reason"),
                          heads={h: v for h, v in r.get("labels", {}).get("heads", {}).items() if v["present"] == 1},
                          clip=r.get("labels", {}).get("clip")), default=str, indent=1))


if __name__ == "__main__":
    main()
