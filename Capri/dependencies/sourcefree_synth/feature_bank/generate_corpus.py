"""Hybrid v3 corpus generator: process pool, sharded, resumable, with fixed hashed splits and an audit.

  python -m feature_bank.generate_corpus --config configs/corpus_hybrid_v3.yaml --out runs/<new_dir> [--train 100000]
  python -m feature_bank.generate_corpus --audit-only --out runs/<dir>

Every image is a deterministic function of (config seed, split, index, attempt), so a restart
re-reads the shard manifests and only renders missing indices. Per image the shard holds
<id>.png, <id>.json (stage case metadata), <id>.labels.npz (opaque / translucent / class maps),
<id>.calls.json.gz (merged call log), and one manifest row with source, chain, seeds, parameters,
fonts per character, all 46 head labels (null = masked), text targets, clipping and image hash.

Splits: train (configurable), steering, selection, calibration, seen_test, generalisation (v2; legacy: locked_test). Fixed sets: no
Python-source images; locked_test also holds the held-out-library and held-out-font strata;
those libraries/fonts never appear anywhere else). Rejections (upstream crashes, clipping, buried
chain layers, off-scale intensity) are redrawn with the next attempt and logged.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from feature_bank.bank02_smoke import ROOT, active_libraries

# Holdout libraries v2 (user, 2026-10-02): yiisoft_yii-captcha and hnhx_captcha moved into training; these two are the
# generalisation libraries (generalisation split only). v1 held out 4 (+ yii, hnhx).
HOLDOUT_LIBRARIES = ["NikolaiT_SVG-Captcha", "sameer-shelavale_multi-captcha"]
FIXED_SPLITS = ["steering", "selection", "calibration", "locked_test", "seen_test", "generalisation"]
SPLIT_ORDER = ["steering", "selection", "calibration", "locked_test", "seen_test", "generalisation", "train"]
FONT_SOURCE_LIBS = {"BSN4_P-Captcha", "arsovskidev_php-captcha", "eusonlito_captcha", "igoshev_laravel-captcha",
                    "innocead_CaptchaBundle", "josecl_cool-php-captcha", "lifei6671_php-captcha", "lovefc_captcha",
                    "mortezaashrafi_filament-shield-captcha", "omranjamal_RealCaptcha", "php-quickorm_Captcha",
                    "takayukister_really-simple-captcha", "top-think_think-captcha", "zerosdev_laravel-captcha"}
ALWAYS_BANK_FONT = {"dapphp_securimage", "laminas_laminas-captcha", "mobicms_captcha", "yasirmturk_simple-php-captcha"}
CASELESS_FONT = {"innocead_CaptchaBundle", "omranjamal_RealCaptcha", "BSN4_P-Captcha"}
VOWELS = set("aeiouAEIOU")


# ------------------------------------------------------------------ sampling

def alphabets():
    return json.loads((ROOT / "resources/feature_bank_02/alphabets.json").read_text(encoding="utf-8"))


def training_libraries(cfg):
    return [l for l in active_libraries() if l not in HOLDOUT_LIBRARIES and l not in set(cfg["libraries"].get("exclude") or [])]


def approved_recipes():
    out = {}
    for p in sorted((ROOT / "docs/feature_bank").glob("INTENSITY_CONTROL_MAP_*.json")):
        out.update({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                    if isinstance(v, dict) and "feature" in v})  # recipe entries only
    return out


# Middle chain stages (text off) must add a layer of their own: settings that switch it on.
MIDDLE_STAGE = {"Gregwar_Captcha": {"max_behind_lines": 6, "max_front_lines": 4},
                "dapphp_securimage": {}, "kolanos_kohana-captcha": {},
                "top-think_think-captcha": {"use_noise": True, "use_curve": True},
                "LordDashMe_php-simple-captcha": {}}


def layer_libraries(libs):
    """Libraries that draw a background or interference layer (first chain stage, text off)."""
    from feature_bank.heads import TAXONOMY, LAYER_HEADS
    return sorted(l for l in libs if any(h in LAYER_HEADS and h != "glyph_shadow" for h in TAXONOMY["libraries"].get(l, {})))


def pick(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[float(weights[k]) for k in keys])[0]


def make_text(rng, lib, cfg, params):
    """Length uniform in [lo, hi] over the text library's alphabet; optional CV word."""
    lo, hi = cfg["text"]["length"]
    n = rng.randint(lo, hi)
    if lib == "sibche2013_pure-captcha":
        ct = params.setdefault("char_type", rng.choice(["lettersB", "lettersS", "numbers"]))
        alpha = {"lettersB": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "lettersS": "abcdefghijklmnopqrstuvwxyz", "numbers": "0123456789"}[ct]
    else:
        alpha = alphabets()[lib]["alphabet"]
    if lib == "NikolaiT_SVG-Captcha":  # upstream stores glyphs in a map: unique characters only
        return "".join(rng.sample(alpha, n)), "random"
    if rng.random() < cfg["text"]["pronounceable_fraction"]:
        for case in (str.lower, str.upper):
            vs = sorted({c for c in alpha if c in VOWELS and case(c) == c})
            cs = sorted({c for c in alpha if c.isalpha() and c not in VOWELS and case(c) == c})
            if len(vs) >= 2 and len(cs) >= 4:
                start = rng.random() < 0.5
                return "".join(rng.choice(cs if (i % 2 == 0) == start else vs) for i in range(n)), "pronounceable"
    return "".join(rng.choice(alpha) for _ in range(n)), "random"


def font_params(rng, lib, cfg, font_split):
    """Font-source choice: held-out stratum forces bank fonts (drawn from the held-out split)."""
    p = {}
    if lib in FONT_SOURCE_LIBS:
        if font_split == "heldout" or rng.random() < cfg["fonts"]["bank_font_probability"]:
            p["font_source"] = "bank"
        else:
            p["font_source"] = "library"
    return p


def stage_seed(rng):
    return rng.randint(1, 2 ** 31 - 2)


def sample_settings(cfg, split, index, attempt, kind=None, stratum="seen", slot=None):
    """Deterministic settings for (split, index, attempt). slot: epoch-batch slot (base kind or targeted spec)."""
    if slot and slot.get("target"):
        fb = int(cfg.get("curriculum", {}).get("target_fallback_attempts", 3))
        if attempt < int(cfg["max_attempts"]) - fb:
            from feature_bank.targeting import targeted_settings
            return targeted_settings(cfg, split, index, attempt, slot["target"], None)
        kind = "python" if slot["target"].get("python") else "library_single"  # fallback: target dropped
    elif slot:
        kind = slot["kind"]
    rng = random.Random(f"{cfg['seed']}|{split}|{index}|{attempt}")
    libs = training_libraries(cfg)
    if kind is None:
        mix = dict(cfg["mix"])
        if split in FIXED_SPLITS:
            mix.pop("python", None)
        kind = pick(rng, mix)
    font_split = "heldout" if stratum == "heldout_font" else "train"
    s = dict(kind=kind, stratum=stratum, font_split=font_split, text_mode="random", recipe=None, source="library")
    if stratum == "holdout_library":
        lib = rng.choice(HOLDOUT_LIBRARIES)
        params = font_params(rng, lib, cfg, "train")
        text, mode = make_text(rng, lib, cfg, params)
        params["text"] = text
        s.update(kind="library_single", text_mode=mode,
                 stages=[dict(source="php", library=lib, seed=stage_seed(rng), params=params)])
        return s
    if stratum == "heldout_font":
        kind = "library_single" if kind not in ("chain",) else "chain"
        s["kind"] = kind
        font_libs = sorted((FONT_SOURCE_LIBS | ALWAYS_BANK_FONT) & set(libs))
    if kind == "intensity":
        recs = approved_recipes()
        name = rng.choice(sorted(n for n, r in recs.items() if r.get("source", "php") == "php"))
        r = recs[name]
        level = rng.choice(cfg["intensity"]["levels"])
        lib = r["library"]
        if r.get("source", "php") != "php":
            from feature_bank import py_sources
            st = py_sources.recipe_stage(name, r, level, rng, cfg)
            s.update(source=r["source"], recipe=dict(name=name, level=level, feature=r["feature"]), stages=[st],
                     text_mode=st.get("_text_mode", "random"))
            return s
        params = dict(r["params_by_level"][str(level)])
        text, mode = make_text(rng, lib, cfg, params)
        params["text"] = text
        s.update(recipe=dict(name=name, level=level, feature=r["feature"]), text_mode=mode,
                 stages=[dict(source="php", library=lib, seed=stage_seed(rng), params=params)])
        return s
    if kind == "python":
        from feature_bank import py_sources
        src = pick(rng, cfg["python_sources"])
        if src == "python_recipe":  # a reviewed Python 0-10 ladder (counts against the Python cap)
            recs = {n: r for n, r in approved_recipes().items() if r.get("source", "php") != "php"}
            name = rng.choice(sorted(recs))
            r = recs[name]
            level = rng.choice(cfg["intensity"]["levels"])
            st = py_sources.recipe_stage(name, r, level, rng, cfg)
            s.update(kind="python", source=r["source"], recipe=dict(name=name, level=level, feature=r["feature"]), stages=[st])
            return s
        s["source"] = src
        chained = rng.random() < cfg.get("python_chain_fraction", 0.0)
        return py_sources.sample(rng, src, s, cfg, libs, chained)
    if kind == "chain":
        from feature_bank.render_chain import BG_LIBRARIES
        depth = int(pick(rng, {int(k): v for k, v in cfg["chain_depth"].items()}))
        tops = sorted(set(BG_LIBRARIES) & set(libs))
        stages = [dict(source="php", library=rng.choice(layer_libraries(libs)), seed=stage_seed(rng), params={})]
        for _ in range(depth - 1):
            mid = rng.choice(sorted(set(MIDDLE_STAGE) & set(tops)))
            stages.append(dict(source="php", library=mid, seed=stage_seed(rng), params=dict(MIDDLE_STAGE[mid])))
        top_pool = sorted(set(tops) & set(font_libs)) if stratum == "heldout_font" else tops
        top = rng.choice(top_pool)
        params = font_params(rng, top, cfg, font_split)
        text, mode = make_text(rng, top, cfg, params)
        params["text"] = text
        stages.append(dict(source="php", library=top, seed=stage_seed(rng), params=params))
        s.update(stages=stages, text_mode=mode)
        return s
    # library_single
    pool = font_libs if stratum == "heldout_font" else libs
    lib = rng.choice(pool)
    params = font_params(rng, lib, cfg, font_split)
    text, mode = make_text(rng, lib, cfg, params)
    params["text"] = text
    s.update(kind="library_single", text_mode=mode, stages=[dict(source="php", library=lib, seed=stage_seed(rng), params=params)])
    return s


def strata_of(cfg, split):
    """Strata of a fixed split: cfg["strata"][split], or the legacy locked_test_strata (v1 config)."""
    st = (cfg.get("strata") or {}).get(split)
    if st is None and split == "locked_test":
        st = cfg.get("locked_test_strata")
    return st


def plan(cfg, train_n):
    """[(split, index, stratum)] for every image."""
    sizes = dict(cfg["sizes"], train=train_n)
    items = []
    for split in SPLIT_ORDER:
        n = int(sizes.get(split, 0))
        st = strata_of(cfg, split)
        if st:
            counts = {k: int(round(n * float(st.get(k, 0)))) for k in ("holdout_library", "heldout_font")}
            counts["seen"] = n - counts["holdout_library"] - counts["heldout_font"]
            i = 0
            for stratum in ("seen", "holdout_library", "heldout_font"):
                for _ in range(counts[stratum]):
                    items.append((split, i, stratum))
                    i += 1
        else:
            items += [(split, i, "seen") for i in range(n)]
    return items


# ------------------------------------------------------------------ one image (worker)

def chain_desc(settings):
    return " > ".join(st.get("library", st["source"]) for st in settings["stages"])


def accept(settings, r, cfg):
    """Reasons to reject a render (empty = accept)."""
    from feature_bank.heads import thresholds
    lab = r["labels"]
    why = []
    if not lab["clip"]["ok"]:
        why.append("clip: " + "; ".join(lab["clip"]["reasons"])[:160])
    rec = settings.get("recipe")
    if rec:
        h = lab["heads"][rec["feature"]]
        if rec["level"] == 0 and h["present"] not in (0,):
            why.append(f"level 0 but {rec['feature']} present={h['present']}")
        elif rec["level"] > 0:
            if h["present"] != 1:
                why.append(f"intended {rec['feature']} not visible (present={h['present']})")
            elif h.get("level") is None:
                why.append(f"{rec['feature']} level not measurable on the image")
            elif abs(h["level"] - rec["level"]) > cfg["intensity"]["max_level_error"]:
                why.append(f"{rec['feature']} measured level {h['level']} vs requested {rec['level']}")
    tgt = settings.get("target")
    if tgt:
        want = 1 if tgt["polarity"] == "pos" else 0
        got = lab["heads"][tgt["head"]]["present"]
        if got != want:
            why.append(f"target {tgt['head']} {tgt['polarity']} not met (present={got})")
    # chains: every intended layer of every stage must survive to the final image
    tau = thresholds()
    for k, layers in (r.get("intended_layers") or {}).items():
        for h, own in layers.items():
            fin = (lab["layers"]["by_stage"].get(k) or {}).get(h, 0)
            if fin < tau.get(h, 12):
                why.append(f"stage {k} {h} buried ({own}px -> {fin}px)")
    return why


def make_image(cfg, split, index, stratum, shard_dir, slot=None):
    from feature_bank.render_chain import render, compress, sha256
    img_id = f"{split}_{index:07d}"
    stem = Path(shard_dir) / img_id
    tries = []
    for attempt in range(int(cfg["max_attempts"])):
        settings = sample_settings(cfg, split, index, attempt, stratum=stratum, slot=slot)
        if attempt and stratum == "seen" and tries and tries[0]["kind"] != settings["kind"]:
            pass  # kind is re-drawn per attempt; recorded in the attempt log
        try:
            r = render(settings, stem)
        except Exception as e:  # noqa: BLE001
            r = dict(ok=False, reason="exception " + repr(e)[:200] + " " + traceback.format_exc()[-300:])
        if not r["ok"]:
            tries.append(dict(attempt=attempt, kind=settings["kind"], what=chain_desc(settings), reason=r["reason"][:300]))
            continue
        why = accept(settings, r, cfg)
        if why:
            tries.append(dict(attempt=attempt, kind=settings["kind"], what=chain_desc(settings), reason="; ".join(why)[:300]))
            for f in Path(shard_dir).glob(img_id + ".*"):
                f.unlink()
            continue
        compress(stem)
        lab = r["labels"]
        fonts = lab["text"]["fonts_per_char"]
        row = dict(
            id=img_id, split=split, index=index, stratum=stratum, kind=settings["kind"], source=settings["source"],
            stem=str(stem.relative_to(Path(shard_dir).parent.parent)).replace("\\", "/"),
            chain_depth=len(settings["stages"]) - 1, chain_desc=chain_desc(settings),
            chain=[dict(source=st["source"], library=st.get("library"), seed=st["seed"], disable=d.get("disable"),
                        params=param_values(d["meta"]), size=d["size"])
                   for st, d in zip(settings["stages"], r["stages"])],
            composited=False, font_split=settings["font_split"], text_mode=settings["text_mode"],
            recipe=settings.get("recipe"),
            fonts_per_char=fonts,
            heads={h: dict(present=v["present"], level=v.get("level")) for h, v in lab["heads"].items()},
            measures={h: {k: v[k] for k in ("value", "px", "level_source", "level_value") if v.get(k) is not None}
                      for h, v in lab["heads"].items() if any(v.get(k) is not None for k in ("value", "px"))},
            text=lab["text"], clip=lab["clip"], glyph_ids=lab["glyph_ids"],
            layers=lab["layers"], size=r["stages"][-1]["size"],
            sha256=sha256(stem.with_suffix(".png")), attempts=attempt + 1, rejected=tries,
            slot=slot, target=settings.get("target"),
            target_met=(None if not (slot and slot.get("target")) else bool(settings.get("target"))))
        return row, tries
    return None, tries


def param_values(meta):
    pc = meta.get("param_control")
    out = {k: v.get("value") for k, v in pc.items()} if isinstance(pc, dict) else {}
    if isinstance(meta.get("params"), dict):
        for k, v in meta["params"].items():
            out.setdefault(k, v)
    return out


def worker(args):
    cfg, split, index, stratum, shard_dir, slot = args
    t = time.time()
    try:
        row, tries = make_image(cfg, split, index, stratum, shard_dir, slot)
    except Exception:  # noqa: BLE001
        row, tries = None, [dict(reason="worker crash " + traceback.format_exc()[-600:])]
    return split, index, stratum, row, tries, time.time() - t


# ------------------------------------------------------------------ driver

def shard_of(split, index, size):
    return f"{split}-{index // size:04d}"


def read_done(out):
    done = {}
    for mf in (out / "shards").glob("*/manifest.jsonl"):
        for line in mf.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line after a crash: that image is re-rendered
            done[(r["split"], r["index"])] = r
    return done


TRAIN_ONLY_KEYS = ("curriculum", "gates", "freezing", "model", "train", "name")


def gen_cfg(cfg):
    """The generator-relevant part of a config (training-only keys may differ between runs sharing a batch)."""
    return {k: v for k, v in cfg.items() if k not in TRAIN_ONLY_KEYS}


def generate(cfg, out, train_n, limit_workers=None, items=None, tag=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    cfg_path = out / "config.yaml"
    if cfg_path.exists():
        old = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if old.get("_train_n") != train_n or gen_cfg({k: v for k, v in old.items() if k not in ("_train_n", "_epoch")}) != gen_cfg(cfg):
            raise SystemExit(f"{out} was started with a different config/--train; use a new --out")
    else:
        cfg_path.write_text(yaml.safe_dump(dict(cfg, _train_n=train_n, **({"_epoch": tag} if tag else {})), sort_keys=False),
                            encoding="utf-8")
    items = items if items is not None else [(sp, i, st, None) for sp, i, st in plan(cfg, train_n)]
    done = read_done(out)
    todo = [it for it in items if (it[0], it[1]) not in done]
    print(f"planned {len(items)}  done {len(done)}  to render {len(todo)}", flush=True)
    fail_log = (out / "failures.jsonl").open("a", encoding="utf-8")
    attempt_log = (out / "attempts.jsonl").open("a", encoding="utf-8")
    handles = {}
    t0 = time.time()
    n_ok = n_fail = 0
    workers = limit_workers or int(cfg["workers"])
    with ProcessPoolExecutor(workers) as ex:
        pending = set()
        it = iter(todo)

        def submit():
            try:
                split, index, stratum, slot = next(it)
            except StopIteration:
                return False
            sd = out / "shards" / shard_of(split, index, int(cfg["shard_size"]))
            sd.mkdir(exist_ok=True)
            pending.add(ex.submit(worker, (cfg, split, index, stratum, str(sd), slot)))
            return True
        for _ in range(workers * 3):
            if not submit():
                break
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            split, index, stratum, row, tries, dt = fut.result()
            for t in tries:
                attempt_log.write(json.dumps(dict(split=split, index=index, **t)) + "\n")
            if row is None:
                n_fail += 1
                fail_log.write(json.dumps(dict(split=split, index=index, stratum=stratum, tries=tries)) + "\n")
                fail_log.flush()
            else:
                n_ok += 1
                sh = shard_of(split, index, int(cfg["shard_size"]))
                if sh not in handles:
                    handles[sh] = (out / "shards" / sh / "manifest.jsonl").open("a", encoding="utf-8")
                handles[sh].write(json.dumps(row) + "\n")
                handles[sh].flush()
            submit()
            if (n_ok + n_fail) % 100 == 0:
                el = time.time() - t0
                print(f"{n_ok + n_fail}/{len(todo)} ok {n_ok} failed {n_fail}  {3600 * (n_ok + n_fail) / max(el, 1e-6):.0f} img/h "
                      f"elapsed {el / 60:.1f} min", flush=True)
    for h in handles.values():
        h.close()
    fail_log.close()
    attempt_log.close()
    el = time.time() - t0
    stats = dict(rendered=n_ok, failed=n_fail, seconds=round(el, 1), images_per_hour=round(3600 * n_ok / max(el, 1e-6)),
                 workers=workers)
    runs = json.loads((out / "throughput.json").read_text()) if (out / "throughput.json").exists() else []
    runs.append(stats)
    (out / "throughput.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print("throughput", stats, flush=True)
    return stats


# ------------------------------------------------------------------ audit

def audit(out):
    from feature_bank.heads import HEADS, ORDINAL_HEADS
    from feature_bank.font_split import load_bank, font_key
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    rows = list(read_done(out).values())
    held_fonts = {font_key(r) for r in load_bank() if r.get("split") == "heldout"}
    a = dict(images=len(rows))
    a["per_split"] = dict(Counter(r["split"] for r in rows))
    a["per_split_kind"] = {s: dict(Counter(r["kind"] for r in rows if r["split"] == s)) for s in a["per_split"]}
    a["per_split_source"] = {s: dict(Counter(r["source"] for r in rows if r["split"] == s)) for s in a["per_split"]}
    a["per_split_stratum"] = {s: dict(Counter(r["stratum"] for r in rows if r["split"] == s)) for s in a["per_split"]}
    a["python_share"] = {s: round(sum(r["source"].startswith("python") for r in rows if r["split"] == s) / max(1, n), 4)
                         for s, n in a["per_split"].items()}
    a["chain_depth"] = {s: dict(Counter(r["chain_depth"] for r in rows if r["split"] == s)) for s in a["per_split"]}
    a["libraries_text_stage"] = dict(Counter(r["chain"][-1]["library"] or r["source"] for r in rows))
    a["text_length"] = dict(sorted(Counter(len(r["text"]["text"]) for r in rows).items()))
    heads = {}
    for h in HEADS:
        c = Counter(str(r["heads"][h]["present"]) for r in rows)
        e = dict(present=c.get("1", 0), absent=c.get("0", 0), masked=c.get("None", 0))
        if h in ORDINAL_HEADS:
            e["levels"] = dict(sorted(Counter(r["heads"][h]["level"] for r in rows if r["heads"][h]["level"] is not None).items()))
        heads[h] = e
    a["heads"] = heads
    a["intensity_requests"] = {}
    for r in rows:
        if r.get("recipe"):
            k = r["recipe"]["name"]
            a["intensity_requests"].setdefault(k, Counter())[r["recipe"]["level"]] += 1
    a["intensity_requests"] = {k: dict(sorted(v.items())) for k, v in a["intensity_requests"].items()}
    # duplicates
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r["id"])
    a["duplicate_hashes"] = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    # holdout leaks
    leaks = []
    for r in rows:
        libs = {st["library"] for st in r["chain"] if st.get("library")}
        fonts = {f for f in (r.get("fonts_per_char") or []) if f}
        if not strata_of(cfg, r["split"]) or r["stratum"] == "seen":
            if libs & set(HOLDOUT_LIBRARIES):
                leaks.append((r["id"], "holdout library", sorted(libs & set(HOLDOUT_LIBRARIES))))
            if fonts & held_fonts:
                leaks.append((r["id"], "held-out font", sorted(fonts & held_fonts)))
        if r["split"] in FIXED_SPLITS and r["source"].startswith("python"):
            leaks.append((r["id"], "python image in fixed set", r["source"]))
        if r.get("composited"):
            leaks.append((r["id"], "composited image", ""))
    a["leaks"] = leaks
    hf = [r for r in rows if r["stratum"] == "heldout_font"]
    a["heldout_font_stratum"] = dict(images=len(hf), with_heldout_font=sum(1 for r in hf if set(r.get("fonts_per_char") or []) & held_fonts),
                                     with_any_train_font=sum(1 for r in hf if any(f and f in {font_key(x) for x in load_bank()} and f not in held_fonts
                                                                                 for f in (r.get("fonts_per_char") or []))))
    # rejections
    att = [json.loads(l) for l in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()] if (out / "attempts.jsonl").exists() else []
    cats = Counter()
    for t in att:
        rs = t.get("reason", "")
        cat = ("clip" if rs.startswith("clip") else "upstream_colour_crash" if "imagecolorallocate" in rs
               else "buried_layer" if "buried" in rs else "level_off_scale" if ("measured level" in rs or "not measurable" in rs)
               else "intended_not_visible" if "not visible" in rs or "level 0 but" in rs else "blank_or_no_text" if ("blank" in rs or "no label" in rs)
               else "other")
        cats[cat] += 1
    total_attempts = len(rows) + len(att)
    a["rejections"] = dict(total_attempts=total_attempts, rejected=len(att), by_cause=dict(cats),
                           clip_reject_rate=round(cats["clip"] / max(1, total_attempts), 4),
                           upstream_crash_rate=round(cats["upstream_colour_crash"] / max(1, total_attempts), 4))
    fails = [json.loads(l) for l in (out / "failures.jsonl").read_text(encoding="utf-8").splitlines()] if (out / "failures.jsonl").exists() else []
    a["unfilled_indices"] = len([f for f in fails if (f["split"], f["index"]) not in {(r["split"], r["index"]) for r in rows}])
    a["other_rejection_examples"] = [t for t in att if not any(k in t.get("reason", "") for k in ("clip", "imagecolorallocate", "buried", "level", "visible"))][:15]
    # fixed-set hashes (frozen identity of each fixed split)
    a["split_hashes"] = {}
    for s in sorted(a["per_split"]):
        h = hashlib.sha256()
        for r in sorted((r for r in rows if r["split"] == s), key=lambda r: r["index"]):
            h.update(f"{r['id']}:{r['sha256']}\n".encode())
        a["split_hashes"][s] = h.hexdigest()
    if cfg.get("_epoch"):
        a["targeting"] = targeting_audit(rows)
        n_planned = int(cfg["_train_n"])
    else:
        n_planned = len(plan(cfg, cfg.get("_train_n", cfg["sizes"]["train"])))
    a["complete"] = len(rows) == n_planned
    a["planned"] = n_planned
    a["ok"] = a["complete"] and not a["duplicate_hashes"] and not leaks
    (out / "audit.json").write_text(json.dumps(a, indent=1, default=str), encoding="utf-8")
    if a["complete"]:
        (out / "splits.json").write_text(json.dumps(dict(hashes=a["split_hashes"], counts=a["per_split"]), indent=1), encoding="utf-8")
    print(json.dumps({k: a[k] for k in ("images", "planned", "complete", "per_split", "python_share", "rejections", "ok")}, indent=1, default=str))
    print("leaks", len(leaks), "duplicate hashes", len(a["duplicate_hashes"]))
    return a


def targeting_audit(rows):
    """Epoch batch: per targeted head, measured presence among its slots and how often the target was met."""
    from feature_bank.heads import HEADS
    t = [r for r in rows if (r.get("slot") or {}).get("target")]
    out = dict(targeted_images=len(t), base_images=len(rows) - len(t),
               python_share=round(sum(r["source"].startswith("python") for r in rows) / max(1, len(rows)), 4),
               target_met_rate=round(sum(bool(r.get("target_met")) for r in t) / max(1, len(t)), 4), heads={})
    by = defaultdict(list)
    for r in t:
        by[r["slot"]["target"]["head"]].append(r)
    for h, rs in sorted(by.items()):
        pres = [r["heads"][h]["present"] for r in rs]
        out["heads"][h] = dict(slots=len(rs), pos_slots=sum(r["slot"]["target"]["polarity"] == "pos" for r in rs),
                               met=sum(bool(r.get("target_met")) for r in rs),
                               present=pres.count(1), absent=pres.count(0), masked=pres.count(None),
                               presence_rate=round(pres.count(1) / max(1, pres.count(1) + pres.count(0)), 3))
    # whole-batch prevalence of every head (base + targeted), for the presence-balance check
    out["batch_presence"] = {h: round(sum(r["heads"][h]["present"] == 1 for r in rows) / max(1, len(rows)), 4) for h in HEADS}
    return out


def epoch_items(cfg, epoch, n, targets, part):
    """[(split, index, stratum, slot)] for an epoch batch (part: base | all)."""
    from feature_bank.targeting import slot_plan, n_base_of
    split = f"epoch{epoch:02d}"
    slots = slot_plan(cfg, split, n, targets)
    upto = n_base_of(n, cfg) if part == "base" else n
    return [(split, i, "seen", slots[i]) for i in range(upto)]


def set_low_priority():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL, inherited
    else:
        os.nice(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, help="curriculum epoch batch (split epochNN); needs --n")
    ap.add_argument("--n", type=int, help="epoch batch size")
    ap.add_argument("--targets", help="targets.json for the targeted half (none: base mix)")
    ap.add_argument("--part", choices=["base", "all"], default="all", help="base: render only the base half")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "configs/corpus_hybrid_v3.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, help="train-pool size (default: config sizes.train)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--fixed-scale", type=float, default=1.0, help="scale the fixed-set sizes (smoke tests only)")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.audit_only:
        audit(out)
        return
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.low_priority:
        set_low_priority()
    if a.epoch is not None:
        from feature_bank.targeting import load_targets
        targets = load_targets(a.targets)
        out.mkdir(parents=True, exist_ok=True)
        tp = out / "targets.json"
        if targets is not None:
            if tp.exists() and json.loads(tp.read_text(encoding="utf-8")) != targets:
                raise SystemExit(f"{out} already has different targets; use a new --out")
            tp.write_text(json.dumps(targets, indent=1), encoding="utf-8")
        elif tp.exists() and a.part == "all":
            targets = json.loads(tp.read_text(encoding="utf-8"))
        items = epoch_items(cfg, a.epoch, a.n, targets, a.part)
        (out / "slots.json").write_text(json.dumps(dict(epoch=a.epoch, n=a.n, part=a.part, targets=targets,
                                                        slots=[it[3] for it in items]), indent=0), encoding="utf-8")
        generate(cfg, out, a.n, a.workers, items=items, tag=f"epoch{a.epoch:02d}")
        audit(out)
        return
    if a.fixed_scale != 1.0:
        cfg["sizes"] = {k: (v if k == "train" else int(round(v * a.fixed_scale))) for k, v in cfg["sizes"].items()}
    train_n = a.train if a.train is not None else int(cfg["sizes"]["train"])
    generate(cfg, out, train_n, a.workers)
    audit(out)
    if not a.no_review:
        from feature_bank.review import corpus_sheets
        for p in corpus_sheets(out):
            print("review sheet", p)


if __name__ == "__main__":
    main()
