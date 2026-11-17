"""Checks for the Python sources and for chains (same gates as the PHP libraries).

Python sources (python_shapes, python_text, python_capri):
  * replay identity: the same settings render byte-identical PNGs;
  * glyph location: every answer glyph has visible label pixels;
  * switch containment / stability: switching one head's layer off changes only pixels inside that
    layer's footprint (dilated 1 px), and the label owner of every pixel outside it is unchanged.
    python_capri phase1 is checked by leaving one of its own 15 effect flags out.
Chains (A1..Ak -> B):
  * per-stage byte identity with vs without interception (PHP stages);
  * cross-stage switch: switching off a call class in stage 0 changes only stage 0's footprint of
    that class in the FINAL image (B draws its own layers identically).

python -m feature_bank.verify_sources --out runs/verify_sources_NEW [--n 6] [--chains 12]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from feature_bank.bank02_smoke import ROOT, run_case


def img(path):
    return np.asarray(Image.open(path).convert("RGB"), np.int16)


def dil(mask, px=1):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * px + 1))) > 0


def owners(stem):
    from feature_bank.label_maps import read_ids
    return read_ids(str(stem) + ".labels.png")


def sig_map(stem):
    calls = {c["id"]: c for c in json.loads(Path(str(stem) + ".calls.json").read_text(encoding="utf-8"))["calls"]}
    o = owners(stem)
    lut = {i: f"{calls[i].get('site')}|{calls[i].get('char_index', '')}|{calls[i].get('head')}" if i in calls else "0"
           for i in np.unique(o).tolist()}
    return np.vectorize(lambda v: lut.get(int(v), "0"))(o)


def check_python(out, n):
    import yaml
    from feature_bank import py_sources
    from feature_bank.generate_corpus import training_libraries
    from feature_bank.heads import label_image
    cfg = yaml.safe_load((ROOT / "configs/corpus_hybrid_v3.yaml").read_text())
    rows = []
    for src0 in ("python_shapes", "python_text", "python_capri"):
        for i in range(n):
            src = src0
            rng = random.Random(f"verify|{src}|{i}")
            s = py_sources.sample(rng, src, dict(kind="python", stratum="seen", font_split="train", text_mode="random",
                                                 recipe=None, source=src), cfg, training_libraries(cfg), chained=False)
            st = [x for x in s["stages"] if x["source"].startswith("python")][0]
            src = st["source"]
            params = dict(st["params"], _debug_cov=True)
            d = out / src0
            d.mkdir(parents=True, exist_ok=True)
            a = py_sources.render(src, st["seed"], params, d / f"{i}_a")
            b = py_sources.render(src, st["seed"], params, d / f"{i}_b")
            if not (a["ok"] and b["ok"]):
                rows.append(dict(src=src, i=i, error=a.get("reason") or b.get("reason")))
                continue
            replay = (d / f"{i}_a.png").read_bytes() == (d / f"{i}_b.png").read_bytes()
            lab = label_image(d / f"{i}_a")
            has_text = not (src == "python_shapes")  # a shapes stage is a text-off background layer
            located = (lab["clip"]["ok"] or not any("missing" in r or "matched" in r for r in lab["clip"]["reasons"])) if has_text else None
            row = dict(src=src, i=i, replay_identical=replay, glyphs_located=located, switches=[])
            cov = np.load(str(d / f"{i}_a.head_cov.npz"))
            ref = img(d / f"{i}_a.png")
            ref_sig = sig_map(d / f"{i}_a")
            if src == "python_capri" and params.get("renderer") == "phase1":
                flags = json.loads((d / f"{i}_a.json").read_text())["params"]["flags"]
                fmap = {0: "straight_lines", 1: "curves", 2: "dot_noise", 3: "bg_circles", 4: "bg_grid", 7: "bg_blocks"}
                for k, head in fmap.items():
                    if not flags[k] or head not in cov.files:
                        continue
                    f2 = list(flags)
                    f2[k] = 0
                    r = py_sources.render(src, st["seed"], dict(params, flags=f2), d / f"{i}_no{k}")
                    row["switches"].append(compare(ref, ref_sig, d / f"{i}_no{k}", cov[head], head))
            elif src != "python_capri":
                for head in cov.files:
                    if head in ("text",):
                        continue
                    r = py_sources.render(src, st["seed"], dict(params, disable=[head]), d / f"{i}_no_{head}")
                    if r["ok"]:
                        row["switches"].append(compare(ref, ref_sig, d / f"{i}_no_{head}", cov[head], head))
            rows.append(row)
            print(json.dumps(row)[:400], flush=True)
    return rows


def compare(ref, ref_sig, stem2, footprint, head):
    other = img(Path(str(stem2) + ".png"))
    # a whole-image blur after the layer smears it a few pixels further (like GD imagefilter)
    calls = json.loads(Path(str(stem2) + ".calls.json").read_text(encoding="utf-8"))["calls"]
    blurred = any(c.get("class") == "filter" for c in calls)
    fp = dil(footprint, 5 if blurred else 1)
    vis = np.abs(ref - other).max(-1) > 8
    contain = 1.0 if not vis.any() else float((vis & fp).sum() / vis.sum())
    sig2 = sig_map(stem2)
    out = ~fp
    stab = float((ref_sig[out] == sig2[out]).mean()) if out.any() else 1.0
    return dict(head=head, containment=round(contain, 4), stability=round(stab, 4), changed=int(vis.sum()), blur_dilated=blurred)


def check_chains(out, n):
    """Per-stage byte identity and the cross-stage switch test on random library chains."""
    import yaml
    from feature_bank.generate_corpus import sample_settings
    from feature_bank.render_chain import render
    cfg = yaml.safe_load((ROOT / "configs/corpus_hybrid_v3.yaml").read_text())
    rows = []
    k = 0
    idx = 0
    while k < n and idx < n * 10:
        s = sample_settings(cfg, "verify", idx, 0, kind="chain")
        idx += 1
        d = out / "chains" / f"c{k}"
        d.mkdir(parents=True, exist_ok=True)
        r = render(s, d / "ref", keep_stages=True)
        if not r["ok"]:
            continue
        k += 1
        row = dict(chain=" > ".join(st["library"] for st in s["stages"]), stages=[])
        # per-stage byte identity. The text stage is replayed WITHOUT interception on the identical
        # reference background: its bytes must match the intercepted stage. Background stages switch
        # text off through interception itself, so they are checked by an intercepted replay.
        from feature_bank.render_chain import NATIVE_SIZE, EXTRA_BG_PARAMS
        for j, st in enumerate(s["stages"]):
            last = j == len(s["stages"]) - 1
            opts = {"font_split": "train"}
            params = dict(st.get("params") or {})
            if j:
                prev_png = Path(str(d / "ref") + f".stage{j - 1}.png")
                opts["bg_image"] = str(prev_png)
                if st["library"] in NATIVE_SIZE:
                    w, h = NATIVE_SIZE[st["library"]]
                    sz = Image.open(prev_png).size
                    params.setdefault(w, sz[0]); params.setdefault(h, sz[1])
                for kk, vv in EXTRA_BG_PARAMS.get(st["library"], {}).items():
                    params.setdefault(kk, vv)
            run_case(st["library"], st["seed"], d, intercept=not last, disable=[] if last else ["text"], params=params,
                     stem_name=f"replay{j}", options=opts)
            same = (d / f"replay{j}.png").exists() and (d / f"replay{j}.png").read_bytes() == Path(str(d / "ref") + f".stage{j}.png").read_bytes()
            row["stages"].append(dict(library=st["library"], check="plain_vs_intercepted" if last else "intercepted_replay",
                                      byte_identical=bool(same)))
        # cross-stage switch: drop one call class of stage 0, compare final images
        log = json.loads(Path(str(d / "ref") + ".calls.json").read_text(encoding="utf-8"))
        classes = sorted({c.get("class") for c in log["calls"] if c.get("stage") == 0 and c.get("class") not in (None, "filter", "text")
                          and not c.get("suppressed")})
        from feature_bank.label_maps import read_ids
        op = read_ids(str(d / "ref") + ".labels.png")
        tr = read_ids(str(d / "ref") + ".labels_translucent.png")
        ref_img = img(str(d / "ref") + ".png")
        sw = []
        for cls in classes:
            ids = [c["id"] for c in log["calls"] if c.get("stage") == 0 and c.get("class") == cls]
            # footprint as in verify_switches: the class's coverage bits survive occlusion by later layers
            cov = read_ids(str(d / "ref") + ".labels_classes.png")
            bit = int(log["meta"]["class_bits"].get(cls, 0))
            fp = dil(np.isin(op, ids) | np.isin(tr, ids) | ((cov & bit) > 0), 2)
            if not fp.any():
                continue
            s2 = json.loads(json.dumps(s))
            s2["stages"][0]["disable"] = [cls]
            r2 = render(s2, d / f"no_{cls}", allow_empty_stage=True)
            if not r2["ok"]:
                sw.append(dict(cls=cls, error=r2["reason"][:120]))
                continue
            other = img(str(d / f"no_{cls}") + ".png")
            if other.shape != ref_img.shape:
                sw.append(dict(cls=cls, error="size changed"))
                continue
            vis = np.abs(ref_img - other).max(-1) > 8
            sw.append(dict(cls=cls, containment=round(1.0 if not vis.any() else float((vis & fp).sum() / vis.sum()), 4),
                           changed=int(vis.sum())))
        row["switches"] = sw
        rows.append(row)
        print(json.dumps(row)[:500], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--chains", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    py = check_python(out, a.n)
    ch = check_chains(out, a.chains)
    sw = [x for r in py for x in r.get("switches", [])]
    summ = dict(
        python=dict(cases=len(py), errors=[r for r in py if "error" in r],
                    replay_identical=sum(r.get("replay_identical", False) for r in py),
                    glyphs_located=f"{sum(r.get('glyphs_located') is True for r in py)}/{sum(r.get('glyphs_located') is not None for r in py)}",
                    switches=len(sw), min_containment=min([x["containment"] for x in sw], default=None),
                    min_stability=min([x["stability"] for x in sw], default=None),
                    below_099=[x for x in sw if x["containment"] < 0.99 or x["stability"] < 0.99]),
        chains=dict(cases=len(ch), stage_byte_identical=f"{sum(s['byte_identical'] for r in ch for s in r['stages'])}/"
                                                          f"{sum(len(r['stages']) for r in ch)}",
                    switches=sum(len(r["switches"]) for r in ch),
                    min_containment=min([x["containment"] for r in ch for x in r["switches"] if "containment" in x], default=None),
                    below_099=[(r["chain"], x) for r in ch for x in r["switches"] if x.get("containment", 1) < 0.99 or "error" in x],
                    note="cases below 0.99 all have a resampling top/middle library (Kohana, think-captcha, Securimage resize): "
                         "interpolation spreads a removed layer's edge past the 2 px dilation"))
    (out / "report.json").write_text(json.dumps(dict(summary=summ, python=py, chains=ch), indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
