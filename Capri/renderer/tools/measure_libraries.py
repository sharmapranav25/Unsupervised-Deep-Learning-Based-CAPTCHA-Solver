"""Measure what the original CAPTCHA libraries actually draw, per effect -> ranges.json (development tool).

Reads the hybrid-v3 generator output (sourcefree_synth: manifests, per-pixel call-id label maps, merged call logs).
Only library-drawn calls are used (Python-source images are skipped). For every effect head (via the corpus rule table
`feature_bank.heads.call_head`) it records, per call and per image:
  per call : visible pixels, bounding box (relative to canvas), stroke thickness (area / long side), aspect,
             colour (RGB), colour distance to the image background, translucency
  per image: number of calls, total coverage fraction
and summarises them as percentiles (p5, p25, p50, p75, p95) per library and pooled.

  python tools/measure_libraries.py --corpus <sourcefree_synth>/runs/r1_main_20261002/data/epoch_01 [...] --n 3000 \
      --out ranges/library_ranges.json
"""
import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HEADS = ["straight_lines", "curves", "strike_through", "dot_noise", "blob_noise", "scribbles", "bg_circles", "bg_grid",
         "bg_blocks", "bg_gradient", "bg_image_texture", "bg_decoy_chars", "border_frame"]
PCT = [5, 25, 50, 75, 95]


def pct(v):
    v = np.asarray(v, float)
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in PCT} if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", default="E:/pranav/Final sem/sourcefree_synth")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.synth_root)
    from feature_bank.heads import call_head, BACKGROUND

    rows = []
    for d in a.corpus:
        for mf in Path(d).glob("shards/*/manifest.jsonl"):
            for line in mf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if not r["source"].startswith("python") and all(st.get("source") == "php" for st in r["chain"]):
                        rows.append((Path(d), r))
    rows = random.Random(0).sample(rows, min(a.n, len(rows)))
    per_call = defaultdict(lambda: defaultdict(list))   # head -> metric -> values
    per_img = defaultdict(lambda: defaultdict(list))
    by_lib = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d, r in rows:
        stem = d / r["stem"]
        try:
            z = np.load(str(stem) + ".labels.npz")
            log = json.loads(gzip.decompress(Path(str(stem) + ".calls.json.gz").read_bytes()))
            img = np.asarray(Image.open(str(stem) + ".png").convert("RGB"), np.float32)
        except Exception:  # noqa: BLE001
            continue
        ids = np.where(z["opaque"] > 0, z["opaque"], z["translucent"]).astype(np.int64)
        H, W = ids.shape
        flat_ids = ids.ravel()
        order = np.argsort(flat_ids, kind="stable")
        sid = flat_ids[order]
        uniq, start = np.unique(sid, return_index=True)
        ends = np.append(start[1:], len(sid))
        pix = {int(u): order[s:e] for u, s, e in zip(uniq, start, ends) if u}
        bg = np.median(img.reshape(-1, 3), 0)
        chain = r["chain"]
        counts = defaultdict(int)
        cover = defaultdict(int)
        for c in log["calls"]:
            if c.get("class") in ("text", "filter") or c.get("chain_bg"):
                continue
            st = int(c.get("stage", 0))
            lib = c.get("library") or chain[min(st, len(chain) - 1)]["library"]
            params = (chain[st].get("params") or {}) if st < len(chain) else {}
            h = call_head(c, lib, params)
            if h not in HEADS or h == BACKGROUND:
                continue
            p = pix.get(int(c["id"]))
            if p is None or len(p) == 0:
                continue
            yy, xx = p // W, p % W
            bw, bh = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
            area = len(p)
            col = img.reshape(-1, 3)[p].mean(0)
            m = dict(area_frac=area / (H * W), bbox_w=bw / W, bbox_h=bh / H, thickness_px=area / max(bw, bh),
                     aspect=max(bw, bh) / max(1, min(bw, bh)), colour_dist=float(np.sqrt(((col - bg) ** 2).sum())),
                     lum=float(0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]),
                     sat=float((col.max() - col.min()) / max(1.0, col.max())), translucent=float(str(c.get("translucent")) == "True"))
            for k, v in m.items():
                per_call[h][k].append(v)
                by_lib[h][lib][k].append(v)
            counts[h] += 1
            cover[h] += area
        for h in counts:
            per_img[h]["calls"].append(counts[h])
            per_img[h]["coverage"].append(cover[h] / (H * W))
    out = dict(method=__doc__, images_measured=len(rows), heads={})
    for h in HEADS:
        if h not in per_call:
            out["heads"][h] = dict(n_calls=0, note="not drawn by any measured library call")
            continue
        out["heads"][h] = dict(
            n_calls=len(per_call[h]["area_frac"]), n_images=len(per_img[h]["calls"]),
            per_image={k: pct(v) for k, v in per_img[h].items()},
            per_call={k: pct(v) for k, v in per_call[h].items()},
            libraries={lib: dict(n_calls=len(m["area_frac"]), thickness_px=pct(m["thickness_px"]), area_frac=pct(m["area_frac"]),
                                 colour_dist=pct(m["colour_dist"])) for lib, m in by_lib[h].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for h, e in out["heads"].items():
        if e.get("n_calls"):
            pc, pi = e["per_call"], e["per_image"]
            print(f"{h:18s} imgs {e['n_images']:5d} calls/img {pi['calls']['p50']:.0f} [{pi['calls']['p5']:.0f}-{pi['calls']['p95']:.0f}] "
                  f"cover {pi['coverage']['p50']:.3f} thick {pc['thickness_px']['p50']:.1f}px "
                  f"libs {sorted(e['libraries'], key=lambda l: -e['libraries'][l]['n_calls'])[:4]}")
        else:
            print(f"{h:18s} none")


if __name__ == "__main__":
    main()
