"""Hybrid v3 data: generator output -> a height-96 cache -> width-bucketed padded batches with masks.

Cache (one per generator output dir, <dir>/cache_h96/, built once, read through memmaps):
  images.u8      all images resized to height 96 (native aspect, width capped), RGB, concatenated
  coverage.f16   per-token layer-coverage fractions (6 rows x ncols x 10 classes), concatenated
  index.npz      per image: offsets, widths (px), ncols, label tensors (masks: -1 / NaN = null)
  meta.json      ids, sha256, splits, strata, sources, recipe, target (row order = cache order)

Labels (null -> masked, never 0):
  presence  [46]  1 / 0 / -1
  level     [27]  0..10 / -1     (ordinal heads, heads.ORDINAL_HEADS order)
  length    []    len - 2 in 0..6
  comp      [3]   digit / lower / upper fractions
  slot_*    [8]   present 1/0 (slot_mask = 0 when the image has no per-character glyphs, e.g. SVG-Captcha),
                  box (cx, w, cy, h) relative to the real image, angle (sin, cos), group, colour /255, outline, char
Batches: images padded on the right by edge replication to a common width (>= real width + MARGIN, a multiple
of 8), so a token's CNN input is identical alone or in any batch; `cols` = real token columns.
"""
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

H = 96
TOK_W = 8
TOK_H = 16
ROWS = H // TOK_H
MARGIN = 32  # edge-replicated columns after every image (> CNN receptive field 25 px)
COVERAGE = ["glyph", "line", "curve", "dot", "blob", "grid", "block", "decoy", "texture", "background"]
HEAD_COV = {"straight_lines": "line", "strike_through": "line", "border_frame": "line", "curves": "curve",
            "scribbles": "curve", "dot_noise": "dot", "blob_noise": "blob", "bg_circles": "blob", "bg_grid": "grid",
            "bg_blocks": "block", "bg_decoy_chars": "decoy", "fixed_caption": "decoy", "bg_image_texture": "texture",
            "bg_gradient": "texture", "glyph_shadow": "texture"}
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_SLOTS = 8


def heads_lists():
    from feature_bank.heads import HEADS, ORDINAL_HEADS
    return list(HEADS), list(ORDINAL_HEADS)


def read_rows(out):
    rows = []
    for mf in sorted(Path(out).glob("shards/*/manifest.jsonl")):
        for line in mf.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    rows.sort(key=lambda r: (r["split"], r["index"]))
    return rows


# ------------------------------------------------------------------ per image (cache worker)

def resized_width(w, h, max_width):
    return max(TOK_W, min(max_width, int(round(w * H / h))))


def coverage_map(row, root, shape):
    """Per-pixel coverage class (uint8 index into COVERAGE) at native resolution."""
    from feature_bank.heads import call_head, BACKGROUND
    stem = Path(root) / row["stem"]
    z = np.load(str(stem) + ".labels.npz")
    opaque, trans = z["opaque"].astype(np.int64), z["translucent"].astype(np.int64)
    ids = np.where(opaque > 0, opaque, trans)
    log = json.loads(gzip.decompress(Path(str(stem) + ".calls.json.gz").read_bytes()).decode("utf-8"))
    glyphs = set(row.get("glyph_ids") or [])
    chain = row.get("chain") or []
    bg = COVERAGE.index("background")
    lut = {}
    for c in log["calls"]:
        cid = c["id"]
        if cid in glyphs:
            lut[cid] = 0
            continue
        if c.get("class") == "filter":
            continue
        if c.get("class") == "text":
            h = c.get("head", "bg_decoy_chars")
        else:
            st = int(c.get("stage", len(chain) - 1)) if chain else 0
            params = (chain[st].get("params") or {}) if st < len(chain) else {}
            h = call_head(c, c.get("library") or (chain[st]["library"] if st < len(chain) else None), params)
        if h == BACKGROUND or h is None:
            continue
        lut[cid] = COVERAGE.index(HEAD_COV.get(h, "background")) if h in HEAD_COV else bg
    out = np.full(ids.shape, bg, np.uint8)
    if lut:
        mx = int(max(ids.max(), max(lut)))
        table = np.full(mx + 1, bg, np.uint8)
        for k, v in lut.items():
            if k <= mx:
                table[k] = v
        out = table[np.clip(ids, 0, mx)]
        out[ids == 0] = bg
    if out.shape != tuple(shape):
        out = np.array(Image.fromarray(out).resize((shape[1], shape[0]), Image.NEAREST))
    return out


def labels_of(row, heads, ordinal):
    pres = np.array([-1 if row["heads"][h]["present"] is None else int(row["heads"][h]["present"]) for h in heads], np.int8)
    lev = np.array([-1 if row["heads"][h]["level"] is None else int(row["heads"][h]["level"]) for h in ordinal], np.int8)
    t = row["text"]
    n = int(t["length"])
    mix = t.get("mix") or {}
    comp = np.array([mix.get("digits", 0), mix.get("lower", 0), mix.get("upper", 0)], np.float32) / max(1, n)
    slots = list(t.get("slots") or [])[:N_SLOTS] + [None] * max(0, N_SLOTS - len(t.get("slots") or []))
    has_glyphs = any(s is not None for s in slots)
    sp = np.zeros(N_SLOTS, np.int8)
    box = np.zeros((N_SLOTS, 4), np.float32)
    ang = np.zeros((N_SLOTS, 2), np.float32)
    grp = np.full(N_SLOTS, -1, np.int8)
    col = np.zeros((N_SLOTS, 3), np.float32)
    outl = np.full(N_SLOTS, -1, np.int8)
    ch = np.full(N_SLOTS, -1, np.int16)
    for i, s in enumerate(slots):
        if s is None:
            continue
        sp[i] = 1
        x0, y0, x1, y1 = s["box"]
        box[i] = [(x0 + x1) / 2, x1 - x0, (y0 + y1) / 2, y1 - y0]
        a = np.deg2rad(float(s.get("angle") or 0.0))
        ang[i] = [np.sin(a), np.cos(a)]
        c = str(s.get("char") or "")
        if c:
            grp[i] = 0 if c.isdigit() else 1 if c.islower() else 2 if c.isupper() else -1
            ch[i] = CHARS.index(c) if c in CHARS else -1
        if s.get("colour") is not None:
            col[i] = np.array(s["colour"][:3], np.float32) / 255.0
        if s.get("outline") is not None:
            outl[i] = int(bool(s["outline"]))
    return dict(presence=pres, level=lev, length=np.int8(min(max(n, 2), 8) - 2), comp=comp, slot_mask=np.int8(has_glyphs),
                slot_present=sp, slot_box=box, slot_angle=ang, slot_group=grp, slot_colour=col, slot_outline=outl, slot_char=ch)


def cache_one(args):
    root, row, max_width, heads, ordinal = args
    img = Image.open(Path(root) / (row["stem"] + ".png")).convert("RGB")
    w0, h0 = img.size
    w = resized_width(w0, h0, max_width)
    im = np.asarray(img.resize((w, H), Image.BILINEAR), np.uint8)
    ncols = -(-w // TOK_W)
    try:
        cov = coverage_map(row, root, (h0, w0))
        cov = np.array(Image.fromarray(cov).resize((w, H), Image.NEAREST))
        wp = ncols * TOK_W
        cov = np.pad(cov, ((0, 0), (0, wp - w)), mode="edge")
        oh = np.eye(len(COVERAGE), dtype=np.float32)[cov]  # H x wp x C
        tok = oh.reshape(ROWS, TOK_H, ncols, TOK_W, len(COVERAGE)).mean(axis=(1, 3))
        cov_ok = 1
    except Exception:  # noqa: BLE001 - coverage is an auxiliary target: mask it
        tok = np.zeros((ROWS, ncols, len(COVERAGE)), np.float32)
        cov_ok = 0
    return im, tok.astype(np.float16), labels_of(row, heads, ordinal), cov_ok


def build_cache(out, max_width=1024, workers=12, splits=None):
    """Build <out>/cache_h96 (idempotent: rebuilt only if the manifest row set changed)."""
    out = Path(out)
    cdir = out / "cache_h96"
    rows = [r for r in read_rows(out) if splits is None or r["split"] in splits]
    ident = sorted(f"{r['id']}:{r['sha256']}" for r in rows)
    import hashlib
    sig = hashlib.sha256("\n".join(ident).encode()).hexdigest()
    if (cdir / "meta.json").exists():
        m = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        if m.get("signature") == sig and m.get("max_width") == max_width:
            return cdir
    cdir.mkdir(exist_ok=True)
    heads, ordinal = heads_lists()
    jobs = [(str(out), r, max_width, heads, ordinal) for r in rows]
    if workers > 1 and len(jobs) > 64:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(cache_one, jobs, chunksize=32))
    else:
        res = [cache_one(j) for j in jobs]
    widths = np.array([r[0].shape[1] for r in res], np.int32)
    ncols = np.array([r[1].shape[1] for r in res], np.int32)
    img_off = np.concatenate([[0], np.cumsum(widths.astype(np.int64) * H * 3)])
    cov_off = np.concatenate([[0], np.cumsum(ncols.astype(np.int64) * ROWS * len(COVERAGE))])
    mm = np.lib.format.open_memmap(cdir / "images.npy", mode="w+", dtype=np.uint8, shape=(int(img_off[-1]),))
    cm = np.lib.format.open_memmap(cdir / "coverage.npy", mode="w+", dtype=np.float16, shape=(int(cov_off[-1]),))
    for i, (im, tok, _, _) in enumerate(res):
        mm[img_off[i]:img_off[i + 1]] = im.ravel()
        cm[cov_off[i]:cov_off[i + 1]] = tok.ravel()
    mm.flush()
    cm.flush()
    del mm, cm
    lab = {k: np.stack([r[2][k] for r in res]) for k in res[0][2]} if res else {}
    np.savez(cdir / "index.npz", img_off=img_off, cov_off=cov_off, widths=widths, ncols=ncols,
             cov_ok=np.array([r[3] for r in res], np.int8), **lab)
    meta = dict(signature=sig, max_width=max_width, heads=heads, ordinal=ordinal, coverage=COVERAGE,
                rows=[dict(id=r["id"], sha256=r["sha256"], split=r["split"], stratum=r["stratum"], source=r["source"],
                           kind=r["kind"], text_lib=(r["chain"][-1].get("library") or r["source"]) if r.get("chain") else r["source"],
                           recipe=r.get("recipe"), target=(r.get("slot") or {}).get("target")) for r in rows])
    (cdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return cdir


# ------------------------------------------------------------------ dataset / batches

class CacheSet:
    """One cache (a generator output dir); items addressed by row number."""

    def __init__(self, cdir):
        self.cdir = Path(cdir)
        self.meta = json.loads((self.cdir / "meta.json").read_text(encoding="utf-8"))
        z = np.load(self.cdir / "index.npz")
        self.ix = {k: z[k] for k in z.files}
        self.images = np.load(self.cdir / "images.npy", mmap_mode="r")
        self.cov = np.load(self.cdir / "coverage.npy", mmap_mode="r")
        self.rows = self.meta["rows"]

    def __len__(self):
        return len(self.rows)

    def select(self, pred):
        return [i for i, r in enumerate(self.rows) if pred(r)]

    def image(self, i):
        w = int(self.ix["widths"][i])
        a, b = self.ix["img_off"][i], self.ix["img_off"][i + 1]
        return np.asarray(self.images[a:b]).reshape(H, w, 3)

    def coverage(self, i):
        n = int(self.ix["ncols"][i])
        a, b = self.ix["cov_off"][i], self.ix["cov_off"][i + 1]
        return np.asarray(self.cov[a:b], np.float32).reshape(ROWS, n, len(COVERAGE))


LABEL_KEYS = ["presence", "level", "length", "comp", "slot_mask", "slot_present", "slot_box", "slot_angle", "slot_group",
              "slot_colour", "slot_outline", "slot_char", "cov_ok"]


def collate(items):
    """items: [(CacheSet, i)] -> dict of numpy arrays (image uint8 B x 3 x 96 x Wp, cols, labels, coverage)."""
    ims = [s.image(i) for s, i in items]
    ncols = np.array([int(s.ix["ncols"][i]) for s, i in items], np.int64)
    wp = int(max(ncols.max() * TOK_W, max(im.shape[1] for im in ims)) + MARGIN)
    wp = -(-wp // TOK_W) * TOK_W
    batch = np.empty((len(items), H, wp, 3), np.uint8)
    for k, im in enumerate(ims):
        batch[k] = np.pad(im, ((0, 0), (0, wp - im.shape[1]), (0, 0)), mode="edge")
    tcols = wp // TOK_W
    cov = np.zeros((len(items), ROWS, tcols, len(COVERAGE)), np.float32)
    for k, (s, i) in enumerate(items):
        c = s.coverage(i)
        cov[k, :, :c.shape[1]] = c
    out = dict(image=batch.transpose(0, 3, 1, 2), cols=ncols, coverage=cov)
    for key in LABEL_KEYS:
        out[key] = np.stack([s.ix[key][i] for s, i in items])
    return out


PIXEL_BUDGET = 128 * 352  # batch x widest width (px at height 96): a 128-image batch of median-width images


def width_batches(items, widths, batch, rng, shuffle=True, budget=PIXEL_BUDGET):
    """Width-bucketed batches: shuffle, sort inside chunks of 32 batches by width, cut batches of at most `batch` images
    and at most `budget` = images x widest padded width (activation memory grows with width: a 128-image batch of 758 px
    images needed 16 GB and spilled out of VRAM on Windows; budgeted it is ~6.5 GB). Shuffle batch order."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    chunk = batch * 32
    out = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda j: widths[j])
        cur = []
        for j in part:
            w = widths[j] + MARGIN
            if cur and (len(cur) >= batch or (len(cur) + 1) * w > budget):
                out.append([items[x] for x in cur])
                cur = []
            cur.append(j)
        if cur:
            out.append([items[x] for x in cur])
    if shuffle:
        rng.shuffle(out)
    return out


def to_torch(b, device):
    import torch
    out = {}
    for k, v in b.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    out["image"] = out["image"].float().div_(255.0)
    return out
