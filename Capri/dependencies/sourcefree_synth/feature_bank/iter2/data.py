"""Iteration 2 cache reader + batches (cache written by feature_bank.iter2.generate).

Batches: images padded on the right by edge replication to a common width (>= real width + MARGIN, multiple of 8), so a
token's CNN input is identical alone or in any batch (same rule as iteration 1); `cols` = real token columns.
"""
import json
import queue
import threading
from pathlib import Path

import numpy as np

from feature_bank.train.data import MARGIN, PIXEL_BUDGET, TOK_W, width_batches  # noqa: F401  (shared rules)

H = 96
LABEL_KEYS = ["eff_present", "eff_level", "txt_present", "txt_level", "chars", "length", "font"]


class Iter2Set:
    def __init__(self, cdir):
        self.cdir = Path(cdir)
        self.meta = json.loads((self.cdir / "meta.json").read_text(encoding="utf-8"))
        z = np.load(self.cdir / "index.npz")
        self.ix = {k: z[k] for k in z.files}
        self.cols = np.memmap(self.cdir / "images.u8", np.uint8, "r").reshape(-1, H, 3)
        self.name = self.cdir.name

    def __len__(self):
        return len(self.ix["widths"])

    def image(self, i):
        o, w = int(self.ix["offsets"][i]), int(self.ix["widths"][i])
        return np.asarray(self.cols[o:o + w]).transpose(1, 0, 2)


def open_split(root, split):
    """All cache dirs of a split (train is stored in parts train_000000, train_010000, ...)."""
    root = Path(root) / "iter2_h96"
    dirs = sorted(root.glob(f"{split}_*")) if split == "train" else [root / split]
    return [Iter2Set(d) for d in dirs if (d / "meta.json").exists()]


def items_of(sets):
    items, widths = [], []
    for s in sets:
        items += [(s, i) for i in range(len(s))]
        widths += s.ix["widths"].tolist()
    return items, np.array(widths)


def collate(items):
    ims = [s.image(i) for s, i in items]
    w = np.array([im.shape[1] for im in ims])
    ncols = -(-w // TOK_W)
    wp = int(max(ncols.max() * TOK_W, w.max()) + MARGIN)
    wp = -(-wp // TOK_W) * TOK_W
    batch = np.empty((len(items), H, wp, 3), np.uint8)
    for k, im in enumerate(ims):
        batch[k] = np.pad(im, ((0, 0), (0, wp - im.shape[1]), (0, 0)), mode="edge")
    out = dict(image=batch.transpose(0, 3, 1, 2), cols=ncols.astype(np.int64))
    for key in LABEL_KEYS:
        out[key] = np.stack([s.ix[key][i] for s, i in items])
    return out


def prefetch(batches, n_threads=6, depth=12):
    """Collate batches in background threads (numpy releases the GIL for the copies); yields in order."""
    q_in, slots = queue.Queue(), {}
    for k, b in enumerate(batches):
        q_in.put((k, b))
    cond = threading.Condition()

    def work():
        while True:
            try:
                k, b = q_in.get_nowait()
            except queue.Empty:
                return
            with cond:
                cond.wait_for(lambda: k < nxt[0] + depth)
            r = collate(b)
            with cond:
                slots[k] = r
                cond.notify_all()

    nxt = [0]
    ts = [threading.Thread(target=work, daemon=True) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for k in range(len(batches)):
        with cond:
            cond.wait_for(lambda: k in slots)
            r = slots.pop(k)
            nxt[0] = k + 1
            cond.notify_all()
        yield r
