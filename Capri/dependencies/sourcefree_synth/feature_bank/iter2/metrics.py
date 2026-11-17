"""Iteration 2 scoring: predictions over a fixed set, then per-head metrics.

Effects / text heads: presence F1 at 0.5 (and AUC-free mean F1); intensity on level > 0 only: exact and within +-1,
predicted level = number of "level >= k" logits above 0. Reading: greedy CTC decode -> exact text, CER, length
(from CTC and from the length head). Fonts: category / stroke accuracy, ordinal within +-1, italic F1.
"""
import numpy as np
import torch

from feature_bank.iter2.data import PIXEL_BUDGET, collate, items_of, width_batches

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@torch.no_grad()
def predict(model, sets, device, batch=128, budget=PIXEL_BUDGET):
    items, widths = items_of(sets)
    bs = width_batches(items, widths, batch, None, shuffle=False, budget=budget)
    model.eval()
    keep = ["eff_presence", "eff_ordinal", "txt_presence", "txt_ordinal", "length", "font_cat", "font_stroke",
            "font_ordinal", "font_italic"]
    P = {k: [] for k in keep}
    P["text"] = []
    L = {k: [] for k in ["eff_present", "eff_level", "txt_present", "txt_level", "chars", "length", "font"]}
    for b in bs:
        nb = collate(b)
        img = torch.from_numpy(nb["image"]).to(device).float().div_(255)
        cols = torch.from_numpy(nb["cols"]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img, cols)
        for k in keep:
            P[k].append(out[k].float().cpu().numpy())
        fr = out["ctc"].shape[1] // (nb["image"].shape[-1] // 8)
        P["text"] += decode(out["ctc"].float().cpu().numpy(), nb["cols"] * fr)
        for k in L:
            L[k].append(nb[k])
    P = {k: (np.concatenate(v) if k != "text" else v) for k, v in P.items()}
    L = {k: np.concatenate(v) for k, v in L.items()}
    L["text"] = ["".join(CHARS[c] for c in row[:n]) for row, n in zip(L["chars"], L["length"])]
    return P, L


def decode(logits, cols):
    out = []
    for lg, c in zip(logits, cols):
        best = lg[:c].argmax(-1)
        s, prev = [], 0
        for k in best:
            if k != prev and k != 0:
                s.append(CHARS[k - 1])
            prev = k
        out.append("".join(s))
    return out


def f1(pred, y):
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def ordinal_level(logits):
    return (logits > 0).sum(-1)


def head_block(pres_logit, ord_logit, present, level, names):
    out = {}
    for i, h in enumerate(names):
        y = present[:, i] == 1
        p = pres_logit[:, i] > 0
        r = dict(f1=round(f1(p, y), 4), prev=round(float(y.mean()), 4))
        lv = level[:, i]
        m = lv > 0
        if m.sum() >= 5:
            pl = ordinal_level(ord_logit[m, i])
            r.update(lvl_exact=round(float((pl == lv[m]).mean()), 4), lvl_pm1=round(float((abs(pl - lv[m]) <= 1).mean()), 4),
                     lvl_mae=round(float(abs(pl - lv[m]).mean()), 3), lvl_n=int(m.sum()))
        out[h] = r
    return out


def scores(P, L, eff_heads, txt_heads):
    s = {}
    s["effects"] = head_block(P["eff_presence"], P["eff_ordinal"], L["eff_present"], L["eff_level"], eff_heads)
    s["text_heads"] = head_block(P["txt_presence"], P["txt_ordinal"], L["txt_present"], L["txt_level"], txt_heads)
    gt, pr = L["text"], P["text"]
    s["reading"] = dict(exact=round(float(np.mean([a == b for a, b in zip(gt, pr)])), 4),
                        exact_caseless=round(float(np.mean([a.lower() == b.lower() for a, b in zip(gt, pr)])), 4),
                        cer=round(float(sum(levenshtein(a, b) for a, b in zip(gt, pr)) / max(1, sum(map(len, gt)))), 4),
                        length_ctc=round(float(np.mean([len(a) == len(b) for a, b in zip(gt, pr)])), 4),
                        length_head=round(float((P["length"].argmax(-1) + 2 == L["length"]).mean()), 4))
    f = L["font"]
    fs = {}
    for j, (k, logit) in enumerate([("category", P["font_cat"]), ("stroke", P["font_stroke"])]):
        m = f[:, j] >= 0
        fs[k] = round(float((logit[m].argmax(-1) == f[m, j]).mean()), 4) if m.any() else None
    for j, k in enumerate(["weight", "width", "x_height", "slant"]):
        m = f[:, 2 + j] >= 0
        pl = ordinal_level(P["font_ordinal"][m, j])
        fs[k + "_pm1"] = round(float((abs(pl - f[m, 2 + j]) <= 1).mean()), 4)
    fs["italic_f1"] = round(f1(P["font_italic"] > 0, f[:, 6] == 1), 4)
    s["font"] = fs
    e = s["effects"]
    lv = [v["lvl_pm1"] for v in e.values() if "lvl_pm1" in v]
    s["summary"] = dict(
        eff_f1=round(float(np.mean([v["f1"] for v in e.values()])), 4),
        eff_ge07=int(sum(v["f1"] >= 0.7 for v in e.values())),
        eff_lvl_pm1=round(float(np.mean(lv)), 4) if lv else None,
        txt_f1=round(float(np.mean([v["f1"] for v in s["text_heads"].values()])), 4),
        read_exact=s["reading"]["exact"], cer=s["reading"]["cer"], length=s["reading"]["length_head"],
        font_cat=fs["category"], font_ord_pm1=round(float(np.mean([fs[k + "_pm1"] for k in ["weight", "width", "x_height", "slant"]])), 4))
    return s
