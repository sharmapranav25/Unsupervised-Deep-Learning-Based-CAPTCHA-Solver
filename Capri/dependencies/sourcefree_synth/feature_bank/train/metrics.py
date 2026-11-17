"""Hybrid v3 metrics and per-head scores / gates / states (handoff Â§3).



Score per head (used for targeting, states and checkpoint choice):

  * presence head: F1 at 0.5. Gate: F1 >= 0.85, recall >= 0.85, FPR <= 0.10.

  * ordinal head with level labels in the scored set (>= MIN_LEVELS labelled images, some above 0):

    score = min(F1, within-Â±1 accuracy); gate: the presence gate and within-Â±1 >= 0.85.

    (Taking the min keeps a head whose levels are mostly 0 from passing on Â±1 alone.)

  * text: length (accuracy, gate 0.95), composition (score 1 - MAE, gate MAE <= 0.10), slots (fraction of present

    slots with box IoU >= 0.5, gate 0.90).

  * a head with no positives or no negatives in the scored set has score None ("no_data"): it cannot be steered.

gap = how far below its gate a head is (max over its gate criteria), 0 when passed.

"""

import numpy as np



MIN_LEVELS = 20





def f1_stats(prob, y):

    v = y >= 0

    p, t = prob[v] >= 0.5, y[v] == 1

    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())

    prec = tp / max(1, tp + fp)

    rec = tp / max(1, tp + fn)

    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    fpr = fp / max(1, fp + tn)

    ap = None

    if tp + fn > 0 and fp + tn > 0:

        from sklearn.metrics import average_precision_score

        ap = float(average_precision_score(t, prob[v]))

    return dict(n_pos=tp + fn, n_neg=fp + tn, f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4),

                fpr=round(fpr, 4), ap=None if ap is None else round(ap, 4))





def ordinal_stats(cum_prob, lev):

    v = lev >= 0

    if v.sum() == 0:

        return dict(n_levels=0)

    pred = (cum_prob[v] >= 0.5).sum(-1)

    err = np.abs(pred - lev[v])

    return dict(n_levels=int(v.sum()), n_levels_pos=int((lev[v] > 0).sum()), mae=round(float(err.mean()), 4),

                within1=round(float((err <= 1).mean()), 4), within2=round(float((err <= 2).mean()), 4),

                exact=round(float((err == 0).mean()), 4))





def box_iou_np(a, b):

    ax0, ax1, ay0, ay1 = a[..., 0] - a[..., 1] / 2, a[..., 0] + a[..., 1] / 2, a[..., 2] - a[..., 3] / 2, a[..., 2] + a[..., 3] / 2

    bx0, bx1, by0, by1 = b[..., 0] - b[..., 1] / 2, b[..., 0] + b[..., 1] / 2, b[..., 2] - b[..., 3] / 2, b[..., 2] + b[..., 3] / 2

    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)

    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)

    inter = iw * ih

    union = np.clip(ax1 - ax0, 0, None) * np.clip(ay1 - ay0, 0, None) + np.clip(bx1 - bx0, 0, None) * np.clip(by1 - by0, 0, None) - inter

    return inter / np.clip(union, 1e-6, None)





def text_stats(P, L):

    out = dict(length_acc=round(float((P["length"].argmax(-1) == L["length"]).mean()), 4) if len(L["length"]) else None)

    comp = np.exp(P["comp"]) / np.exp(P["comp"]).sum(-1, keepdims=True)

    out["composition_mae"] = round(float(np.abs(comp - L["comp"]).mean()), 4)

    on = (L["slot_mask"][:, None] > 0) & (L["slot_present"] > 0)

    if on.sum():

        iou = box_iou_np(P["slot_box"], L["slot_box"])[on]

        out["slot_iou_mean"] = round(float(iou.mean()), 4)

        out["slot_iou_frac"] = round(float((iou >= 0.5).mean()), 4)

        sm = L["slot_mask"][:, None].repeat(L["slot_present"].shape[1], 1) > 0

        out["slot_present_acc"] = round(float(((P["slot_present"] > 0) == (L["slot_present"] > 0))[sm].mean()), 4)

        ch = L["slot_char"][on]

        okc = ch >= 0

        out["slot_char_acc"] = round(float((P["slot_char"].argmax(-1)[on][okc] == ch[okc]).mean()), 4) if okc.any() else None

        out["n_slots"] = int(on.sum())

    else:

        out.update(slot_iou_mean=None, slot_iou_frac=None, n_slots=0)

    return out





def head_scores(P, L, heads, ordinal, gates, use_ordinal=True):

    """Per-head stats + score + gap + passed, for the 46 feature heads and the 3 text heads."""

    gp = gates["presence"]

    res = {}

    oi = {h: j for j, h in enumerate(ordinal)}

    prob = 1 / (1 + np.exp(-P["presence"]))

    cum = 1 / (1 + np.exp(-P["ordinal"]))

    for j, h in enumerate(heads):

        st = f1_stats(prob[:, j], L["presence"][:, j])

        if h in oi:

            st.update(ordinal_stats(cum[:, oi[h]], L["level"][:, oi[h]]))

        if st["n_pos"] == 0 or st["n_neg"] == 0:

            st.update(score=None, gap=None, passed=None, kind="no_data")

            res[h] = st

            continue

        gaps = [gp["f1"] - st["f1"], gp["recall"] - st["recall"], st["fpr"] - gp["fpr"]]

        score = st["f1"]

        kind = "presence"

        if use_ordinal and h in oi and st.get("n_levels", 0) >= MIN_LEVELS and st.get("n_levels_pos", 0) > 0:

            score = min(score, st["within1"])

            gaps.append(gates["ordinal_within1"] - st["within1"])

            kind = "ordinal"

        gap = max(0.0, max(gaps))

        st.update(score=round(score, 4), gap=round(gap, 4), passed=gap <= 0, kind=kind)

        res[h] = st

    t = text_stats(P, L)

    res["length"] = dict(score=t["length_acc"], gap=round(max(0.0, gates["length_acc"] - t["length_acc"]), 4), kind="text")

    cs = 1 - t["composition_mae"]

    res["composition"] = dict(score=round(cs, 4), mae=t["composition_mae"], gap=round(max(0.0, t["composition_mae"] - gates["composition_mae"]), 4), kind="text")

    if t["slot_iou_frac"] is None:

        res["slots"] = dict(score=None, gap=None, passed=None, kind="no_data")

    else:

        res["slots"] = dict(score=t["slot_iou_frac"], gap=round(max(0.0, gates["slot_iou_frac"] - t["slot_iou_frac"]), 4), kind="text")

    for k in ("length", "composition", "slots"):

        if res[k]["gap"] is not None:

            res[k]["passed"] = res[k]["gap"] <= 0

    return res, t





def mean_score(scores):

    v = [s["score"] for s in scores.values() if s.get("score") is not None]

    return round(float(np.mean(v)), 5) if v else None





def update_states(history, scores, k, cfg_cur):

    """history: {head: [{"epoch", "score", "state", "nonimp"}...]}; returns {head: state} for epoch k (handoff §3).



    passed: at/above gate. improving: below gate and score >= best of the previous 2 epochs + 0.01 (every head counts as

    improving until it has 3 epochs of history). flagged: below gate and not improving for `flag_after` consecutive

    epochs (gets no targeted data; returns to improving on a +0.01 rise). stalled: below gate, not improving for fewer

    than `flag_after` epochs (still targeted). no_data: no positives or no negatives in the scored set.

    """

    delta, flag_after = float(cfg_cur["improve_delta"]), int(cfg_cur["flag_after"])

    states = {}

    for h, s in scores.items():

        hist = history.setdefault(h, [])

        sc = s.get("score")

        prev = [x["score"] for x in hist if x["score"] is not None]

        nonimp = 0

        if sc is None:

            st = "no_data"

        elif s.get("passed"):

            st = "passed"

        else:

            improving = not prev or sc >= max(prev[-2:]) + delta

            nonimp = 0 if improving else (hist[-1].get("nonimp", 0) if hist else 0) + 1

            if len(prev) < 2 or improving:

                st = "improving"

            elif nonimp >= flag_after:

                st = "flagged"

            else:

                st = "stalled"

        hist.append(dict(epoch=k, score=sc, state=st, nonimp=nonimp, gap=s.get("gap")))

        states[h] = st

    return states





def flagged_streak(history, h):

    n = 0

    for x in reversed(history.get(h, [])):

        if x["state"] != "flagged":

            break

        n += 1

    return n

