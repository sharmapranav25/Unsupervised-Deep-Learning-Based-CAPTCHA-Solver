"""Hybrid v3 losses (handoff §2): every head has its own loss (logged); the global loss weights the 4 groups
(presence, ordinal, text, coverage) by learned uncertainty. Null labels are masked, never 0. No manual per-head weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = ["presence", "ordinal", "text", "coverage"]


def masked_mean(x, m):
    m = m.float()
    return (x * m).sum() / m.sum().clamp(min=1.0), m.sum()


def box_iou(a, b):
    """(cx, w, cy, h) boxes -> IoU (same shape[:-1])."""
    ax0, ax1 = a[..., 0] - a[..., 1] / 2, a[..., 0] + a[..., 1] / 2
    ay0, ay1 = a[..., 2] - a[..., 3] / 2, a[..., 2] + a[..., 3] / 2
    bx0, bx1 = b[..., 0] - b[..., 1] / 2, b[..., 0] + b[..., 1] / 2
    by0, by1 = b[..., 2] - b[..., 3] / 2, b[..., 2] + b[..., 3] / 2
    iw = (torch.minimum(ax1, bx1) - torch.maximum(ax0, bx0)).clamp(min=0)
    ih = (torch.minimum(ay1, by1) - torch.maximum(ay0, by0)).clamp(min=0)
    inter = iw * ih
    union = (ax1 - ax0).clamp(min=0) * (ay1 - ay0).clamp(min=0) + (bx1 - bx0).clamp(min=0) * (by1 - by0).clamp(min=0) - inter
    return inter / union.clamp(min=1e-6)


class UncertaintyWeights(nn.Module):
    def __init__(self, n=len(GROUPS)):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n))

    def forward(self, group_losses, active=None):
        tot = 0.0
        for i, g in enumerate(GROUPS):
            if active is not None and g not in active:
                continue
            tot = tot + torch.exp(-self.log_var[i]) * group_losses[g] + self.log_var[i]
        return tot


def compute(out, b, focal_mask, char_weight=0.2, gamma=2.0, head_mask=None, ord_mask=None):
    """Returns (group_losses: dict of scalars, per: presence/ordinal -> (per-head loss vector, valid counts),
    text -> dict of component losses, coverage -> scalar)."""
    per = {}
    # ---- presence: BCE, focal for rare heads (focal_mask: bool [46])
    logit = out["presence"].float()
    y = b["presence"].long()
    valid = y >= 0
    yt = y.clamp(min=0).float()
    bce = F.binary_cross_entropy_with_logits(logit, yt, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(yt > 0.5, p, 1 - p)
    foc = bce * (1 - pt).pow(gamma)
    pl = torch.where(focal_mask.view(1, -1), foc, bce)
    vf = valid.float()
    pres_heads = (pl * vf).sum(0) / vf.sum(0).clamp(min=1.0)
    per["presence"] = (pres_heads, vf.sum(0))
    hm = head_mask.float() if head_mask is not None else torch.ones_like(pres_heads)
    pres = (pres_heads * hm).sum() / hm.sum().clamp(min=1.0)  # mean over the active heads
    # ---- ordinal: cumulative-logit BCE over thresholds 1..10, only where a level exists
    ol = out["ordinal"].float()  # B x 27 x 10
    lev = b["level"].long()
    lv = lev >= 0
    thr = torch.arange(1, 11, device=ol.device).view(1, 1, 10)
    tgt = (lev.clamp(min=0).unsqueeze(-1) >= thr).float()
    obce = F.binary_cross_entropy_with_logits(ol, tgt, reduction="none").mean(-1)
    lf = lv.float()
    ord_heads = (obce * lf).sum(0) / lf.sum(0).clamp(min=1.0)
    per["ordinal"] = (ord_heads, lf.sum(0))
    om = ord_mask.float() if ord_mask is not None else torch.ones_like(ord_heads)
    ordl = (ord_heads * om).sum() / om.sum().clamp(min=1.0)
    # ---- text
    ln = F.cross_entropy(out["length"].float(), b["length"].long())
    comp = (torch.softmax(out["comp"].float(), -1) - b["comp"].float()).abs().sum(-1).mean()
    sm = b["slot_mask"].bool().view(-1, 1).expand_as(b["slot_present"])
    sp = b["slot_present"].float()
    s_pres, _ = masked_mean(F.binary_cross_entropy_with_logits(out["slot_present"].float(), sp, reduction="none"), sm)
    on = sm & (sp > 0.5)
    pb, tb = out["slot_box"].float(), b["slot_box"].float()
    s_box, _ = masked_mean((pb - tb).abs().sum(-1) + (1 - box_iou(pb, tb)), on)
    s_ang, _ = masked_mean(((out["slot_angle"].float() - b["slot_angle"].float()) ** 2).sum(-1), on)
    g = b["slot_group"].long()
    s_grp, _ = masked_mean(F.cross_entropy(out["slot_group"].float().reshape(-1, 3), g.clamp(min=0).reshape(-1),
                                           reduction="none").view_as(g), on & (g >= 0))
    s_col, _ = masked_mean((out["slot_colour"].float() - b["slot_colour"].float()).abs().sum(-1), on)
    o = b["slot_outline"].long()
    s_out, _ = masked_mean(F.binary_cross_entropy_with_logits(out["slot_outline"].float(), o.clamp(min=0).float(),
                                                              reduction="none"), on & (o >= 0))
    c = b["slot_char"].long()
    s_chr, _ = masked_mean(F.cross_entropy(out["slot_char"].float().reshape(-1, out["slot_char"].shape[-1]),
                                           c.clamp(min=0).reshape(-1), reduction="none").view_as(c), on & (c >= 0))
    text_parts = dict(length=ln, composition=comp, slot_present=s_pres, slot_box=s_box, slot_angle=s_ang, slot_group=s_grp,
                      slot_colour=s_col, slot_outline=s_out, slot_char=s_chr)
    text = ln + comp + s_pres + s_box + s_ang + s_grp + s_col + s_out + char_weight * s_chr
    per["text"] = text_parts
    # ---- coverage: per-token BCE (soft targets), real tokens of images with a coverage map
    cm = out["token_mask"] & b["cov_ok"].bool().view(-1, 1, 1)
    cv = out["coverage"].float()
    tgt = b["coverage"].float()[:, :, :cv.shape[2]]
    cl = F.binary_cross_entropy_with_logits(cv, tgt, reduction="none").mean(-1)
    cov, _ = masked_mean(cl, cm)
    per["coverage"] = cov
    return dict(presence=pres, ordinal=ordl, text=text, coverage=cov), per
