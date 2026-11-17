"""Hybrid v3 model (docs/HYBRID_V3_TRAINING_HANDOFF.md §2): local CNN -> fine tokens -> transformer -> every output.

Padding: every image is right-padded by edge replication (data.collate), GroupNorm statistics are taken over the real
columns only, attention keys / pools / losses mask padded tokens, so an image's outputs do not depend on its batch.
Freezing unit: one output channel of a CNN conv (its filter + bias + following GroupNorm scale/shift): CNN.units().
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COV = 10
N_CHARS = 62


class MaskedGroupNorm(nn.GroupNorm):
    """GroupNorm whose statistics use only the valid columns (mask B x 1 x 1 x W, 1 = real)."""
    detach_stats = False

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        B, C, Hh, W = x.shape
        g = self.num_groups
        xg = x.float().view(B, g, C // g, Hh, W)
        m = mask.float().view(B, 1, 1, 1, W)
        n = m.sum(dim=(2, 3, 4), keepdim=True) * (C // g) * Hh
        mean = (xg * m).sum(dim=(2, 3, 4), keepdim=True) / n
        var = (((xg - mean) ** 2) * m).sum(dim=(2, 3, 4), keepdim=True) / n
        if MaskedGroupNorm.detach_stats:  # receptive-field probe: only the local (conv) gradient path
            mean, var = mean.detach(), var.detach()
        y = ((xg - mean) / torch.sqrt(var + self.eps)).view(B, C, Hh, W).to(x.dtype)
        return y * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)


class ConvGN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2)
        self.gn = MaskedGroupNorm(8 if cout % 8 == 0 else 4, cout)
        self.act = act
        self.stride = stride

    def forward(self, x, mask):
        y = self.gn(self.conv(x), mask)
        return F.silu(y) if self.act else y


def col_mask(cols, width, scale):
    """B x 1 x 1 x width mask of real columns at a feature resolution of `scale` px per column."""
    w = (cols.float() * 8 / scale).ceil().long()
    ar = torch.arange(width, device=cols.device).view(1, -1)
    return (ar < w.view(-1, 1)).view(-1, 1, 1, width)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = ConvGN(cin, cout, 3)
        self.b = ConvGN(cout, cout, 1, act=False)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None
        self.grow = nn.ModuleList()  # stuck-head growth branches (zero output at creation)

    def forward(self, x, mask):
        y = self.b(self.a(x, mask), mask)
        for g in self.grow:
            y = y + g(x, mask)
        return F.silu(y + (self.skip(x) if self.skip is not None else x))


class GrowBranch(nn.Module):
    def __init__(self, cin, n, cout):
        super().__init__()
        self.a = ConvGN(cin, n, 3)
        self.out = nn.Conv2d(n, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, mask):
        return self.out(self.a(x, mask))


class CNN(nn.Module):
    """Receptive field 25 px; output stride 4 (24 x W/4 x 96)."""

    def __init__(self):
        super().__init__()
        self.s1 = ConvGN(3, 32, 3, 1)
        self.s2 = ConvGN(32, 64, 3, 2)
        self.s3 = ConvGN(64, 64, 3, 2)
        self.r1 = ResBlock(64, 96)
        self.r2 = ResBlock(96, 96)

    def forward(self, x, cols, return_acts=False):
        acts = {}
        W = x.shape[-1]
        m1 = col_mask(cols, W, 1)
        x = self.s1(x, m1); acts["s1"] = x
        m2 = col_mask(cols, x.shape[-1] // 2 + x.shape[-1] % 2, 2)
        x = self.s2(x, m2); acts["s2"] = x
        m4 = col_mask(cols, x.shape[-1] // 2 + x.shape[-1] % 2, 4)
        x = self.s3(x, m4); acts["s3"] = x
        x = self.r1(x, m4); acts["r1"] = x
        x = self.r2(x, m4); acts["r2"] = x
        return (x, acts) if return_acts else x

    def units(self):
        """Freezing units: (name, conv, gn) per conv followed by a GroupNorm; channel c of each is one unit."""
        out = []
        for name, cg in [("s1", self.s1), ("s2", self.s2), ("s3", self.s3), ("r1.a", self.r1.a), ("r1.b", self.r1.b),
                         ("r2.a", self.r2.a), ("r2.b", self.r2.b)]:
            out.append((name, cg.conv, cg.gn))
        return out


class RelAttention(nn.Module):
    def __init__(self, d, heads, rows, max_cols, drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.rows, self.max_cols = rows, max_cols
        self.bias = nn.Parameter(torch.zeros(heads, 2 * rows - 1, 2 * max_cols - 1))
        self.drop = drop

    def rel_bias(self, R, C, device):
        r = torch.arange(R, device=device)
        c = torch.arange(C, device=device)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        rr, cc = rr.reshape(-1), cc.reshape(-1)
        dr = (rr.view(-1, 1) - rr.view(1, -1)) + self.rows - 1
        dc = (cc.view(-1, 1) - cc.view(1, -1)).clamp(-(self.max_cols - 1), self.max_cols - 1) + self.max_cols - 1
        return self.bias[:, dr, dc]  # heads x N x N

    def forward(self, x, key_mask, R, C):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        bias = self.rel_bias(R, C, x.device).unsqueeze(0).to(q.dtype)
        mask = bias + torch.where(key_mask.view(B, 1, 1, N), 0.0, float("-inf")).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, N, D))


def drop_path(x, p, training):
    if not training or p == 0:
        return x
    keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= p
    return x * keep / (1 - p)


class Block(nn.Module):
    def __init__(self, d, heads, mlp, rows, max_cols, drop, dp):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = RelAttention(d, heads, rows, max_cols, drop)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, d))
        self.drop = nn.Dropout(drop)
        self.dp = dp

    def forward(self, x, key_mask, R, C):
        x = x + drop_path(self.drop(self.attn(self.n1(x), key_mask, R, C)), self.dp, self.training)
        return x + drop_path(self.drop(self.mlp(self.n2(x))), self.dp, self.training)


class QueryPool(nn.Module):
    """One learned query per head attention-pools the tokens (padding masked)."""

    def __init__(self, n, d, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, key_mask):
        q = self.q.unsqueeze(0).expand(x.shape[0], -1, -1)
        y, _ = self.attn(q, x, x, key_padding_mask=~key_mask, need_weights=False)
        return self.norm(y)


class GroupedMLP(nn.Module):
    """n independent MLPs d -> d -> out (one per head), batched."""

    def __init__(self, n, d, out):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, d, d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(n, d))
        self.w2 = nn.Parameter(torch.randn(n, d, out) / math.sqrt(d) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n, out))

    def forward(self, x):  # x: B x n x d
        h = F.gelu(torch.einsum("bnd,nde->bne", x, self.w1) + self.b1)
        return torch.einsum("bnd,nde->bne", h, self.w2) + self.b2


class SlotDecoder(nn.Module):
    def __init__(self, slots, d, heads, layers, drop):
        super().__init__()
        self.q = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True,
                                                                activation="gelu") for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, key_mask):
        q = self.q.unsqueeze(0).expand(x.shape[0], -1, -1)
        for l in self.layers:
            q = l(q, x, memory_key_padding_mask=~key_mask)
        return self.norm(q)


class Hybrid(nn.Module):
    def __init__(self, n_heads=46, n_ordinal=27, ordinal_index=None, d=160, layers=5, heads=4, mlp=640, dropout=0.1,
                 drop_path=0.1, slots=8, slot_layers=3, max_cols=128, rows=6, **_):
        super().__init__()
        self.n_heads, self.n_ordinal = n_heads, n_ordinal
        self.register_buffer("ordinal_index", torch.tensor(ordinal_index or list(range(n_ordinal))), persistent=False)
        self.rows, self.max_cols = rows, max_cols
        self.cnn = CNN()
        self.proj = nn.Linear(96, d)
        self.width_emb = nn.Embedding(max_cols + 1, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp, rows, max_cols, dropout, drop_path * i / max(1, layers - 1))
                                     for i in range(layers)])
        self.norm = nn.LayerNorm(d)
        # 46 feature heads + 2 whole-image text queries (length, composition)
        self.pool = QueryPool(n_heads + 2, d, heads)
        self.head_mlp = GroupedMLP(n_heads, d, 11)  # per head: [presence logit, 10 cumulative "level >= k" logits]
        self.length = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))
        self.comp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.slots = SlotDecoder(slots, d, heads, slot_layers, dropout)
        self.slot_out = nn.ModuleDict(dict(present=nn.Linear(d, 1), box=nn.Linear(d, 4), angle=nn.Linear(d, 2),
                                           group=nn.Linear(d, 3), colour=nn.Linear(d, 3), outline=nn.Linear(d, 1),
                                           char=nn.Linear(d, N_CHARS)))
        self.coverage = nn.Linear(d, N_COV)

    def tokens(self, image, cols, return_acts=False):
        f = self.cnn(image, cols, return_acts)
        f, acts = f if return_acts else (f, None)
        B, Cc, Hf, Wf = f.shape
        t = F.avg_pool2d(f, (4, 2))  # 24 x W/4 -> 6 x W/8: one token = 16 px tall x 8 px wide
        R, C = t.shape[-2], t.shape[-1]
        x = self.proj(t.permute(0, 2, 3, 1)).reshape(B, R * C, -1)
        x = x + self.width_emb(cols.clamp(max=self.max_cols)).unsqueeze(1)
        km = (torch.arange(C, device=x.device).view(1, 1, C) < cols.view(-1, 1, 1)).expand(B, R, C).reshape(B, R * C)
        return x, km, R, C, acts

    def forward(self, image, cols, return_acts=False):
        x, km, R, C, acts = self.tokens(image, cols, return_acts)
        for b in self.blocks:
            x = b(x, km, R, C)
        x = self.norm(x)
        pooled = self.pool(x, km)
        hv = pooled[:, :self.n_heads]
        ho = self.head_mlp(hv)
        out = dict(presence=ho[..., 0],
                   ordinal=ho[:, self.ordinal_index, 1:],  # non-ordinal heads' level outputs are unused
                   length=self.length(pooled[:, self.n_heads]),
                   comp=self.comp(pooled[:, self.n_heads + 1]),
                   coverage=self.coverage(x).view(x.shape[0], R, C, N_COV),
                   token_mask=km.view(-1, R, C))
        s = self.slots(x, km)
        for k, m in self.slot_out.items():
            out["slot_" + k] = m(s).squeeze(-1) if k in ("present", "outline") else m(s)
        out["slot_box"] = torch.sigmoid(out["slot_box"])
        out["slot_colour"] = torch.sigmoid(out["slot_colour"])
        if return_acts:
            out["acts"] = acts
        return out


def build(cfg_model, ordinal_index):
    return Hybrid(ordinal_index=ordinal_index, **cfg_model)


def param_counts(m):
    tot = sum(p.numel() for p in m.parameters())
    cnn = sum(p.numel() for p in m.cnn.parameters())
    tr = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("proj", "width_emb", "blocks", "norm")))
    return dict(total=tot, cnn=cnn, transformer=tr, heads=tot - cnn - tr)
