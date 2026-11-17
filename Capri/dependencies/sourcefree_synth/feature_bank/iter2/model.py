"""Iteration 2 baseline model (docs/ITERATION2_PLAN.md, decision 5): shared CNN stem, then two branches.

  stem (iteration-1 CNN, receptive field 25 px, stride 4)
   ├─ effect branch (CNN): one residual block -> 6 x W/8 tokens -> per-head query pool -> GroupedMLP
   │     37 effect heads: presence + 10 cumulative "level >= k" logits
   └─ text branch (transformer): 5 blocks, 2-D relative position bias, width embedding
         CTC reading (per token column, 62 characters + blank), length (2..8),
         16 text heads (presence + 10 ordinal), image font heads: category (5), stroke class (3),
         weight / width / x_height / slant (10 ordinal each), italic (1)

Each head's loss reaches only its own branch and the shared stem. Padding is masked exactly as in iteration 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_bank.train.model import CNN, Block, GroupedMLP, QueryPool, ResBlock, col_mask

N_CHARS = 62
N_CAT, N_STROKE, N_FONT_ORD = 5, 3, 4


def token_mask(cols, R, C):
    B = cols.shape[0]
    return (torch.arange(C, device=cols.device).view(1, 1, C) < cols.view(-1, 1, 1)).expand(B, R, C).reshape(B, R * C)


class Iter2Model(nn.Module):
    def __init__(self, n_effects=37, n_text=16, d=160, d_eff=128, layers=5, heads=4, mlp=640, dropout=0.1,
                 drop_path=0.1, max_cols=128, rows=6, ctc_frames=1, **_):
        super().__init__()
        self.n_effects, self.n_text, self.max_cols = n_effects, n_text, max_cols
        self.cnn = CNN()
        # ---- effect branch (CNN)
        self.eff_block = ResBlock(96, d_eff)
        self.eff_pos = nn.Parameter(torch.zeros(rows, d_eff))  # row position only (effects are not column-anchored)
        self.eff_norm = nn.LayerNorm(d_eff)
        self.eff_pool = QueryPool(n_effects, d_eff, 4)
        self.eff_head = GroupedMLP(n_effects, d_eff, 11)
        # ---- text branch (transformer)
        self.proj = nn.Linear(96, d)
        self.width_emb = nn.Embedding(max_cols + 1, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp, rows, max_cols, dropout, drop_path * i / max(1, layers - 1))
                                     for i in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.col_pool = nn.Linear(rows * d, d)  # a token column (6 rows) -> one CTC frame
        self.ctc_frames = ctc_frames  # CTC frames per token column (2 = 4 px frames)
        self.ctc_up = nn.ConvTranspose1d(d, d, ctc_frames, ctc_frames) if ctc_frames > 1 else None
        self.ctc = nn.Sequential(nn.GELU(), nn.Linear(d, N_CHARS + 1))  # index 0 = blank
        n_q = n_text + 1 + 7  # text heads, length, font heads
        self.txt_pool = QueryPool(n_q, d, heads)
        self.txt_head = GroupedMLP(n_text, d, 11)
        self.length = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))
        self.font_cat = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_CAT))
        self.font_stroke = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_STROKE))
        self.font_ord = GroupedMLP(N_FONT_ORD, d, 10)
        self.font_italic = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, image, cols):
        f = self.cnn(image, cols)  # B x 96 x 24 x W/4
        B = f.shape[0]
        m4 = col_mask(cols, f.shape[-1], 4)
        # effect branch
        e = self.eff_block(f, m4)
        e = F.avg_pool2d(e, (4, 2))  # 6 x W/8 tokens
        R, C = e.shape[-2], e.shape[-1]
        km = token_mask(cols, R, C)
        et = self.eff_norm(e.permute(0, 2, 3, 1) + self.eff_pos.view(1, R, 1, -1)).reshape(B, R * C, -1)
        eo = self.eff_head(self.eff_pool(et, km))
        # text branch
        t = F.avg_pool2d(f, (4, 2))
        x = self.proj(t.permute(0, 2, 3, 1)).reshape(B, R * C, -1)
        x = x + self.width_emb(cols.clamp(max=self.max_cols)).unsqueeze(1)
        for b in self.blocks:
            x = b(x, km, R, C)
        x = self.norm(x)
        colf = self.col_pool(x.view(B, R, C, -1).permute(0, 2, 1, 3).reshape(B, C, -1))
        if self.ctc_up is not None:
            colf = self.ctc_up(colf.transpose(1, 2)).transpose(1, 2)  # B x C*f x d
        q = self.txt_pool(x, km)
        nt = self.n_text
        to = self.txt_head(q[:, :nt])
        fq = q[:, nt + 1:]
        return dict(eff_presence=eo[..., 0], eff_ordinal=eo[..., 1:],
                    ctc=self.ctc(colf),  # B x C*ctc_frames x 63
                    txt_presence=to[..., 0], txt_ordinal=to[..., 1:],
                    length=self.length(q[:, nt]),
                    font_cat=self.font_cat(fq[:, 0]), font_stroke=self.font_stroke(fq[:, 1]),
                    font_ordinal=self.font_ord(fq[:, 2:6]), font_italic=self.font_italic(fq[:, 6]).squeeze(-1))

    def branches(self):
        """Parameter groups: stem, effect branch, text branch (for logging / checks)."""
        g = dict(stem=[], effect=[], text=[])
        for n, p in self.named_parameters():
            g["stem" if n.startswith("cnn.") else "effect" if n.startswith("eff_") else "text"].append((n, p))
        return g


def param_counts(m):
    return {k: sum(p.numel() for _, p in v) for k, v in m.branches().items()}
