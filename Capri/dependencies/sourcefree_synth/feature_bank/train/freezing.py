"""Dynamic CNN-channel freezing (handoff §3). Unit = one output channel of a CNN conv (filter + bias + its GroupNorm
scale/shift). Frozen channels get zero gradient and are restored after every optimiser step (so AdamW momentum and
weight decay cannot move them). The transformer is never frozen.
"""
import numpy as np
import torch


class Freezer:
    def __init__(self, model, cfg_f):
        self.cfg = cfg_f
        self.units = model.cnn.units()
        self.frozen = {n: np.zeros(c.out_channels, bool) for n, c, _ in self.units}
        self.saved = {}
        self.grown = []          # [(epoch, head)] growth events
        self.released = {}       # head -> epoch of the last stuck release
        self.events = []

    # ---------------------------------------------------------------- state
    def n_channels(self):
        return sum(len(v) for v in self.frozen.values())

    def n_frozen(self):
        return int(sum(v.sum() for v in self.frozen.values()))

    def state(self):
        return dict(frozen={k: v.tolist() for k, v in self.frozen.items()}, grown=self.grown, released=self.released)

    def load(self, st, model):
        self.units = model.cnn.units()
        self.frozen = {k: np.array(v, bool) for k, v in st["frozen"].items()}
        self.grown = [tuple(x) for x in st.get("grown", [])]
        self.released = st.get("released", {})
        self.snapshot_frozen()

    def snapshot_frozen(self):
        self.saved = {}
        for n, conv, gn in self.units:
            idx = np.nonzero(self.frozen[n])[0]
            if len(idx):
                t = torch.as_tensor(idx, device=conv.weight.device)
                self.saved[n] = (t, conv.weight.detach()[t].clone(), conv.bias.detach()[t].clone(),
                                 gn.weight.detach()[t].clone(), gn.bias.detach()[t].clone())

    # ---------------------------------------------------------------- training hooks
    def mask_grads(self):
        for n, conv, gn in self.units:
            if n in self.saved:
                t = self.saved[n][0]
                for p in (conv.weight, conv.bias, gn.weight, gn.bias):
                    if p.grad is not None:
                        p.grad[t] = 0

    @torch.no_grad()
    def restore(self):
        for n, conv, gn in self.units:
            if n in self.saved:
                t, w, b, gw, gb = self.saved[n]
                conv.weight[t] = w
                conv.bias[t] = b
                gn.weight[t] = gw
                gn.bias[t] = gb

    # ---------------------------------------------------------------- per-epoch statistics
    @torch.no_grad()
    def weights(self):
        out = {}
        for n, conv, gn in self.units:
            w = torch.cat([conv.weight.flatten(1), conv.bias[:, None], gn.weight[:, None], gn.bias[:, None]], 1)
            out[n] = w.float().cpu().numpy()
        return out

    def update(self, k, stats, prev_stats, prev2_stats, states, scores, best, history_flag_streak):
        """stats: dict(weights, act, importance{head: {unit: vec}}) for this epoch; prev/prev2: earlier epochs.
        states: head -> state; scores: head -> score; best: head -> best score so far.
        Returns the events of this update."""
        cf = self.cfg
        ev = []
        if k < int(cf["start_epoch"]) or prev_stats is None or prev2_stats is None:
            return ev
        # dependent heads per channel: heads for which it is in the top 20% importance (over all CNN channels)
        dep = {n: [set() for _ in range(len(self.frozen[n]))] for n in self.frozen}
        for h, per_unit in stats["importance"].items():
            allv = np.concatenate([per_unit[n] for n in self.frozen])
            if allv.max() <= 0:
                continue
            thr = np.quantile(allv, 1 - float(cf["importance_top"]))
            for n in self.frozen:
                for c in np.nonzero(per_unit[n] >= thr)[0]:
                    dep[n][c].add(h)

        def rel(a, b):
            return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-8)

        cand = []
        for n in self.frozen:
            dw1 = rel(stats["weights"][n], prev_stats["weights"][n])
            dw2 = rel(prev_stats["weights"][n], prev2_stats["weights"][n])
            da1 = np.abs(stats["act"][n] - prev_stats["act"][n]) / np.maximum(np.abs(prev_stats["act"][n]), 1e-6)
            da2 = np.abs(prev_stats["act"][n] - prev2_stats["act"][n]) / np.maximum(np.abs(prev2_stats["act"][n]), 1e-6)
            for c in range(len(self.frozen[n])):
                heads = dep[n][c]
                if self.frozen[n][c]:
                    # unfreeze when a dependent head drops >= 0.02 below its best
                    drop = [h for h in heads if scores.get(h) is not None and best.get(h) is not None
                            and scores[h] <= best[h] - float(cf["unfreeze_drop"])]
                    if drop:
                        self.frozen[n][c] = False
                        ev.append(dict(event="unfreeze", unit=n, channel=int(c), heads=sorted(drop)))
                    continue
                stable = max(dw1[c], dw2[c]) < float(cf["weight_change"]) and max(da1[c], da2[c]) < float(cf["act_drift"])
                if stable and all(states.get(h) == "passed" for h in heads):
                    cand.append((float(max(dw1[c], dw2[c])), n, int(c)))
        cap = int(float(cf["max_frozen_fraction"]) * self.n_channels())
        room = cap - self.n_frozen()
        for _, n, c in sorted(cand)[:max(0, room)]:
            self.frozen[n][c] = True
            ev.append(dict(event="freeze", unit=n, channel=c))
        if len(cand) > max(0, room):
            ev.append(dict(event="cap_reached", candidates=len(cand), cap=cap))
        # stuck heads
        for h, st in states.items():
            if st != "flagged":
                continue
            streak = history_flag_streak(h)
            imp = stats["importance"].get(h)
            if streak == 3 and imp is not None:
                fz = [(imp[n][c], n, c) for n in self.frozen for c in np.nonzero(self.frozen[n])[0]]
                rel_ = sorted(fz)[:int(cf["stuck_release"])]
                for _, n, c in rel_:
                    self.frozen[n][c] = False
                self.released[h] = k
                ev.append(dict(event="stuck_release", head=h, channels=[(n, int(c)) for _, n, c in rel_]))
            if streak == 3 + int(cf["stuck_grow_after"]) and h not in {g[1] for g in self.grown}:
                self.grown.append((k, h))
                ev.append(dict(event="grow", head=h, unit="r2", channels=int(cf["grow_channels"])))
        self.snapshot_frozen()
        self.events += [dict(epoch=k, **e) for e in ev]
        return ev

    def freeze_map(self):
        return dict(frozen={n: np.nonzero(v)[0].tolist() for n, v in self.frozen.items()},
                    counts={n: int(v.sum()) for n, v in self.frozen.items()},
                    total_frozen=self.n_frozen(), total_channels=self.n_channels(),
                    fraction=round(self.n_frozen() / max(1, self.n_channels()), 4), grown=self.grown)
