"""Synthetic-only data access: DarkBay A+B training LMDBs and the SynthTIGER validation set."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
TRAIN_LMDBS = {
    'A': ROOT / 'datasets/darkbay/A/train_lmdb',
    'B': ROOT / 'datasets/darkbay/B/train_lmdb',
}
VAL_DIR = ROOT / 'datasets/darkbay/validation/controller4_synthtiger_1000'
CHARSET = ''.join((ROOT / 'datasets/darkbay/A/charset.txt').read_text(encoding='utf-8').split())
ORDER_SEED = 20261114


class LmdbUnion(Dataset):
    """Concatenation of the A and B LMDBs; index i -> (arm, key). Opens LMDB lazily per worker."""

    def __init__(self, transform, arms=('A', 'B')):
        import lmdb
        self.transform = transform
        self.arms = list(arms)
        self.sizes = []
        for arm in self.arms:
            env = lmdb.open(str(TRAIN_LMDBS[arm]), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                self.sizes.append(int(txn.get(b'num-samples')))
            env.close()
        self.offsets = np.cumsum([0] + self.sizes)
        self._envs = None

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, i):
        a = int(np.searchsorted(self.offsets, i, side='right') - 1)
        return a, int(i - self.offsets[a]) + 1  # LMDB keys are 1-based

    def __getitem__(self, i):
        import lmdb
        if self._envs is None:
            self._envs = [lmdb.open(str(TRAIN_LMDBS[a]), readonly=True, lock=False, readahead=False, meminit=False)
                          for a in self.arms]
        a, k = self.locate(i)
        with self._envs[a].begin() as txn:
            img = Image.open(io.BytesIO(txn.get(b'image-%09d' % k))).convert('RGB')
            label = txn.get(b'label-%09d' % k).decode('utf-8')
        return self.transform(img), label


class ValSet(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.rows = []
        for line in (VAL_DIR / 'labels.tsv').read_text(encoding='utf-8').splitlines():
            if line.strip():
                rel, label = line.split('\t')
                self.rows.append((VAL_DIR / rel, label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, label = self.rows[i]
        return self.transform(Image.open(p).convert('RGB')), label


def fixed_order(n, seed=ORDER_SEED):
    """One shared permutation so every teacher sees the same images in the same order."""
    return np.random.default_rng(seed).permutation(n)


def collate(batch):
    import torch
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)


def data_identity():
    receipts = {}
    for arm in ('A', 'B'):
        r = json.loads((ROOT / f'datasets/darkbay/{arm}/data_receipt.json').read_text(encoding='utf-8'))
        receipts[arm] = {'images': r['images'], 'lmdb_sha256': r['lmdb_sha256']}
    return {'train': receipts, 'validation': str(VAL_DIR.relative_to(ROOT)), 'charset': CHARSET,
            'order_seed': ORDER_SEED, 'augmentation': None}
