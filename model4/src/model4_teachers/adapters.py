"""One adapter per pretrained STR. Each exposes the same small interface used by run.py.

LR rule (uniform): peak LR = 0.1 x the model's own published from-scratch training LR.
Fallback where no comparable AdamW recipe exists (MGP-STR: Adadelta; TrOCR: published LR is
already a fine-tuning LR): 5e-5. All use AdamW, 5% linear warmup then cosine to zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from .data import CHARSET, ROOT

TP = ROOT / 'model4/third_party'
W = ROOT / 'model4/weights'
BICUBIC = T.InterpolationMode.BICUBIC


def _prepend(paths):
    for p in reversed([str(p) for p in paths]):
        if p not in sys.path:
            sys.path.insert(0, p)


def _decay_groups(named, lr, wd=0.01):
    decay, no_decay = [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith('.bias') else decay).append(p)
    return [{'params': decay, 'lr': lr, 'weight_decay': wd}, {'params': no_decay, 'lr': lr, 'weight_decay': 0.0}]


class Adapter:
    name = ''
    family = ''
    native_lr = None
    lr_rule = '0.1 x native from-scratch LR'
    micro_batch = 128
    weight_files: list[Path] = []
    source = ''

    @property
    def lr(self):
        return 0.1 * self.native_lr

    def param_groups(self, m):
        return _decay_groups(m.named_parameters(), self.lr)

    def notes(self):
        return {}


# ---------------------------------------------------------------- baudm/parseq strhub family
class Strhub(Adapter):
    family = 'strhub (baudm/parseq @1902db0)'
    source = 'https://github.com/baudm/parseq/releases/tag/v1.0.0'
    _lrs = {'parseq': 7e-4, 'abinet': 3.4e-4, 'trba': 6.9e-4, 'vitstr': 8.9e-4, 'crnn': 5.1e-4}
    _files = {'parseq': 'parseq/parseq-bb5792a6.pt', 'abinet': 'abinet/abinet-1d1e373e.pt',
              'trba': 'trba/trba-cfaed284.pt', 'vitstr': 'vitstr/vitstr-26d0fcf4.pt', 'crnn': 'crnn/crnn-679d0e31.pt'}

    def __init__(self, exp):
        self.name = self.exp = exp
        self.native_lr = self._lrs[exp]
        self.weight_files = [W / self._files[exp]]

    def build(self):
        _prepend([TP / 'parseq'])
        from strhub.models.utils import create_model
        m = create_model(self.exp, pretrained=False)
        sd = torch.load(self.weight_files[0], map_location='cpu', weights_only=True)
        (m.model if self.exp == 'parseq' else m).load_state_dict(sd)
        if self.exp == 'abinet':
            # Fine-tune the full pretrained model: skip the from-scratch vision/LM pretraining phase and
            # do NOT re-initialise the pretrained alignment module (upstream does so at its first joint step).
            m.__class__ = type('ABINetFineTune', (m.__class__,), {'_pretraining': property(lambda s: False)})
            m._reset_alignment = False
            m._reset_optimizers = False
        m.log = lambda *a, **k: None
        self.img_size = tuple(m.hparams.img_size)
        return m

    def transform(self):
        return T.Compose([T.Resize(self.img_size, BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def loss(self, m, x, labels):
        return m.training_step((x, labels), 0)

    @torch.no_grad()
    def predict(self, m, x):
        preds, _ = m.tokenizer.decode(m(x).softmax(-1))
        return preds

    def notes(self):
        return {'charset': 'pretrained 94_full tokenizer kept; plain decoding'}


# ---------------------------------------------------------------- CLIP4STR (strhub fork)
class Clip4str(Adapter):
    name = 'clip4str_b'
    family = 'CLIP4STR (VamosC/CLIP4STR @9730b64), strhub fork'
    source = 'https://github.com/VamosC/CLIP4STR/releases/tag/1.0.0 (CLIP4STR-B, Real 3.3M)'
    native_lr = 8.4e-5
    micro_batch = 32
    weight_files = [W / 'clip4str_b/clip4str_base16x16_d70bde1f2d.ckpt', W / 'clip_vit_b16/ViT-B-16.pt']
    MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def build(self):
        _prepend([TP / 'CLIP4STR'])
        import typing
        import pytorch_lightning.utilities.types as plt
        if not hasattr(plt, 'EPOCH_OUTPUT'):  # removed in Lightning 2.x; only used as a type alias upstream
            plt.EPOCH_OUTPUT = typing.Any
        # vl_str/system.py asserts a hard-coded author path at import time; the real CLIP weights path is
        # passed explicitly below, so satisfy only that import-time assert without editing pinned source.
        import os
        real_exists = os.path.exists
        os.path.exists = lambda p: True if str(p).startswith('/home/shu') else real_exists(p)
        try:
            from strhub.models.vl_str import system as _vl  # noqa: F401
        finally:
            os.path.exists = real_exists
        from strhub.models.utils import load_from_checkpoint
        m = load_from_checkpoint(str(self.weight_files[0]), clip_pretrained=str(self.weight_files[1]))
        m.log = lambda *a, **k: None
        self.img_size = tuple(m.hparams.img_size)
        self.coef_lr = float(m.hparams.coef_lr)
        return m

    def transform(self):
        return T.Compose([T.Resize(self.img_size, BICUBIC), T.ToTensor(), T.Normalize(self.MEAN, self.STD)])

    def param_groups(self, m):
        enc = [(n, p) for n, p in m.named_parameters() if 'clip_model' in n]
        dec = [(n, p) for n, p in m.named_parameters() if 'clip_model' not in n]
        return _decay_groups(enc, self.lr) + _decay_groups(dec, self.lr * self.coef_lr)

    def loss(self, m, x, labels):
        out = m.training_step((x, labels), 0)
        return out['loss'] if isinstance(out, dict) else out

    @torch.no_grad()
    def predict(self, m, x):
        preds, _ = m.tokenizer.decode(m(x).softmax(-1))
        return preds

    def notes(self):
        return {'charset': 'pretrained 94-char tokenizer kept; plain decoding',
                'lr_groups': 'CLIP encoder at peak LR, decoder at peak LR x native coef_lr (19), as in the published recipe'}


# ---------------------------------------------------------------- MGP-STR (HF transformers)
class MgpStr(Adapter):
    name = 'mgp_str_base'
    family = 'MGP-STR (HF transformers, alibaba-damo/mgp-str-base @5d06493)'
    source = 'https://huggingface.co/alibaba-damo/mgp-str-base'
    lr_rule = 'fallback 5e-5 (native recipe is Adadelta)'
    micro_batch = 64
    weight_files = [W / 'mgp_str_base/model.safetensors']

    @property
    def lr(self):
        return 5e-5

    def build(self):
        from transformers import MgpstrForSceneTextRecognition
        m = MgpstrForSceneTextRecognition.from_pretrained(str(W / 'mgp_str_base'))
        import json
        vocab = json.loads((W / 'mgp_str_base/vocab.json').read_text(encoding='utf-8'))
        self.itos = [None] * len(vocab)
        for k, v in vocab.items():
            self.itos[v] = k
        # Pretrained char head is lowercase+digits only. Add uppercase rows initialised from lowercase.
        old = m.char_head
        upper = [c for c in CHARSET if c.isupper()]
        new = torch.nn.Linear(old.in_features, old.out_features + len(upper))
        with torch.no_grad():
            new.weight[:old.out_features] = old.weight
            new.bias[:old.out_features] = old.bias
            for j, c in enumerate(upper):
                src = vocab[c.lower()]
                new.weight[old.out_features + j] = old.weight[src]
                new.bias[old.out_features + j] = old.bias[src]
        m.char_head = new
        m.config.num_character_labels = new.out_features
        self.itos += upper
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.T = m.config.max_token_length
        return m

    def transform(self):
        return T.Compose([T.Resize((32, 128), BICUBIC), T.ToTensor()])  # do_normalize: false

    def loss(self, m, x, labels):
        logits = m(pixel_values=x).logits[0]  # char head only
        tgt = torch.full((len(labels), self.T), -100, dtype=torch.long, device=x.device)
        for b, s in enumerate(labels):
            ids = [self.stoi[c] for c in s] + [self.stoi['[s]']]
            tgt[b, 1:1 + len(ids)] = torch.tensor(ids[:self.T - 1], device=x.device)  # pos 0 = [GO], skipped
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def predict(self, m, x):
        idx = m(pixel_values=x).logits[0].argmax(-1)[:, 1:].tolist()
        out = []
        for row in idx:
            s = []
            for i in row:
                t = self.itos[i]
                if t == '[s]':
                    break
                if t != '[GO]':
                    s.append(t)
            out.append(''.join(s))
        return out

    def notes(self):
        return {'charset': 'char head extended 38->64 (uppercase rows initialised from lowercase); BPE/WordPiece heads unused; char-head decoding only'}


# ---------------------------------------------------------------- TrOCR-small (HF transformers)
class TrOcrSmall(Adapter):
    name = 'trocr_small_printed'
    family = 'TrOCR (HF transformers, microsoft/trocr-small-printed @04e994a)'
    source = 'https://huggingface.co/microsoft/trocr-small-printed'
    lr_rule = 'fallback 5e-5 (published LR is already a fine-tuning LR)'
    micro_batch = 64
    weight_files = [W / 'trocr_small_printed/model.safetensors']

    @property
    def lr(self):
        return 5e-5

    def build(self):
        from types import SimpleNamespace
        from transformers import VisionEncoderDecoderModel, XLMRobertaTokenizer
        d = str(W / 'trocr_small_printed')
        # Slow sentencepiece tokenizer only (fast conversion needs protobuf); images use our own transform,
        # which matches preprocessor_config.json (384x384 bicubic, mean/std 0.5).
        self.proc = SimpleNamespace(tokenizer=XLMRobertaTokenizer.from_pretrained(d))
        m = VisionEncoderDecoderModel.from_pretrained(d)
        tok = self.proc.tokenizer
        # Match the checkpoint's own generation convention (verified: decoder input [</s>=2, t1..tn],
        # target [t1..tn, </s>], no leading <s>; teacher-forced loss on its own generation = 0.014).
        m.config.decoder_start_token_id = m.generation_config.decoder_start_token_id
        m.config.pad_token_id = tok.pad_token_id
        assert tok.bos_token_id == 0 and m.config.decoder_start_token_id == tok.eos_token_id == 2
        return m

    def transform(self):
        return T.Compose([T.Resize((384, 384), BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def loss(self, m, x, labels):
        # Explicit teacher-forced seq2seq loss. (transformers 4.57 resolves VisionEncoderDecoder's default
        # loss to ForCausalLMLoss, which shifts labels a second time, so we do not pass labels to the model.)
        tok = self.proc.tokenizer
        ids = tok(labels, padding=True, return_tensors='pt').input_ids.to(x.device)
        assert bool((ids[:, 0] == tok.bos_token_id).all())
        ids = ids[:, 1:]  # drop leading <s>
        start = torch.full((ids.size(0), 1), m.config.decoder_start_token_id, dtype=ids.dtype, device=x.device)
        dec_in = torch.cat([start, ids[:, :-1]], 1)
        logits = m(pixel_values=x, decoder_input_ids=dec_in).logits
        tgt = ids.masked_fill(ids == tok.pad_token_id, -100)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def predict(self, m, x):
        gen = m.generate(pixel_values=x, max_new_tokens=16, num_beams=1, do_sample=False)
        return [s.replace(' ', '') for s in self.proc.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    def notes(self):
        return {'charset': 'pretrained subword tokenizer kept; spaces stripped from decoded text'}


# ---------------------------------------------------------------- SVTRv2 (OpenOCR, Model1 encoder class)
class Svtrv2Ch(Adapter):
    name = 'svtrv2_ch'
    family = 'SVTRv2 CTC (OpenOCR vendored in model1/src/OpenOCR; encoder class SVTRv2LNConvTwo33)'
    source = 'https://github.com/Topdu/OpenOCR/releases/tag/develop0.0.1 (openocr_svtrv2_ch.pth)'
    native_lr = 1e-4
    weight_files = [W / 'svtrv2_ch/openocr_svtrv2_ch.pth']
    H, Wd = 48, 192  # native height 48; DarkBay images are 4:1

    def build(self):
        import yaml
        oo = ROOT / 'model1/src/OpenOCR'
        _prepend([oo])
        from openrec.modeling import build_model
        cfg = yaml.safe_load((oo / 'configs/rec/svtrv2/svtrv2_ch.yml').read_text(encoding='utf-8'))
        chars = [l.rstrip('\n') for l in (oo / 'tools/utils/ppocr_keys_v1.txt').read_text(encoding='utf-8').splitlines()]
        self.itos = ['blank'] + chars + [' ']
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        missing = [c for c in CHARSET if c not in self.stoi]
        assert not missing, missing
        arch = cfg['Architecture']
        arch['Decoder']['out_channels'] = len(self.itos)
        m = build_model(arch)
        sd = torch.load(self.weight_files[0], map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        # Checkpoint was trained with a GTC guidance branch; fine-tune and infer with its CTC path only.
        dropped = sorted(k for k in sd if k.startswith(('decoder.gtc_head.', 'decoder.before_gtc.')))
        res = m.load_state_dict({k: v for k, v in sd.items() if k not in dropped}, strict=True)
        self.dropped_keys = len(dropped)
        allowed = torch.zeros(len(self.itos), dtype=torch.bool)
        allowed[0] = True
        for c in CHARSET:
            allowed[self.stoi[c]] = True
        self.allowed = allowed
        return m

    def transform(self):
        return T.Compose([T.Resize((self.H, self.Wd), BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def loss(self, m, x, labels):
        logits = m(x)
        logp = logits.float().log_softmax(-1).permute(1, 0, 2)  # T,B,C
        tgt = torch.tensor([self.stoi[c] for s in labels for c in s], dtype=torch.long)
        tl = torch.tensor([len(s) for s in labels], dtype=torch.long)
        il = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
        return F.ctc_loss(logp, tgt, il, tl, blank=0, zero_infinity=True)

    @torch.no_grad()
    def predict(self, m, x):
        p = m(x)  # eval mode returns softmax
        p = p.masked_fill(~self.allowed.to(p.device), -1.0)
        out = []
        for row in p.argmax(-1).tolist():
            s, prev = [], 0
            for i in row:
                if i != prev and i != 0:
                    s.append(self.itos[i])
                prev = i
            out.append(''.join(s))
        return out

    def notes(self):
        return {'charset': 'pretrained 6625-class CTC head kept; greedy decoding restricted to blank + 62-char charset',
                'input': '48x192 (native height 48, no padding)',
                'checkpoint': f'GTC guidance branch dropped ({getattr(self, "dropped_keys", "?")} tensors); encoder + CTC head loaded strictly'}


# ---------------------------------------------------------------- docTR MASTER
class DoctrMaster(Adapter):
    name = 'doctr_master'
    family = 'MASTER (mindee/doctr v0.8.1 @62d94ff; last line supporting Python 3.9)'
    source = 'https://doctr-static.mindee.com/models?id=v0.7.0/master-fde31e4a.pt'
    native_lr = 1e-3
    micro_batch = 64
    weight_files = [W / 'doctr_master/master-fde31e4a.pt']

    def build(self):
        _prepend([TP / 'doctr'])
        from doctr.models.recognition.master.pytorch import default_cfgs, master
        m = master(pretrained=False)
        m.load_state_dict(torch.load(self.weight_files[0], map_location='cpu', weights_only=True))
        cfg = default_cfgs['master']
        self.mean, self.std = cfg['mean'], cfg['std']
        missing = [c for c in CHARSET if c not in m.vocab]
        assert not missing, missing
        return m

    def transform(self):
        return T.Compose([T.Resize((32, 128), BICUBIC), T.ToTensor(), T.Normalize(self.mean, self.std)])

    def loss(self, m, x, labels):
        return m(x, target=labels)['loss']

    @torch.no_grad()
    def predict(self, m, x):
        return [w for w, _ in m(x, return_preds=True)['preds']]

    def notes(self):
        return {'charset': 'pretrained french vocab kept; plain decoding'}


def get(name):
    if name in Strhub._lrs:
        return Strhub(name)
    return {'clip4str_b': Clip4str, 'mgp_str_base': MgpStr, 'trocr_small_printed': TrOcrSmall,
            'svtrv2_ch': Svtrv2Ch, 'doctr_master': DoctrMaster}[name]()


ALL = ['parseq', 'abinet', 'trba', 'vitstr', 'crnn', 'clip4str_b', 'mgp_str_base', 'trocr_small_printed',
       'svtrv2_ch', 'doctr_master']
