# coding: utf-8
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

# Mevcut (audio, text, idx) sağlayan modül
from data_handling.datamodule import AudioCaptionDataModule


def _pad_or_truncate(x: np.ndarray, target_dim: int) -> np.ndarray:
    """x: (F,), target_dim'e sağdan 0 pad veya truncate."""
    F = x.shape[0]
    if F == target_dim:
        return x
    if F < target_dim:
        out = np.zeros((target_dim,), dtype=x.dtype)
        out[:F] = x
        return out
    else:
        return x[:target_dim]


class _PerceptualBank:
    """PKL'den (N_audio, F) yükler, torch.Tensor halde tutar. Boyutu target_dim'e uyarlar."""
    def __init__(self, pkl_path: str, target_dim: int = None,
                 apply_log1p: bool = False, l2_normalize: bool = True):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Perceptual PKL not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Handle dict structure (e.g., {"features": array, "audio_paths": list})
        if isinstance(data, dict):
            feats = data["features"]
        elif isinstance(data, list):
            feats = np.stack(data, axis=0)  # (N, F)
        else:
            feats = data

        # hedef boyuta uydur
        if target_dim is not None:
            feats = np.stack([_pad_or_truncate(v, target_dim) for v in feats], axis=0)

        # ölçekleme / normalize (opsiyonel)
        if apply_log1p:
            feats = np.log1p(np.maximum(feats, 0.0))  # negatif varsa sıfıra clamp
        if l2_normalize:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            feats = feats / norms

        self.bank = torch.tensor(feats, dtype=torch.float32)  # (N, F)

    def __len__(self):
        return self.bank.shape[0]

    @property
    def dim(self):
        return self.bank.shape[1]

    def get_by_audio_idx(self, audio_idx: int) -> torch.Tensor:
        return self.bank[audio_idx]


# --------- TOP-LEVEL collate (picklenebilir) ---------
def hybrid_collate(batch, perceptual_bank: _PerceptualBank):
    """
    Varolan (audio, text, idx) batch'ine pvec ekler.
    Clotho varsayımı: 1 audio -> 5 caption, dolayısıyla audio_idx = idx // 5
    """
    audio, text, idx = zip(*batch)

    # audio zaten Tensor -> stack
    audio = torch.stack(audio)  # (B, ...)

    # idx bazen int olabilir -> güvenli tensora çevir
    idx = torch.as_tensor(idx, dtype=torch.long)  # (B,)

    # text: string list kalmalı; model encode_text içinde tokenize edecek
    text = list(text)

    # audio_idx eşlemesi:
    audio_idx = (idx // 5).tolist()
    pvec = torch.stack([perceptual_bank.get_by_audio_idx(i) for i in audio_idx])  # (B, Fp)

    return audio, text, idx, pvec
# -----------------------------------------------------


class HybridDataModule:
    """
    Mevcut AudioCaptionDataModule'u sarar; sadece collate_fn ekleyip
    batch'e perceptual vektörleri (pvec) enjekte eder.
    """
    def __init__(self, config, dataset_name="Clotho"):
        self.cfg = config
        self.base = AudioCaptionDataModule(config, dataset_name)

        # PKL yolları (iki isimlendirmeyi de destekleyelim)
        perc_paths_cfg = config.get("perceptual_paths", {}) or config.get("hybrid", {}).get("pkl", {})
        train_p = perc_paths_cfg.get("train")
        val_p   = perc_paths_cfg.get("val")
        test_p  = perc_paths_cfg.get("test")
        if not (train_p and val_p and test_p):
            raise KeyError("perceptual_paths / hybrid.pkl içinde 'train', 'val', 'test' yolları eksik.")

        # hedef dim: üç dosyanın en büyüğü → hepsini buna pad'le
        dims = []
        for p in [train_p, val_p, test_p]:
            with open(p, "rb") as f:
                data = pickle.load(f)
            # Handle dict structure (e.g., {"features": array, "audio_paths": list})
            if isinstance(data, dict):
                arr = data["features"]
            elif isinstance(data, list):
                arr = np.stack(data, axis=0)
            else:
                arr = data
            dims.append(arr.shape[1])
        target_dim = max(dims)

        # opsiyonel ölçekleme / normalize
        hybcfg = config.get("hybrid", {})
        normalize = bool(hybcfg.get("normalize_perceptual", True))
        apply_log1p = bool(hybcfg.get("log1p", False))

        self.perc_train = _PerceptualBank(train_p, target_dim=target_dim,
                                          apply_log1p=apply_log1p, l2_normalize=normalize)
        self.perc_val   = _PerceptualBank(val_p,   target_dim=target_dim,
                                          apply_log1p=apply_log1p, l2_normalize=normalize)
        self.perc_test  = _PerceptualBank(test_p,  target_dim=target_dim,
                                          apply_log1p=apply_log1p, l2_normalize=normalize)

        # Top-level collate + partial => picklenebilir
        self._collate_train = partial(hybrid_collate, perceptual_bank=self.perc_train)
        self._collate_val   = partial(hybrid_collate, perceptual_bank=self.perc_val)
        self._collate_test  = partial(hybrid_collate, perceptual_bank=self.perc_test)

        # Data args
        da = config.get("data_args", {})
        self.batch_size  = da.get("batch_size", 32)
        self.num_workers = da.get("num_workers", 0)  # MPS + tokenizer ile 0 güvenli; hız için 4–8 deneyebilirsin.

    @property
    def perc_dim(self):
        return self.perc_train.dim

    def train_dataloader(self, is_distributed=False, num_tasks=1, global_rank=0):
        ds = self.base.train_dataloader(is_distributed=is_distributed,
                                        num_tasks=num_tasks,
                                        global_rank=global_rank).dataset
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self._collate_train,
                          pin_memory=False)

    def val_dataloader(self):
        ds = self.base.val_dataloader().dataset
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self._collate_val,
                          pin_memory=False)

    def test_dataloader(self):
        ds = self.base.test_dataloader().dataset
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self._collate_test,
                          pin_memory=False)