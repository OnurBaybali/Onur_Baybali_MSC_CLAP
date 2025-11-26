# retrieval/data_handling/reduced_perceptual_datamodule.py

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class ReducedPerceptualDataset(Dataset):
    """
    PKL → direkt feature tensörleri (57 no-CLAP features)
    Caption'ları Clotho json'larından okur.
    Tokenizer: bert-base-uncased (config'ten geliyor)
    Dönüş: (audio_feat_tensor, input_ids_tensor, idx_tensor)
    Features: Tonnetz (12) + Chroma (24) + Contrast (14) + Scalar (7) = 57
    """
    def __init__(self, pkl_path, tokenizer_name="bert-base-uncased", split="train"):
        self.pkl_path = pkl_path
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.audio_paths = self.data["audio_paths"]
        self.audio_feats = self.data["features"]  # (N, 57) direkt tensor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # captions
        self.captions = self._load_captions(split)

        # basit hizalama kontrolü
        exp_caps = len(self.audio_paths) * 5
        if len(self.captions) != exp_caps:
            print(f"[WARN] {split} split: caption sayısı {len(self.captions)}; "
                  f"beklenen {exp_caps} (={len(self.audio_paths)} * 5). "
                  "JSON ile PKL listesi uyuşmuyor olabilir.")


    def _load_captions(self, split):
        # file adları: train.json / val.json / test.json
        split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
        mapped = split_map.get(split, split)
        json_path = f"data/Clotho/json_files/{mapped}.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Caption JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            obj = json.load(f)

        # dosya formatı: {"data":[{caption_1..caption_5, audio..}, ...], "num_captions_per_audio":5}
        items = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        caps = []
        for item in items:
            for i in range(1, 6):
                caps.append(item[f"caption_{i}"])
        return caps

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        cap = self.captions[idx]
        audio_idx = idx // 5  # 5 caption / audio
        audio_feat = self.audio_feats[audio_idx]  # np.float32 vector - already loaded

        tokens = self.tokenizer(
            cap,
            truncation=True,
            padding="max_length",
            max_length=30,
            return_tensors="pt",
            return_attention_mask=False  # mask'e ihtiyac yok; encoder içeride üretir
        )
        # squeeze batch dim
        input_ids = tokens["input_ids"].squeeze(0)

        return (
            torch.tensor(audio_feat, dtype=torch.float32),  # (60,)
            input_ids.to(dtype=torch.long),                 # (L,)
            torch.tensor(audio_idx, dtype=torch.long)       # () - FIXED: Return audio_idx, not caption idx!
        )


def collate_fn(batch):
    audio, input_ids, idx = zip(*batch)
    return (
        torch.stack(audio),        # (B, 60) float32
        torch.stack(input_ids),    # (B, L) long
        torch.stack(idx)           # (B,)   long
    )


class ReducedPerceptualDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg["data_args"]["batch_size"]
        # multiprocessing kapalı -> MPS/Tokenizer fork sorunları yok
        self.num_workers = 0

        self.paths = {
            "train_pkl": cfg["data_args"]["train_pkl"],
            "val_pkl":   cfg["data_args"]["val_pkl"],
            "test_pkl":  cfg["data_args"]["test_pkl"],
        }
        self.tokenizer_name = cfg["text_encoder_args"]["type"]

        self.train_dataset = ReducedPerceptualDataset(self.paths["train_pkl"], self.tokenizer_name, "train")
        self.val_dataset   = ReducedPerceptualDataset(self.paths["val_pkl"],   self.tokenizer_name, "validation")
        self.test_dataset  = ReducedPerceptualDataset(self.paths["test_pkl"],  self.tokenizer_name, "test")

        # feature_dim auto
        self.feature_dim = self.train_dataset[0][0].numel()
        print(f"[ReducedPerceptualDataModule] inferred feature_dim: {self.feature_dim}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=False)