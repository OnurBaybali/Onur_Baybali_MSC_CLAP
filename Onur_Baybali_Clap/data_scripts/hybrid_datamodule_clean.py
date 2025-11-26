#!/usr/bin/env python3
# coding: utf-8
"""
Clean HybridDataModule: Combines baseline audio loading + perceptual features
Based on working AudioCaptionDataModule + perceptual PKL loading
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial

# Import the working baseline datamodule
from data_handling.datamodule import AudioCaptionDataModule


class HybridDataset(Dataset):
    """
    Hybrid dataset that combines:
    1. Audio loading (from baseline AudioCaptionDataModule)
    2. Perceptual features (from PKL files)
    """
    
    def __init__(self, base_dataset, perceptual_features, perceptual_paths):
        self.base_dataset = base_dataset
        self.perceptual_features = perceptual_features  # (N, F) tensor
        self.perceptual_paths = perceptual_paths  # list of audio paths
        
        # Create mapping from audio path to perceptual feature index
        self.path_to_idx = {path: idx for idx, path in enumerate(perceptual_paths)}
        
        print(f"[HybridDataset] Base dataset: {len(base_dataset)} samples")
        print(f"[HybridDataset] Perceptual features: {perceptual_features.shape}")
        print(f"[HybridDataset] Perceptual paths: {len(perceptual_paths)}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample (audio, text, base_idx)
        audio, text, base_idx = self.base_dataset[idx]
        
        # Get corresponding perceptual features
        # For Clotho: 1 audio = 5 captions, so audio_idx = base_idx // 5
        audio_idx = base_idx // 5
        perceptual_feat = self.perceptual_features[audio_idx].clone()  # (F,) - ensure it's a copy
        
        # Return audio_idx instead of base_idx for correct positive pair matching
        return audio, text, audio_idx, perceptual_feat


def hybrid_collate_fn(batch):
    """Collate function for hybrid batches"""
    audio, text, idx, pvec = zip(*batch)
    
    # Stack tensors
    audio = torch.stack(audio)  # (B, T)
    idx = torch.tensor(idx, dtype=torch.long)  # (B,) - convert int to tensor
    pvec = torch.stack(pvec)    # (B, F)
    
    # Text remains as list of strings
    text = list(text)
    
    return audio, text, pvec, idx  # FIXED: Return pvec, idx in correct order


class HybridDataModuleClean:
    """
    Clean hybrid datamodule that properly combines:
    - Baseline audio loading (AudioCaptionDataModule)
    - Perceptual features (from PKL files)
    """
    
    def __init__(self, config, dataset_name="Clotho"):
        self.config = config
        self.dataset_name = dataset_name
        
        # Get PKL paths from config
        hybrid_cfg = config.get("hybrid", {})
        pkl_cfg = hybrid_cfg.get("pkl", {})
        
        train_pkl = pkl_cfg.get("train")
        val_pkl = pkl_cfg.get("val") 
        test_pkl = pkl_cfg.get("test")
        
        if not all([train_pkl, val_pkl, test_pkl]):
            raise ValueError("Missing PKL paths in hybrid.pkl config")
        
        print(f"[HybridDataModuleClean] Loading PKL files...")
        print(f"  Train: {train_pkl}")
        print(f"  Val:   {val_pkl}")
        print(f"  Test:  {test_pkl}")
        
        # Load perceptual features
        self.perc_train = self._load_perceptual_features(train_pkl)
        self.perc_val = self._load_perceptual_features(val_pkl)
        self.perc_test = self._load_perceptual_features(test_pkl)
        
        # Create base datamodule (for audio loading)
        self.base_dm = AudioCaptionDataModule(config, dataset_name)
        
        # Get feature dimension
        self.feature_dim = self.perc_train["features"].shape[1]
        print(f"[HybridDataModuleClean] Feature dimension: {self.feature_dim}")
        
        # Data args
        data_args = config.get("data_args", {})
        self.batch_size = data_args.get("batch_size", 32)
        self.num_workers = data_args.get("num_workers", 0)
    
    def _load_perceptual_features(self, pkl_path):
        """Load perceptual features from PKL file"""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # Handle dict structure
        if isinstance(data, dict):
            features = torch.tensor(data["features"], dtype=torch.float32)
            paths = data["audio_paths"]
        else:
            raise ValueError(f"Unexpected PKL structure: {type(data)}")
        
        return {
            "features": features,
            "paths": paths
        }
    
    def train_dataloader(self, is_distributed=False, num_tasks=1, global_rank=0):
        """Create training dataloader"""
        # Get base dataset
        base_dataset = self.base_dm.train_dataloader(
            is_distributed=is_distributed,
            num_tasks=num_tasks, 
            global_rank=global_rank
        ).dataset
        
        # Create hybrid dataset
        hybrid_dataset = HybridDataset(
            base_dataset, 
            self.perc_train["features"],
            self.perc_train["paths"]
        )
        
        return DataLoader(
            hybrid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=hybrid_collate_fn,
            pin_memory=False
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        base_dataset = self.base_dm.val_dataloader().dataset
        
        hybrid_dataset = HybridDataset(
            base_dataset,
            self.perc_val["features"], 
            self.perc_val["paths"]
        )
        
        return DataLoader(
            hybrid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=hybrid_collate_fn,
            pin_memory=False
        )
    
    def test_dataloader(self):
        """Create test dataloader"""
        base_dataset = self.base_dm.test_dataloader().dataset
        
        hybrid_dataset = HybridDataset(
            base_dataset,
            self.perc_test["features"],
            self.perc_test["paths"]
        )
        
        return DataLoader(
            hybrid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=hybrid_collate_fn,
            pin_memory=False
        )
    
    @property
    def perc_dim(self):
        """Get perceptual feature dimension"""
        return self.feature_dim
