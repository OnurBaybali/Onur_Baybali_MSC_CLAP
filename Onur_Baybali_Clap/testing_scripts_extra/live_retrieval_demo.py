#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live CLAP Retrieval Demo
Test your perceptual-only CLAP model with real examples!
"""

import torch
import pickle
import numpy as np
from models.ase_perceptual import ASEPerceptual
from transformers import AutoTokenizer
from ruamel.yaml import YAML
import json

def load_model(config_path, checkpoint_path, device='mps'):
    """Load trained perceptual-only CLAP model"""
    
    # Load config
    with open(config_path, 'r') as f:
        yaml_loader = YAML(typ='safe', pure=True)
        config = yaml_loader.load(f)
    
    # Load perceptual features to get feature_dim
    with open(config['data_args']['test_pkl'], 'rb') as f:
        data = pickle.load(f)
        feature_dim = data['features'].shape[1]
    
    print(f"✅ Feature dimension: {feature_dim}")
    
    # Initialize model
    model = ASEPerceptual(config, feature_dim=feature_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"✅ Trained for {checkpoint['epoch']} epochs")
    
    return model, config, feature_dim


def load_test_data(config):
    """Load test audio features and captions"""
    
    # Load audio features
    with open(config['data_args']['test_pkl'], 'rb') as f:
        data = pickle.load(f)
        audio_feats = data['features']
        audio_paths = data['audio_paths']
    
    print(f"✅ Loaded {len(audio_paths)} test audio samples")
    
    # Load captions
    with open('data/Clotho/json_files/test.json', 'r') as f:
        json_data = json.load(f)
        captions = []
        for item in json_data['data']:
            for i in range(1, 6):
                captions.append(item[f'caption_{i}'])
    
    print(f"✅ Loaded {len(captions)} test captions")
    
    return audio_feats, audio_paths, captions


@torch.no_grad()
def text_to_audio_retrieval(model, query_text, audio_feats, audio_paths, tokenizer, device, top_k=10):
    """Given a text query, find the most similar audios"""
    
    # Tokenize query
    tokens = tokenizer(
        query_text,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    
    # Encode query text
    text_emb = model.encode_text(input_ids)  # (1, 1024)
    
    # Encode all audios
    audio_feats_tensor = torch.tensor(audio_feats, dtype=torch.float32).to(device)
    audio_embs = model.encode_audio(audio_feats_tensor)  # (N, 1024)
    
    # Compute similarities
    similarities = torch.matmul(text_emb, audio_embs.T).squeeze(0)  # (N,)
    
    # Get top-K
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        results.append({
            'rank': rank,
            'audio_path': audio_paths[idx.item()],
            'similarity': score.item(),
            'audio_name': audio_paths[idx.item()].split('/')[-1]
        })
    
    return results


@torch.no_grad()
def audio_to_text_retrieval(model, audio_idx, audio_feats, captions, tokenizer, device, top_k=10):
    """Given an audio, find the most similar captions"""
    
    # Encode query audio
    audio_feat = torch.tensor(audio_feats[audio_idx:audio_idx+1], dtype=torch.float32).to(device)
    audio_emb = model.encode_audio(audio_feat)  # (1, 1024)
    
    # Encode all captions
    all_input_ids = []
    for cap in captions:
        tokens = tokenizer(
            cap,
            truncation=True,
            padding='max_length',
            max_length=30,
            return_tensors='pt'
        )
        all_input_ids.append(tokens['input_ids'])
    
    all_input_ids = torch.cat(all_input_ids, dim=0).to(device)  # (N, 30)
    
    # Encode in batches to avoid OOM
    batch_size = 64
    text_embs = []
    for i in range(0, len(all_input_ids), batch_size):
        batch_ids = all_input_ids[i:i+batch_size]
        batch_embs = model.encode_text(batch_ids)
        text_embs.append(batch_embs)
    
    text_embs = torch.cat(text_embs, dim=0)  # (N, 1024)
    
    # Compute similarities
    similarities = torch.matmul(audio_emb, text_embs.T).squeeze(0)  # (N,)
    
    # Get top-K
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        results.append({
            'rank': rank,
            'caption': captions[idx.item()],
            'similarity': score.item()
        })
    
    return results


def main():
    print("\n" + "="*80)
    print("🎵 LIVE CLAP RETRIEVAL DEMO 🎵")
    print("Testing Perceptual-Only CLAP Model")
    print("="*80 + "\n")
    
    # Configuration
    config_path = 'settings/perceptual.yaml'
    checkpoint_path = 'outputs/perceptual_only_full_training/models/perceptual_only_best_model.pt'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"🖥️  Device: {device}\n")
    
    # Load model
    print("📥 Loading model...")
    model, config, feature_dim = load_model(config_path, checkpoint_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load test data
    print("\n📥 Loading test data...")
    audio_feats, audio_paths, captions = load_test_data(config)
    
    print("\n" + "="*80)
    print("🎯 READY FOR LIVE TESTING!")
    print("="*80 + "\n")
    
    # Example 1: Text → Audio
    print("\n" + "─"*80)
    print("📝 TEST 1: Text → Audio Retrieval")
    print("─"*80)
    
    queries = [
        "A dog is barking loudly",
        "Rain falling on the ground",
        "A car engine starting",
        "Birds chirping in the morning",
        "Someone typing on a keyboard"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("   Top 5 most similar audios:")
        
        results = text_to_audio_retrieval(
            model, query, audio_feats, audio_paths, 
            tokenizer, device, top_k=5
        )
        
        for res in results:
            print(f"      {res['rank']}. {res['audio_name']:<40} (similarity: {res['similarity']:.4f})")
    
    # Example 2: Audio → Text
    print("\n\n" + "─"*80)
    print("🔊 TEST 2: Audio → Text Retrieval")
    print("─"*80)
    
    # Pick some random test audios
    test_audio_indices = [0, 10, 50, 100, 200]
    
    for audio_idx in test_audio_indices:
        audio_name = audio_paths[audio_idx].split('/')[-1]
        print(f"\n🔍 Query Audio: {audio_name}")
        print("   Top 5 most similar captions:")
        
        results = audio_to_text_retrieval(
            model, audio_idx, audio_feats, captions,
            tokenizer, device, top_k=5
        )
        
        for res in results:
            print(f"      {res['rank']}. {res['caption']:<70} (sim: {res['similarity']:.4f})")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

