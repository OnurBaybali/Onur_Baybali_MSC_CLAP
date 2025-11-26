#!/usr/bin/env python3
"""
Test Custom Audio: Test a WAV file that's NOT in the dataset!
"""

import torch
import pickle
import json
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from ruamel.yaml import YAML
from tqdm import tqdm
import sys

# Add perceptual_model to path
sys.path.insert(0, str(Path(__file__).parent))
from perceptual_model.extract_perceptual_full import extract_all_features, vectorize_feats
from models.ase_perceptual import ASEPerceptual


def load_model(config_path, checkpoint_path, device='mps'):
    """Load trained perceptual-only CLAP model"""
    print("📥 Loading model...")
    
    with open(config_path, "r") as f:
        yaml_loader = YAML(typ='safe', pure=True)
        config = yaml_loader.load(f)
    
    # Load test PKL to get feature_dim
    test_pkl = config['data_args']['test_pkl']
    with open(test_pkl, 'rb') as f:
        data = pickle.load(f)
        feature_dim = data['features'].shape[1]
    
    print(f"   ✅ Feature dimension: {feature_dim}")
    
    model = ASEPerceptual(config, feature_dim=feature_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"   ✅ Model loaded from: {Path(checkpoint_path).name}\n")
    
    return model, config, feature_dim


def load_test_captions(config):
    """Load all test captions"""
    json_path = Path(__file__).parent / "data" / "Clotho" / "json_files" / "test.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Caption JSON not found: {json_path}")
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    all_captions = []
    for item in json_data['data']:
        for i in range(1, 6):
            all_captions.append(item[f'caption_{i}'])
    
    print(f"   ✅ Loaded {len(all_captions)} test captions (5 per audio)\n")
    
    return all_captions


@torch.no_grad()
def encode_audio_from_file(audio_path, model, device):
    """Extract features from audio file and encode"""
    print(f"🎵 Processing audio: {Path(audio_path).name}")
    
    # Extract perceptual features
    print("   📊 Extracting perceptual features...")
    feats_dict = extract_all_features(str(audio_path))
    features, feat_names = vectorize_feats(feats_dict)
    
    print(f"   ✅ Extracted {len(feat_names)} features: {len(feat_names)}D")
    
    # Encode to embedding
    audio_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    audio_emb = model.encode_audio(audio_tensor)  # (1, 1024)
    
    return audio_emb, features, feat_names


@torch.no_grad()
def encode_all_captions(model, tokenizer, captions, device, batch_size=64):
    """Encode all captions to embeddings"""
    print(f"📝 Encoding {len(captions)} captions...")
    
    caption_embs = []
    for i in tqdm(range(0, len(captions), batch_size), desc="   Encoding"):
        batch_captions = captions[i:i+batch_size]
        
        tokens = tokenizer(
            batch_captions,
            truncation=True,
            padding='max_length',
            max_length=30,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(device)
        
        batch_embs = model.encode_text(input_ids)  # (B, 1024)
        caption_embs.append(batch_embs.cpu())
    
    caption_embs = torch.cat(caption_embs, dim=0)  # (N, 1024)
    print(f"   ✅ Encoded {len(caption_embs)} captions\n")
    
    return caption_embs.to(device)


@torch.no_grad()
def retrieve_top_captions(audio_emb, caption_embs, captions, top_k=10):
    """Find top-K most similar captions for audio"""
    
    similarities = torch.matmul(audio_emb, caption_embs.T).squeeze(0)  # (N,)
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        caption = captions[idx.item()]
        word_count = len(caption.split())
        
        words = caption.split()[:5]
        short_caption = ' '.join(words)
        
        results.append({
            'rank': rank,
            'caption': caption,
            'short_caption': short_caption,
            'similarity': score.item(),
            'word_count': word_count
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Custom Audio (Not in Dataset)')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to WAV file (not in dataset)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top captions to retrieve')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (mps, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Check if audio exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print("=" * 80)
    print(f"🎹 TESTING CUSTOM AUDIO (NOT IN DATASET)")
    print("=" * 80)
    print(f"📁 Audio file: {audio_path.name}")
    print(f"📍 Full path: {audio_path}")
    print()
    
    # Load model
    model, config, feature_dim = load_model(args.config, args.model_path, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['text_encoder_args']['type'])
    print(f"✅ Tokenizer loaded: {config['text_encoder_args']['type']}\n")
    
    # Extract features and encode audio
    audio_emb, features, feat_names = encode_audio_from_file(audio_path, model, args.device)
    
    # Load test captions
    print("📂 Loading test captions...")
    all_captions = load_test_captions(config)
    
    # Encode all captions
    caption_embs = encode_all_captions(model, tokenizer, all_captions, args.device)
    
    # Retrieve top captions
    print("🔍 Searching for top matching captions...\n")
    top_captions = retrieve_top_captions(audio_emb, caption_embs, all_captions, top_k=args.top_k)
    
    print("=" * 80)
    print(f"🎯 TOP {args.top_k} GENERATED CAPTIONS FOR: {audio_path.name}")
    print("=" * 80)
    print()
    
    for result in top_captions:
        print(f"   {result['rank']:2d}. \"{result['short_caption']}\"")
        print(f"       (similarity: {result['similarity']:.4f}, words: {result['word_count']})")
        if result['word_count'] > 5:
            print(f"       Full: \"{result['caption']}\"")
        print()


if __name__ == '__main__':
    main()

