#!/usr/bin/env python3
"""
Simple Live Demo: Test Single Audio File with Custom Text Queries

Usage:
    python test_single_audio_demo.py \
        --audio_path "path/to/your/audio.wav" \
        --queries "A dog barking" "Rain falling" "Birds chirping" \
        --model_path "outputs/full_perceptual_121_features_rerun/models/perceptual_only_best_model.pt" \
        --config "settings/perceptual.yaml"

Or interactive mode:
    python test_single_audio_demo.py --interactive
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer
from ruamel.yaml import YAML

# Import your model and feature extraction
from models.ase_perceptual import ASEPerceptual

# Import feature extraction functions
sys.path.insert(0, str(Path(__file__).parent / 'perceptual_model'))
from extract_perceptual_full import extract_all_features, vectorize_feats


def load_model(config_path, checkpoint_path, device='cuda'):
    """Load trained perceptual-only CLAP model"""
    print("📥 Loading model...")
    
    # Load config
    with open(config_path, "r") as f:
        yaml_loader = YAML(typ='safe', pure=True)
        config = yaml_loader.load(f)
    
    # Load checkpoint to infer feature_dim
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to infer feature_dim from model state_dict
    feature_dim = None
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Find audio_proj layer
        for key in state_dict.keys():
            if 'audio_proj.0.weight' in key:
                feature_dim = state_dict[key].shape[1]
                print(f"   ✅ Inferred feature_dim: {feature_dim}")
                break
    
    if feature_dim is None:
        # Fallback: try to load from config or use default
        feature_dim = config.get('perceptual_args', {}).get('feature_dim', 121)
        print(f"   ⚠️  Using feature_dim from config/default: {feature_dim}")
    
    # Create model
    model = ASEPerceptual(config, feature_dim=feature_dim)
    
    # Load weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"   ✅ Model loaded successfully (feature_dim={feature_dim})")
    
    return model, config, feature_dim


def extract_audio_features(audio_path, sr=32000):
    """Extract perceptual features from audio file"""
    print(f"🎵 Extracting features from: {audio_path}")
    
    # Extract features using the same pipeline as training (takes file path)
    feats_dict = extract_all_features(audio_path)  # extract_all_features expects file path
    features, feat_names = vectorize_feats(feats_dict)
    
    # Convert to tensor
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    elif isinstance(features, torch.Tensor):
        pass
    else:
        raise ValueError(f"Unexpected feature type: {type(features)}")
    
    # Ensure 1D vector
    if features.dim() > 1:
        features = features.flatten()
    
    print(f"   ✅ Extracted {len(features)} features")
    if len(feat_names) > 0:
        print(f"   📋 Feature names: {', '.join(feat_names[:5])}...")
    return features.unsqueeze(0)  # (1, feature_dim)


@torch.no_grad()
def encode_text(model, tokenizer, text_query, device):
    """Encode text query to embedding"""
    # Tokenize
    tokens = tokenizer(
        text_query,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    
    # Encode
    text_emb = model.encode_text(input_ids)  # (1, 1024)
    return text_emb


@torch.no_grad()
def encode_audio(model, audio_features, device):
    """Encode audio features to embedding"""
    audio_tensor = audio_features.to(device)
    audio_emb = model.encode_audio(audio_tensor)  # (1, 1024)
    return audio_emb


@torch.no_grad()
def compute_similarity(text_emb, audio_emb, temperature=None):
    """Compute cosine similarity between text and audio embeddings"""
    # Normalize (already normalized in encode_* methods, but let's be safe)
    text_emb = F.normalize(text_emb, dim=-1)
    audio_emb = F.normalize(audio_emb, dim=-1)
    
    # Cosine similarity
    similarity = (text_emb * audio_emb).sum(dim=-1).item()
    
    # Apply temperature if provided (model's learned temperature)
    if temperature is not None:
        similarity = similarity / temperature
    
    return similarity


def test_audio_with_texts(audio_path, text_queries, model_path, config_path, device='mps'):
    """Main function: Test audio file with multiple text queries"""
    
    print("\n" + "="*80)
    print("🎵 PERCEPTUAL CLAP LIVE DEMO")
    print("="*80 + "\n")
    
    # Load model
    model, config, feature_dim = load_model(config_path, model_path, device)
    
    # Load tokenizer
    text_encoder_type = config.get('text_encoder_args', {}).get('type', 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    print(f"✅ Tokenizer loaded: {text_encoder_type}\n")
    
    # Extract audio features
    audio_features = extract_audio_features(audio_path)
    
    # Encode audio
    audio_emb = encode_audio(model, audio_features, device)
    print(f"✅ Audio encoded to embedding shape: {audio_emb.shape}\n")
    
    # Get model temperature (if available)
    temperature = model.temp.item() if hasattr(model, 'temp') else None
    print(f"🌡️  Model temperature: {temperature}\n")
    
    # Process each text query
    print("="*80)
    print("📝 TEXT → AUDIO SIMILARITY RESULTS")
    print("="*80 + "\n")
    
    results = []
    for query in text_queries:
        # Encode text
        text_emb = encode_text(model, tokenizer, query, device)
        
        # Compute similarity
        similarity = compute_similarity(text_emb, audio_emb, temperature=None)
        
        # Store result
        results.append({
            'query': query,
            'similarity': similarity,
            'similarity_scaled': similarity / temperature if temperature else similarity
        })
        
        # Print
        print(f"🔍 Query: \"{query}\"")
        print(f"   Cosine Similarity: {similarity:.4f}")
        if temperature:
            print(f"   Scaled Similarity: {similarity/temperature:.4f} (÷ temp={temperature:.4f})")
        print()
    
    # Sort by similarity
    results_sorted = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    print("="*80)
    print("📊 RANKED RESULTS (by similarity)")
    print("="*80 + "\n")
    
    for rank, res in enumerate(results_sorted, 1):
        print(f"{rank}. \"{res['query']}\" → {res['similarity']:.4f}")
    
    return results


def interactive_mode(model_path, config_path, device='mps'):
    """Interactive mode: continuously test audio files with text queries"""
    
    print("\n" + "="*80)
    print("🎵 INTERACTIVE PERCEPTUAL CLAP DEMO")
    print("="*80 + "\n")
    
    # Load model once
    model, config, feature_dim = load_model(config_path, model_path, device)
    text_encoder_type = config.get('text_encoder_args', {}).get('type', 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    print(f"✅ Tokenizer loaded: {text_encoder_type}\n")
    
    print("💡 Interactive mode ready!")
    print("   - Enter audio file path to load")
    print("   - Enter text queries (one per line, empty line to finish)")
    print("   - Type 'quit' to exit\n")
    
    current_audio_path = None
    current_audio_emb = None
    
    while True:
        # Get audio path
        audio_path = input("\n🎵 Audio file path (or 'quit'): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not Path(audio_path).exists():
            print(f"❌ File not found: {audio_path}")
            continue
        
        # Extract and encode audio
        try:
            audio_features = extract_audio_features(audio_path)
            current_audio_emb = encode_audio(model, audio_features, device)
            current_audio_path = audio_path
            print(f"✅ Audio loaded: {Path(audio_path).name}\n")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            continue
        
        # Get text queries
        print("📝 Enter text queries (one per line, empty line when done):")
        queries = []
        while True:
            query = input("   Query: ").strip()
            if not query:
                break
            queries.append(query)
        
        if not queries:
            print("⚠️  No queries entered. Skipping...\n")
            continue
        
        # Test each query
        print("\n" + "─"*80)
        print(f"🔍 Testing: {Path(current_audio_path).name}")
        print("─"*80 + "\n")
        
        for query in queries:
            text_emb = encode_text(model, tokenizer, query, device)
            similarity = compute_similarity(text_emb, current_audio_emb, temperature=None)
            
            print(f"   \"{query}\" → {similarity:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test single audio file with text queries')
    parser.add_argument('--audio_path', type=str, help='Path to audio file')
    parser.add_argument('--queries', nargs='+', help='Text queries to test')
    parser.add_argument('--model_path', type=str, 
                       default='/Users/onur/Desktop/SMC_CodecCLAP-main 3/retrieval/outputs/max536_perceptual_run/models/perceptual_only_best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='settings/perceptual.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Interactive mode
    if args.interactive:
        interactive_mode(args.model_path, args.config, device)
        return
    
    # Normal mode
    if not args.audio_path or not args.queries:
        print("❌ Error: --audio_path and --queries are required (or use --interactive)")
        parser.print_help()
        return
    
    # Run test
    test_audio_with_texts(
        args.audio_path,
        args.queries,
        args.model_path,
        args.config,
        device
    )


if __name__ == '__main__':
    main()

