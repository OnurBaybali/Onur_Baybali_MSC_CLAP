#!/usr/bin/env python3
"""
Retrieval with Alignment Explanation

This script:
1. Performs audio-to-text retrieval (finds top-K captions for a given audio)
2. For each retrieved caption, shows alignment explanation:
   - Top contributing tokens
   - Top contributing perceptual features

Usage:
    cd retrieval
    conda run -n clap310 python retrieval_with_explanation.py \
        --audio_idx 88 \
        --top_k 5
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from transformers import AutoTokenizer

from models.ase_perceptual import ASEPerceptual
from data_handling.perceptual_datamodule import PerceptualDataModule


ROOT = Path(__file__).parent

# Basic English stop words for token filtering
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "where", "when", "why", "how", "all", "each", "every",
    "some", "any", "no", "not", "only", "just", "also", "very", "too", "so", "than",
    "then", "there", "their", "them", "one", "two", "more", "most", "other", "such",
    "same", "different", "many", "much", "few", "little", "own", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "what", "which", "who", "where",
    "when", "why", "how", "as", "while", "then", "other", "some", "any", "all",
    "both", "each", "few", "more", "most", "such", "only", "own", "same", "so",
    "than", "too", "very", "just", "now",
}


def load_config_and_model(config_path: Path, model_path: Path, device: str = "mps"):
    """Load YAML config and trained ASEPerceptual model."""
    yaml_loader = YAML(typ="safe", pure=True)
    with config_path.open("r") as f:
        config = yaml_loader.load(f)

    dm = PerceptualDataModule(config)
    feature_dim = dm.feature_dim

    model = ASEPerceptual(config, feature_dim=feature_dim)
    ckpt = torch.load(model_path, map_location=device)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return config, model, dm


def load_test_data(config):
    """Load test features and captions."""
    # Load features
    test_pkl_path = Path(config["data_args"]["test_pkl"])
    if not test_pkl_path.is_absolute():
        test_pkl_path = ROOT / test_pkl_path

    with test_pkl_path.open("rb") as f:
        test_data = pickle.load(f)

    features = test_data["features"]
    audio_paths = test_data["audio_paths"]
    feat_names = test_data.get("feat_names", [])

    # Load captions
    json_path = ROOT / "data" / "Clotho" / "json_files" / "test.json"
    with json_path.open("r") as f:
        caption_data = json.load(f)

    # Flatten captions: each audio has 5 captions
    all_captions = []
    captions_per_audio = []
    for item in caption_data["data"]:
        caps = [item[f"caption_{i}"] for i in range(1, 6)]
        captions_per_audio.append(caps)
        all_captions.extend(caps)

    return features, audio_paths, all_captions, captions_per_audio, feat_names


@torch.no_grad()
def audio_to_text_retrieval(model, audio_idx, audio_feats, captions, tokenizer, device, top_k=10):
    """Retrieve top-K captions for a given audio."""
    # Encode query audio
    audio_feat = torch.tensor(audio_feats[audio_idx:audio_idx+1], dtype=torch.float32).to(device)
    audio_emb = model.encode_audio(audio_feat)  # (1, 1024)

    # Encode all captions in batches
    all_input_ids = []
    for cap in captions:
        tokens = tokenizer(
            cap,
            truncation=True,
            padding="max_length",
            max_length=30,
            return_tensors="pt",
        )
        all_input_ids.append(tokens["input_ids"])

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
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(captions)))

    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        results.append({
            "rank": rank,
            "caption": captions[idx.item()],
            "similarity": score.item(),
            "caption_idx": idx.item(),
        })

    return results


def explain_alignment_for_caption(
    model, audio_feat_np, caption_text, feat_names, tokenizer, device, top_tokens=10, top_features=10
):
    """Explain alignment between audio and a specific caption."""
    device_t = torch.device(device)

    # Prepare audio
    audio_feat = torch.tensor(audio_feat_np, dtype=torch.float32, device=device_t).unsqueeze(0)

    # Tokenize caption
    tokens = tokenizer(
        caption_text,
        truncation=True,
        padding="max_length",
        max_length=30,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device_t)

    # Get embeddings
    with torch.no_grad():
        audio_emb = model.encode_audio(audio_feat)  # (1, 1024)
        text_emb = model.encode_text(input_ids)     # (1, 1024)

    audio_emb_vec = audio_emb[0]  # (1024,)
    text_emb_vec = text_emb[0]    # (1024,)

    sim = float((audio_emb_vec * text_emb_vec).sum().item())

    # Per-token contributions
    with torch.no_grad():
        token_feats = model.text_encoder(input_ids)  # (1, L, H)
    token_feats = token_feats[0]  # (L, H)

    token_embs = model.text_proj(token_feats)        # (L, 1024)
    token_embs = F.normalize(token_embs, dim=-1)     # L2 normalize

    sim_per_token = torch.matmul(token_embs, audio_emb_vec)  # (L,)

    # Decode tokens
    token_ids = input_ids[0].cpu().tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    token_scores = []
    for i, (tok, s) in enumerate(zip(token_strs, sim_per_token.tolist())):
        if tok in tokenizer.all_special_tokens:
            continue
        tok_clean = tok.replace("##", "").strip()
        if not tok_clean:
            continue
        if not any(ch.isalnum() for ch in tok_clean):
            continue
        if tok_clean.lower() in STOP_WORDS:
            continue
        token_scores.append((i, tok_clean, s))

    token_scores.sort(key=lambda x: x[2], reverse=True)
    top_tokens_list = token_scores[:top_tokens]

    # Per-feature contributions
    W = model.audio_proj[0].weight.detach().cpu().numpy()  # (1024, F)
    f = audio_feat_np  # (F,)
    u = text_emb_vec.detach().cpu().numpy()  # (1024,)

    col_contrib = W.T @ u  # (F,)
    feat_contrib = f * col_contrib  # (F,)
    feat_importance = np.abs(feat_contrib)

    feature_rows = []
    for j in range(len(f)):
        name = feat_names[j] if j < len(feat_names) else f"feat_{j}"
        feature_rows.append((name, feat_contrib[j], feat_importance[j]))

    feature_rows.sort(key=lambda x: x[2], reverse=True)
    top_features_list = feature_rows[:top_features]

    return {
        "similarity": sim,
        "top_tokens": top_tokens_list,
        "top_features": top_features_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval with alignment explanation for max536 perceptual model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="settings/perceptual_max536.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/max536_perceptual_run/models/perceptual_only_best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--audio_idx",
        type=int,
        required=True,
        help="Index of test audio (0-based)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top captions to retrieve and explain",
    )
    parser.add_argument(
        "--top_tokens",
        type=int,
        default=10,
        help="Number of top tokens to show per caption",
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=10,
        help="Number of top features to show per caption",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RETRIEVAL WITH ALIGNMENT EXPLANATION")
    print("=" * 80)

    config_path = ROOT / args.config
    model_path = ROOT / args.model_path

    # Load model
    print("\n📥 Loading model...")
    config, model, dm = load_config_and_model(config_path, model_path, args.device)
    print("✅ Model loaded")

    # Load data
    print("\n📥 Loading test data...")
    features, audio_paths, all_captions, captions_per_audio, feat_names = load_test_data(config)
    print(f"✅ Loaded {len(features)} test audios")
    print(f"✅ Loaded {len(all_captions)} test captions")

    # Check audio index
    if args.audio_idx < 0 or args.audio_idx >= len(features):
        raise IndexError(f"audio_idx {args.audio_idx} out of range (0..{len(features)-1})")

    audio_name = Path(audio_paths[args.audio_idx]).name
    print(f"\n🎵 Query Audio: {audio_name} (index {args.audio_idx})")

    # Show ground-truth captions
    gt_captions = captions_per_audio[args.audio_idx]
    print(f"\n📝 Ground-truth captions for this audio:")
    for i, cap in enumerate(gt_captions, 1):
        print(f"  [{i}] {cap}")

    # Load tokenizer
    tokenizer_name = config["text_encoder_args"]["type"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Perform retrieval
    print(f"\n🔍 Retrieving top-{args.top_k} captions...")
    retrieved_results = audio_to_text_retrieval(
        model, args.audio_idx, features, all_captions, tokenizer, args.device, top_k=args.top_k
    )

    # Explain each retrieved caption
    audio_feat_np = features[args.audio_idx].astype(np.float32)

    for result in retrieved_results:
        rank = result["rank"]
        caption = result["caption"]
        similarity = result["similarity"]

        print("\n" + "=" * 80)
        print(f"RANK {rank}: Similarity = {similarity:.4f}")
        print("=" * 80)
        print(f'Caption: "{caption}"')

        # Explain alignment
        explanation = explain_alignment_for_caption(
            model,
            audio_feat_np,
            caption,
            feat_names,
            tokenizer,
            args.device,
            top_tokens=args.top_tokens,
            top_features=args.top_features,
        )

        print(f"\n  Global Similarity: {explanation['similarity']:.4f}")

        print(f"\n  Top {len(explanation['top_tokens'])} Tokens:")
        for i, (pos, tok, score) in enumerate(explanation["top_tokens"], 1):
            print(f"    {i:2d}. {tok:15s} (position {pos:2d})  sim={score:.4f}")

        print(f"\n  Top {len(explanation['top_features'])} Features:")
        for i, (name, contrib, imp) in enumerate(explanation["top_features"], 1):
            sign = "+" if contrib >= 0 else "-"
            print(f"    {i:2d}. {name:35s} contrib={sign}{abs(contrib):.4f}  |contrib|={imp:.4f}")

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

