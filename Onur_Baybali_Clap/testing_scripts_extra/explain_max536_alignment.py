#!/usr/bin/env python3
"""
Explain a single audio–text alignment for the 536D perceptual-max model.

Given:
  - a trained perceptual-only CLAP model (max536_perceptual_run)
  - an audio index from the Clotho test set
  - an optional custom text query (otherwise uses the ground-truth caption)

This script:
  1. Loads the 536D perceptual features for the selected audio.
  2. Loads the trained perceptual-only CLAP model.
  3. Computes the audio and text embeddings and their cosine similarity.
  4. Computes per-token contributions: which BERT tokens align most with the audio.
  5. Computes per-feature contributions: which perceptual features contribute most
     to the similarity for this specific audio–text pair.

Usage example:

    cd retrieval

    # Use ground-truth caption_1 of test audio index 0
    conda run -n clap310 python explain_max536_alignment.py \
        --audio_idx 0

    # Use a custom query instead of ground-truth caption
    conda run -n clap310 python explain_max536_alignment.py \
        --audio_idx 0 \
        --query "A dog is barking loudly in a park"

"""

import argparse
import json
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
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    "including",
    "against",
    "among",
    "throughout",
    "despite",
    "towards",
    "upon",
    "concerning",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "can",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "as",
    "while",
    "then",
    "other",
    "some",
    "any",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "now",
}


def load_config_and_model(
    config_path: Path,
    model_path: Path,
    device: str = "mps",
):
    """Load YAML config and trained ASEPerceptual model for max536 run."""
    yaml_loader = YAML(typ="safe", pure=True)
    with config_path.open("r") as f:
        config = yaml_loader.load(f)

    # Use PerceptualDataModule to infer feature_dim and keep things consistent
    dm = PerceptualDataModule(config)
    feature_dim = dm.feature_dim

    print(f"✅ Loaded config from: {config_path}")
    print(f"✅ Inferred feature_dim from PKL: {feature_dim}")

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

    print(f"✅ Loaded model from: {model_path}")

    return config, model, dm


def load_test_captions():
    """Load test captions from Clotho JSON, grouped per audio."""
    json_path = ROOT / "data" / "Clotho" / "json_files" / "test.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Caption JSON not found: {json_path}")

    with json_path.open("r") as f:
        data = json.load(f)

    items = data["data"]
    # captions_per_audio[i] = [caption_1, ..., caption_5]
    captions_per_audio = []
    for item in items:
        caps = [item[f"caption_{k}"] for k in range(1, 6)]
        captions_per_audio.append(caps)

    return captions_per_audio


def explain_alignment(
    config,
    model: ASEPerceptual,
    dm: PerceptualDataModule,
    audio_idx: int,
    query: str | None,
    device: str = "mps",
    top_tokens: int = 10,
    top_features: int = 20,
):
    """Explain a single audio–text alignment."""
    # --- 1) Load test features and feat_names from PKL ---
    test_pkl_path = Path(config["data_args"]["test_pkl"])
    if not test_pkl_path.is_absolute():
        test_pkl_path = ROOT / test_pkl_path

    import pickle

    with test_pkl_path.open("rb") as f:
        test_data = pickle.load(f)

    features = test_data["features"]  # (N_test, 536)
    audio_paths = test_data["audio_paths"]
    feat_names = test_data.get("feat_names", [])

    num_test = features.shape[0]
    if audio_idx < 0 or audio_idx >= num_test:
        raise IndexError(f"audio_idx {audio_idx} out of range (0..{num_test-1})")

    print("\n" + "=" * 70)
    print("SELECTED AUDIO")
    print("=" * 70)
    print(f"Index: {audio_idx}")
    print(f"Path : {audio_paths[audio_idx]}")
    print(f"Feature dim: {features.shape[1]}")

    # --- 2) Choose caption / query text ---
    captions_per_audio = load_test_captions()
    if len(captions_per_audio) != num_test:
        print(
            f"⚠️  Warning: test PKL has {num_test} audios but JSON has {len(captions_per_audio)} items"
        )

    gt_captions = captions_per_audio[audio_idx]
    if query is None:
        text = gt_captions[0]
        print(f"\nUsing ground-truth caption_1:\n  \"{text}\"")
    else:
        text = query
        print(f"\nUsing custom query:\n  \"{text}\"")
        print(f"(Ground-truth captions for this audio were:)")
        for i, c in enumerate(gt_captions, 1):
            print(f"  [{i}] {c}")

    # --- 3) Prepare tensors ---
    device_t = torch.device(device)

    # Single audio feature
    audio_feat_np = features[audio_idx].astype(np.float32)
    audio_feat = torch.tensor(audio_feat_np, dtype=torch.float32, device=device_t).unsqueeze(0)

    # Tokenizer (same as training)
    tokenizer_name = config["text_encoder_args"]["type"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=30,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device_t)

    # --- 4) Get embeddings ---
    with torch.no_grad():
        audio_emb = model.encode_audio(audio_feat)  # (1, 1024), L2-normalized
        text_emb = model.encode_text(input_ids)     # (1, 1024), L2-normalized

    audio_emb_vec = audio_emb[0]  # (1024,)
    text_emb_vec = text_emb[0]    # (1024,)

    sim = float((audio_emb_vec * text_emb_vec).sum().item())

    print("\n" + "=" * 70)
    print("GLOBAL SIMILARITY")
    print("=" * 70)
    print(f"Cosine(audio, text) = {sim:.4f}")

    # --- 5) Per-token contributions (text side) ---
    print("\n" + "=" * 70)
    print("PER-TOKEN CONTRIBUTIONS (TEXT SIDE)")
    print("=" * 70)

    # Get token-level hidden states from text encoder
    with torch.no_grad():
        token_feats = model.text_encoder(input_ids)  # (1, L, H)
    token_feats = token_feats[0]  # (L, H)

    # Project each token to embedding space using text_proj
    token_embs = model.text_proj(token_feats)        # (L, 1024)
    token_embs = F.normalize(token_embs, dim=-1)     # L2 normalize

    # Similarity of each token embedding to audio embedding
    sim_per_token = torch.matmul(token_embs, audio_emb_vec)  # (L,)

    # Decode tokens to strings
    token_ids = input_ids[0].cpu().tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    token_scores = []
    for i, (tok, s) in enumerate(zip(token_strs, sim_per_token.tolist())):
        # Skip padding and special tokens
        if tok in tokenizer.all_special_tokens:
            continue
        # Merge BERT subword prefix
        tok_clean = tok.replace("##", "").strip()
        if not tok_clean:
            continue
        # Skip pure punctuation (e.g. "." or ",")
        if not any(ch.isalnum() for ch in tok_clean):
            continue
        # Skip stop words (case-insensitive)
        if tok_clean.lower() in STOP_WORDS:
            continue
        token_scores.append((i, tok_clean, s))

    # Sort by score descending
    token_scores.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {top_tokens} tokens by similarity to audio embedding:")
    for rank, (idx_tok, tok, score) in enumerate(token_scores[:top_tokens], 1):
        print(f"  {rank:2d}. {tok:15s}  (position {idx_tok:2d})  sim={score:.4f}")

    # --- 6) Per-feature contributions (audio side) ---
    print("\n" + "=" * 70)
    print("PER-FEATURE CONTRIBUTIONS (AUDIO SIDE)")
    print("=" * 70)

    # First linear layer in audio_proj: (embed_size, feature_dim)
    W = model.audio_proj[0].weight.detach().cpu().numpy()  # (1024, F)
    f = audio_feat_np  # (F,)
    u = text_emb_vec.detach().cpu().numpy()  # (1024,)

    # Column j of W is w_j; compute projection of w_j onto text direction u
    # col_contrib_j = w_j^T u
    col_contrib = W.T @ u  # (F,)

    # Feature contribution ~ feature value * sensitivity along text direction
    feat_contrib = f * col_contrib  # (F,)
    feat_importance = np.abs(feat_contrib)

    # Build list of (name, contrib, importance)
    feature_rows = []
    for j in range(len(f)):
        name = feat_names[j] if j < len(feat_names) else f"feat_{j}"
        feature_rows.append((name, feat_contrib[j], feat_importance[j]))

    # Sort by importance
    feature_rows.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {top_features} perceptual features for this alignment:")
    for rank, (name, val, imp) in enumerate(feature_rows[:top_features], 1):
        print(f"  {rank:2d}. {name:35s} contrib={val:+.4f}  |contrib|={imp:.4f}")

    print("\n✅ Explanation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Explain a single audio–text alignment for the 536D perceptual-max model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="settings/perceptual_max536.yaml",
        help="Path to YAML config file (default: settings/perceptual_max536.yaml)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/max536_perceptual_run/models/perceptual_only_best_model.pt",
        help="Path to trained perceptual-only model checkpoint.",
    )
    parser.add_argument(
        "--audio_idx",
        type=int,
        required=True,
        help="Index of test audio in the max536 test PKL (0-based).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional custom query text. If omitted, uses ground-truth caption_1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (e.g., 'mps', 'cuda', or 'cpu').",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    model_path = ROOT / args.model_path

    config, model, dm = load_config_and_model(config_path, model_path, device=args.device)
    explain_alignment(
        config=config,
        model=model,
        dm=dm,
        audio_idx=args.audio_idx,
        query=args.query,
        device=args.device,
    )


if __name__ == "__main__":
    main()


