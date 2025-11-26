import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder_perceptual import TextEncoder as TextEncoderPerceptual
from tools.losses import AudioTextContrastiveLoss


class ASEPerceptual(nn.Module):
    """
    Audio: perceptual vektör (B, F)  → MLP → (B, D)
    Text : input_ids (B, L)          → HF model → [CLS] → proj → (B, D)
    Loss : symmetric contrastive
    """
    def __init__(self, config, feature_dim=None):
        super().__init__()
        self.config = config

        # text encoder (sadece input_ids alıyor, attention_mask'i içeride üretiyor)
        self.text_encoder = TextEncoderPerceptual(config)
        text_width = self.text_encoder.text_width

        embed_size = config["embed_size"]
        if feature_dim is None:
            feature_dim = config["perceptual_args"]["feature_dim"]

        # projections
        self.audio_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])
        self.embed_reg = config.get("embed_regularization", False)
        self.atc_loss = AudioTextContrastiveLoss()

    def encode_audio(self, audio_vec):
        emb = self.audio_proj(audio_vec)          # (B, D)
        emb = F.normalize(emb, dim=-1)
        return emb

    def encode_text(self, input_ids):
        feats = self.text_encoder(input_ids)      # (B, L, H)  attention_mask içeride
        cls = feats[:, 0, :]                      # (B, H)
        emb = self.text_proj(cls)                 # (B, D)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, audio_vec, input_ids, idx):
        audio_embeds = self.encode_audio(audio_vec)
        text_embeds  = self.encode_text(input_ids)

        # aynı audio için 5 caption → pozitif maske
        idx = idx.view(-1, 1)
        pos = torch.eq(idx, idx.t()).float()                      # (B, B)
        sim_targets = pos / (pos.sum(1, keepdim=True) + 1e-8)

        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)

        if self.embed_reg:
            loss += (audio_embeds.abs().mean() / (audio_embeds.pow(2).sum().sqrt() + 1e-8) +
                     text_embeds.abs().mean()  / (text_embeds.pow(2).sum().sqrt()  + 1e-8))

        return loss