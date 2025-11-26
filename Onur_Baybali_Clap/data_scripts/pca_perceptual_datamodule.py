import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class PcaPerceptualDataset(Dataset):
    """
    PKL → PCA-reduced feature tensors (K components)
    Captions: read from Clotho JSONs (handled elsewhere in existing pipeline)
    Tokenizer: bert-base-uncased (from config)
    Return: (audio_feat_tensor, input_ids_tensor, idx_tensor)
    """
    def __init__(self, pkl_path, tokenizer_name="bert-base-uncased", split="train"):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.audio_feats = self.data["features"]  # (N, K)
        self.paths = self.data["audio_paths"]
        self.feature_dim = self.audio_feats.shape[1]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.split = split

    def __len__(self):
        return self.audio_feats.shape[0]

    def __getitem__(self, idx):
        import torch
        feat = torch.tensor(self.audio_feats[idx], dtype=torch.float32)
        # Dummy text; existing pipeline will replace via collate if needed
        tokens = self.tokenizer(" ", return_tensors="pt", padding='max_length', truncation=True)
        input_ids = tokens['input_ids'][0]
        return feat, input_ids, idx


def make_pca_perceptual_loaders(train_pkl, val_pkl, test_pkl, batch_size=32, num_workers=8, tokenizer_name="bert-base-uncased"):
    train_ds = PcaPerceptualDataset(train_pkl, tokenizer_name, split="train")
    val_ds = PcaPerceptualDataset(val_pkl, tokenizer_name, split="validation")
    test_ds = PcaPerceptualDataset(test_pkl, tokenizer_name, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.feature_dim


