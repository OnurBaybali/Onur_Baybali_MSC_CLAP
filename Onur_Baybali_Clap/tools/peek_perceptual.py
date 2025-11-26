# tools/peek_perceptual.py
import os, sys, pickle, numpy as np

PKLS = {
    "train": "retrieval/perceptual_model/perceptual_model/outputs/train_perceptual_features.pkl",
    "val":   "retrieval/perceptual_model/perceptual_model/outputs/validation_perceptual_features.pkl",
    "test":  "retrieval/perceptual_model/perceptual_model/outputs/test_perceptual_features.pkl",
}

def load_pkl(p):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        obj = np.stack(obj, axis=0)
    return obj

for split, path in PKLS.items():
    print(f"\n=== {split.upper()} ===")
    if not os.path.exists(path):
        print(f"[X] Bulunamadı: {path}")
        continue
    arr = load_pkl(path)
    print(f"[OK] {path}")
    print(f"shape: {arr.shape} (N_audio, F)")
    print(f"dtype: {arr.dtype}")
    # basit kontrol: NaN/Inf var mı?
    nan_cnt = np.isnan(arr).sum()
    inf_cnt = np.isinf(arr).sum()
    print(f"NaN: {nan_cnt}, Inf: {inf_cnt}")
    # norm istatistik
    norms = np.linalg.norm(arr, axis=1)
    print(f"||x|| mean: {norms.mean():.4f}  std: {norms.std():.4f}  min: {norms.min():.4f}  max: {norms.max():.4f}")
    # First 2 examples
    for i in range(min(2, arr.shape[0])):
        print(f"sample[{i}] head: {np.array2string(arr[i][:8], precision=3)}")