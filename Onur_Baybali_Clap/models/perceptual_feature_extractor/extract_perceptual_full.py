#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perceptual feature extractor (robust, parallel, NaN/Inf-safe)

Output format (pickle):
{
  "audio_paths": np.ndarray[object] shape (N,),
  "features":    np.ndarray[float32] shape (N, D),
  "feat_names":  List[str] length D
}

Usage:
  cd retrieval
  python perceptual_model/extract_perceptual_full.py
"""

import os
import json
import pickle
import warnings
from typing import Tuple, Dict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import librosa

# --- scipy beat_track compat: some librosa versions expect scipy.signal.hann
try:
    import scipy
    import scipy.signal as sig
    if not hasattr(sig, "hann"):
        from numpy import hanning as _np_hann
        def _hann(M):  # type: ignore
            return _np_hann(M)
        sig.hann = _hann  # type: ignore[attr-defined]
except Exception:
    pass

# --- pyloudnorm optional
HAS_PYLN = False
try:
    import pyloudnorm as pyln
    HAS_PYLN = True
    print("Optional deps → pyloudnorm: OK")
except Exception:
    print("Optional deps → pyloudnorm: NOT FOUND (LUFS/LRA will be computed with limited accuracy)")

# ---------------------------- CONFIG ----------------------------
SR = 32000
HOP_LENGTH = 512
N_FFT = 2048
WIN_LENGTH = 2048
N_MELS = 64
N_MFCC = 13
N_JOBS = 8    # Adjust based on CPU core count

FEATURE_ROOT = os.path.dirname(__file__)
BASE_DIR = os.path.normpath(os.path.join(FEATURE_ROOT, ".."))

BASE_JSON_DIR = os.path.join(BASE_DIR, "data", "Clotho", "json_files")
SPLITS = {
    "train":      os.path.join(BASE_JSON_DIR, "train.json"),
    "validation": os.path.join(BASE_JSON_DIR, "val.json"),
    "test":       os.path.join(BASE_JSON_DIR, "test.json"),
}

OUTPUT_DIR = os.path.join(FEATURE_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-12


# ---------------------------- Utils ----------------------------
def _safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _stats_1d(x: np.ndarray) -> np.ndarray:
    """1D time series summaries (mean,std,min,max,median,p10,p90), NaN/Inf safe."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(7, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        vals = np.array([
            np.nanmean(x),
            np.nanstd(x),
            np.nanmin(x),
            np.nanmax(x),
            np.nanmedian(x),
            np.nanpercentile(x, 10),
            np.nanpercentile(x, 90),
        ], dtype=np.float32)
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _stats_2d(X: np.ndarray) -> np.ndarray:
    """
    For 2D (C,T), compute mean and std per channel → (2*C,)
    Safe fallback for empty/singular cases.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.size == 0:
        return np.zeros(2, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(X, axis=1)
        stds = np.nanstd(X, axis=1)
    out = np.concatenate([means, stds]).astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_audio(path: str, sr: int = SR) -> Tuple[np.ndarray, int]:
    """
    Load WAV file (float32), normalize silent/very short cases.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WAV not found: {path}")
    y, srr = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
    if y.ndim != 1:
        y = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)
    if y.size == 0 or np.allclose(y, 0.0):
        # tamamen sessizse min dump ekle
        y = np.zeros(int(0.5 * sr), dtype=np.float32)
    return y, srr


# ---------------------------- Feature primitives ----------------------------
def compute_attack_decay(y: np.ndarray, sr: int, hop_len: int = HOP_LENGTH):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len)
    if onset_env.size == 0 or not np.isfinite(onset_env).any():
        return 0.0, 0.0
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_len)
    thr = 0.1 * np.nanmax(onset_env) if np.nanmax(onset_env) > 0 else 0.0
    idx = np.nonzero(onset_env > thr)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    return float(times[idx[0]]), float(times[idx[-1]])


def spectral_flux(y: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    S = S / (np.linalg.norm(S, axis=0, keepdims=True) + EPS)
    diff = np.diff(S, axis=1)
    flux = np.sqrt((diff * diff).sum(axis=0))
    return _safe_float(np.nanmean(flux))


def crest_factor_db(y: np.ndarray) -> float:
    peak = np.max(np.abs(y)) + EPS
    rms = np.sqrt(np.mean(np.square(y)) + EPS)
    cf = 20.0 * np.log10(peak / (rms + EPS))
    return _safe_float(cf)


def f0_features(y: np.ndarray, sr: int) -> Tuple[float, float]:
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH
        )
        if f0 is None or f0.size == 0:
            return 0.0, 0.0
        f0 = np.asarray(f0, dtype=np.float32)
        med = np.nanmedian(f0)
        vr = np.nanvar(f0)
        return _safe_float(med), _safe_float(vr)
    except Exception:
        return 0.0, 0.0


def harmonic_ratio(y: np.ndarray, sr: int) -> float:
    """
    Basit HR: H ve P ayrımı sonrası enerji oranı H / (H+P)
    """
    try:
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        H, P = librosa.decompose.hpss(S)
        eH = np.sum(np.abs(H)**2)
        eP = np.sum(np.abs(P)**2)
        hr = eH / (eH + eP + EPS)
        return _safe_float(hr)
    except Exception:
        return 0.0


def lufs_integrated(y: np.ndarray, sr: int) -> float:
    if not HAS_PYLN:
        return 0.0
    try:
        meter = pyln.Meter(sr)  # EBU R128
        return _safe_float(meter.integrated_loudness(y.astype(np.float64)))
    except Exception:
        return 0.0


def loudness_range(y: np.ndarray, sr: int) -> float:
    if not HAS_PYLN:
        return 0.0
    try:
        meter = pyln.Meter(sr)
        return _safe_float(pyln.loudness_range(y.astype(np.float64), meter.block_size))
    except Exception:
        try:
            return _safe_float(pyln.loudness_range(y.astype(np.float64)))
        except Exception:
            return 0.0


def tempo_bpm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        return _safe_float(tempo)
    except Exception:
        return 0.0


# ---------------------------- Full feature set ----------------------------
def extract_all_features(wav_path: str) -> Dict[str, np.ndarray]:
    y, sr = load_audio(wav_path, SR)

    # STFT & Mel
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Not just for visualization, sometimes useful for contrast

    # 1D series features
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff  = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    rms      = librosa.feature.rms(S=S)[0]

    # 2D matrix features
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    chroma   = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=HOP_LENGTH, n_chroma=12)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, hop_length=HOP_LENGTH)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Ek skalarlar
    f0_med, f0_vr = f0_features(y, sr)
    harm_ratio    = harmonic_ratio(y, sr)
    lufs          = lufs_integrated(y, sr)
    lra           = loudness_range(y, sr)
    atk, dcy      = compute_attack_decay(y, sr, HOP_LENGTH)
    tempo         = tempo_bpm(y, sr)
    sflux         = spectral_flux(y, sr)
    crest_db      = crest_factor_db(y)

    feats = {
        "centroid":  centroid,
        "bandwidth": bandwidth,
        "rolloff":   rolloff,
        "zcr":       zcr,
        "rms":       rms,
        "mfcc":      mfcc,
        "chroma":    chroma,
        "contrast":  contrast,
        "tonnetz":   tonnetz,
        "f0_med":    f0_med,
        "f0_vr":     f0_vr,
        "harm_ratio": harm_ratio,
        "lufs":      lufs,
        "lra":       lra,
        "attack_time": atk,
        "decay_time":  dcy,
        "tempo_bpm":   tempo,
        "spec_flux":   sflux,
        "crest_db":    crest_db,
        # You can also add mel_db if needed (vectorization needs separate consideration)
    }
    return feats


# ---------------------------- Vectorization ----------------------------
def vectorize_feats(feats: dict):
    """
    Summarize time series into fixed-size vector.
    Returns: (vec, names) -> (np.ndarray[D], [str]*D)
    """
    vec = []
    names = []

    # 1D series → 7 features (mean/std/min/max/median/p10/p90)
    for key in ("centroid", "bandwidth", "rolloff", "zcr", "rms"):
        x = feats.get(key, np.array([], dtype=np.float32))
        stats = _stats_1d(x)
        vec.append(stats)
        names += [f"{key}_{s}" for s in ["mean","std","min","max","median","p10","p90"]]

    # 2D (C,T) → mean & std per channel
    def add_2d(name, X, cname_prefix):
        X = feats.get(name, np.empty((0, 0), dtype=np.float32))
        stats = _stats_2d(X)  # (2*C,) -> [means..., stds...]
        vec.append(stats)

        if X.ndim == 2 and X.shape[0] > 0:
            C = X.shape[0]
            nm = [f"{cname_prefix}_c{i}_mean" for i in range(C)] + \
                 [f"{cname_prefix}_c{i}_std"  for i in range(C)]
        else:
            nm = [f"{cname_prefix}_mean", f"{cname_prefix}_std"]

        if len(nm) != stats.shape[0]:
            nm = [f"{cname_prefix}_{i}" for i in range(stats.shape[0])]
        return nm

    names += add_2d("mfcc",     feats.get("mfcc"),     "mfcc")
    names += add_2d("chroma",   feats.get("chroma"),   "chroma")
    names += add_2d("contrast", feats.get("contrast"), "contrast")
    names += add_2d("tonnetz",  feats.get("tonnetz"),  "tonnetz")

    # Skalarlar
    scalar_keys = [
        ("f0_med", "f0_med"),
        ("f0_vr", "f0_vr"),
        ("harm_ratio", "harm_ratio"),
        ("lufs", "lufs"),
        ("lra", "lra"),
        ("attack_time", "attack"),
        ("decay_time", "decay"),
        ("tempo_bpm", "tempo"),
        ("spec_flux", "spec_flux"),
        ("crest_db", "crest_db"),
    ]
    for k, n in scalar_keys:
        v = feats.get(k, np.nan)
        vec.append(np.array([_safe_float(v)], dtype=np.float32))
        names.append(n)

    # NaN/Inf cleaning (neutral=0.0)
    vec = np.concatenate(vec).astype(np.float32, copy=False)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec, names


# ---------------------------- Worker ----------------------------
def process_one(wav_path: str):
    feats = extract_all_features(wav_path)
    vec, names = vectorize_feats(feats)
    return {
        "path": wav_path,
        "vec": vec,
        "names": names,
    }


# ---------------------------- Save helpers ----------------------------
def save_pkl(out_path: str, paths, X, feat_names):
    obj = {
        "audio_paths": np.array(paths, dtype=object),
        "features": X.astype(np.float32),
        "feat_names": feat_names,
    }
    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------- Main per split ----------------------------
def run_split(split: str, json_path: str):
    with open(json_path, "r") as f:
        items = json.load(f)["data"]
    audio_paths = [it["audio"] for it in items]

    print(f"\n[{split}] Extracting with {N_JOBS} workers…  (N={len(audio_paths)})")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_one)(p) for p in tqdm(audio_paths, desc=f"{split} split")
    )

    if not results:
        print(f"[{split}] warning: no results!")
        return

    feat_names = results[0]["names"]
    X = np.stack([r["vec"] for r in results], axis=0).astype(np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # toplu temizlik
    P = [r["path"] for r in results]

    # Expected filename for hybrid.yaml: train_perceptual_features.pkl / validation_perceptual_features.pkl / test_perceptual_features.pkl
    out_name = "validation_perceptual_features.pkl" if split == "validation" else f"{split}_perceptual_features.pkl"
    out_pkl = os.path.join(OUTPUT_DIR, out_name)
    save_pkl(out_pkl, P, X, feat_names)
    print(f"[{split}] saved → {out_pkl} (shape: {X.shape})")


# ---------------------------- Entrypoint ----------------------------
def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.seterr(all="ignore")

    for split, jpath in SPLITS.items():
        run_split(split, jpath)


if __name__ == "__main__":
    main()