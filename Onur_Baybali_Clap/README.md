# Paper Reproduction Package

This package contains the code to reproduce the experiments from the thesis on statistical perceptual audio representations for CLAP models.

## Directory Structure

```
paper_reproduction/
├── main_settings/                    # Configuration files
│   ├── baseline.yaml                 # HTSAT baseline model config
│   ├── mel.yaml                      # Mel-based model config
│   └── perceptual.yaml               # Perceptual feature-based model config
├── training_scripts/                 # Training scripts
│   ├── train_baseline.py             # Train HTSAT baseline
│   ├── train_mel.py                  # Train Mel-based model
│   └── train_perceptual.py           # Train perceptual feature-based model
├── models/                           # Model architectures
│   ├── audio_embedding_model.py      # Baseline HTSAT model (renamed from ase_model.py)
│   ├── audio_embedding_perceptual.py # Perceptual model (renamed from ase_perceptual.py)
│   ├── audio_encoder.py              # Audio encoder (HTSAT + Mel)
│   ├── feature_extractor.py          # Log-mel spectrogram extractor
│   ├── htsat.py                      # HTSAT Swin Transformer
│   ├── text_encoder.py               # BERT text encoder
│   └── text_encoder_perceptual.py    # BERT text encoder for perceptual model
├── data_scripts/                     # Data loading modules
│   ├── datamodule.py                 # Main datamodule for baseline/mel
│   ├── perceptual_datamodule.py      # Datamodule for perceptual features
│   ├── caption_dataset.py            # Caption dataset
│   └── ...
├── tools/                            # Utility functions
│   ├── losses.py                     # Contrastive loss functions
│   ├── optim_utils.py                # Optimizer utilities
│   └── utils.py                      # General utilities
├── perceptual_feature_extraction/    # Perceptual feature extraction
│   └── extract_perceptual_full.py    # Extract 536 perceptual features
└── testing_scripts_extra/            # Testing and evaluation scripts
    ├── explain_max536_alignment.py   # Explain audio-text alignment
    ├── retrieval_with_explanation.py # Retrieval with detailed explanation
    ├── test_single_audio_demo.py     # Test single audio file
    ├── test_custom_audio.py          # Test custom WAV file
    └── live_retrieval_demo.py        # Live retrieval demo
```

## Models

### 1. Baseline HTSAT
- **Config:** `main_settings/baseline.yaml`
- **Training script:** `training_scripts/train_baseline.py`
- **Model:** HTSAT Swin Transformer (pretrained=False)
- **Input:** Log-mel spectrogram (64,064D)

### 2. Mel-based
- **Config:** `main_settings/mel.yaml`
- **Training script:** `training_scripts/train_mel.py`
- **Model:** Direct log-mel features + MLP
- **Input:** Log-mel spectrogram (64,064D)

### 3. Perceptual Feature-based
- **Config:** `main_settings/perceptual.yaml`
- **Training script:** `training_scripts/train_perceptual.py`
- **Model:** Handcrafted perceptual features + MLP
- **Input:** 536 perceptual features

## Usage

### Step 1: Extract Perceptual Features (for perceptual model only)
```bash
cd perceptual_feature_extraction
python extract_perceptual_full.py
```

This will generate:
- `train_perceptual_features.pkl`
- `validation_perceptual_features.pkl`
- `test_perceptual_features.pkl`

### Step 2: Train Models

#### Train Baseline HTSAT
```bash
cd training_scripts
python train_baseline.py --config ../main_settings/baseline.yaml --exp_name baseline_run
```

#### Train Mel-based Model
```bash
cd training_scripts
python train_mel.py --config ../main_settings/mel.yaml --exp_name mel_run
```

#### Train Perceptual Model
```bash
cd training_scripts
python train_perceptual.py --config ../main_settings/perceptual.yaml --exp_name perceptual_run
```

## Requirements

### Option 1: Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate clap_reproduction
```

### Option 2: Pip Only
```bash
pip install -r requirements.txt
```

**Note:** Some packages (soundfile, ffmpeg) work better when installed via conda.

### Key Dependencies:
- Python 3.10
- PyTorch 2.1.0 (with CUDA support if available)
- librosa 0.10.0
- torchlibrosa 0.1.0
- transformers 4.41.2 (Hugging Face)
- pytorch-lightning 2.2.5
- pyloudnorm 0.1.1 (optional, for LUFS/LRA features)
- wandb 0.16.5 (optional, for experiment tracking)

## Testing & Evaluation

After training, use the testing scripts in `testing_scripts_extra/`:

### 1. Audio-Text Alignment Explanation
Explain how the model aligns a specific audio with its caption:
```bash
cd testing_scripts_extra
python explain_max536_alignment.py --audio_idx 0 --top_tokens 8 --top_features 8
```

### 2. Audio-to-Text Retrieval with Explanation
Retrieve top-K captions for an audio and explain each match:
```bash
python retrieval_with_explanation.py --audio_idx 400 --top_k 10 --top_tokens 8 --top_features 8
```

### 3. Test Single Audio (Interactive)
Test a single audio file with custom text queries:
```bash
python test_single_audio_demo.py \
  --audio_path "path/to/audio.wav" \
  --queries "a dog barking" "rain falling" \
  --model_path "../outputs/perceptual_run/models/perceptual_only_best_model.pt" \
  --config "../main_settings/perceptual.yaml"
```

### 4. Test Custom WAV File
Test a custom WAV file against all test captions:
```bash
python test_custom_audio.py \
  --audio_path "path/to/custom_audio.wav" \
  --model_path "../outputs/perceptual_run/models/perceptual_only_best_model.pt" \
  --config "../main_settings/perceptual.yaml"
```

### 5. Live Retrieval Demo
Interactive text-to-audio and audio-to-text retrieval:
```bash
python live_retrieval_demo.py \
  --model_path "../outputs/perceptual_run/models/perceptual_only_best_model.pt" \
  --config "../main_settings/perceptual.yaml"
```

## Notes

- All models trained on Clotho dataset
- Training logs will be saved in `outputs/{exp_name}/logging/`
- Best models will be saved in `outputs/{exp_name}/models/`

