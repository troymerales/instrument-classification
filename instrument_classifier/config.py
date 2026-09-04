"""
Constants transcribed verbatim from ``new_methods.ipynb``. Every value here must
trace back to a specific notebook cell — do not add or tune parameters here.
"""

import os
from pathlib import Path

# --- Cell 9 / 10: production preprocessing + feature extraction -----------------
SR = 22050                # librosa.load(filepath, sr=sr) in the main extraction loop
N_FFT = 2048               # extract_mfcc, extract_stft
HOP_LENGTH = 512            # extract_summary, extract_mfcc, extract_stft, extract_cqt
N_MFCC = 13                  # extract_mfcc
ROLL_PERCENT = 0.85           # extract_summary -> spectral_rolloff

# --- Cell 6 / 7: mel spectrogram visualization (display-only; NOT the same
#     n_mels used internally by librosa.feature.mfcc's default mel step) ---------
VIZ_N_MELS = 26

# --- Cell 9: label map (feature-set independent) --------------------------------
LABEL_MAP = {"cel": 0, "gac": 1, "gel": 2, "vio": 3}
CLASS_NAMES = {
    0: "Cello",
    1: "Acoustic Guitar",
    2: "Electric Guitar",
    3: "Violin",
}

# --- Cell 13: model hyperparameters (already baked into the saved pipelines;
#     restated here only because scripts/build_viz_artifacts.py needs to
#     reproduce cell 14's separate 2D visualization SVM with the same config) ----
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"

FEATURE_SETS = ("mfcc", "stft", "cqt")

# --- Paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models" / "instrument_classifier"))
SAVED_DATA_DIR = REPO_ROOT / "saved_data"
SAVED_MODELS_DIR = REPO_ROOT / "saved_models"


def model_path(feature_set: str) -> Path:
    return MODEL_DIR / f"{feature_set}_svm.pkl"


def labels_path() -> Path:
    return MODEL_DIR / "labels.json"


def viz_dir(feature_set: str) -> Path:
    return MODEL_DIR / "viz" / feature_set
