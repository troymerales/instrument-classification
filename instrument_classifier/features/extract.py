"""
Feature extraction — a 1:1 port of ``new_methods.ipynb`` cell 9
(``extract_summary``, ``extract_mfcc``, ``extract_stft``, ``extract_cqt``).

Do not add, remove, or re-parameterize any feature here without updating the
notebook first. This module must stay numerically identical to the notebook
functions it mirrors.
"""

import numpy as np
import librosa

from instrument_classifier.config import HOP_LENGTH, N_FFT, N_MFCC, ROLL_PERCENT


def extract_summary(y: np.ndarray, sr: int) -> np.ndarray:
    """5 summary statistics, in the exact order the notebook concatenates them."""
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    )

    rms = np.mean(
        librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=ROLL_PERCENT)
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )

    return np.array([zcr, rms, centroid, rolloff, bandwidth])


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.mean(mfcc, axis=1)


def extract_stft(y: np.ndarray, sr: int) -> np.ndarray:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    return np.mean(mag, axis=1)


def extract_cqt(y: np.ndarray, sr: int) -> np.ndarray:
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH)
    mag = np.abs(cqt)
    return np.mean(mag, axis=1)


TRANSFORM_EXTRACTORS = {
    "mfcc": extract_mfcc,
    "stft": extract_stft,
    "cqt": extract_cqt,
}


def extract_feature_vector(y: np.ndarray, sr: int, feature_set: str) -> np.ndarray:
    """concat(summary(5), transform(N)) — matches the concatenation in cell 10."""
    if feature_set not in TRANSFORM_EXTRACTORS:
        raise ValueError(f"Unknown feature_set '{feature_set}', expected one of {list(TRANSFORM_EXTRACTORS)}")

    summary_feat = extract_summary(y, sr)
    transform_feat = TRANSFORM_EXTRACTORS[feature_set](y, sr)
    return np.concatenate([summary_feat, transform_feat])
