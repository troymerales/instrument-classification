"""
Audio decoding for uploaded files.

The notebook's own preprocessing (cell 9/10) is just
``librosa.load(filepath, sr=22050)`` — mono + resample to 22050 Hz, no trim,
no pad, no normalization. librosa already dispatches to soundfile or the
audioread/ffmpeg backend depending on container format, so accepting more
input formats than the original all-WAV training set (wav/mp3/flac/ogg/m4a/
aac/aiff) is purely a deployment/decoding concern (Rule 19) — the audio that
reaches ``extract_feature_vector`` is identical in shape/semantics regardless
of upload format.
"""

import io

import librosa
import numpy as np

from backend.app.config import SR

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}


class UnsupportedAudioError(ValueError):
    pass


def load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary-format audio bytes into the notebook's production signal:
    mono, resampled to SR=22050. Returns (y, sr)."""
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=SR, mono=True)
    except Exception as exc:  # librosa/audioread raise a variety of error types
        raise UnsupportedAudioError(f"Could not decode audio file: {exc}") from exc

    if y.size == 0:
        raise UnsupportedAudioError("Decoded audio is empty.")

    return y, sr
