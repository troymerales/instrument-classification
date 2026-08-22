import io

import numpy as np
import pytest
import soundfile as sf

from backend.app.config import SR


def _sine_wave(freq: float, seconds: float = 1.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture
def sine_audio_array():
    """A short synthetic mono signal at the production sample rate."""
    return _sine_wave(220.0), SR


@pytest.fixture
def sine_wav_bytes():
    y = _sine_wave(220.0)
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def other_sine_wav_bytes():
    y = _sine_wave(880.0)
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")
    return buf.getvalue()
