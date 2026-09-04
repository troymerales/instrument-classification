import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from instrument_classifier.audio.loader import UnsupportedAudioError, load_audio
from instrument_classifier.config import SR


def _encode(y: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["WAV", "FLAC", "OGG"])
def test_decodes_soundfile_native_formats(sine_audio_array, fmt):
    y, sr = sine_audio_array
    encoded = _encode(y, sr, fmt)

    decoded_y, decoded_sr = load_audio(encoded)

    assert decoded_sr == SR
    assert decoded_y.ndim == 1
    assert decoded_y.size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="mp3 decoding requires ffmpeg")
def test_decodes_mp3_when_ffmpeg_available(tmp_path, sine_audio_array):
    import subprocess

    y, sr = sine_audio_array
    wav_path = tmp_path / "tone.wav"
    mp3_path = tmp_path / "tone.mp3"
    sf.write(wav_path, y, sr, format="WAV")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)],
        check=True, capture_output=True,
    )

    decoded_y, decoded_sr = load_audio(mp3_path.read_bytes())
    assert decoded_sr == SR
    assert decoded_y.size > 0


def test_rejects_garbage_bytes():
    with pytest.raises(UnsupportedAudioError):
        load_audio(b"this is not an audio file")
