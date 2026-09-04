import librosa
import numpy as np

from instrument_classifier.config import HOP_LENGTH, N_FFT, N_MFCC, ROLL_PERCENT
from instrument_classifier.features.extract import (
    extract_cqt,
    extract_feature_vector,
    extract_mfcc,
    extract_stft,
    extract_summary,
)


def test_summary_matches_notebook_formula(sine_audio_array):
    y, sr = sine_audio_array

    expected = np.array([
        np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)),
        np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=ROLL_PERCENT)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
    ])

    actual = extract_summary(y, sr)

    assert actual.shape == (5,)
    np.testing.assert_allclose(actual, expected)


def test_mfcc_shape_and_values(sine_audio_array):
    y, sr = sine_audio_array
    expected = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH), axis=1
    )
    actual = extract_mfcc(y, sr)
    assert actual.shape == (13,)
    np.testing.assert_allclose(actual, expected)


def test_stft_shape(sine_audio_array):
    y, sr = sine_audio_array
    actual = extract_stft(y, sr)
    assert actual.shape == (N_FFT // 2 + 1,)


def test_cqt_shape(sine_audio_array):
    y, sr = sine_audio_array
    actual = extract_cqt(y, sr)
    assert actual.shape == (84,)  # librosa.cqt default n_bins


def test_feature_vector_concatenation_order(sine_audio_array):
    y, sr = sine_audio_array
    summary = extract_summary(y, sr)
    mfcc = extract_mfcc(y, sr)

    combined = extract_feature_vector(y, sr, "mfcc")

    assert combined.shape == (18,)
    np.testing.assert_allclose(combined[:5], summary)
    np.testing.assert_allclose(combined[5:], mfcc)
