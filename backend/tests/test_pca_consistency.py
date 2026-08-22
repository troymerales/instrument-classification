"""
Verifies the uploaded audio's PCA coordinates are actually derived from its own
extracted features via the existing fitted PCA (.transform() only), and that
different audio produces different PCA coordinates -- i.e. the point is not a
static placeholder.
"""

import numpy as np

from backend.app.features.extract import extract_feature_vector
from backend.app.inference.pipeline import analyze
from backend.app.pca.viz import load_viz_artifacts, project


def test_project_matches_manual_transform(sine_audio_array):
    y, sr = sine_audio_array
    feat = extract_feature_vector(y, sr, "mfcc")

    artifacts = load_viz_artifacts("mfcc")
    expected = artifacts.pca_viz.transform(artifacts.scaler_viz.transform(feat.reshape(1, -1)))

    pc1, pc2 = project(feat, "mfcc")

    np.testing.assert_allclose([pc1, pc2], expected[0])


def test_different_audio_yields_different_pca_point(sine_wav_bytes, other_sine_wav_bytes):
    result_a = analyze(sine_wav_bytes, "mfcc")
    result_b = analyze(other_sine_wav_bytes, "mfcc")

    assert (result_a.pc1, result_a.pc2) != (result_b.pc1, result_b.pc2)
