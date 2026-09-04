"""
Guards Rule 2/3/4: the request-time analysis path must never call .fit()/
.fit_transform() on StandardScaler, PCA, or SVC. Everything used there must
already be fitted and loaded from disk.
"""

import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from instrument_classifier.inference.pipeline import analyze


@pytest.fixture(autouse=True)
def forbid_fitting(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise AssertionError("fit()/fit_transform() must not be called during request-time analysis")

    monkeypatch.setattr(StandardScaler, "fit", _raise)
    monkeypatch.setattr(StandardScaler, "fit_transform", _raise)
    monkeypatch.setattr(PCA, "fit", _raise)
    monkeypatch.setattr(PCA, "fit_transform", _raise)
    monkeypatch.setattr(SVC, "fit", _raise)


@pytest.mark.parametrize("feature_set", ["mfcc", "stft", "cqt"])
def test_analyze_never_fits_anything(sine_wav_bytes, feature_set):
    result = analyze(sine_wav_bytes, feature_set)
    assert result.predicted_index in {0, 1, 2, 3}
