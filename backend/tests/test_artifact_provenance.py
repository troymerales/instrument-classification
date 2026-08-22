"""
Guards against silently loading a stale/mismatched artifact: the labels.json
class ordering must match each model's SVC.classes_, and each model's expected
input dimensionality must match what extract_feature_vector actually produces.
"""

import numpy as np
import pytest

from backend.app.features.extract import extract_feature_vector
from backend.app.models.registry import get_labels, get_pipeline

EXPECTED_DIMS = {"mfcc": 18, "stft": 1030, "cqt": 89}


@pytest.mark.parametrize("feature_set,expected_dim", EXPECTED_DIMS.items())
def test_pipeline_input_dim_matches_feature_vector(feature_set, expected_dim, sine_audio_array):
    y, sr = sine_audio_array
    feat = extract_feature_vector(y, sr, feature_set)
    pipeline = get_pipeline(feature_set)

    assert feat.shape[0] == expected_dim
    assert pipeline.n_features_in_ == expected_dim


@pytest.mark.parametrize("feature_set", EXPECTED_DIMS.keys())
def test_labels_cover_all_pipeline_classes(feature_set):
    pipeline = get_pipeline(feature_set)
    labels = get_labels()

    svm = pipeline.named_steps["svm"]
    assert set(int(c) for c in svm.classes_) <= set(labels.keys())


def test_labels_json_matches_notebook_class_names():
    from backend.app.config import CLASS_NAMES

    labels = get_labels()
    assert labels == CLASS_NAMES
