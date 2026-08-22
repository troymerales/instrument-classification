"""
Orchestrates: raw audio bytes -> notebook preprocessing -> notebook feature
extraction -> existing fitted pipeline.predict(). No fitting happens anywhere
in this module.
"""

from dataclasses import dataclass

import numpy as np

from backend.app.audio.loader import load_audio
from backend.app.features.extract import extract_feature_vector
from backend.app.models.registry import class_name, get_pipeline
from backend.app.pca.viz import project


@dataclass(frozen=True)
class AnalysisResult:
    feature_set: str
    feature_vector: np.ndarray
    y: np.ndarray
    sr: int
    predicted_index: int
    predicted_class: str
    pc1: float
    pc2: float


def analyze(file_bytes: bytes, feature_set: str) -> AnalysisResult:
    y, sr = load_audio(file_bytes)

    feature_vector = extract_feature_vector(y, sr, feature_set)

    pipeline = get_pipeline(feature_set)
    predicted_index = int(pipeline.predict(feature_vector.reshape(1, -1))[0])
    predicted_class = class_name(predicted_index)

    pc1, pc2 = project(feature_vector, feature_set)

    return AnalysisResult(
        feature_set=feature_set,
        feature_vector=feature_vector,
        y=y,
        sr=sr,
        predicted_index=predicted_index,
        predicted_class=predicted_class,
        pc1=pc1,
        pc2=pc2,
    )
