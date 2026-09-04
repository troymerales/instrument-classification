"""
Loads the already-fitted classifier pipelines and label map exactly once at
startup. Nothing in this module ever calls .fit()/.fit_transform() — only
joblib.load() and json.load().
"""

import json
from functools import lru_cache

import joblib
from sklearn.pipeline import Pipeline

from instrument_classifier.config import FEATURE_SETS, labels_path, model_path


@lru_cache(maxsize=1)
def get_labels() -> dict[int, str]:
    with open(labels_path(), encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


@lru_cache(maxsize=None)
def get_pipeline(feature_set: str) -> Pipeline:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}', expected one of {FEATURE_SETS}")
    return joblib.load(model_path(feature_set))


def class_name(index: int) -> str:
    return get_labels()[index]
