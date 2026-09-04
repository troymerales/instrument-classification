"""
Loads the offline-precomputed PCA visualization artifacts (scripts/
build_viz_artifacts.py) and projects a new feature vector into that space.

Only .transform()/.predict() are ever called here on new data. scaler_viz,
pca_viz, svm_viz, and the decision-region grid are fit once, offline, from the
notebook's own saved train/test arrays (see build_viz_artifacts.py) -- never
refit per request.
"""

from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np

from instrument_classifier.config import FEATURE_SETS, viz_dir


@dataclass(frozen=True)
class VizArtifacts:
    scaler_viz: object
    pca_viz: object
    svm_viz: object
    xx: np.ndarray
    yy: np.ndarray
    Z: np.ndarray
    pc_train: np.ndarray
    pc_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@lru_cache(maxsize=None)
def load_viz_artifacts(feature_set: str) -> VizArtifacts:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}', expected one of {FEATURE_SETS}")

    d = viz_dir(feature_set)
    grid = np.load(d / "grid.npz")

    return VizArtifacts(
        scaler_viz=joblib.load(d / "scaler_viz.pkl"),
        pca_viz=joblib.load(d / "pca_viz.pkl"),
        svm_viz=joblib.load(d / "svm_viz.pkl"),
        xx=grid["xx"],
        yy=grid["yy"],
        Z=grid["Z"],
        pc_train=np.load(d / "pc_train.npy"),
        pc_test=np.load(d / "pc_test.npy"),
        y_train=np.load(d / "y_train.npy"),
        y_test=np.load(d / "y_test.npy"),
    )


def project(feature_vector: np.ndarray, feature_set: str) -> tuple[float, float]:
    """Project a single new raw feature vector into the precomputed 2D PCA
    space via .transform() only. Returns (pc1, pc2)."""
    artifacts = load_viz_artifacts(feature_set)
    x = feature_vector.reshape(1, -1)
    x_scaled = artifacts.scaler_viz.transform(x)
    pc = artifacts.pca_viz.transform(x_scaled)
    return float(pc[0, 0]), float(pc[0, 1])
