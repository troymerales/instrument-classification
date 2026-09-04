"""
OFFLINE artifact builder — run once, not at request time.

Reproduces ``new_methods.ipynb`` cell 14's PCA/decision-region procedure
*exactly*, using the already-committed ``saved_data/{feat}_X_{train,test}.npy``
/ ``saved_data/{feat}_y_{train,test}.npy`` arrays (the same arrays the notebook
itself saved during training). No raw audio is touched, no new data is
generated, and the real classifier (``saved_models/{feat}_svm.pkl``) is never
refit — this only recreates the notebook's own *separate*, visualization-only
2D PCA + 2D SVM, and persists it so the web app can load it read-only.

Usage:
    python scripts/build_viz_artifacts.py

Output (per feature set), under models/instrument_classifier/viz/{feat}/:
    scaler_viz.pkl   - StandardScaler fit on vstack(X_train, X_test)          (cell 14)
    pca_viz.pkl      - PCA(n_components=2) fit on the scaled features         (cell 14)
    svm_viz.pkl      - SVC(kernel='rbf', C=10, gamma='scale') fit on the 2D
                       PCA coordinates only -- illustrative decision-region
                       model, NOT the real classifier                        (cell 14)
    grid.npz         - xx, yy, Z meshgrid+predictions for the contour plot    (cell 14)
    pc_train.npy     - 2D PCA coords of the training rows (for scatter)
    pc_test.npy      - 2D PCA coords of the test rows (for scatter)
    y_train.npy      - copy of saved_data/{feat}_y_train.npy (for scatter color)
    y_test.npy       - copy of saved_data/{feat}_y_test.npy (for scatter color)

Also (re)writes models/instrument_classifier/labels.json (transcribed from the
notebook's own label_map / class_names dicts) and copies the three real
classifier pipelines from saved_models/ into models/instrument_classifier/.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from instrument_classifier.config import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_SETS,
    MODEL_DIR,
    SAVED_DATA_DIR,
    SAVED_MODELS_DIR,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
)

GRID_POINTS = 300  # matches cell 14's np.linspace(..., 300)


def build_labels_json() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels = {str(idx): name for idx, name in sorted(CLASS_NAMES.items())}
    out_path = MODEL_DIR / "labels.json"
    out_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def copy_classifier_pipelines() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for feat in FEATURE_SETS:
        src = SAVED_MODELS_DIR / f"{feat}_svm.pkl"
        dst = MODEL_DIR / f"{feat}_svm.pkl"
        shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst}")


def build_viz_artifacts_for(feature_set: str) -> None:
    print(f"\n=== building viz artifacts: {feature_set} ===")

    X_train = np.load(SAVED_DATA_DIR / f"{feature_set}_X_train.npy")
    X_test = np.load(SAVED_DATA_DIR / f"{feature_set}_X_test.npy")
    y_train = np.load(SAVED_DATA_DIR / f"{feature_set}_y_train.npy")
    y_test = np.load(SAVED_DATA_DIR / f"{feature_set}_y_test.npy")

    n_train = len(X_train)

    # --- cell 14: STANDARDIZE + PCA on vstack(train, test) ---
    scaler_viz = StandardScaler()
    X_scaled = scaler_viz.fit_transform(np.vstack([X_train, X_test]))

    pca_viz = PCA(n_components=2)
    X_pca = pca_viz.fit_transform(X_scaled)

    pc_train = X_pca[:n_train]
    pc_test = X_pca[n_train:]

    # --- cell 14: separate 2D SVM fit only on PCA coords, for decision regions ---
    y_all = np.concatenate([y_train, y_test])
    svm_viz = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)
    svm_viz.fit(X_pca, y_all)

    # --- cell 14: meshgrid + predictions ---
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_POINTS),
        np.linspace(y_min, y_max, GRID_POINTS),
    )

    Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    out_dir = MODEL_DIR / "viz" / feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler_viz, out_dir / "scaler_viz.pkl")
    joblib.dump(pca_viz, out_dir / "pca_viz.pkl")
    joblib.dump(svm_viz, out_dir / "svm_viz.pkl")
    np.savez(out_dir / "grid.npz", xx=xx, yy=yy, Z=Z)
    np.save(out_dir / "pc_train.npy", pc_train)
    np.save(out_dir / "pc_test.npy", pc_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)

    print(f"  X shape: train={X_train.shape} test={X_test.shape}")
    print(f"  PCA explained_variance_ratio_: {pca_viz.explained_variance_ratio_}")
    print(f"  wrote artifacts to {out_dir}")


def main() -> None:
    build_labels_json()
    copy_classifier_pipelines()
    for feat in FEATURE_SETS:
        build_viz_artifacts_for(feat)
    print("\nDone.")


if __name__ == "__main__":
    main()
