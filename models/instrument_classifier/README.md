# `models/instrument_classifier/` — deployed artifacts

Every file in this directory is derived from `new_methods.ipynb` and/or the
repo's existing `saved_models/` / `saved_data/` outputs. Nothing here was fit
on uploaded audio, and nothing here was invented independent of the notebook.

## `{mfcc,stft,cqt}_svm.pkl`

Unmodified copies of `saved_models/{feat}_svm.pkl`, produced by notebook cell 13:
`Pipeline([StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")])`, fit on
the concatenated `(summary(5), transform(N))` feature vector (18/1030/89 dims
for mfcc/stft/cqt respectively). This is the real classifier — the app only
ever calls `.predict()` on it.

**Excluded on purpose:** `saved_models/gfcc_svm.pkl` and
`saved_models/vggish_svm.pkl` also exist in this repo but are **not** copied
here, because GFCC and VGGish feature extraction do not appear anywhere in
`new_methods.ipynb` — they were produced by a different notebook. Per the
project's source-of-truth rule, only artifacts traceable to
`new_methods.ipynb` are deployed.

## `labels.json`

`{"0": "Cello", "1": "Acoustic Guitar", "2": "Electric Guitar", "3": "Violin"}`
— transcribed verbatim from the notebook's own `label_map` (cell 9) /
`class_names` (cells 11/13). This mapping does not exist as a persisted
artifact anywhere else; it was never saved by the notebook, so it is
regenerated (not invented) directly from the notebook's source dicts.

## `viz/{mfcc,stft,cqt}/`

PCA is **not** part of the trained classifier above — in the notebook it is
used only to visualize the feature space (cell 14). These files reproduce
that cell's procedure exactly, once, offline, against the already-committed
`saved_data/{feat}_X_{train,test}.npy` / `_y_{train,test}.npy` arrays (the
same arrays the notebook itself saved during training — no raw audio or
notebook re-run involved):

- `scaler_viz.pkl` — `StandardScaler` fit on `vstack(X_train, X_test)`
- `pca_viz.pkl` — `PCA(n_components=2)` fit on the scaled features
- `svm_viz.pkl` — a **separate** `SVC(kernel="rbf", C=10, gamma="scale")` fit
  only on the 2D PCA coordinates, used solely to render decision regions.
  **This is not the real classifier** — see cell 14, which does exactly this
  to draw its decision-region plot. The real prediction always comes from
  `{feat}_svm.pkl` above.
- `grid.npz` (`xx`, `yy`, `Z`) — the 300×300 meshgrid and `svm_viz` predictions
  over it, for the decision-region contour.
- `pc_train.npy` / `pc_test.npy` / `y_train.npy` / `y_test.npy` — the training
  and test set's 2D PCA coordinates and labels, for the background scatter.

Regenerate all of the above (idempotent, deterministic) with:

```
python scripts/build_viz_artifacts.py
```

## Known caveat

The pickled pipelines were originally saved with scikit-learn 1.8.0; this
repo's environment during artifact generation had scikit-learn 1.7.1
installed, which emits an `InconsistentVersionWarning` on load. This predates
this deployment work — pin `scikit-learn==1.8.0` in your environment if you
want to silence it, but predictions are unaffected.
