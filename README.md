# String Instrument Classifier

Predominant string-instrument recognition from a short audio clip —
**cello, acoustic guitar, electric guitar, or violin** — served as a
single Streamlit app.

The modelling methodology (preprocessing, feature extraction, PCA, SVM)
is developed in `new_methods.ipynb` on the
[IRMAS](https://www.upf.edu/web/mtg/irmas) dataset. The app is a faithful
deployment of that pipeline: it loads the already-fitted scaler + SVM and
only ever calls `.predict()` — nothing is retrained, no scaler or PCA is
refit on uploaded audio.

## Problem

IRMAS clips are 3-second excerpts of real recordings labelled by the
*predominant* instrument. The task: take one new clip and predict which
of the four string instruments dominates.

## Method

Each clip is reduced to a single feature vector:

1. **Preprocessing** — `librosa.load(sr=22050)`, mono, no trimming/padding/normalisation.
2. **Summary statistics (5, always)** — mean over frames of zero-crossing rate,
   RMS energy, spectral centroid, spectral roll-off (85th percentile), spectral bandwidth.
3. **One time-frequency transform**, selectable:
   - **MFCC** — 13 coefficients (`n_fft=2048`, `hop=512`), averaged over time
   - **STFT** — 1025-bin magnitude spectrum, averaged over time
   - **CQT** — 84-bin constant-Q magnitude, averaged over time
4. **Classifier** — scikit-learn `Pipeline(StandardScaler → SVC(kernel="rbf", C=10, gamma="scale"))`,
   fitted once on the IRMAS training split and loaded from `models/instrument_classifier/`.

PCA is **not** part of the classifier. A separate `StandardScaler` +
`PCA(n_components=2)` were fit offline on the notebook's saved train/test
feature arrays purely to visualise the feature space in 2-D; a further
2-D SVM on those PCA coordinates draws the illustrative decision regions.

## Input / Output

- **Input** — one audio file (`wav/mp3/flac/ogg/m4a/aac/aiff`).
- **Output** — the predicted instrument, feature-extraction visualisations
  (waveform + spectral features, mel spectrogram, MFCC, mel filter bank),
  and the 2-D PCA projection with the uploaded clip plotted in it.

## Results

Accuracy on the IRMAS held-out test split (`all_results.csv`):

| Feature set | Test accuracy | Test F1 |
|---|---:|---:|
| MFCC | 0.786 | 0.784 |
| STFT | 0.774 | 0.768 |
| CQT  | 0.755 | 0.749 |

Per-instrument and pairwise-separability breakdowns for the MFCC model
are in `mfcc_per_instrument_metrics.csv` and `mfcc_pairwise_accuracy.csv`
(also shown in the app's *Model performance* tab).

## Tech stack

Python · Streamlit · librosa · scikit-learn · matplotlib · NumPy

## Project structure

```
app.py                        Streamlit UI — orchestration and rendering only
instrument_classifier/        deployment package (no .fit() anywhere)
├── audio/loader.py           decode arbitrary audio -> mono 22.05 kHz signal
├── features/extract.py       1:1 port of the notebook's feature functions
├── inference/pipeline.py     bytes -> features -> pipeline.predict()
├── models/registry.py        loads the pre-fitted SVM pipelines + label map
├── pca/viz.py                projects a vector into the precomputed 2-D PCA space
├── visualization/plots.py    matplotlib figure builders (one per notebook plot)
└── config.py                 constants transcribed from new_methods.ipynb
models/instrument_classifier/ deployed artefacts: {feat}_svm.pkl, labels.json, viz/
saved_models/ saved_data/     the notebook's own fitted models and feature arrays
scripts/build_viz_artifacts.py  regenerates models/…/viz/* from saved_data/
tests/                        provenance, no-refit, audio-format, feature, PCA-consistency
new_methods.ipynb             source-of-truth methodology
main.ipynb                    earlier feature-extraction / experiment notebook
figures/                      exported figures from the notebooks
```

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py          # opens http://localhost:8501
```

Pick a feature set in the sidebar, upload an audio clip, and read the
prediction and visualisations.

Run the tests:

```bash
pip install pytest
pytest tests
```

If the notebook model is ever retrained, regenerate the deployed viz
artefacts with `python scripts/build_viz_artifacts.py`.

## Example usage (programmatic)

```python
from instrument_classifier.inference.pipeline import analyze

result = analyze(open("clip.wav", "rb").read(), feature_set="mfcc")
print(result.predicted_class)   # e.g. "Violin"
```

## Limitations

- IRMAS labels the *predominant* instrument only; dense polyphonic mixes
  are out of distribution.
- Only four string instruments are modelled — anything else (piano,
  voice, brass, a second guitar type) will still be forced into one of them.
- Very short clips give unstable frame-averaged features.
- The PCA panel is a 2-D visual approximation, not the classifier's real
  (18–1030-dimensional) decision boundary.

## Future improvements

- Multi-label output for polyphonic clips.
- Calibrated probabilities / abstention when confidence is low.
- Broaden beyond the four IRMAS string classes.

## Data and licensing

IRMAS has its own [conditions of use](https://www.upf.edu/web/mtg/irmas)
(non-commercial, attribution, no modified redistribution). The `dataset/`
and `features/` directories are intentionally not version-controlled;
add the data locally and run the notebooks to rebuild features. This
project's code is separate from the dataset.
