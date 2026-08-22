# Instrument Classifier

Jupyter notebooks for **predominant instrument recognition** in musical audio, using the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset and features extracted with **librosa** (e.g. CQT, MFCC), **PCA**, and **scikit-learn** for modeling.

## What’s in this repository

- Notebooks such as `main.ipynb`, `flattened.ipynb`, and `full.ipynb` for feature extraction, flattening, and experiments.
- `new_methods.ipynb` — the **source-of-truth methodology** (preprocessing, feature extraction, PCA, SVM) for the deployed app below.
- `backend/`, `frontend/`, `scripts/`, `models/instrument_classifier/` — a FastAPI + vanilla-JS app that deploys `new_methods.ipynb`'s existing trained pipeline for inference on new audio. See "Running the app" below.

## Running the app

The app reuses the already-fitted classifier pipelines in `saved_models/` — it
never retrains, refits a scaler, or refits PCA. See
`models/instrument_classifier/README.md` for exactly which artifacts are used
and where they come from.

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000, upload an audio file (wav/mp3/flac/ogg/m4a/aac/aiff),
choose a feature set (MFCC/STFT/CQT), and click "Analyze Audio".

Run the test suite:

```bash
pytest backend/tests
```

Or with Docker:

```bash
docker build -t instrument-classifier .
docker run -p 8000:8000 instrument-classifier
```

If the underlying dataset/model is ever retrained, regenerate the deployed
artifacts with `python scripts/build_viz_artifacts.py` (see that script and
`models/instrument_classifier/README.md` for what it does and why).

## What is not in version control (by choice)

The **`features/`** and **`dataset/`** directories are **not pushed** to git: they are too large for a typical remote, so they stay only on the machine where you generate or download them.

- **`dataset/`** — place the IRMAS audio data here (or point your notebooks at the path you use). See the official IRMAS page for download and license terms.
- **`features/`** — holds extracted feature files (e.g. raw, flattened, PCA outputs) produced by the notebooks; regenerate locally after you have the data.

If you clone this repo, add the dataset and run the notebooks to rebuild features as needed.

If you store outputs under other directory names (for example `features_cqt`, `features_mfcc`, or `dataset_full`), add those paths to `.gitignore` as well if you do not want them in git.

## Requirements

Use a Python environment with at least **NumPy**, **librosa**, **scikit-learn**, and a Jupyter runtime. Exact versions are not pinned here; match them to your existing setup or export a `requirements.txt` when you lock versions.

## License and data

IRMAS has its own [conditions of use](https://www.upf.edu/web/mtg/irmas) (non-commercial, attribution, no redistribution in modified form, etc.). This project code is separate from the dataset; comply with IRMAS terms when using their files.
