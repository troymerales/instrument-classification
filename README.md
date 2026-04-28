# Instrument Classifier

Jupyter notebooks for **predominant instrument recognition** in musical audio, using the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset and features extracted with **librosa** (e.g. CQT, MFCC), **PCA**, and **scikit-learn** for modeling.

## What’s in this repository

- Notebooks such as `main.ipynb`, `flattened.ipynb`, and `full.ipynb` for feature extraction, flattening, and experiments.

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
