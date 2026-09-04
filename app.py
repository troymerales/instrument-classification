"""Streamlit front end for the string-instrument classifier.

All modelling logic lives in the framework-independent
``instrument_classifier`` package and in the pre-fitted artefacts under
``models/`` — this file only handles UI, orchestration and rendering.
Nothing here ever calls ``.fit()``. Run with::

    streamlit run app.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from instrument_classifier.audio.loader import SUPPORTED_EXTENSIONS, UnsupportedAudioError
from instrument_classifier.config import CLASS_NAMES, FEATURE_SETS
from instrument_classifier.inference.pipeline import analyze
from instrument_classifier.pca.viz import load_viz_artifacts
from instrument_classifier.visualization.plots import (
    plot_mel_filterbank,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_pca_decision_regions,
    plot_waveform_with_features,
)

st.set_page_config(page_title="String Instrument Classifier", page_icon="🎻", layout="wide")

ROOT = Path(__file__).parent
FEATURE_LABELS = {"mfcc": "MFCC (13 coeff.)", "stft": "STFT magnitude (1025 bins)", "cqt": "CQT magnitude (84 bins)"}


@st.cache_data(show_spinner=False)
def _load_csv(name: str) -> pd.DataFrame | None:
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading audio and running the classifier…")
def _analyze(data: bytes, feature_set: str):
    result = analyze(data, feature_set)
    artifacts = load_viz_artifacts(feature_set)
    figs = {
        "waveform": plot_waveform_with_features(result.y, result.sr),
        "mel": plot_mel_spectrogram(result.y, result.sr),
        "mfcc": plot_mfcc(result.y, result.sr),
        "filterbank": plot_mel_filterbank(),
        "pca": plot_pca_decision_regions(artifacts, result.pc1, result.pc2, result.predicted_index),
    }
    return result, figs


def _png(b64: str):
    return base64.b64decode(b64)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🎻 Instrument Classifier")
st.sidebar.caption("Predominant string-instrument recognition from a short audio clip.")

feature_set = st.sidebar.selectbox(
    "Feature set", list(FEATURE_SETS), format_func=lambda k: FEATURE_LABELS.get(k, k)
)

exts = sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)
upload = st.sidebar.file_uploader("Audio clip", type=exts)

samples = sorted(ROOT.glob("sample_audio/*"))
sample_choice = None
if samples:
    sample_choice = st.sidebar.selectbox(
        "…or a bundled sample", ["—"] + [p.name for p in samples]
    )

audio_bytes: bytes | None = None
audio_name: str | None = None
if upload is not None:
    audio_bytes, audio_name = upload.getvalue(), upload.name
elif sample_choice and sample_choice != "—":
    p = ROOT / "sample_audio" / sample_choice
    audio_bytes, audio_name = p.read_bytes(), sample_choice

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("String Instrument Classifier")
st.markdown(
    "Faithful deployment of the `new_methods.ipynb` pipeline: **summary spectral "
    "statistics + one time-frequency transform → StandardScaler → SVM (RBF)**, "
    "trained on the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset. "
    "The scaler and SVM are loaded pre-fitted and only ever `.predict()`-ed — "
    "nothing is retrained on your audio."
)

with st.expander("Problem, method, and how to read the results", expanded=audio_bytes is None):
    st.markdown(
        """
**Problem** — given a short musical clip, identify the *predominant* string
instrument: **cello, acoustic guitar, electric guitar, or violin**.

**Method** — each clip is reduced to one feature vector: 5 summary statistics
(zero-crossing rate, RMS energy, spectral centroid, roll-off, bandwidth) plus
the time-averaged MFCC / STFT / CQT you pick in the sidebar. A scikit-learn
`Pipeline(StandardScaler → SVC(kernel="rbf", C=10))` fitted on IRMAS makes the
prediction.

**Input** — one audio file (`wav/mp3/flac/ogg/m4a/aac/aiff`), mono, resampled
to 22 050 Hz. No trimming, padding or normalisation.

**Output** — the predicted instrument, the feature-extraction visualisations,
and a 2-D PCA projection of the classifier's feature space with your clip
plotted in it.

**Evaluation** — see the *Model performance* tab: held-out accuracy is
~0.79 (MFCC), ~0.77 (STFT), ~0.75 (CQT) on the IRMAS test split.

**Limitations** — IRMAS labels the *predominant* instrument only; polyphonic
mixes, instruments outside the four classes, and very short clips are all
out of distribution. The PCA panel is a visualisation aid, not the real
decision boundary (which lives in 18–1030 dimensions).
        """
    )

if audio_bytes is None:
    st.info("Upload an audio clip or pick a bundled sample from the sidebar.")
    perf = _load_csv("all_results.csv")
    if perf is not None:
        st.subheader("Model performance (IRMAS test split)")
        st.dataframe(perf, use_container_width=True, hide_index=True)
    st.stop()

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
try:
    result, figs = _analyze(audio_bytes, feature_set)
except UnsupportedAudioError as exc:
    st.error(f"Could not decode this audio: {exc}")
    st.stop()
except FileNotFoundError as exc:
    st.error(f"A model artefact is missing: {exc}. Check the `models/` directory.")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unexpected error: {exc}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Prediction")
    st.metric("Predicted instrument", result.predicted_class)
    st.caption(f"Feature set: {FEATURE_LABELS.get(feature_set, feature_set)} · file: {audio_name}")
    st.audio(audio_bytes)
with right:
    st.subheader("Classifier feature space (2-D PCA)")
    st.image(_png(figs["pca"]), use_container_width=True)
    st.caption(
        "Background regions come from a separate 2-D SVM fit on PCA coordinates — "
        "an approximation for visualisation, not the real boundary. ★ is your clip."
    )

tab_feat, tab_perf = st.tabs(["Feature extraction", "Model performance"])

with tab_feat:
    st.image(_png(figs["waveform"]), use_container_width=True,
             caption="Waveform with normalised ZCR, RMS, spectral centroid, bandwidth and roll-off.")
    c1, c2 = st.columns(2)
    c1.image(_png(figs["mel"]), use_container_width=True, caption="Mel spectrogram (dB).")
    c2.image(_png(figs["mfcc"]), use_container_width=True, caption="MFCC (n_mfcc=13) — the model's input parameters.")
    with st.expander("Mel filter bank (fixed diagram)"):
        st.image(_png(figs["filterbank"]), use_container_width=True)

with tab_perf:
    overall = _load_csv("all_results.csv")
    if overall is not None:
        st.markdown("**Overall accuracy by feature set (IRMAS train/test split)**")
        st.dataframe(overall, use_container_width=True, hide_index=True)
    per_inst = _load_csv("mfcc_per_instrument_metrics.csv")
    if per_inst is not None:
        st.markdown("**Per-instrument metrics (MFCC model)**")
        st.dataframe(per_inst, use_container_width=True, hide_index=True)
    pairwise = _load_csv("mfcc_pairwise_accuracy.csv")
    if pairwise is not None:
        st.markdown("**Pairwise separability (MFCC model)**")
        st.dataframe(pairwise, use_container_width=True, hide_index=True)
    st.caption(f"Classes: {', '.join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES))}")
