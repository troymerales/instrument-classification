"""
Matplotlib figure builders, one per notebook visualization actually present in
``new_methods.ipynb``. Every plot here corresponds to a specific cell; no plot
type is added that the notebook doesn't already produce. All figures are
rendered server-side (Agg backend) to PNG bytes for the frontend to embed.
"""

import base64
import io

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from instrument_classifier.config import CLASS_NAMES, HOP_LENGTH, N_FFT, N_MFCC, SR, VIZ_N_MELS  # noqa: E402
from instrument_classifier.pca.viz import VizArtifacts  # noqa: E402


def _normalize(x: np.ndarray) -> np.ndarray:
    """Matches the notebook's `normalize` helper in cell 6."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def _fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_waveform_with_features(y: np.ndarray, sr: int) -> str:
    """Cell 7, plot 1: waveform + ZCR/RMS/centroid/bandwidth/rolloff overlay."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    feature_times = librosa.times_like(zcr, sr=sr, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(11, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax)

    ax.plot(feature_times, _normalize(centroid), label="Spectral Centroid", linewidth=2)
    ax.plot(feature_times, _normalize(bandwidth), label="Spectral Bandwidth", linewidth=2)
    ax.plot(feature_times, _normalize(rolloff), label="Spectral Roll-Off", linewidth=2)
    ax.plot(feature_times, _normalize(zcr), label="Zero Crossing Rate", linewidth=2)
    ax.plot(feature_times, _normalize(rms), label="Root Mean Square Energy", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Waveform and Audio Features")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return _fig_to_base64_png(fig)


def plot_mel_spectrogram(y: np.ndarray, sr: int) -> str:
    """Cell 7, plot 2: mel spectrogram in dB (the notebook's only log spectrogram)."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=VIZ_N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Scale Frequency Spectrogram (log, dB)")
    fig.tight_layout()

    return _fig_to_base64_png(fig)


def plot_mfcc(y: np.ndarray, sr: int) -> str:
    """MFCC matrix using the PRODUCTION extractor parameters (n_mfcc=13,
    n_fft=2048, hop_length=512) -- the same parameters that feed the model,
    not the demo cell's separate n_mels=26 variant."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC")
    ax.set_xlabel("Time")
    fig.tight_layout()

    return _fig_to_base64_png(fig)


def plot_mel_filterbank() -> str:
    """Cell 7, plot 4: mel filter bank. Depends only on SR/N_FFT/VIZ_N_MELS, not
    on the uploaded audio -- a static methodology diagram."""
    mel_filters = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=VIZ_N_MELS)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(mel_filters.shape[0]):
        ax.plot(mel_filters[i])
    ax.set_title("Mel Filter Bank")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    return _fig_to_base64_png(fig)


def plot_pca_decision_regions(
    artifacts: VizArtifacts, pc1: float, pc2: float, predicted_index: int
) -> str:
    """Cell 14's PCA scatter + decision-region contour, precomputed offline,
    with the new audio's projected point overlaid as a distinct marker.

    This is an illustrative 2D projection of the classifier's feature space
    (built from a SEPARATE 2D SVM fit only on PCA coordinates for this plot).
    It is NOT the real classifier's decision boundary -- the real prediction
    comes from the full-dimensional pipeline, shown elsewhere.
    """
    class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    contour = ax.contourf(artifacts.xx, artifacts.yy, artifacts.Z, alpha=0.3)

    pc_all = np.vstack([artifacts.pc_train, artifacts.pc_test])
    y_all = np.concatenate([artifacts.y_train, artifacts.y_test])

    for cls in np.unique(y_all):
        idx = y_all == cls
        ax.scatter(pc_all[idx, 0], pc_all[idx, 1], label=class_names[cls], alpha=0.6, s=25)

    ax.scatter(
        [pc1], [pc2],
        marker="*", s=400, c="black", edgecolors="white", linewidths=1.2,
        label=f"New Audio ({class_names[predicted_index]})", zorder=5,
    )

    region_colors = contour.cmap(contour.norm(np.unique(artifacts.Z)))
    region_patches = [
        Patch(color=region_colors[i], alpha=0.3, label=f"{class_names[i]} Region")
        for i in range(len(class_names))
    ]

    point_legend = ax.legend(loc="upper left", title="Samples", fontsize=8)
    ax.add_artist(point_legend)
    ax.legend(handles=region_patches, loc="upper right", title="Decision Regions", fontsize=8)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Projection — 2D Decision Region Approximation")
    ax.grid(True)
    fig.tight_layout()

    return _fig_to_base64_png(fig)
