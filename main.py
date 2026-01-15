import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

B = 12                     # bins per octave
fmin = 32.7                # minimum frequency (C1)
hop_length = 512           # time resolution

filename = "dataset/openmic-2018/audio/000/000189_207360.ogg"  
x, fs = librosa.load(filename, sr=None)

X_cq = librosa.cqt(
    x,
    sr=fs,
    hop_length=hop_length,
    fmin=fmin,
    bins_per_octave=B
)
X_cq_db = librosa.amplitude_to_db(np.abs(X_cq), ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(
    X_cq_db,
    sr=fs,
    hop_length=hop_length,
    x_axis="time",
    y_axis="cqt_hz",
    fmin=fmin,
    bins_per_octave=B
)
plt.colorbar(format="%+2.0f dB")
plt.title("CQT of Guitar Sample 1")
plt.tight_layout()
plt.show()
plt.savefig('pr1.png', dpi=300, bbox_inches='tight')
plt.close()

filename2 = "dataset/openmic-2018/audio/000/000403_0.ogg"  
x, fs = librosa.load(filename2, sr=None)

X_cq = librosa.cqt(
    x,
    sr=fs,
    hop_length=hop_length,
    fmin=fmin,
    bins_per_octave=B
)
X_cq_db = librosa.amplitude_to_db(np.abs(X_cq), ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(
    X_cq_db,
    sr=fs,
    hop_length=hop_length,
    x_axis="time",
    y_axis="cqt_hz",
    fmin=fmin,
    bins_per_octave=B
)
plt.colorbar(format="%+2.0f dB")
plt.title("CQT of Guitar Sample 2")
plt.tight_layout()
plt.show()
plt.savefig('pr2.png', dpi=300, bbox_inches='tight')
plt.close()
