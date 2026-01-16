import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def compute_cqt(filename, B, fmin, hop_length, plot,title):
    x, fs = librosa.load(filename, sr=None)
    X_cq = librosa.cqt(
        x,
        sr=fs,
        hop_length=hop_length,
        fmin=fmin,
        bins_per_octave=B
    )
    X_cq_db = librosa.amplitude_to_db(np.abs(X_cq), ref=np.max)
    X_mag = np.abs(X_cq)
    features = np.mean(X_mag, axis=1)
    features = np.log1p(features)
    
    if plot:
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
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()
    return features

# Parameters (same as before)
B = 12
fmin = 32.7
hop_length = 512

# File paths
guitar_files = [
    'dataset/openmic-2018/audio/000/000189_207360.ogg',
    'dataset/openmic-2018/audio/000/000914_69120.ogg'
]

ukulele_files = [
    'dataset/openmic-2018/audio/000/000144_30720.ogg',
    'dataset/openmic-2018/audio/000/000252_42240.ogg'
]

# Extract features
X = []
y = []

for f in guitar_files:
    X.append(compute_cqt(f, B, fmin, hop_length,plot=False,title=""))
    y.append(0)

for f in ukulele_files:
    X.append(compute_cqt(f, B, fmin, hop_length,plot=False,title=""))
    y.append(1)

X = np.array(X)
y = np.array(y)

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# SVM pipeline
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Train
svm.fit(X_train, y_train)

# Evaluate
accuracy = svm.score(X_test, y_test)
print("Test accuracy:", accuracy)
print("Feature shape:", X.shape)
print("Labels:", y)
print("Feature variance:", np.var(X, axis=0).mean())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_pca = PCA(n_components=2).fit_transform(X)

plt.scatter(X_pca[:2,0], X_pca[:2,1], label="Guitar")
plt.scatter(X_pca[2:,0], X_pca[2:,1], label="Ukulele")
plt.legend()
plt.title("PCA of CQT Features")
plt.show()

for k, val in enumerate(X[0]):
    print(f"Bin {k:02d}: {val:.4f}")
