const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileNameEl = document.getElementById("fileName");
const featureSetEl = document.getElementById("featureSet");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorBox = document.getElementById("errorBox");
const results = document.getElementById("results");
const player = document.getElementById("player");

let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    setFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    setFile(fileInput.files[0]);
  }
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  analyzeBtn.disabled = false;
  player.src = URL.createObjectURL(file);
  hideError();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
}

function hideError() {
  errorBox.style.display = "none";
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("feature_set", featureSetEl.value);

    const resp = await fetch("/api/analyze", { method: "POST", body: formData });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }

    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Audio";
  }
});

function renderResults(data) {
  document.getElementById("predictedClass").textContent = data.predicted_class;
  document.getElementById("pcCoords").textContent =
    `PC1 = ${data.pc1.toFixed(3)}   PC2 = ${data.pc2.toFixed(3)}   (feature set: ${data.feature_set.toUpperCase()})`;

  document.getElementById("imgWaveform").src = `data:image/png;base64,${data.waveform_features_png}`;
  document.getElementById("imgMel").src = `data:image/png;base64,${data.mel_spectrogram_png}`;
  document.getElementById("imgMfcc").src = `data:image/png;base64,${data.mfcc_png}`;
  document.getElementById("imgFilterbank").src = `data:image/png;base64,${data.mel_filterbank_png}`;
  document.getElementById("imgPca").src = `data:image/png;base64,${data.pca_decision_regions_png}`;

  results.style.display = "block";
  results.scrollIntoView({ behavior: "smooth" });
}
