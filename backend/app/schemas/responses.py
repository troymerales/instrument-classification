from pydantic import BaseModel


class ModelsResponse(BaseModel):
    feature_sets: list[str]
    class_names: dict[int, str]


class AnalyzeResponse(BaseModel):
    feature_set: str
    predicted_index: int
    predicted_class: str
    pc1: float
    pc2: float
    waveform_features_png: str
    mel_spectrogram_png: str
    mfcc_png: str
    mel_filterbank_png: str
    pca_decision_regions_png: str
