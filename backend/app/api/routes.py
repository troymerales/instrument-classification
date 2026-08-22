from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.audio.loader import UnsupportedAudioError
from backend.app.config import FEATURE_SETS
from backend.app.inference.pipeline import analyze
from backend.app.models.registry import get_labels
from backend.app.pca.viz import load_viz_artifacts
from backend.app.schemas.responses import AnalyzeResponse, ModelsResponse
from backend.app.visualization.plots import (
    plot_mel_filterbank,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_pca_decision_regions,
    plot_waveform_with_features,
)

router = APIRouter(prefix="/api")


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(feature_sets=list(FEATURE_SETS), class_names=get_labels())


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    feature_set: str = Form("mfcc"),
) -> AnalyzeResponse:
    if feature_set not in FEATURE_SETS:
        raise HTTPException(status_code=422, detail=f"feature_set must be one of {FEATURE_SETS}")

    file_bytes = await file.read()

    try:
        result = analyze(file_bytes, feature_set)
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    artifacts = load_viz_artifacts(feature_set)

    return AnalyzeResponse(
        feature_set=result.feature_set,
        predicted_index=result.predicted_index,
        predicted_class=result.predicted_class,
        pc1=result.pc1,
        pc2=result.pc2,
        waveform_features_png=plot_waveform_with_features(result.y, result.sr),
        mel_spectrogram_png=plot_mel_spectrogram(result.y, result.sr),
        mfcc_png=plot_mfcc(result.y, result.sr),
        mel_filterbank_png=plot_mel_filterbank(),
        pca_decision_regions_png=plot_pca_decision_regions(
            artifacts, result.pc1, result.pc2, result.predicted_index
        ),
    )
