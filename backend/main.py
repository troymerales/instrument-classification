from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend" / "static"

app = FastAPI(title="Instrument Classifier")

app.include_router(router)

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
