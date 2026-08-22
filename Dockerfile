FROM python:3.11-slim

# ffmpeg backs librosa/audioread for mp3/m4a/aac decoding; libsndfile1 backs
# soundfile for wav/flac/ogg/aiff.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY models/instrument_classifier/ models/instrument_classifier/

ENV MODEL_DIR=/srv/models/instrument_classifier
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
