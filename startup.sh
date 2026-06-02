#!/bin/bash
set -e

if [ "${SKIP_GCS_DOWNLOAD}" != "true" ]; then
  echo "[Startup] Downloading data from GCS bucket: ${GCS_BUCKET}"
  
  # 1. Ensure target directories exist before downloading to prevent "Not Found" crashes
  mkdir -p data/processed models/saved
  
  # 2. Use modern 'gcloud storage' instead of 'gsutil' to natively handle Cloud Run authentication and fast multithreading
  gcloud storage cp -r gs://${GCS_BUCKET}/chroma_db data/
  gcloud storage cp -r gs://${GCS_BUCKET}/processed data/
  gcloud storage cp -r gs://${GCS_BUCKET}/models/saved models/
  gcloud storage cp gs://${GCS_BUCKET}/rights_guide.json data/
  gcloud storage cp gs://${GCS_BUCKET}/legal_graph.json data/
  
  echo "[Startup] Data download complete."
else
  echo "[Startup] Skipping GCS download (local dev mode)"
fi

exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1