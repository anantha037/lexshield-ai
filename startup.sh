#!/bin/bash
set -e

if [ "${SKIP_GCS_DOWNLOAD}" != "true" ]; then
  echo "[Startup] Downloading data from GCS bucket: ${GCS_BUCKET}"
  gsutil -m cp -r gs://${GCS_BUCKET}/chroma_db data/
  gsutil -m cp -r gs://${GCS_BUCKET}/processed data/
  gsutil -m cp -r gs://${GCS_BUCKET}/models/saved models/
  gsutil cp gs://${GCS_BUCKET}/rights_guide.json data/
  gsutil cp gs://${GCS_BUCKET}/legal_graph.json data/
  echo "[Startup] Data download complete."
else
  echo "[Startup] Skipping GCS download (local dev mode)"
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1