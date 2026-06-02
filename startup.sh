#!/bin/bash
set -e

if [ "${SKIP_GCS_DOWNLOAD}" != "true" ]; then
  echo "[Startup] Authenticating gcloud..."
  
  # Securely fetch the ambient Cloud Run token using Python's built-in JSON parser
  export CLOUDSDK_AUTH_ACCESS_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
  
  echo "[Startup] Downloading data from GCS bucket: ${GCS_BUCKET}"
  mkdir -p data/processed models/saved
  
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